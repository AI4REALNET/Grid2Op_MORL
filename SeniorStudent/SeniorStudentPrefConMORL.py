"""
In this file, a multi-process training for PPO model is designed.
training process:
    The environment steps “do nothing” action (except reconnection of lines)
    until encountering a dangerous scenario, then its observation is sent to
    the Senior Student to get a “do something” action. After stepping this
    action, the reward is calculated and fed back to the Senior Student for
    network updating.

author: chen binbin
mail: cbb@cbb1996.com
"""
import os
import time
import grid2op
import numpy as np
from PPO import PPO
import tensorflow as tf
from PPO_Reward import PPO_Reward
from multiprocessing import cpu_count
from grid2op.Environment import SingleEnvMultiProcess
import importlib.util

# Path to the file one level up
PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MORL_OBJ_PATH = os.path.join(PARENT, "morl_objectives.py")

spec = importlib.util.spec_from_file_location("morl_objectives", MORL_OBJ_PATH)
morl_objectives = importlib.util.module_from_spec(spec)
spec.loader.exec_module(morl_objectives)



# Now import functions from that module
build_morl_params_from_dataset = morl_objectives.build_morl_params_from_dataset
compute_morl_metrics = morl_objectives.compute_morl_metrics
build_gated_scalar_reward = morl_objectives.build_gated_scalar_reward
scalarize_with_preferences = morl_objectives.scalarize_with_preferences

import sys
# Optional: wandb logging
try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False

# ---- cluster time cap (shared with orchestrator via ORCH_START_TIME) ----
MAX_RUNTIME_SECONDS = ((12)-0.5) * 3600  # 23h30

_orch_start = os.environ.get("ORCH_START_TIME")
try:
    # If orchestrate_training started us, use its start time
    start_time_global = float(_orch_start) if _orch_start is not None else time.time()
except ValueError:
    # Fallback if env var is corrupted or non-numeric
    start_time_global = time.time()


def curriculum_schedule(epoch: int):
    """
    Simple curriculum for SeniorStudentPrefConMORL:
    - Early: shorter chronics + fewer env steps per epoch
    - Later: gradually increase both to train long-term behaviour.
    Returns (num_env_steps_each_epoch, max_iter_or_None).
    """
    if epoch < 10:
        # Very short episodes, fast feedback
        return 5000, 288   # ~1 day at 5-min resolution, adjust if needed
    elif epoch < 30:
        # Medium episodes
        return 10000, 576  # ~2 days
    else:
        # Full episodes
        return 20000, None  # None = don't override max_iter


class Run_env(object):
    def __init__(self, envs, agent, n_steps=2000, n_cores=12, gamma=0.99, lam=0.95, action_space_path='../', morl_params=None,
        morl_log_interval=1000,):
        self.envs = envs
        self.agent = agent
        self.n_steps = n_steps
        self.n_cores = n_cores
        self.gamma = gamma
        self.lam = lam
        self.chosen = list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
        self.chosen += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
        self.chosen += list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(range(1164, 1223))
        self.chosen = np.asarray(self.chosen, dtype=np.int32) - 1  # (1221,)
        self.actions62 = np.load(os.path.join(action_space_path, 'actions62.npy'))
        self.actions146 = np.load(os.path.join(action_space_path, 'actions146.npy'))
        self.actions = np.concatenate((self.actions62, self.actions146), axis=0)
        self.batch_reward_records = []
        self.aspace = self.envs.action_space[0]
        self.rec_rewards = []
        self.worker_alive_steps = np.zeros(self.n_cores)
        self.alive_steps_record = []
        self.global_step = 0  # for progress / wandb logging
        # --- NEW: MORL logging state ---
        self.morl_params = morl_params  # None = disabled
        self.morl_log_interval = morl_log_interval
        self.morl_metrics_buffer = []  # list of per-step dicts
        self.morl_step_counter = 0
        self.pref_dim = 4
        self.pref_vectors = np.zeros((n_cores, self.pref_dim), dtype=np.float32)
        self._resample_prefs(np.arange(self.n_cores))
        # --- NEW: chronic / termination stats ---
        self.chronic_stats = {
            "total": 0,
            "completed": 0,
            "game_over": 0,
            "thermal": 0,
            "overload": 0,
            "islanding": 0,
            "divergence": 0,
            "other_fail": 0,
        }

    def _resample_prefs(self, env_ids):
        """
        Sample new preference vectors for the given env ids.
        Survival always gets a strong minimum share.
        """
        min_surv = 0.4  # ensure survival is always important
        for i in env_ids:
            base = np.random.dirichlet(np.array([4.0, 1.0, 1.0, 1.0]))
            # Enforce survival >= min_surv
            if base[0] < min_surv:
                deficit = min_surv - base[0]
                rest = base[1:].sum()
                if rest > 0:
                    base[1:] = base[1:] * (1.0 - min_surv) / rest
                base[0] = min_surv
            self.pref_vectors[i] = base.astype(np.float32)

    def run_n_steps(self, n_steps=None):
        def swap_and_flatten(arr):
            shape = arr.shape
            return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

        self.n_steps = n_steps if n_steps is not None else self.n_steps
        # mb for mini-batch
        mb_obs, mb_rewards, mb_actions = [[] for _ in range(self.n_cores)], [[] for _ in range(self.n_cores)], [[] for _ in range(self.n_cores)]
        # For progress logging
        step_rewards = []
        mb_values, mb_dones, mb_neg_log_p = [[] for _ in range(self.n_cores)], [[] for _ in range(self.n_cores)], [[] for _ in range(self.n_cores)]

        # start sampling
        obs_objs = self.envs.get_obs()
        obss = np.asarray([obs.to_vect()[self.chosen] for obs in obs_objs])  # (12, 1221,)
        # NEW: inject preferences into last 4 dims
        obs_aug = obss.copy()
        obs_aug[:, -self.pref_dim:] = self.pref_vectors  # (self.n_cores, 4)

        dones = np.asarray([False for _ in range(self.n_cores)])  # (12,)
        agent_step_rs = np.asarray([0 for _ in range(self.n_cores)], dtype=np.float64)  # (12,)
        for _ in range(self.n_steps):
            self.worker_alive_steps += 1
            actions = np.asarray([None for _ in range(self.n_cores)])
            values = np.asarray([None for _ in range(self.n_cores)])
            neg_log_ps = np.asarray([None for _ in range(self.n_cores)])

            for id in range(self.n_cores):
                if obs_aug[id, 654:713].max() >= ACTION_THRESHOLD:
                    actions[id], values[id], neg_log_ps[id], _ = map(
                        lambda x: x._numpy(), self.agent.step(tf.constant(obs_aug[[id], :]))
                    )
                    if dones[id] == False and len(mb_obs[id]) > 0:
                        mb_rewards[id].append(agent_step_rs[id])
                    agent_step_rs[id] = 0
                    mb_obs[id].append(obs_aug[[id], :])
                    mb_dones[id].append(dones[[id]])
                    dones[id] = False
                    mb_actions[id].append(actions[id])
                    mb_values[id].append(values[id])
                    mb_neg_log_p[id].append(neg_log_ps[id])
                else:
                    pass

            actions_array = []
            for idx, act_id in enumerate(actions):
                if act_id is not None:
                    # Use the selected high-level action
                    actions_array.append(
                        self.array2action(self.actions[act_id][0])
                    )
                else:
                    # Do-nothing action + reconnect logic
                    reconnect_arr = Run_env.reconnect_array(obs_objs[idx])
                    empty = np.zeros(self.actions.shape[1], dtype=np.float32)
                    actions_array.append(self.array2action(empty, reconnect_arr))

            # Keep a copy of previous observations for MORL metrics
            prev_obs_objs = obs_objs

            # Step all environments
            obs_objs, rs, env_dones, infos = self.envs.step(actions_array)
            obs_aug = np.asarray([obs.to_vect()[self.chosen] for obs in obs_objs])
            # NEW: re-inject preferences every step
            obs_aug = obs_aug.copy()
            obs_aug[:, -self.pref_dim:] = self.pref_vectors
            # Effective reward that will actually be used for PPO.
            # By default this is the original env reward.
            effective_rs = np.asarray(rs, dtype=np.float64)

            # --- NEW: MORL metrics accumulation + scalar reward ---
            if self.morl_params is not None:
                for idx in range(self.n_cores):
                    try:
                        metrics = compute_morl_metrics(
                            prev_obs_objs[idx],  # obs before step
                            obs_objs[idx],  # obs after step
                            actions_array[idx],  # grid2op Action
                            float(rs[idx]),  # original scalar env reward
                            bool(env_dones[idx]),
                            infos[idx],
                            self.morl_params,
                        )

                        # Build gated scalar reward from transformed metrics
                        scalar_info = scalarize_with_preferences(metrics, self.pref_vectors[idx])

                        # Add scalarization components into the metrics dict for logging
                        metrics.update(scalar_info)

                        # Use the gated scalar reward as the effective reward for PPO
                        effective_rs[idx] = scalar_info["scalar_reward"]

                        # Buffer metrics for periodic wandb logging
                        self.morl_metrics_buffer.append(metrics)
                        self.morl_step_counter += 1
                    except Exception as e:
                        # If anything fails, fall back to original reward for this env
                        # and skip metrics for that step.
                        # print(f"[MORL] metric computation error on env {idx}: {e}")
                        effective_rs[idx] = float(rs[idx])
                        continue

                # Log average metrics every `morl_log_interval` steps
                if USE_WANDB and self.morl_step_counter % self.morl_log_interval == 0:
                    if self.morl_metrics_buffer:
                        # Aggregate metrics in the buffer
                        sums = {}
                        n = len(self.morl_metrics_buffer)
                        for m in self.morl_metrics_buffer:
                            for k, v in m.items():
                                sums[k] = sums.get(k, 0.0) + float(v)
                        avg = {
                            f"morl/{k}_mean_{self.morl_log_interval}": val / n
                            for k, val in sums.items()
                        }
                        wandb.log(avg, step=self.global_step)
                    # Clear buffer after logging
                    self.morl_metrics_buffer.clear()

            # Progress logging: record mean reward and log every 100 env steps
            mean_r = float(np.mean(rs))
            step_rewards.append(mean_r)
            self.global_step += 1
            # --- NEW: handle terminations / chronic stats / final agent_step reward ---
            for id in range(self.n_cores):
                if env_dones[id]:
                    # Record how long this worker survived
                    self.alive_steps_record.append(self.worker_alive_steps[id])
                    self.worker_alive_steps[id] = 0

                    # Termination classification
                    info = infos[id]
                    exc = str(info.get("exception", "")) if info is not None else ""
                    self.chronic_stats["total"] += 1

                    # --- Termination classification (detailed) ---
                    info = infos[id]
                    self.chronic_stats["total"] += 1

                    # Normalize exception into a list
                    exc_obj = None
                    if info is not None:
                        exc_obj = info.get("exception", None)

                    if exc_obj is None:
                        exc_list = []
                    elif isinstance(exc_obj, (list, tuple)):
                        exc_list = list(exc_obj)
                    else:
                        exc_list = [exc_obj]

                    if exc_list:
                        # We had a real failure (game over)
                        self.chronic_stats["game_over"] += 1

                        # Combine all exception messages, lowercased
                        exc_str = " ".join(str(e) for e in exc_list).lower()

                        tagged = False

                        # 1) Islanding / isolated buses
                        if "islanding" in exc_str or "isolated bus" in exc_str:
                            self.chronic_stats["islanding"] += 1
                            tagged = True

                        # 2) Power-flow divergence / NR did not converge
                        if ("did not converge" in exc_str
                                or "power flow nr" in exc_str
                                or "powerflow diverged" in exc_str):
                            self.chronic_stats["divergence"] += 1
                            tagged = True

                        # 3) Thermal / overload / limit issues
                        if ("thermal" in exc_str
                                or "overflow" in exc_str
                                or "overload" in exc_str
                                or "rho" in exc_str
                                or "limit" in exc_str):
                            self.chronic_stats["overload"] += 1
                            tagged = True

                        # 4) Fallback bucket if nothing matched
                        if not tagged:
                            self.chronic_stats["other_fail"] += 1

                        term_bonus = -50.0
                    else:
                        # No exception → we truly completed the chronic
                        self.chronic_stats["completed"] += 1
                        term_bonus = +50.0

                    mb_rewards[id].append(agent_step_rs[id] + term_bonus)
                    agent_step_rs[id] = 0.0

                    # Mark rollout done flag for this trajectory
                    # (used later when building GAE targets)
                    dones[id] = True

                    # OPTIONAL: resample a new preference vector for this worker
                    self._resample_prefs([id])


            if USE_WANDB and (self.global_step % 100 == 0):
                # average over the last 100 (or fewer, at the very beginning) steps
                wandb.log({"sampling/mean_reward_100_steps": float(np.mean(step_rewards))},
                          step=self.global_step)
                step_rewards = []

            # --- NEW: log chronic stats periodically (scalars, no histograms) ---
            if USE_WANDB and (self.global_step % 1000 == 0):
                cs = self.chronic_stats
                wandb.log(
                    {
                        "chronic/total": cs["total"],
                        "chronic/completed": cs["completed"],
                        "chronic/game_over": cs["game_over"],
                        "chronic/thermal": cs["thermal"],
                        "chronic/islanding": cs["islanding"],
                        "chronic/overload": cs["overload"],
                        "chronic/other_fail": cs["other_fail"],
                    },
                    step=self.global_step,
                )

            # Accumulate MORL-based scalar reward across this "agent step"
            agent_step_rs += effective_rs

        # end sampling

        # batch to trajectory
        for id in range(self.n_cores):
            if mb_obs[id] == []:
                continue
            if dones[id]:
                mb_dones[id].append(np.asarray([True]))
                mb_values[id].append(np.asarray([0]))
            else:
                mb_obs[id].pop()
                mb_actions[id].pop()
                mb_neg_log_p[id].pop()
        obs2ret, done2ret, action2ret, value2ret, neglogp2ret, return2ret = ([] for _ in range(6))
        for id in range(self.n_cores):
            if mb_obs[id] == []:
                continue
            mb_obs_i = np.asarray(mb_obs[id], dtype=np.float32)
            mb_rewards_i = np.asarray(mb_rewards[id], dtype=np.float32)
            mb_actions_i = np.asarray(mb_actions[id], dtype=np.float32)
            mb_values_i = np.asarray(mb_values[id][:-1], dtype=np.float32)
            mb_neg_log_p_i = np.asarray(mb_neg_log_p[id], dtype=np.float32)
            mb_dones_i = np.asarray(mb_dones[id][:-1], dtype=bool)
            last_done = mb_dones[id][-1][0]
            last_value = mb_values[id][-1][0]

            # calculate R and A
            mb_advs_i = np.zeros_like(mb_values_i)
            last_gae_lam = 0
            for t in range(len(mb_obs[id]))[::-1]:
                if t == len(mb_obs[id]) - 1:
                    # last step
                    next_non_terminal = 1 - last_done
                    next_value = last_value
                else:
                    next_non_terminal = 1 - mb_dones_i[t + 1]
                    next_value = mb_values_i[t + 1]
                # calculate delta：r + gamma * v' - v
                delta = mb_rewards_i[t] + self.gamma * next_value * next_non_terminal - mb_values_i[t]
                mb_advs_i[t] = last_gae_lam = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            mb_returns_i = mb_advs_i + mb_values_i
            obs2ret.append(mb_obs_i)
            action2ret.append(mb_actions_i)
            value2ret.append(mb_values_i)
            done2ret.append(mb_dones_i)
            neglogp2ret.append(mb_neg_log_p_i)
            return2ret.append(mb_returns_i)
        obs2ret = np.concatenate(obs2ret, axis=0)
        action2ret = np.concatenate(action2ret, axis=0)
        value2ret = np.concatenate(value2ret, axis=0)
        done2ret = np.concatenate(done2ret, axis=0)
        neglogp2ret = np.concatenate(neglogp2ret, axis=0)
        return2ret = np.concatenate(return2ret, axis=0)
        self.rec_rewards.append(sum([sum(i) for i in mb_rewards]) / action2ret.shape[0])
        return *map(swap_and_flatten, (obs2ret, return2ret, done2ret, action2ret, value2ret, neglogp2ret)), (sum([sum(i) for i in mb_rewards]) / action2ret.shape[0])

    @staticmethod
    def reconnect_array(obs):
        """
        Build an integer status vector for set_line_status:
        0 = do nothing, 1 = reconnect this line.
        """
        # Make sure we create an *integer* array, as required by grid2op
        new_line_status_array = np.zeros_like(obs.line_status, dtype=int)

        # Find currently disconnected lines
        disconnected_lines = np.where(obs.line_status == False)[0]

        # Try to reconnect the most recently disconnected (reverse order)
        for line in disconnected_lines[::-1]:
            if not obs.time_before_cooldown_line[line]:
                new_line_status_array[line] = 1  # mark this line to reconnect
                break

        return new_line_status_array

    def array2action(self, total_array, reconnect_array=None):
        # Ensure we pass a boolean change_bus vector directly to the action constructor
        change_bus = total_array[236:413].astype(bool)
        action = self.aspace({'change_bus': change_bus})

        if reconnect_array is None:
            return action

        action.update({'set_line_status': reconnect_array})
        return action


if __name__ == '__main__':
    # hyper-parameters
    ACTION_THRESHOLD = 0.9
    DATA_PATH = '../training_data_track1'  # for demo only, use your own dataset
    SCENARIO_PATH = '../training_data_track1/chronics'
    EPOCHS = 1000
    NUM_ENV_STEPS_EACH_EPOCH = 20000 # larger is better
    # Use a capped number of parallel environments to avoid hitting OS "open files" limits.
    max_envs = 16  # you can tune this; 8 or 16 is usually safe
    NUM_CORE = min(cpu_count(), max_envs)
    print('CPU counts (capped): %d' % NUM_CORE, flush=True)

    # Build single-process environment
    try:
        # if lightsim2grid is available, use it.
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend, reward_class=PPO_Reward)
    except:
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, reward_class=PPO_Reward)
    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])
    # Convert to multi-process environment
    envs = SingleEnvMultiProcess(env=env, nb_env=NUM_CORE)
    envs.reset()

    # Build PPO agent
    agent = PPO(coef_entropy=1e-3, coef_value_func=0.01)

    # --- NEW: build MORL params ONCE for this stage ---
    morl_params = build_morl_params_from_dataset(DATA_PATH)

    # Build a runner (MORL logging every 1000 env steps)
    runner = Run_env(
        envs,
        agent,
        gamma=0.999,  # or 0.997
        lam=0.99,  # or 0.97
        action_space_path='../ActionSpace',
        morl_params=morl_params,
        morl_log_interval=1000,
        n_cores = NUM_CORE,
    )

    # --- wandb init (optional) ---
    if USE_WANDB:
        run_name = f"senior_student_{time.strftime('%m-%d-%H-%M', time.localtime())}"
        wandb.init(
            project="vt1_grid2op_senior_ppo",
            name=run_name,
            config={
                "baseline": "morl_pref_con",
                "action_threshold": ACTION_THRESHOLD,
                "epochs": EPOCHS,
                "num_env_steps_each_epoch": NUM_ENV_STEPS_EACH_EPOCH,
                "num_cores": NUM_CORE,
                "coef_entropy": 1e-3,
                "coef_value_func": 0.01,
                "gamma": runner.gamma,
                "lam": runner.lam,
            },
        )

    # Ensure log directory exists
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    logfile = os.path.join(log_dir, 'log-%s.txt' % time.strftime('%m-%d-%H-%M', time.localtime()))
    with open(logfile, 'w') as f:
        f.writelines('epoch, ave_r, ave_alive, policy_loss, value_loss, entropy, kl, clipped_ratio, time\n')

    print(f'start training... logging to {logfile}', flush=True)
    for update in range(EPOCHS):
        # ---- SAFETY CHECK: global time limit (from orchestrator) ----
        elapsed = time.time() - start_time_global
        if elapsed > MAX_RUNTIME_SECONDS:
            print("\n=======================", flush=True)
            print(" Reached global 23h30 runtime limit — exiting SeniorStudent gracefully.", flush=True)
            print(" Saving final checkpoint before exit.", flush=True)
            print("=======================\n", flush=True)

            ckpt_dir = './ckpt'
            os.makedirs(ckpt_dir, exist_ok=True)
            final_ckpt = os.path.join(ckpt_dir, f'FINAL_{update}.keras')
            runner.agent.model.model.save(final_ckpt)

            if USE_WANDB:
                try:
                    wandb.log({
                        "status": "graceful_exit_due_to_time_limit",
                        "epoch_finished": update,
                    }, step=update)
                    wandb.finish()
                except Exception:
                    pass

            sys.exit(0)
        # --- NEW: curriculum for chronic length and steps per epoch ---
        steps_this_epoch, max_iter = curriculum_schedule(update)
        NUM_ENV_STEPS_EACH_EPOCH = steps_this_epoch
        if max_iter is not None:
            try:
                def _set_max_iter(real_env, m):
                    real_env.chronics_handler.max_iter = m


                envs.call(_set_max_iter, max_iter)
            except Exception as e:
                print(f"[CURRICULUM] Could not propagate max_iter={max_iter} to workers: {e}", flush=True)

        # Optionally log curriculum parameters
        if USE_WANDB:
            wandb.log(
                {
                    "curriculum/steps_per_epoch": NUM_ENV_STEPS_EACH_EPOCH,
                    "curriculum/max_iter": float(max_iter if max_iter is not None else -1),
                },
                step=update,
            )

        # update learning rate
        lr_now = 6e-5 * np.linspace(1, 0.025, 500)[np.clip(update, 0, 499)]
        if update < 5:
            lr_now = 1e-4
        clip_range_now = 0.2

        # generate a new batch
        tick = time.time()
        obs, returns, dones, actions, values, neg_log_p, ave_r = runner.run_n_steps(NUM_ENV_STEPS_EACH_EPOCH)
        returns /= 20
        print('sampling number in this epoch: %d' % obs.shape[0])

        # update policy-value-network
        n = obs.shape[0]
        advs = returns - values
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
        for _ in range(2):
            ind = np.arange(n)
            np.random.shuffle(ind)
            for batch_id in range(10):
                slices = (tf.constant(arr[ind[batch_id::10]]) for arr in (obs, returns, actions, values, neg_log_p, advs))
                policy_loss, value_loss, entropy, approx_kl, clip_ratio = agent.train(*slices,
                                                                                      lr=lr_now,
                                                                                      clip_range=clip_range_now)

        # logging
        ave_alive = float(np.average(runner.alive_steps_record[-1000:])) if len(runner.alive_steps_record) > 0 else 0.0
        duration = float(time.time() - tick)

        print(
            'epoch-%d, policy loss: %5.3f, value loss: %.5f, entropy: %.5f, approximate kl-divergence: %.5g, clipped ratio: %.5g'
            % (update, policy_loss, value_loss, entropy, approx_kl, clip_ratio))
        print('epoch-%d, ave_r: %5.3f, ave_alive: %5.3f, duration: %5.3f'
              % (update, ave_r, ave_alive, duration))

        with open(logfile, 'a') as f:
            f.writelines('%d, %.2f, %.2f, %.3f, %.3f, %.3f, %.3f, %.3f, %.2f\n'
                         % (update, ave_r, ave_alive,
                            policy_loss, value_loss, entropy, approx_kl, clip_ratio,
                            duration))

        # wandb logging
        if USE_WANDB:
            wandb.log({
                "epoch": update,
                "lr": float(lr_now),
                "policy_loss": float(policy_loss),
                "value_loss": float(value_loss),
                "entropy": float(entropy),
                "approx_kl": float(approx_kl),
                "clip_ratio": float(clip_ratio),
                "ave_r": float(ave_r),
                "ave_alive": ave_alive,
                "sampling_num": int(obs.shape[0]),
                "duration": duration,
            })

        ckpt_dir = './ckpt'
        os.makedirs(ckpt_dir, exist_ok=True)
        # Keras 3 requires a proper extension
        ckpt_path = os.path.join(ckpt_dir, '%d-%.2f.keras' % (update, ave_r))
        runner.agent.model.model.save(ckpt_path)



