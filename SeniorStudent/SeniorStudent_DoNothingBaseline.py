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

class Run_env(object):
    def __init__(self, envs, agent, n_steps=2000, n_cores=12, gamma=0.99, lam=0.95, action_space_path='../', morl_params=None,
        morl_log_interval=1000, never_act=False,):
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
        self.worker_alive_steps = np.zeros(NUM_CORE)
        self.alive_steps_record = []
        self.global_step = 0  # for progress / wandb logging
        # --- NEW: MORL logging state ---
        self.morl_params = morl_params  # None = disabled
        self.morl_log_interval = morl_log_interval
        self.morl_metrics_buffer = []  # list of per-step dicts
        self.morl_step_counter = 0
        self.never_act = bool(never_act)  # if True: never query agent, always do-nothing (+reconnect)
        self.ave_alive_log_interval = 20000

    @staticmethod
    def reconnect_array(obs):
        """
        Build an integer status vector for set_line_status:
        0 = do nothing, 1 = reconnect this line.
        """
        new_line_status_array = np.zeros_like(obs.line_status, dtype=int)
        disconnected_lines = np.where(obs.line_status == False)[0]

        for line in disconnected_lines[::-1]:
            if not obs.time_before_cooldown_line[line]:
                new_line_status_array[line] = 1
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


    def run_n_steps(self, n_steps=None):
        def swap_and_flatten(arr):
            shape = arr.shape
            return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

        self.n_steps = n_steps if n_steps is not None else self.n_steps
        # --- DO-NOTHING BASELINE MODE ---
        if self.never_act:
            return self._run_n_steps_do_nothing()

        # mb for mini-batch
        mb_obs, mb_rewards, mb_actions = [[] for _ in range(NUM_CORE)], [[] for _ in range(NUM_CORE)], [[] for _ in range(NUM_CORE)]
        # For progress logging
        step_rewards = []
        mb_values, mb_dones, mb_neg_log_p = [[] for _ in range(NUM_CORE)], [[] for _ in range(NUM_CORE)], [[] for _ in range(NUM_CORE)]

        # start sampling
        obs_objs = self.envs.get_obs()
        obss = np.asarray([obs.to_vect()[self.chosen] for obs in obs_objs])  # (12, 1221,)
        dones = np.asarray([False for _ in range(NUM_CORE)])  # (12,)
        agent_step_rs = np.asarray([0 for _ in range(NUM_CORE)], dtype=np.float64)  # (12,)
        for _ in range(self.n_steps):
            self.worker_alive_steps += 1
            actions = np.asarray([None for _ in range(NUM_CORE)])
            values = np.asarray([None for _ in range(NUM_CORE)])
            neg_log_ps = np.asarray([None for _ in range(NUM_CORE)])

            for id in range(NUM_CORE):
                if obss[id, 654:713].max() >= ACTION_THRESHOLD:
                    actions[id], values[id], neg_log_ps[id], _ = map(
                        lambda x: x._numpy(), self.agent.step(tf.constant(obss[[id], :]))
                    )
                    if dones[id] == False and len(mb_obs[id]) > 0:
                        mb_rewards[id].append(agent_step_rs[id])
                    agent_step_rs[id] = 0
                    mb_obs[id].append(obss[[id], :])
                    mb_dones[id].append(dones[[id]])
                    dones[id] = False
                    mb_actions[id].append(actions[id])
                    mb_values[id].append(values[id])
                    mb_neg_log_p[id].append(neg_log_ps[id])
                else:
                    pass

            actions_array = [
                self.array2action(self.actions[i][0]) if i is not None
                else self.array2action(np.zeros(494), Run_env.reconnect_array(obs_objs[idx]))
                for idx, i in enumerate(actions)
            ]

            # Keep a copy of previous observations for MORL metrics
            prev_obs_objs = obs_objs

            # Step all environments
            obs_objs, rs, env_dones, infos = self.envs.step(actions_array)
            obss = np.asarray([obs.to_vect()[self.chosen] for obs in obs_objs])

            # --- NEW: MORL metrics accumulation ---
            if self.morl_params is not None:
                for idx in range(NUM_CORE):
                    try:
                        metrics = compute_morl_metrics(
                            prev_obs_objs[idx],  # obs before step
                            obs_objs[idx],  # obs after step
                            actions_array[idx],  # grid2op Action
                            float(rs[idx]),  # scalar reward
                            bool(env_dones[idx]),  # done flag
                            infos[idx],  # info dict
                            self.morl_params,
                        )
                        self.morl_metrics_buffer.append(metrics)
                        self.morl_step_counter += 1
                    except Exception as e:
                        # You can print once if you want to debug
                        # print(f"[MORL] metric computation error on env {idx}: {e}")
                        pass

                # Log average metrics every `morl_log_interval` steps
                if USE_WANDB and self.morl_step_counter % self.morl_log_interval == 0:
                    if self.morl_metrics_buffer:
                        agg = {}
                        n = len(self.morl_metrics_buffer)
                        for m in self.morl_metrics_buffer:
                            for k, v in m.items():
                                agg[k] = agg.get(k, 0.0) + float(v)
                        avg = {f"morl/{k}_mean_{self.morl_log_interval}": val / n
                               for k, val in agg.items()}
                        wandb.log(avg, step=self.global_step)
                    self.morl_metrics_buffer.clear()

            # Progress logging: record mean reward and log every 100 env steps
            mean_r = float(np.mean(rs))
            step_rewards.append(mean_r)
            self.global_step += 1

            if USE_WANDB and (self.global_step % 100 == 0):
                # average over the last 100 (or fewer, at the very beginning) steps
                wandb.log({"sampling/mean_reward_100_steps": float(np.mean(step_rewards))},
                          step=self.global_step)
                step_rewards = []
            # Log ave_alive every 20k env steps (same cadence as other scripts)
            if USE_WANDB and (self.global_step % self.ave_alive_log_interval == 0):
                ave_alive = float(np.average(self.alive_steps_record[-self.ave_alive_log_interval:])) if len(
                    self.alive_steps_record) > 0 else 0.0
                wandb.log({"ave_alive": ave_alive}, step=self.global_step)

            for id in range(NUM_CORE):
                if env_dones[id]:
                    # death or end
                    self.alive_steps_record.append(self.worker_alive_steps[id])
                    self.worker_alive_steps[id] = 0
                    if 'GAME OVER' in str(infos[id]['exception']):
                        dones[id] = True
                        mb_rewards[id].append(agent_step_rs[id] - 300)
                    else:
                        dones[id] = True
                        mb_rewards[id].append(agent_step_rs[id] + 500)
            agent_step_rs += rs

        # end sampling

        # batch to trajectory
        for id in range(NUM_CORE):
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
        for id in range(NUM_CORE):
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

    def _run_n_steps_do_nothing(self):
        """Step envs with strict do-nothing actions (plus reconnection),
        while keeping the same MORL logging hooks and alive-step tracking.

        Returns dummy arrays (empty) plus ave_r to keep the main loop unchanged.
        """
        step_rewards = []

        obs_objs = self.envs.get_obs()
        agent_step_rs = np.asarray([0 for _ in range(NUM_CORE)], dtype=np.float64)

        for _ in range(self.n_steps):
            self.worker_alive_steps += 1

            # Always do-nothing (+ reconnect if possible)
            actions_array = [
                self.array2action(np.zeros(494), Run_env.reconnect_array(obs))
                for obs in obs_objs
            ]

            prev_obs_objs = obs_objs
            obs_objs, rs, env_dones, infos = self.envs.step(actions_array)

            # MORL metrics accumulation (same pattern as the main path)
            if self.morl_params is not None:
                for idx in range(NUM_CORE):
                    try:
                        metrics = compute_morl_metrics(
                            prev_obs_objs[idx],
                            obs_objs[idx],
                            actions_array[idx],
                            float(rs[idx]),
                            bool(env_dones[idx]),
                            infos[idx],
                            self.morl_params,
                        )
                        self.morl_metrics_buffer.append(metrics)
                        self.morl_step_counter += 1
                    except Exception:
                        pass

                if USE_WANDB and (self.morl_step_counter % self.morl_log_interval == 0):
                    if self.morl_metrics_buffer:
                        agg = {}
                        n = len(self.morl_metrics_buffer)
                        for m in self.morl_metrics_buffer:
                            for k, v in m.items():
                                agg[k] = agg.get(k, 0.0) + float(v)
                        avg = {f"morl/{k}_mean_{self.morl_log_interval}": val / n for k, val in agg.items()}
                        wandb.log(avg, step=self.global_step)
                    self.morl_metrics_buffer.clear()

            mean_r = float(np.mean(rs))
            step_rewards.append(mean_r)
            self.global_step += 1

            if USE_WANDB and (self.global_step % 100 == 0):
                wandb.log({"sampling/mean_reward_100_steps": float(np.mean(step_rewards))}, step=self.global_step)
                step_rewards = []
            # Log ave_alive every 20k env steps (same cadence as other scripts)
            if USE_WANDB and (self.global_step % self.ave_alive_log_interval == 0):
                ave_alive = float(np.average(self.alive_steps_record[-self.ave_alive_log_interval:])) if len(
                    self.alive_steps_record) > 0 else 0.0
                wandb.log({"ave_alive": ave_alive}, step=self.global_step)

            for idx in range(NUM_CORE):
                if env_dones[idx]:
                    self.alive_steps_record.append(self.worker_alive_steps[idx])
                    self.worker_alive_steps[idx] = 0

                    # keep your original "death vs end" shaping idea
                    if 'GAME OVER' in str(infos[idx].get('exception', '')):
                        agent_step_rs[idx] += float(rs[idx]) - 300.0
                    else:
                        agent_step_rs[idx] += float(rs[idx]) + 500.0

            agent_step_rs += rs

        ave_r = float(np.mean(agent_step_rs)) if len(agent_step_rs) else 0.0
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty.astype(bool), empty, empty, empty, ave_r


if __name__ == '__main__':
    # hyper-parameters
    ACTION_THRESHOLD = 0.9
    DO_NOTHING_BASELINE = True  # set True for do-nothing baseline with identical logging
    # --- Adaptive threshold parameters ---
    ENTROPY_TARGET = 2.5  # if entropy_ema falls below this, lower threshold
    MIN_ACTION_THRESHOLD = 0.80  # do not go below this
    THRESHOLD_STEP = 0.005  # how much to lower per epoch when triggered
    ENTROPY_EMA_ALPHA = 0.1  # smoothing factor for running entropy estimate

    DATA_PATH = '../training_data_track1'  # for demo only, use your own dataset
    SCENARIO_PATH = '../training_data_track1/chronics'
    EPOCHS = 1000
    NUM_ENV_STEPS_EACH_EPOCH = 40000 # larger is better
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
    coef_entropy = 1e-3
    coef_value_func = 0.01
    agent = PPO(coef_entropy=coef_entropy, coef_value_func=coef_value_func)
    print("SeniorStudent: number of trainable params:",
          sum(int(tf.size(v)) for v in agent.model.model.trainable_variables))

    # --- NEW: build MORL params ONCE for this stage ---
    morl_params = build_morl_params_from_dataset(DATA_PATH)

    # Build a runner (MORL logging every 1000 env steps)
    runner = Run_env(
        envs,
        agent,
        action_space_path='../ActionSpace',
        morl_params=morl_params,
        morl_log_interval=1000,
        never_act=DO_NOTHING_BASELINE,
    )

    # --- wandb init (optional) ---
    if USE_WANDB:
        run_name = f"senior_student_{time.strftime('%m-%d-%H-%M', time.localtime())}"
        wandb.init(
            project="vt1_grid2op_senior_ppo",
            name=run_name,
            config={
                "baseline": "do_nothing",
                "action_threshold": ACTION_THRESHOLD,
                "epochs": EPOCHS,
                "num_env_steps_each_epoch": NUM_ENV_STEPS_EACH_EPOCH,
                "num_cores": NUM_CORE,
                "coef_entropy":coef_entropy,
                "coef_value_func": coef_value_func,
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
    # running estimate of entropy for adaptive threshold
    entropy_ema = None
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
if not DO_NOTHING_BASELINE and obs.shape[0] > 0:
    n = obs.shape[0]
    advs = returns - values
    advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)
    for _ in range(2):
        ind = np.arange(n)
        np.random.shuffle(ind)
        for batch_id in range(10):
            slices = (tf.constant(arr[ind[batch_id::10]]) for arr in
                      (obs, returns, actions, values, neg_log_p, advs))
            policy_loss, value_loss, entropy, approx_kl, clip_ratio = agent.train(
                *slices,
                lr=lr_now,
                clip_range=clip_range_now,
            )
else:
    # Baseline mode: no training updates (keep last values for logging)
    policy_loss, value_loss, entropy, approx_kl, clip_ratio = 0.0, 0.0, float(entropy_ema or 0.0), 0.0, 0.0
    # --- Adaptive threshold based on entropy ---
    entropy_val = float(entropy)
    if entropy_ema is None:
        entropy_ema = entropy_val
    else:
        entropy_ema = (
                (1.0 - ENTROPY_EMA_ALPHA) * entropy_ema
                + ENTROPY_EMA_ALPHA * entropy_val
        )



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
            "entropy_ema": float(entropy_ema),
            "approx_kl": float(approx_kl),
            "clip_ratio": float(clip_ratio),
            "ave_r": float(ave_r),
            "ave_alive": ave_alive,
            "sampling_num": int(obs.shape[0]),
            "duration": duration,
            "action_threshold": float(ACTION_THRESHOLD),
        })

    ckpt_dir = './ckpt'
    os.makedirs(ckpt_dir, exist_ok=True)
    # Keras 3 requires a proper extension
    ckpt_path = os.path.join(ckpt_dir, '%d-%.2f.keras' % (update, ave_r))
    runner.agent.model.model.save(ckpt_path)



