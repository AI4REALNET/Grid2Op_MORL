"""
SeniorStudent_RandomAction.py

Random-action baseline for the SeniorStudent pipeline.

Logic:
- Below ACTION_THRESHOLD: do nothing (+ reconnect one eligible disconnected line)
- At / above ACTION_THRESHOLD: select a RANDOM action from the predefined action set

No PPO training, no policy updates.
MORL metrics and ave_alive logging are preserved for comparison.
"""

import os
import time
import importlib.util
from multiprocessing import cpu_count

import grid2op
import numpy as np
from grid2op.Environment import SingleEnvMultiProcess

from PPO_Reward import PPO_Reward

# ---------------- wandb ----------------
try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False

# ---------------- runtime guard (shared with orchestrator) ----------------
MAX_RUNTIME_SECONDS = (12.0 - 0.5) * 3600  # 11h30

_orch_start = os.environ.get("ORCH_START_TIME")
try:
    start_time_global = float(_orch_start) if _orch_start is not None else time.time()
except ValueError:
    start_time_global = time.time()

# ---------------- MORL objectives (import from repo root) ----------------
PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MORL_OBJ_PATH = os.path.join(PARENT, "morl_objectives.py")

spec = importlib.util.spec_from_file_location("morl_objectives", MORL_OBJ_PATH)
morl_objectives = importlib.util.module_from_spec(spec)
spec.loader.exec_module(morl_objectives)

build_morl_params_from_dataset = morl_objectives.build_morl_params_from_dataset
compute_morl_metrics = morl_objectives.compute_morl_metrics


# =========================================================
# Runner
# =========================================================
class Run_env:
    def __init__(
        self,
        envs,
        n_steps,
        n_cores,
        action_space_path,
        action_threshold,
        morl_params=None,
        morl_log_interval=1000,
        ave_alive_log_interval=20000,
    ):
        self.envs = envs
        self.n_steps = int(n_steps)
        self.n_cores = int(n_cores)
        self.action_threshold = float(action_threshold)

        # Observation indices (match SeniorStudent*)
        chosen = list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
        chosen += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
        chosen += list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(range(1164, 1223))
        self.chosen = np.asarray(chosen, dtype=np.int32) - 1

        # Action space arrays
        self.actions62 = np.load(os.path.join(action_space_path, "actions62.npy"))
        self.actions146 = np.load(os.path.join(action_space_path, "actions146.npy"))
        self.actions = np.concatenate((self.actions62, self.actions146), axis=0)

        self.aspace = self.envs.action_space[0]

        # Bookkeeping
        self.worker_alive_steps = np.zeros(self.n_cores, dtype=np.float64)
        self.alive_steps_record = []
        self.global_step = 0

        # MORL logging
        self.morl_params = morl_params
        self.morl_log_interval = int(morl_log_interval)
        self.morl_metrics_buffer = []
        self.morl_step_counter = 0

        # Align with other scripts: log ave_alive every 20k env-steps
        self.ave_alive_log_interval = int(ave_alive_log_interval)

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------
    @staticmethod
    def reconnect_array(obs):
        """Build an integer status vector for set_line_status:
        0 = do nothing, 1 = reconnect this line (at most one).
        """
        arr = np.zeros_like(obs.line_status, dtype=int)
        disconnected = np.where(obs.line_status == False)[0]
        for line in disconnected[::-1]:
            if not obs.time_before_cooldown_line[line]:
                arr[line] = 1
                break
        return arr

    def _action_vector_from_store(self, act_id: int) -> np.ndarray:
        """Robustly extract the 494-dim vector from actions arrays.

        The stored arrays can have shape:
          - (N, 494)         -> self.actions[act_id] is (494,)
          - (N, 1, 494)      -> self.actions[act_id] is (1, 494)
          - (N, k, 494)      -> take the first entry (k>1 shouldn't happen, but be defensive)
        """
        vec = np.asarray(self.actions[int(act_id)])
        while vec.ndim > 1:
            vec = vec[0]
        vec = np.asarray(vec, dtype=np.float32)
        if vec.shape[0] != 494:
            raise ValueError(f"Unexpected action vector shape: {vec.shape} (expected (494,))")
        return vec

    def array2action(self, total_array: np.ndarray, reconnect=None):
        """Convert our 494-dim action vector into a Grid2Op Action.

        IMPORTANT: we only use the change_bus slice here, matching the SeniorStudent setup.
        """
        total_array = np.asarray(total_array, dtype=np.float32)
        # Ensure slicing works even if the caller accidentally passes a list
        change_bus = total_array[236:413].astype(bool)
        action = self.aspace({"change_bus": change_bus})
        if reconnect is not None:
            action.update({"set_line_status": reconnect})
        return action

    # -----------------------------------------------------
    # Logging
    # -----------------------------------------------------
    def _maybe_log_morl(self, prev_obs, obs, actions, rs, dones, infos):
        if self.morl_params is None:
            return

        for i in range(self.n_cores):
            try:
                m = compute_morl_metrics(
                    prev_obs[i],
                    obs[i],
                    actions[i],
                    float(rs[i]),
                    bool(dones[i]),
                    infos[i],
                    self.morl_params,
                )
                self.morl_metrics_buffer.append(m)
                self.morl_step_counter += 1
            except Exception:
                # stay robust; MORL metrics are auxiliary
                pass

        if USE_WANDB and self.morl_step_counter % self.morl_log_interval == 0:
            if self.morl_metrics_buffer:
                agg = {}
                n = len(self.morl_metrics_buffer)
                for m in self.morl_metrics_buffer:
                    for k, v in m.items():
                        agg[k] = agg.get(k, 0.0) + float(v)
                wandb.log(
                    {f"morl/{k}_mean_{self.morl_log_interval}": v / n for k, v in agg.items()},
                    step=self.global_step,
                )
            self.morl_metrics_buffer.clear()

    def _maybe_log_sampling(self, step_rewards):
        # mean reward over last 100 env steps
        if USE_WANDB and self.global_step % 100 == 0:
            wandb.log({"sampling/mean_reward_100_steps": float(np.mean(step_rewards))}, step=self.global_step)
            step_rewards.clear()

        # ave_alive every 20k env steps (like the other scripts)
        if USE_WANDB and self.global_step % self.ave_alive_log_interval == 0:
            ave_alive = float(np.average(self.alive_steps_record[-1000:])) if self.alive_steps_record else 0.0
            wandb.log({"ave_alive": ave_alive}, step=self.global_step)

    # -----------------------------------------------------
    # Core loop (random baseline)
    # -----------------------------------------------------
    def run_n_steps(self, n_steps: int):
        self.n_steps = int(n_steps)

        step_rewards = []
        obs = self.envs.get_obs()
        obss = np.asarray([o.to_vect()[self.chosen] for o in obs], dtype=np.float32)

        agent_step_rs = np.zeros(self.n_cores, dtype=np.float64)
        n_actions = int(self.actions.shape[0])

        for _ in range(self.n_steps):
            self.worker_alive_steps += 1

            action_ids = [None] * self.n_cores
            for i in range(self.n_cores):
                if obss[i, 654:713].max() >= self.action_threshold:
                    action_ids[i] = int(np.random.randint(0, n_actions))

            actions = []
            for i, a in enumerate(action_ids):
                if a is not None:
                    vec = self._action_vector_from_store(a)
                    actions.append(self.array2action(vec))
                else:
                    # do nothing + reconnect
                    reconnect = Run_env.reconnect_array(obs[i])
                    actions.append(self.array2action(np.zeros(494, dtype=np.float32), reconnect))

            prev_obs = obs
            obs, rs, dones, infos = self.envs.step(actions)
            obss = np.asarray([o.to_vect()[self.chosen] for o in obs], dtype=np.float32)

            self._maybe_log_morl(prev_obs, obs, actions, rs, dones, infos)

            step_rewards.append(float(np.mean(rs)))
            self.global_step += 1
            self._maybe_log_sampling(step_rewards)

            for i in range(self.n_cores):
                if dones[i]:
                    self.alive_steps_record.append(self.worker_alive_steps[i])
                    self.worker_alive_steps[i] = 0
                    if "GAME OVER" in str(infos[i].get("exception", "")):
                        agent_step_rs[i] += float(rs[i]) - 300.0
                    else:
                        agent_step_rs[i] += float(rs[i]) + 500.0

            agent_step_rs += rs

        ave_r = float(np.mean(agent_step_rs)) if len(agent_step_rs) else 0.0
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty.astype(bool), empty, empty, empty, ave_r


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    ACTION_THRESHOLD = 0.9

    DATA_PATH = "../training_data_track1"
    SCENARIO_PATH = "../training_data_track1/chronics"

    EPOCHS = 1000
    NUM_ENV_STEPS_EACH_EPOCH = 20000  # match SeniorStudent default cadence

    # Cap envs to avoid open-files issues
    NUM_CORE = min(cpu_count(), 16)
    print(f"CPU cores used: {NUM_CORE}", flush=True)

    # Build env (+ LightSim if available)
    try:
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend, reward_class=PPO_Reward)
    except Exception:
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, reward_class=PPO_Reward)

    # Shuffle chronics like SeniorStudent scripts
    env.chronics_handler.shuffle(shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)])

    envs = SingleEnvMultiProcess(env=env, nb_env=NUM_CORE)
    envs.reset()

    # MORL params once
    morl_params = build_morl_params_from_dataset(DATA_PATH)

    runner = Run_env(
        envs=envs,
        n_steps=2000,
        n_cores=NUM_CORE,
        action_space_path="../ActionSpace",
        action_threshold=ACTION_THRESHOLD,
        morl_params=morl_params,
        morl_log_interval=1000,
        ave_alive_log_interval=20000,
    )

    if USE_WANDB:
        wandb.init(
            project="vt1_grid2op_senior_ppo",
            name=f"senior_student_random_{time.strftime('%m-%d-%H-%M')}",
            config={
                "baseline": "random_action",
                "action_threshold": ACTION_THRESHOLD,
                "num_env_steps_each_epoch": NUM_ENV_STEPS_EACH_EPOCH,
                "num_cores": NUM_CORE,
            },
        )

    for epoch in range(EPOCHS):
        if time.time() - start_time_global > MAX_RUNTIME_SECONDS:
            print("Time limit reached, exiting.", flush=True)
            break

        tick = time.time()
        _, _, _, _, _, _, ave_r = runner.run_n_steps(NUM_ENV_STEPS_EACH_EPOCH)
        ave_alive = float(np.average(runner.alive_steps_record[-1000:])) if runner.alive_steps_record else 0.0
        duration = time.time() - tick

        print(
            f"epoch {epoch:04d} | ave_r={ave_r:7.2f} | ave_alive={ave_alive:7.1f} | {duration:6.1f}s",
            flush=True,
        )

        if USE_WANDB:
            wandb.log(
                {
                    "epoch": epoch,
                    "ave_r": ave_r,
                    "ave_alive": ave_alive,
                    "duration": duration,
                },
                step=epoch,
            )

    print("SeniorStudent_RandomAction finished.", flush=True)
