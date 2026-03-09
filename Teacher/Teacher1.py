"""
In this file, we do the following thing repeatedly:
    1. choose a scenario
    2. sample a time-step every 6 hours
    3. disconnect a line which is under possible attack
    4. search a greedy action to minimize the max rho (~60k possible actions)
    5. save the tuple of (attacked line, observation, action) to a csv file.

author: chen binbin
mail: cbb@cbb1996.com
"""
import os
import time
import random
import grid2op
import numpy as np
import pandas as pd
import time
import argparse
from grid2op.Exceptions import Grid2OpException, EnvError


EXPERIENCES1_FILENAME = "Experiences1.csv"  # <-- new global default



def topology_search(obs, env, dst_step):
    min_rho, overflow_id = obs.rho.max(), obs.rho.argmax()
    print("step-%s, line-%s(from bus-%d to bus-%d) overflows, max rho is %.5f" %
          (dst_step, overflow_id, env.line_or_to_subid[overflow_id],
           env.line_ex_to_subid[overflow_id], obs.rho.max()))
    all_actions = env.action_space.get_all_unitary_topologies_change(env.action_space)
    action_chosen = env.action_space({})
    tick = time.time()
    for action in all_actions:
        if not env._game_rules(action, env):
            continue
        obs_, _, done, _ = obs.simulate(action)
        if (not done) and (obs_.rho.max() < min_rho):
            min_rho = obs_.rho.max()
            action_chosen = action
    print("find a greedy action and max rho decreases to %.5f, search duration: %.2f" %
          (min_rho, time.time() - tick))
    return action_chosen



def save_sample(save_path):
    # not necessary to save a "do nothing" action
    if action == env.action_space({}):
        return None

    # Collect which objects were moved on which side
    act_or, act_ex, act_gen, act_load = [], [], [], []
    change_bus_dict = action.as_dict()['change_bus_vect']
    modif_sub = change_bus_dict['modif_subs_id'][0]

    for key, val in change_bus_dict[modif_sub].items():
        if val['type'] == 'line (extremity)':
            act_ex.append(key)
        elif val['type'] == 'line (origin)':
            act_or.append(key)
        elif val['type'] == 'load':
            act_load.append(key)
        else:
            # treat everything else as generator
            act_gen.append(key)

    # ---- meta info row (scenario, step, attacked line, etc.) ----
    row_meta = [
        env.chronics_handler.get_name(),                       # scenario name
        dst_step,                                              # time step
        line_to_disconnect,                                    # attacked line id
        env.line_or_to_subid[line_to_disconnect],             # from-bus
        env.line_ex_to_subid[line_to_disconnect],             # to-bus
        str(np.where(obs.rho > 1)[0].tolist()),               # overloaded lines indices
        str([float(i) for i in np.around(
            obs.rho[np.where(obs.rho > 1)], 2)]),             # their rho values
        modif_sub,                                            # modified substation id
        str(act_or),                                          # moved origins (as string)
        str(act_ex),                                          # moved extremities (as string)
        str(act_gen),                                         # moved generators (as string)
        str(act_load),                                        # moved loads (as string)
        float(obs.rho.max()),                                 # max rho before action
        int(obs.rho.argmax()),                                # argmax before action
        float(obs_.rho.max()),                                # max rho after action
        int(obs_.rho.argmax()),                               # argmax after action
    ]

    # Create DataFrames:
    #  - one for the meta info (single row)
    #  - one for the full numeric vectors (obs, obs_, action)
    df_meta = pd.DataFrame([row_meta])
    df_vect = pd.DataFrame(
        [np.concatenate((obs.to_vect(), obs_.to_vect(), action.to_vect()))]
    )

    pd.concat((df_meta, df_vect), axis=1).to_csv(
        os.path.join(save_path, EXPERIENCES1_FILENAME),
        index=False,
        header=False,
        mode='a'
    )


if __name__ == "__main__":
    # hyper-parameters
    DATA_PATH = '../training_data_track1'  # one level up from Teacher/
    SCENARIO_PATH = '../training_data_track1/chronics'
    SAVE_PATH = './'
    LINES2ATTACK = [45, 56, 0, 9, 13, 14, 18, 23, 27, 39]
    NUM_EPISODES = 1000  # total episodes

    # --- argument parsing for parallel splits ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-start", type=int, default=0,
                        help="Inclusive start episode index")
    parser.add_argument("--episode-end", type=int, default=NUM_EPISODES,
                        help="Exclusive end episode index")
    parser.add_argument("--job-id", type=str, default="job0",
                        help="Job id suffix for output CSV")
    args = parser.parse_args()

    # make sure globals are set for save_sample
    EXPERIENCES1_FILENAME = f"Experiences1_{args.job_id}.csv"

    # precompute number of chronics once
    n_chronics = len(os.listdir(SCENARIO_PATH))

    # build the environment ONCE and reuse it
    try:
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend)
    except Exception:
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH)

    total_blocks = (args.episode_end - args.episode_start) * len(LINES2ATTACK)

    for episode in range(args.episode_start, args.episode_end):
        # traverse all attacks
        for idx_line, line_to_disconnect in enumerate(LINES2ATTACK):

            # ---- TIMING START ----
            block_start = time.time()
            block_id = (episode - args.episode_start) * len(LINES2ATTACK) + idx_line + 1
            print(
                f"[Teacher1 {args.job_id}] Starting block {block_id}/{total_blocks} "
                f"(episode {episode}, line {line_to_disconnect})",
                flush=True,
            )
            # ---- END TIMING HEADER ----

            # reshuffle the order of chronics for this (episode, line)
            env.chronics_handler.shuffle(
                shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)]
            )

            # traverse all scenarios (we reuse env instead of recreating it)
            for chronic in range(n_chronics):
                # env.reset() will move to the next chronic in the shuffled order
                obs = env.reset()
                # episode_local makes episodes "relative" to this split
                episode_local = episode - args.episode_start
                dst_step = episode_local * 72 + random.randint(0, 72)  # a random sampling every 6 hours

                print(
                    '\n\n' + '*' * 50 +
                    '\nScenario[%s]: at step[%d], disconnect line-%d(from bus-%d to bus-%d]' % (
                        env.chronics_handler.get_name(), dst_step, line_to_disconnect,
                        env.line_or_to_subid[line_to_disconnect],
                        env.line_ex_to_subid[line_to_disconnect]
                    )
                )

                # to the destination time-step
                try:
                    env.fast_forward_chronics(dst_step - 1)
                except StopIteration:
                    # This chronic is too short for this dst_step; skip to next chronic
                    print(
                        f"[Teacher1 {args.job_id}] fast_forward_chronics reached end "
                        f"of chronic at step {dst_step - 1}, skipping this chronic.",
                        flush=True,
                    )
                    continue

                try:
                    obs, reward, done, _ = env.step(env.action_space({}))
                except Grid2OpException as e:
                    print(
                        f"[Teacher1 {args.job_id}] Grid2OpException right after "
                        f"fast_forward_chronics at step {dst_step}: {e}. "
                        f"Resetting env and skipping this chronic.",
                        flush=True,
                    )
                    env.reset()
                    continue

                # If the no-op step itself ended the episode, do NOT continue.
                if done:
                    print(
                        f"[Teacher1 {args.job_id}] Episode ended immediately after "
                        f"no-op step at step {dst_step}; moving to next chronic.",
                        flush=True,
                    )
                    continue

                # disconnect the targeted line
                new_line_status_array = np.zeros(obs.rho.shape, dtype=int)
                new_line_status_array[line_to_disconnect] = -1
                action = env.action_space({"set_line_status": new_line_status_array})

                try:
                    obs, reward, done, _ = env.step(action)
                except Grid2OpException as e:
                    print(
                        f"[Teacher1 {args.job_id}] Grid2OpException when disconnecting "
                        f"line {line_to_disconnect} at step {dst_step}: {e}. "
                        f"Resetting env and skipping this chronic.",
                        flush=True,
                    )
                    env.reset()
                    continue

                if done:
                    print(
                        f"[Teacher1 {args.job_id}] Episode ended immediately after "
                        f"disconnecting line {line_to_disconnect} at step {dst_step}; "
                        f"moving to next chronic.",
                        flush=True,
                    )
                    continue

                if obs.rho.max() < 1:
                    # not necessary to do a dispatch
                    continue
                else:
                    # search a greedy action (unchanged logic)
                    action = topology_search(obs, env, dst_step)
                    try:
                        obs_, reward, done, _ = env.step(action)
                    except Grid2OpException as e:
                        print(
                            f"[Teacher1 {args.job_id}] Grid2OpException when applying "
                            f"topology action at step {dst_step}: {e}. "
                            f"Resetting env and skipping this chronic.",
                            flush=True,
                        )
                        env.reset()
                        continue

                    save_sample(SAVE_PATH)

            # ---- TIMING FOOTER PRINT ----
            block_end = time.time()
            duration = block_end - block_start
            print(
                f"[Teacher1 {args.job_id}] Finished block {block_id}/{total_blocks} "
                f"in {duration:.2f} seconds",
                flush=True,
            )
            # ---- END TIMING FOOTER ----



