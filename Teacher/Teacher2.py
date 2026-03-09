"""
In this file, we do the following thing repeatedly:
    1. choose a scenario
    2. while not game over:
    3.     if not overflow:
    4.         step a "reconnect disconnected line" or "do nothing" action
    5.     else:
    6.         search a greedy action to minimize the max rho (~60k possible actions)
    7.         save the tuple of (None, observation, action) to a csv file.

author: chen binbin
mail: cbb@cbb1996.com
"""
import os
import time
import grid2op
import numpy as np
import pandas as pd
from copy import deepcopy
import argparse  # <-- add this

EXPERIENCES2_FILENAME = "Experiences2.csv"  # <-- new global



def topology_search(env):
    obs = env.get_obs()
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


def save_sample(save_path='./'):
    # Not necessary to save a "do nothing" action
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

    # ---- meta info row (scenario, step, etc.) ----
    # Note: columns 2–4 are None here by design (no "attacked line" in Teacher2)
    row_meta = [
        env.chronics_handler.get_name(),                       # 0: scenario name
        dst_step,                                              # 1: time step
        None,                                                  # 2: attacked line id (None)
        None,                                                  # 3: from-bus (None)
        None,                                                  # 4: to-bus (None)
        str(np.where(obs.rho > 1)[0].tolist()),               # 5: overloaded lines indices
        str([float(i) for i in np.around(
            obs.rho[np.where(obs.rho > 1)], 2)]),             # 6: their rho values
        modif_sub,                                            # 7: modified substation id
        str(act_or),                                          # 8: moved origins
        str(act_ex),                                          # 9: moved extremities
        str(act_gen),                                         # 10: moved generators
        str(act_load),                                        # 11: moved loads
        float(obs.rho.max()),                                 # 12: max rho before action
        int(obs.rho.argmax()),                                # 13: argmax before
        float(obs_.rho.max()),                                # 14: max rho after action
        int(obs_.rho.argmax()),                               # 15: argmax after
    ]

    # Create DataFrames:
    df_meta = pd.DataFrame([row_meta])
    df_vect = pd.DataFrame(
        [np.concatenate((obs.to_vect(), obs_.to_vect(), action.to_vect()))]
    )

    pd.concat((df_meta, df_vect), axis=1).to_csv(
        os.path.join(save_path, EXPERIENCES2_FILENAME),
        index=False,
        header=False,
        mode='a'
    )


def find_best_line_to_reconnect(obs, original_action):
    disconnected_lines = np.where(obs.line_status == False)[0]
    if not len(disconnected_lines):
        return original_action
    o, _, _, _ = obs.simulate(original_action)
    min_rho = o.rho.max()
    line_to_reconnect = -1
    for line in disconnected_lines:
        if not obs.time_before_cooldown_line[line]:
            # MUST be int, not float
            reconnect_array = np.zeros(obs.rho.shape, dtype=int)
            reconnect_array[line] = 1
            reconnect_action = deepcopy(original_action)
            reconnect_action.update({'set_line_status': reconnect_array})
            if not is_legal(reconnect_action, obs):
                continue
            o, _, _, _ = obs.simulate(reconnect_action)
            if o.rho.max() < min_rho:
                line_to_reconnect = line
                min_rho = o.rho.max()
    if line_to_reconnect != -1:
        # MUST be int here as well
        reconnect_array = np.zeros(obs.rho.shape, dtype=int)
        reconnect_array[line_to_reconnect] = 1
        original_action.update({'set_line_status': reconnect_array})
    return original_action


def is_legal(action, obs):
    if 'change_bus_vect' not in action.as_dict():
        return True
    substation_to_operate = int(action.as_dict()['change_bus_vect']['modif_subs_id'][0])
    if obs.time_before_cooldown_sub[substation_to_operate]:
        return False
    for line in [eval(key) for key, val in action.as_dict()['change_bus_vect'][str(substation_to_operate)].items() if 'line' in val['type']]:
        if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
            return False
    return True


if __name__ == "__main__":
    # hyper-parameters
    DATA_PATH = '../training_data_track1'  # one level up from Teacher/
    SCENARIO_PATH = '../training_data_track1/chronics'
    SAVE_PATH = './'
    NUM_EPISODE = 100  # total episodes

    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-end", type=int, default=NUM_EPISODE)
    parser.add_argument("--job-id", type=str, default="job0")
    args = parser.parse_args()

    EXPERIENCES2_FILENAME = f"Experiences2_{args.job_id}.csv"

    n_chronics = len(os.listdir(SCENARIO_PATH))

    # build env once and reuse it
    try:
        from lightsim2grid import LightSimBackend
        backend = LightSimBackend()
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH, backend=backend)
    except Exception:
        env = grid2op.make(dataset=DATA_PATH, chronics_path=SCENARIO_PATH)

    for episode in range(args.episode_start, args.episode_end):
        print(f"[Teacher2 {args.job_id}] Starting episode {episode}", flush=True)

        env.chronics_handler.shuffle(
            shuffler=lambda x: x[np.random.choice(len(x), size=len(x), replace=False)]
        )

        for chronic in range(n_chronics):
            obs = env.reset()
            dst_step = 0
            print('Scenario to test is [%s], start from step-%d. .' %
                  (env.chronics_handler.get_name(), dst_step))
            env.fast_forward_chronics(dst_step)
            obs, done = env.get_obs(), False
            while not done:
                if obs.rho.max() >= 1:
                    action = topology_search(env)
                    obs_, reward, done, _ = env.step(action)
                    save_sample(SAVE_PATH)
                    obs = obs_
                else:
                    action = env.action_space({})
                    action = find_best_line_to_reconnect(obs, action)
                    obs, reward, done, _ = env.step(action)
                dst_step += 1

