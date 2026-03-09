"""
In this file, an expert agent (named Tutor), which does a greedy search
in the reduced action space (208 actions), is built.
It receives an observation, and returns the action that decreases the rho
most, as well as its index [api: Tutor.act(obs)].

author: chen binbin
mail: cbb@cbb1996.com
"""
import os
import time
import numpy as np
from grid2op.Agent import BaseAgent
from grid2op.Exceptions import Grid2OpException, AmbiguousAction



class Tutor(BaseAgent):
    def __init__(self, action_space, action_space_directory):
        BaseAgent.__init__(self, action_space=action_space)
        self.actions62 = np.load(os.path.join(action_space_directory, 'actions62.npy'))
        self.actions146 = np.load(os.path.join(action_space_directory, 'actions146.npy'))

    @staticmethod
    def reconnect_array(obs):
        new_line_status_array = np.zeros_like(obs.rho)
        disconnected_lines = np.where(obs.line_status==False)[0]
        for line in disconnected_lines[::-1]:
            if not obs.time_before_cooldown_line[line]:
                # this line is disconnected, and, it is not cooling down.
                line_to_reconnect = line
                new_line_status_array[line_to_reconnect] = 1
                break  # reconnect the first one
        return new_line_status_array

    def array2action(self, total_array, reconnect_array):
        """
        Build a Grid2Op action from the concatenated action vector.

        total_array: full action descriptor (length ~494 in the original paper code)
        reconnect_array: line-status changes (0/1) to feed into set_line_status
        """
        # ensure the "change_bus" part is a boolean vector (as required by Grid2Op)
        change_bus_part = np.asarray(total_array[236:413], dtype=bool)

        # ensure line-status IDs are integers (as required by newer Grid2Op)
        reconnect_array = np.asarray(reconnect_array, dtype=np.int64)

        # Build the action *once* through the public API, no manual property poking
        action = self.action_space({
            'change_bus': change_bus_part,
            'set_line_status': reconnect_array
        })
        return action

    @staticmethod
    def old_is_legal(action, obs):
        substation_to_operate = int(action.as_dict()['change_bus_vect']['modif_subs_id'][0])
        if obs.time_before_cooldown_sub[substation_to_operate]:
            # substation is cooling down
            return False
        for line in [eval(key) for key, val in action.as_dict()['change_bus_vect'][str(substation_to_operate)].items() if 'line' in val['type']]:
            if obs.time_before_cooldown_line[line] or not obs.line_status[line]:
                # line is cooling down, or line is disconnected
                return False
        return True
    @staticmethod
    def is_legal(action, obs):
        """Legacy placeholder: we now rely on observation.simulate + exception handling
        to filter out illegal / ambiguous actions."""
        return True


    def act(self, observation):
        tick = time.time()
        reconnect_array = self.reconnect_array(observation)

        if observation.rho.max() < 0.925:
            # secure, return "do nothing" in bus switches.
            # Grid2Op (newer versions) requires integer status IDs, not floats
            reconnect_array = np.asarray(reconnect_array, dtype=np.int64)
            return self.action_space({'set_line_status': reconnect_array}), -1

        # not secure, do a greedy search
        # not secure, do a greedy search
        min_rho = observation.rho.max()
        print('%s: overload! line-%d has a max. rho of %.2f'
              % (str(observation.get_time_stamp()),
                 observation.rho.argmax(),
                 observation.rho.max()))
        action_chosen = None
        return_idx = -1

        # hierarchy-1: 62 actions.
        for idx, action_array in enumerate(self.actions62):
            a = self.array2action(action_array, reconnect_array)
            try:
                obs_sim, _, done_sim, _ = observation.simulate(a)
            except (Grid2OpException, AmbiguousAction):
                # illegal / ambiguous in this state -> skip
                continue
            if done_sim:
                # leads to game over -> skip
                continue
            if obs_sim.rho.max() < min_rho:
                min_rho = obs_sim.rho.max()
                action_chosen = a
                return_idx = idx

        if min_rho <= 0.999:
            print('    Action %d decreases max. rho to %.2f, search duration is %.2fs'
                  % (return_idx, min_rho, time.time() - tick))
            return (action_chosen if action_chosen
                    else self.array2action(np.zeros(494), reconnect_array)), return_idx

        # hierarchy-2: 146 actions.
        for idx, action_array in enumerate(self.actions146):
            a = self.array2action(action_array, reconnect_array)
            try:
                obs_sim, _, done_sim, _ = observation.simulate(a)
            except (Grid2OpException, AmbiguousAction):
                continue
            if done_sim:
                continue
            if obs_sim.rho.max() < min_rho:
                min_rho = obs_sim.rho.max()
                action_chosen = a
                return_idx = idx + 62

        print('    Action %d decreases max. rho to %.2f, search duration is %.2fs' % (return_idx, min_rho, time.time() - tick))
        return action_chosen if action_chosen else self.array2action(np.zeros(494), reconnect_array), return_idx
