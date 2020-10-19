from grid2op.Agent import BaseAgent
import numpy as np


class RecoPowerlineAgent(BaseAgent):

    def __init__(self, action_space):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=action_space)

    def act(self, observation, reward, done):

        new_line_status_array = np.zeros_like(observation.rho)
        disconnected_lines = np.where(observation.line_status == False)[0]

        minrho = observation.rho.max()
        minidx = -1

        for line in disconnected_lines[::-1]:
            if not observation.time_before_cooldown_line[line]:
                # 若有线路是断开的, 且, 该线路的冷却回合数为零
                line_to_reconnect = line  # 对其重合闸
                new_line_status_array[line_to_reconnect] = 1
                obs_, _, done, _ = observation.simulate(self.action_space({'set_line_status': new_line_status_array}))
                if not done and obs_.rho.max() < minrho:
                    minidx = line
                    minrho = obs_.rho.max()
                new_line_status_array[line_to_reconnect] = 0

        if minidx != -1:
            new_line_status_array[minidx] = 1
            return self.action_space({'set_line_status': new_line_status_array})
        else:
            return self.action_space({})

def make_agent(env, this_directory_path):
    my_agent = RecoPowerlineAgent(env.action_space)
    return my_agent
