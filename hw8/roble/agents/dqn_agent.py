import numpy as np


from hw8.roble.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from hw8.roble.policies.argmax_policy import ArgMaxPolicy
from hw8.roble.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['alg']['batch_size']
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['alg']['ac_dim']
        self.learning_starts = agent_params['alg']['learning_starts']
        self.learning_freq = agent_params['alg']['learning_freq']
        self.target_update_freq = agent_params['alg']['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(self.critic)

        lander = agent_params['env']['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        for path in paths:
            self.replay_buffer_idx = self.replay_buffer.store_frame(path["observation"])
            self.replay_buffer.encode_recent_observation()
            self.replay_buffer.store_effect(self.replay_buffer_idx, path["action"], path["reward"], path["terminal"])

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """
        #might need epsilon?
        action = self.actor.get_action(self.last_obs)
        new_obs, reward, terminated, _, _, _ = self.env.step(action)

        path = {
            'observation' : self.last_obs,
            'action' : action,
            'reward' : reward,
            'terminated' : terminated
        }
        self.add_to_replay_buffer(path)
        self.last_obs = new_obs

    ####################################
    ####################################

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        raise NotImplementedError
        # Not needed for this homework

    ####################################
    ####################################