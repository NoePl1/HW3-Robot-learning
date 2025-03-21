import numpy as np
from gym.wrappers.frame_stack import LazyFrames

from hw8.roble.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from hw8.roble.policies.argmax_policy import ArgMaxPolicy
from hw8.roble.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['alg']['batch_size']
        self.last_obs = self.env.reset()
        if isinstance(self.last_obs, LazyFrames):
            self.last_obs = np.asarray(self.last_obs).squeeze(axis=3)

        self.num_actions = agent_params['alg']['ac_dim']
        self.learning_starts = agent_params['alg']['learning_starts']
        self.learning_freq = agent_params['alg']['learning_freq']
        self.target_update_freq = agent_params['alg']['target_update_freq']
        self.eps = agent_params['alg']['eps']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(**agent_params)
        self.actor = ArgMaxPolicy(self.critic)

        lander = agent_params['env']['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['alg']['replay_buffer_size'], agent_params['alg']['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        if paths is not None:
            for path in paths:
                self.replay_buffer_idx = self.replay_buffer.store_frame(path["observation"])
                self.replay_buffer.encode_recent_observation()
                self.replay_buffer.store_effect(self.replay_buffer_idx, path["action"], path["reward"], path["terminated"])

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """
        #might need epsilon?
        self.t += 1

        perform_random_action = np.random.random() < self.eps or self.t < self.learning_starts

        if perform_random_action:
            action = self.env.action_space.sample()
        else:
            action = self.actor.get_action(self.last_obs)
        #print("lastobs shape: ", self.last_obs.shape)
        new_obs, reward, terminated, _ = self.env.step(action)
        if isinstance(new_obs, LazyFrames):
            new_obs = np.asarray(new_obs).squeeze(axis=3)

        if len(self.last_obs.shape) == 3:
            path = {
                'observation' : np.expand_dims(self.last_obs[-1],axis=-1),
                'action' : action,
                'reward' : reward,
                'terminated' : terminated
            }
        else:
            path = {
                'observation' : self.last_obs,
                'action' : action,
                'reward' : reward,
                'terminated' : terminated
            }
        self.add_to_replay_buffer([path])

        if terminated:
            self.last_obs = self.env.reset()
            if isinstance(self.last_obs, LazyFrames):
                self.last_obs = np.asarray(self.last_obs).squeeze(axis=3)
        else:
            self.last_obs = new_obs.copy()

    ####################################
    ####################################

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        #print("next_ob_no.shape in train:", next_ob_no.shape)
        #print("ob_no.shape in train:", ob_no.shape)

        loss = self.critic.update(ob_no, ac_na,next_ob_no, re_n, terminal_n)
        return loss
    ####################################
    ####################################