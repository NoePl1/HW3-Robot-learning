from collections import OrderedDict
import pickle
import os
import sys
import time


import gym
from gym import wrappers
import numpy as np
import torch

from hw8.roble.agents.dqn_agent import DQNAgent
from hw8.roble.infrastructure import pytorch_util as ptu
from hw8.roble.infrastructure.atari_wrappers import ReturnWrapper

from hw8.roble.infrastructure import utils
from hw8.roble.infrastructure.logger import Logger

from hw8.roble.agents.explore_or_exploit_agent import ExplorationOrExploitationAgent
from hw8.roble.infrastructure.dqn_utils import (
        get_wrapper_by_name,
        register_custom_envs,
)

#register all of our envs
import hw8.roble.envs

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40 # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, params, agent_class=None):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params['logging']['logdir'])

        # Set random seeds
        seed = self.params['logging']['random_seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['alg']['no_gpu'],
            gpu_id=self.params['alg']['which_gpu']
        )

        #############
        ## ENV
        #############

        # Make the gym environment
        register_custom_envs()
        self.env = gym.make(self.params['env']['env_name'])
        self.eval_env = gym.make(self.params['env']['env_name'])
        print(self.params['env']['env_name'])

        #if not ('pointmass' in self.params['env']['env_name']):
        #    import matplotlib
        #    #matplotlib.use('Agg')
        #    self.env.set_logdir(self.params['logging']['logdir'] + '/expl_')
        #    self.eval_env.set_logdir(self.params['logging']['logdir'] + '/eval_')

        if ('Pointmass' in self.params['env']['env_name']):
            import matplotlib
            matplotlib.use('Agg')
            self.env.set_logdir(self.params['logging']['logdir'] + '/expl_')
            self.eval_env.set_logdir(self.params['logging']['logdir'] + '/eval_')

        if self.params['logging']['video_log_freq'] > 0:
            self.episode_trigger = lambda episode: episode % self.params['logging']['video_log_freq'] == 0
        else:
            self.episode_trigger = lambda episode: False
            
        if 'env_wrappers' in self.params:
            # These operations are currently only for Atari envs
            self.env = wrappers.RecordEpisodeStatistics(self.env, deque_size=1000)
            self.env = ReturnWrapper(self.env)
            #self.env = wrappers.RecordVideo(self.env, os.path.join(self.params['logging']['logdir'], "gym"), episode_trigger=self.episode_trigger)
            self.env = params['env_wrappers'](self.env)

            if self.params['env']['env_name'] != 'LunarLander-v3' and ('Pointmass' not in self.params['env']['env_name']):
                self.env = wrappers.FrameStack(self.env, params['alg']['frame_history_len'])


            self.eval_env = wrappers.RecordEpisodeStatistics(self.eval_env, deque_size=1000)
            self.eval_env = ReturnWrapper(self.eval_env)
            #self.eval_env = wrappers.RecordVideo(self.eval_env, os.path.join(self.params['logging']['logdir'], "gym"), episode_trigger=self.episode_trigger)
            self.eval_env = params['env_wrappers'](self.eval_env)
            if self.params['env']['env_name'] != 'LunarLander-v3' and ('Pointmass' not in self.params['env']['env_name']):
                self.eval_env = wrappers.FrameStack(self.eval_env, params['alg']['frame_history_len'])

            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        if 'non_atari_colab_env' in self.params and self.params['logging']['video_log_freq'] > 0:
            #self.env = wrappers.RecordVideo(self.env, os.path.join(self.params['logging']['logdir'], "gym"), episode_trigger=self.episode_trigger)
            #self.eval_env = wrappers.RecordVideo(self.eval_env, os.path.join(self.params['logging']['logdir'], "gym"), episode_trigger=self.episode_trigger)

            self.mean_episode_reward = -float('nan')
            self.best_mean_episode_reward = -float('inf')
        self.env.seed(seed)
        self.eval_env.seed(seed)

        # Maximum length for episodes
        self.params['env']['max_episode_length'] = self.params['env']['max_episode_length'] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['env']['max_episode_length']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params['alg']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params['alg']['ac_dim'] = ac_dim
        self.params['alg']['ob_dim'] = ob_dim

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10
    
        #############
        ## AGENT
        #############

        self.agent = agent_class(self.env, self.params)


    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                          buffer_name=None,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 1000 if isinstance(self.agent, ExplorationOrExploitationAgent) else 1

        for itr in range(n_iter):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['logging']['video_log_freq'] == 0 and self.params['logging']['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params['logging']['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['logging']['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            if isinstance(self.agent, ExplorationOrExploitationAgent) or isinstance(self.agent, DQNAgent):
                self.agent.step_env()
                envsteps_this_batch = 1
                train_video_paths = None
                paths = None
            else:
                use_batchsize = self.params['alg']['batch_size']
                if itr==0:
                    use_batchsize = self.params['alg']['batch_size_initial']
                paths, envsteps_this_batch, train_video_paths = (
                    self.collect_training_trajectories(
                        itr, initial_expertdata, collect_policy, use_batchsize)
                )

            if isinstance(self.agent, ExplorationOrExploitationAgent):
                if (not self.agent.offline_exploitation) or (self.agent.t <= self.agent.num_exploration_steps):
                    self.total_envsteps += envsteps_this_batch

            # relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # add collected data to replay buffer
            if isinstance(self.agent, ExplorationOrExploitationAgent):
                if (not self.agent.offline_exploitation) or (self.agent.t <= self.agent.num_exploration_steps):
                    self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            if itr % print_period == 0:
                print("\nTraining agent...")
            all_logs = self.train_agent()

            # Log densities and output trajectories
            if isinstance(self.agent, ExplorationOrExploitationAgent) and (itr % print_period == 0):
                #self.dump_density_graphs(itr)
                pass

            # log/save
            if self.logvideo or self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                if isinstance(self.agent, ExplorationOrExploitationAgent) or isinstance(self.agent, DQNAgent):
                    
                    self.perform_dqn_logging(all_logs)
                else:
                    self.perform_logging(itr, paths, eval_policy, train_video_paths, all_logs)

                if self.params['logging']['save_params']:
                    self.agent.save('{}/agent_itr_{}.pt'.format(self.params['logging']['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(self, itr, initial_expertdata, collect_policy, num_transitions_to_sample, save_expert_data_to_disk=False):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param num_transitions_to_sample:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # TODO: get this from hw1 or hw2
        if itr == 0:
            if self._params['env']['env_name'] == 'Visual-cube':
                ob_dtype = np.uint8
                action_dtype = np.float32

                loaded_paths = og.load_dataset(
                    initial_expertdata,
                    ob_dtype=ob_dtype,
                    action_dtype=action_dtype,
                    compact_dataset=False,
                )
            else:
                print(f'itr = {itr}, Loading initial expert data')
                with open(load_initial_expertdata, 'rb') as f:
                    loaded_paths = pickle.load(f)  # Load expert data from a file
            return loaded_paths, 0, None  # Return loaded data

        print("\nCollecting data to be used for training...")

        paths, envsteps_this_batch = (utils.sample_trajectories(
            self._env,
            collect_policy,
            batch_size,
            self._params['env']['max_episode_length']))
        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN

        train_video_paths = None
        if self._log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = utils.sample_n_trajectories(self._env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
        return paths, envsteps_this_batch, train_video_paths
    
    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = [{'loss': 0}]
        for t in range(1, self.params['alg']['num_timesteps']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params["alg"]["train_batch_size"])  # M

            if ob_batch != np.array([]):
                #print("next_ob_batch in train_agent: ", next_ob_batch.shape )
                train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)  # M
                if t % self.params['alg']['target_update_freq'] == 0:
                    self.agent.critic.update_target_network()
                all_logs.append(train_log)

            self.agent.step_env()

        return all_logs

    ####################################
    ####################################

    def do_relabel_with_expert(self, expert_policy, paths):
        raise NotImplementedError
        # hw1/hw2, can ignore it b/c it's not used for this hw

    ####################################
    ####################################
    
    def perform_dqn_logging(self, all_logs):
        last_log = all_logs[-1]

        episode_rewards = self.env.get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

        logs = OrderedDict()

        logs["Train_EnvstepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)

        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start

        logs.update(last_log)
        
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.eval_env, self.agent.actor, self.params['alg']['eval_batch_size'], self.params['env']['max_episode_length'])
        
        eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
        eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

        logs["Eval_AverageReturn"] = np.mean(eval_returns)
        logs["Eval_StdReturn"] = np.std(eval_returns)
        logs["Eval_MaxReturn"] = np.max(eval_returns)
        logs["Eval_MinReturn"] = np.min(eval_returns)
        logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)
        
        logs['Buffer_size'] = self.agent.replay_buffer.num_in_buffer

        sys.stdout.flush()

        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print('Done logging...\n\n')

        self.logger.flush()

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(self.env, eval_policy, self.params['alg']['eval_batch_size'], self.params['env']['max_episode_length'])

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                try:
                    self.logger.log_scalar(value, key, itr)
                except:
                    pdb.set_trace()
            print('Done logging...\n\n')

            self.logger.flush()

    def dump_density_graphs(self, itr):
        import matplotlib.pyplot as plt
        self.fig = plt.figure()
        filepath = lambda name: self.params['logging']['logdir']+'/curr_{}.png'.format(name)

        num_states = self.agent.replay_buffer.num_in_buffer - 2
        states = self.agent.replay_buffer.obs[:num_states]
        if num_states <= 0: return
        
        H, xedges, yedges = np.histogram2d(states[:,0], states[:,1], range=[[0., 1.], [0., 1.]], density=True)
        plt.imshow(np.rot90(H), interpolation='bicubic')
        plt.colorbar()
        plt.title('State Density')
        self.fig.savefig(filepath('state_density'), bbox_inches='tight')

        plt.clf()
        ii, jj = np.meshgrid(np.linspace(0, 1), np.linspace(0, 1))
        obs = np.stack([ii.flatten(), jj.flatten()], axis=1)
        density = self.agent.exploration_model.forward_np(obs)
        density = density.reshape(ii.shape)
        plt.imshow(density[::-1])
        plt.colorbar()
        plt.title('RND Value')
        self.fig.savefig(filepath('rnd_value'), bbox_inches='tight')

        plt.clf()
        exploitation_values = self.agent.exploitation_critic.qa_values(obs).mean(-1)
        exploitation_values = exploitation_values.reshape(ii.shape)
        plt.imshow(exploitation_values[::-1])
        plt.colorbar()
        plt.title('Predicted Exploitation Value')
        self.fig.savefig(filepath('exploitation_value'), bbox_inches='tight')

        plt.clf()
        exploration_values = self.agent.exploration_critic.qa_values(obs).mean(-1)
        exploration_values = exploration_values.reshape(ii.shape)
        plt.imshow(exploration_values[::-1])
        plt.colorbar()
        plt.title('Predicted Exploration Value')
        self.fig.savefig(filepath('exploration_value'), bbox_inches='tight')
