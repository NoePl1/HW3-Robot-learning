import numpy as np



class ArgMaxPolicy(object):

    def __init__(self, critic, use_boltzmann=False):
        self.critic = critic
        self.use_boltzmann = use_boltzmann

    def set_critic(self, critic):
        self.critic = critic

    def get_action(self, obs):
        # MJ: changed the dimension check to a 3
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output

        ## TODO return the action that maxinmizes the Q-value 
        # at the current observation as the output (get it from hw3)
        # NOTE: you should adapt your code so that it considers the boltzmann distribution case

        q_values = self.critic.qa_values(obs)

        if self.use_boltzmann:
            distribution = np.exp(q_values) / np.sum(np.exp(q_values))
            action = self.sample_discrete(distribution)

        else:
            action = q_values.argmax()

        return action

    def sample_discrete(self, p):
        # https://stackoverflow.com/questions/40474436/how-to-apply-numpy-random-choice-to-a-matrix-of-probability-values-vectorized-s
        c = p.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        choices = (u < c).argmax(axis=1)
        return choices

    ####################################
    ####################################