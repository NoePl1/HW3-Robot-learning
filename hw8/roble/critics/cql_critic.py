from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn


from hw8.roble.infrastructure import pytorch_util as ptu


class CQLCritic(BaseCritic):

    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.env_name = hparams['env']['env_name']
        self.ob_dim = hparams['alg']['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim = hparams['alg']['ac_dim']
        self.double_q = hparams['alg']['double_q']
        self.grad_norm_clipping = hparams['alg']['grad_norm_clipping']
        self.gamma = hparams['alg']['gamma']

        self.optimizer_spec = hparams['optimizer_spec']
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)
        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.cql_alpha = hparams['alg']['cql_alpha']

    def dqn_loss(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """ Implement DQN Loss """
        qa_t_values = self.q_net(ob_no)
        q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
        qa_tp1_values = self.q_net_target(next_ob_no)
        if self.double_q:
            q_best_action = torch.argmax(qa_t_values, dim=1, keepdim=True)
            q_tp1 = torch.gather(qa_tp1_values, dim=1, index=q_best_action).squeeze(1)
        else:
            q_tp1, _ = qa_tp1_values.max(dim=1)
        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        target = target.detach()

        assert q_t_values.shape == target.shape
        loss = self.loss(q_t_values, target)
        return loss, qa_t_values, q_t_values


    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        """
            Update the parameters of the critic.
            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories
            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end
            returns:
                nothing
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)

        # Compute the DQN Loss 
        loss, qa_t_values, q_t_values = self.dqn_loss(ob_no, ac_na, next_ob_no, reward_n, terminal_n)
        
        # CQL Implementation
        q_all_values = self.q_net(ob_no)
        q_t_logsumexp = torch.logsumexp(q_all_values, dim=1).mean()
        cql_loss = self.cql_alpha * (q_t_logsumexp - q_t_values.mean())

        # Total loss: DQN loss + CQL loss
        total_loss = loss + cql_loss

        info = {'Training_Loss': ptu.to_numpy(total_loss)}

        info['CQL_Loss'] = ptu.to_numpy(cql_loss)
        info['Data_q-values'] = ptu.to_numpy(q_t_values).mean()
        info['OOD_q-values'] = ptu.to_numpy(q_t_logsumexp).mean()

        # Perform gradient update
        self.optimizer.zero_grad()
        total_loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        self.learning_rate_scheduler.step()

        return info

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs):
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
