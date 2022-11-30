import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal


class ActorCriticMLP(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim, action_std_init=None, continuous=False, use_cuda=True):
        super(ActorCriticMLP, self).__init__()
        self.continuous = continuous
        self.use_cuda = use_cuda
        if continuous:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim, ), action_std_init * action_std_init)
            if use_cuda:
                self.action_var = self.action_var.cuda()

            self.actor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, action_dim),
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        raise NotImplementedError

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)
            if self.use_cuda:
                self.action_var = self.action_var.cuda()
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def act(self, state):
        if self.continuous:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_prob = self.actor(state)
            dist = Categorical(action_prob)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.detach(), action_log_prob.detach()

    def evaluate(self, state, action):
        if self.continuous:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            if self.use_cuda:
                cov_mat = cov_mat.cuda()
            dist = MultivariateNormal(action_mean, cov_mat)
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_prob = self.actor(state)
            dist = Categorical(action_prob)
        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.critic(state)
        return action_log_prob, values, dist_entropy




