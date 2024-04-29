import torch.nn as nn
import torch

class GoalCondObsEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # self.obs_enc = nn.Linear(input_dim, hidden_dim)
        self.x_emb = nn.Embedding(10, 12)
        self.y_emb = nn.Embedding(10, 12)
        self.d_emb = nn.Embedding(4, 6)
    
    def forward(self, states, goals = None):
        state_x = self.x_emb(states[:, 0])
        state_y = self.y_emb(states[:, 1])
        state_d = self.d_emb(states[:, 2])
        if goals:
            goal_x = self.x_emb(goals[:, 0])
            goal_y = self.y_emb(goals[:, 1])
        # state_goal = torch.cat([states, goals], dim=-1).float()
        # obs_emb = self.obs_enc(state_goal)
            obs_emb = torch.cat([state_x, state_y, state_d, goal_x, goal_y], dim=-1).float()
        else:
            obs_emb = torch.cat([state_x, state_y, state_d], dim=-1).float()
        

        return obs_emb


class ObsEncoder(nn.Module):
    def __init__(self, num_obs_unique, hidden_dim):
        super().__init__()
        self.obs_embedding = nn.Embedding(num_obs_unique, hidden_dim)
        # self.obs_embedding = nn.Linear(num_obs_unique, hidden_dim)
    
    def forward(self, obs, goal=None):
        obs_emb = self.obs_embedding(obs)

        return obs_emb


class BCPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        # self.obs_enc = GoalCondObsEncoder(input_dim, hidden_dim)
        # self.obs_enc = ObsEncoder(input_dim, hidden_dim)
        
        self.policy_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    
    def forward(self, states, goals):
        # obs_emb = self.obs_enc(states, goals)
        actions = self.policy_network(states)

        return actions
    


class DoubleCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # DoubleCritic
        self.critic1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, observations, actions):
        inputs = torch.cat([observations, actions], dim=-1)
        q1 = self.critic1(inputs)
        q2 = self.critic2(inputs)

        return q1, q2



class IQL(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, tau):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.tau = tau

        # self.obs_enc = GoalCondObsEncoder(input_dim, self.hidden_dim)
        # self.obs_enc = ObsEncoder(self.input_dim, self.hidden_dim)
        self.critic = DoubleCritic(input_dim = self.obs_dim + self.action_dim, hidden_dim=self.hidden_dim)
        self.target_critic = DoubleCritic(input_dim = self.obs_dim + self.action_dim, hidden_dim=self.hidden_dim)

        self.actor = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        # Value function
        self.value = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.target_update()
    
    def critic_forward(self, observations, actions):
        q1, q2 = self.critic(observations, actions)
        return q1, q2
    
    def target_critic_forward(self, observations, actions):
        target_q1, target_q2 = self.target_critic(observations, actions)

        return target_q1, target_q2

    def actor_forward(self, observations):
        logits = self.actor(observations)

        return logits

    def value_forward(self, observations):
        v = self.value(observations)

        return v
    
    def forward(self, observations, goals):
        # obs_emb = self.obs_enc(observations, goals)
        # actions = self.actor(observations)

        return self.actor_forward(observations)
        
    
    def target_update(self):  # update target
        for target_param, source_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)