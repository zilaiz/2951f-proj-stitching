import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from tqdm import tqdm
import cv2
import shutil

class MiniGridDataset(Dataset):
    def __init__(self, env, trajs):
        super().__init__()
        self.env = env
        self.observations = np.array(trajs['observations'])
        self.actions = np.array(trajs['actions'])
        # self.rewards = trajs['rewards']
        self.dones = np.array(trajs['dones'])
        self.rewards = np.full_like(self.actions, 0)

        self.all_unique_obs = [[pos[0], pos[1], d] for d in range(0, 4) for pos in self.env.all_empty_cells]
    
    def update_data(self, eval_goal=None):
        self.eval_goal = eval_goal
        if self.eval_goal:
            self.calculate_rewards()

        redundant_indices = np.argwhere(self.actions == 3).reshape(-1)
        self.updated_observations = np.delete(self.observations, redundant_indices, axis=0).tolist()
        self.updated_actions = np.delete(self.actions, redundant_indices, axis=0).tolist()
        self.updated_rewards = np.delete(self.rewards, redundant_indices, axis=0).tolist()
        self.updated_dones = np.delete(self.dones, redundant_indices, axis=0).tolist()
        
    def calculate_rewards(self):
        if self.eval_goal:
            for idx, obs in enumerate(self.observations):
                if obs[0] == self.eval_goal[0] and obs[1] == self.eval_goal[1]:
                    self.rewards[idx - 1] = 1.

    def obs_to_one_hot(self, observation):
        # identity_matrix = np.eye(len(self.all_unique_obs))
        # return identity_matrix[self.all_unique_obs.index(observations)]
        return self.all_unique_obs.index(observation)

    def __len__(self):
        return len(self.updated_actions) - 1

    def __getitem__(self, index):
        # i = index // len(self.actions[0])
        # j = index - i * len(self.actions[0])

        done = torch.tensor(self.updated_dones[index], dtype=torch.float32)
        obs = torch.tensor(self.obs_to_one_hot(self.updated_observations[index])).long()
        action = torch.tensor([self.updated_actions[index]])
        next_obs = torch.tensor(self.obs_to_one_hot(self.updated_observations[index + 1])).long()
        goal = next_obs.clone()
        one_hot_action = torch.eye(4)[action].view(-1)
        reward = torch.tensor(self.updated_rewards[index])
        mask = 1 - (reward or done)
        
        return obs, action, reward, next_obs, goal, mask, one_hot_action