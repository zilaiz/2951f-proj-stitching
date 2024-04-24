import hydra
import wandb
import random
import minari
import numpy as np
import gymnasium as gym

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import DecisionMLP
from utils import MinariEpisodicDataset, convert_remote_to_local, get_test_start_state_goals, get_lr, AntmazeWrapper 
from rlkit.torch.sac.policies import MakeDeterministic
import pdb, imageio
import utils, os
import os.path as osp
import wandb

def eval_env_gciql_luo(cfg, model, device, render=False):
    '''model is a goal conditioned policy'''
    if render:
        render_mode = 'human'
    else:
        render_mode = 'rgb_array' # None

    if "pointmaze" in cfg.dataset_name:
        env = env = gym.make(cfg.env_name, continuing_task=False, render_mode=render_mode)
    elif "antmaze" in cfg.dataset_name:
        env = AntmazeWrapper(env = gym.make(cfg.env_name, continuing_task=False, render_mode=render_mode))
    else:
        raise NotImplementedError    

    test_start_state_goal = get_test_start_state_goals(cfg)

    os.makedirs(osp.join(cfg.save_path, 'eval'), exist_ok=True)
    model = MakeDeterministic(model)
    model.eval()
    results = dict()
    with torch.no_grad():
        cum_reward = 0
        for idx_prob, ss_g in enumerate(test_start_state_goal):
            total_reward = 0
            total_timesteps = 0
            env_imgs = [] # save the whole rollout
            save_img_freq = 5
            
            print(ss_g['name'] + ':')
            for i_step in range(cfg.num_eval_ep):
                obs, _ = env.reset(options=ss_g)
                done = False
                for t in range(env.spec.max_episode_steps):
                    total_timesteps += 1

                    running_state = torch.tensor(obs['observation'], dtype=torch.float32, device=device).view(1, -1)
                    target_goal = torch.tensor(obs['desired_goal'], dtype=torch.float32, device=device).view(1, -1)
                    
                    act_preds = model.forward(
                            running_state,
                            target_goal,
                            )
                    ## TODO: can develop to a probabilistic policy
                    act_preds = act_preds.mean # extract from the delta distribution
                    # pdb.set_trace()
                    act = act_preds[0].detach()

                    obs, running_reward, done, _, _ = env.step(act.cpu().numpy())

                    total_reward += running_reward

                    if i_step % save_img_freq == 0:
                        env_imgs.append(env.render())

                    if done:
                        break
                
                print('Achievied goal: ', tuple(obs['achieved_goal'].tolist()))
                print('Desired goal: ', tuple(obs['desired_goal'].tolist()))
                
                eval_save_st_gl_imgs(cfg, env_imgs, idx_prob)
                eval_save_video(cfg, env_imgs, idx_prob, freq=5)
            
            print("=" * 60)
            cum_reward += total_reward
            results['eval/' + str(ss_g['name']) + '_avg_reward'] = total_reward / cfg.num_eval_ep
            results['eval/' + str(ss_g['name']) + '_avg_ep_len'] = total_timesteps / cfg.num_eval_ep
        
        results['eval/avg_reward'] = cum_reward / (cfg.num_eval_ep * len(test_start_state_goal))
        env.close()
    return results


def eval_save_st_gl_imgs(cfg, env_imgs, idx_prob):
    '''save the start and final frames of the whole rollouts'''
    save_path = osp.join(cfg.save_path, 'eval', f'st_gl_{idx_prob}.png')
    utils.create_imgs_grid([env_imgs[0], env_imgs[-1]], n_cols=2, captions=None, save_path=save_path)


def eval_save_video(cfg, env_imgs, idx_prob, freq):
    '''save an video to gif format'''
    env_imgs = env_imgs[::freq]
    save_path = osp.join(cfg.save_path, 'eval', f'rollout_{idx_prob}.gif')
    imageio.mimsave(save_path, env_imgs, duration=0.1)
    print(f'[Save gif] to {save_path}')