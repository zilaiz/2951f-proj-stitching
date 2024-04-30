import hydra, wandb, random, minari
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
from models.gc_gaussian_policy import GC_GaussianPolicy

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'

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
    if isinstance(model, GC_GaussianPolicy):
        model = MakeDeterministic(model)
    model.eval()
    results = dict()
    save_ep_freq = 5

    with torch.no_grad():
        cum_reward = 0
        ## loop through group of problems
        for idx_group, ss_g in enumerate(test_start_state_goal):
            total_reward = 0
            total_timesteps = 0
            
            print(ss_g['name'] + ':')
            for i_ep in range(cfg.num_eval_ep):
                env_imgs = [] # save the whole rollout
                # pdb.set_trace()
                obs, _ = env.reset(options=ss_g)
                # pdb.set_trace()
                done = False
                ## rollout one episode (problem) below
                for t in range(env.spec.max_episode_steps):
                    total_timesteps += 1

                    running_state = torch.tensor(obs['observation'], dtype=torch.float32, device=device).view(1, -1)
                    target_goal = torch.tensor(obs['desired_goal'], dtype=torch.float32, device=device).view(1, -1)
                    
                    act_preds = model.forward(
                            running_state,
                            target_goal,
                            )
                    if not torch.is_tensor(act_preds): # a distribution
                        ## TODO: can develop to a probabilistic policy
                        act_preds = act_preds.mean # extract from the delta distribution
                    # pdb.set_trace()
                    act = act_preds[0].detach()

                    obs, running_reward, done, _, _ = env.step(act.cpu().numpy())

                    total_reward += running_reward

                    env_imgs.append(env.render())

                    if done:
                        break
                
                ## Finished rollout one episode
                print('Achievied goal: ', tuple(obs['achieved_goal'].tolist()))
                print('Desired goal: ', tuple(obs['desired_goal'].tolist()))
                
                ## vis when success
                save_vis = done and \
                        ("antmaze" in cfg.dataset_name or "large" in cfg.dataset_name.lower())
                if i_ep % save_ep_freq == 0 or save_vis:
                    eval_save_st_gl_imgs(cfg, env_imgs, f"{ss_g['name']}-{i_ep}-{done}")
                    eval_save_video(cfg, env_imgs, f"{ss_g['name']}-{i_ep}-{done}", freq=5)
            
            print("=" * 60)
            cum_reward += total_reward
            results['eval/' + str(ss_g['name']) + '_avg_reward'] = total_reward / cfg.num_eval_ep
            results['eval/' + str(ss_g['name']) + '_avg_ep_len'] = total_timesteps / cfg.num_eval_ep
        
        results['eval/avg_reward'] = cum_reward / (cfg.num_eval_ep * len(test_start_state_goal))
        env.close()
    return results


def eval_save_st_gl_imgs(cfg, env_imgs, prob_str):
    '''save the start and final frames of the whole rollouts'''
    save_path = osp.join(cfg.save_path, 'eval', f'{prob_str}_StEnd.png')
    utils.create_imgs_grid([env_imgs[0], env_imgs[-1]], n_cols=2, captions=None, save_path=save_path)


def eval_save_video(cfg, env_imgs, prob_str, freq):
    '''save an video to gif format'''
    env_imgs = env_imgs[::freq]
    save_path = osp.join(cfg.save_path, 'eval', f'{prob_str}_rollout.gif')
    imageio.mimsave(save_path, env_imgs, duration=0.1)
    print(f'[Save gif] to {save_path}')