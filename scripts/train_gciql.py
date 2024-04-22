import sys; sys.path.append('./'); # print(sys.path)
import hydra
import wandb
import random
import minari
import numpy as np
import gymnasium as gym
from pathlib import Path
from datetime import datetime
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import DecisionMLP
from utils import MinariEpisodicDataset, convert_remote_to_local, get_test_start_state_goals, get_lr, AntmazeWrapper 
import pdb
import os.path as osp

import utils
from utils.eval_utils import eval_env_gciql_luo
from utils.misc import set_seed, get_current_time
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("now", get_current_time)

def train(cfg, hydra_cfg):

    #set seed
    set_seed(cfg.seed)

    #set device
    device = torch.device(cfg.device)

    if cfg.save_snapshot:
        # checkpoint_path = Path(hydra_cfg['runtime']['output_dir']) / Path('checkpoint')
        checkpoint_path = Path(cfg.save_path) / Path('checkpoint')
        checkpoint_path.mkdir(exist_ok=True)
        best_eval_returns = -100
    # pdb.set_trace()

    start_time = datetime.now().replace(microsecond=0)
    time_elapsed = start_time - start_time
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")

    if "pointmaze" in cfg.dataset_name:
        if "umaze" in cfg.dataset_name:
            cfg.env_name = 'PointMaze_UMaze-v3'
            cfg.nclusters = 20 if cfg.nclusters is None else cfg.nclusters
        elif "medium" in cfg.dataset_name:
            cfg.env_name = 'PointMaze_Medium-v3'
            cfg.nclusters = 40 if cfg.nclusters is None else cfg.nclusters
        elif "large" in cfg.dataset_name:
            cfg.env_name = 'PointMaze_Large-v3'
            cfg.nclusters = 80 if cfg.nclusters is None else cfg.nclusters
        env = gym.make(cfg.env_name, continuing_task=False)

    elif "antmaze" in cfg.dataset_name:
        if "umaze" in cfg.dataset_name:
            cfg.env_name = 'AntMaze_UMaze-v4'
            cfg.nclusters = 20 if cfg.nclusters is None else cfg.nclusters
        elif "medium" in cfg.dataset_name:
            cfg.env_name = 'AntMaze_Medium-v4'
            cfg.nclusters = 40 if cfg.nclusters is None else cfg.nclusters
        elif "large" in cfg.dataset_name:
            cfg.env_name = 'AntMaze_Large-v4'
            cfg.nclusters = 80 if cfg.nclusters is None else cfg.nclusters
        else:
            raise NotImplementedError
        env = AntmazeWrapper(gym.make(cfg.env_name, continuing_task=False))

    else:
        raise NotImplementedError
    
    env.action_space.seed(cfg.seed)
    env.observation_space.seed(cfg.seed)

    # ----------- create dataset -------------
    if cfg.remote_data:
        convert_remote_to_local(cfg.dataset_name, env)
   
    train_dataset = MinariEpisodicDataset(cfg.dataset_name, cfg.remote_data, cfg.augment_data, cfg.augment_prob, cfg.nclusters, return_reward=cfg.return_reward)
    
    train_data_loader = DataLoader(
                            train_dataset,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            num_workers=cfg.num_workers
                        )
    train_data_iter = iter(train_data_loader)

    # --------------- create model -----------------
    # model = DecisionMLP(cfg.env_name, env, goal_dim=train_dataset.goal_dim).to(device)
    from models.iql.iql_prep import get_gciql_models
    from hydra.utils import instantiate

    obs_shape = env.observation_space['observation'].shape
    assert len(obs_shape) == 1
    obs_dim = obs_shape[0]
    goal_dim = env.observation_space['achieved_goal'].shape[0]
    action_dim = env.action_space.shape[0]

    # qf_kwargs=dict(hidden_sizes=[256, 256, ],)
    # vf_kwargs = dict(hidden_sizes=[256, 256, ],)
    qf1, qf2, target_qf1, target_qf2, vf = \
                get_gciql_models(obs_dim, action_dim, goal_dim, cfg.qf_kwargs, cfg.vf_kwargs)
    
    gc_policy = instantiate(cfg.policy, obs_dim=obs_dim, action_dim=action_dim, goal_dim=goal_dim)

    ### IQL trainer
    from models.iql.iql_trainer import IQLTrainer
    trainer: IQLTrainer = instantiate(cfg.trainer,
        env=env,
        policy=gc_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vf=vf,)
    trainer.to(device)

    total_updates = 0
    utils.print_color('Start Training')
    for i_train_iter in range(cfg.max_train_iters):

        log_action_losses = []
        trainer.set_models_train_mode()
        
        for i in range(cfg.num_updates_per_iter):            
            try:
                batch = next(train_data_iter)

            except StopIteration:
                train_data_iter = iter(train_data_loader)
                batch = next(train_data_iter)

            trainer.train_from_torch(batch)
            
            ### *** torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25) ***
            log_action_losses.append(trainer.eval_statistics['Policy Loss'])
            # pdb.set_trace()

        time = datetime.now().replace(microsecond=0) - start_time - time_elapsed
        time_elapsed = datetime.now().replace(microsecond=0) - start_time

        total_updates += cfg.num_updates_per_iter
        
        mean_action_loss = np.mean(log_action_losses)

        results = eval_env_gciql_luo(cfg, gc_policy, device, render=cfg.render)

        log_str = ("=" * 60 + '\n' +
                "time elapsed: " + str(time_elapsed)  + '\n' +
                "num of updates: " + str(total_updates) + '\n' +
                "train action loss: " +  format(mean_action_loss, ".5f") #+ '\n' +
            )
        
        print(results)
        print(log_str)
        
        if cfg.wandb_log:
            log_data = dict()
            log_data['time'] =  time.total_seconds()
            log_data['time_elapsed'] =  time_elapsed.total_seconds()
            log_data['total_updates'] =  total_updates
            log_data['mean_action_loss'] =  mean_action_loss
            # log_data['lr'] = get_lr(optimizer)
            log_data['training_iter'] = i_train_iter
            log_data.update(results)
            wandb.log(log_data)
            wandb.log(trainer.eval_statistics)

        trainer_params = trainer.get_trainer_snapshot()
        if cfg.save_snapshot and (1+i_train_iter)%cfg.save_snapshot_interval == 0:
            snapshot = Path(checkpoint_path) / Path(str(i_train_iter)+'.pt')
            torch.save(trainer_params, snapshot)

        if cfg.save_snapshot and results['eval/avg_reward'] >= best_eval_returns:
            print("=" * 60)
            print("saving best model!")
            print("=" * 60)
            best_eval_returns = results['eval/avg_reward']
            snapshot = Path(checkpoint_path) / 'best.pt'
            torch.save(trainer_params, snapshot)

    print("=" * 60)
    print("finished training!")
    print("=" * 60)
    end_time = datetime.now().replace(microsecond=0)
    time_elapsed = str(end_time - start_time)
    end_time_str = end_time.strftime("%y-%m-%d-%H-%M-%S")
    print("started training at: " + start_time_str)
    print("finished training at: " + end_time_str)
    print("total training time: " + time_elapsed)
    print("=" * 60)
    
@hydra.main(config_path='../', version_base=None)
def main(cfg: DictConfig):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    # print(cfg)
    # osp.basename(cfg.cfg_path) # yaml filename
    cfg_name = osp.splitext(hydra_cfg.job.config_name)[0]
    cfg.save_path = osp.join(cfg.log_dir, cfg.dataset_name, cfg_name, cfg.log_time)
    
    if cfg.wandb_log:
        if cfg.wandb_dir is None:
            # cfg.wandb_dir = hydra_cfg['runtime']['output_dir']
            cfg.wandb_dir = cfg.save_path
                # hydra_cfg['runtime']['output_dir']
        
        # project_name = cfg.dataset_name
        wandb.init(project='2951f-stitch-2024spring', entity=cfg.wandb_entity, 
                   config=dict(cfg), dir=cfg.wandb_dir, group=cfg.dataset_name,
                   name=cfg_name, id=cfg_name)
        # wandb.run.name = cfg.wandb_run_name
    # pdb.set_trace()
    
    train(cfg, hydra_cfg)
        
if __name__ == "__main__":
    main()