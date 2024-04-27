import gymnasium as gym
from utils import AntmazeWrapper 

def get_env(cfg):
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
    
    return env