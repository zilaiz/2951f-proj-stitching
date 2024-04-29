from data.policies import RandomPolicy, ExpertPolicy
import numpy as np

from tqdm import tqdm

def data_gen(env, total_ep = 2400, is_random = False):
    # step_lower_bound = 1_000_000 if is_random else 10_000
    ep_id = 0
    step_id = 0
    max_length = 30
    
    obs_list = []
    done_list = []
    action_list = []
    goal_list = []

    collected_start_goal_pair = set()
    with tqdm(total=total_ep, leave=True) as pbar:
        while ep_id < total_ep:
            # print("STEP", step_id)
            seed = np.random.randint(2**31 - 1)
            env.reset(seed=seed)
            if (env.agent_pos, env.agent_dir, env.goal_pos) in collected_start_goal_pair:
                continue
            collected_start_goal_pair.add((env.agent_pos, env.agent_dir, env.goal_pos))
            env.action_space.seed(seed)
            policy = RandomPolicy(env) if is_random else ExpertPolicy(env)
            done = False

            ep_obs_list = [[env.agent_pos[0], env.agent_pos[1], env.agent_dir]]
            ep_done_list = []
            ep_action_list = []

            while not done:
                step_id += 1
                action = policy.get_action()
                obs, rew, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_obs_list.append([env.agent_pos[0], env.agent_pos[1], env.agent_dir])
                ep_action_list.append(action)
                ep_done_list.append(done)
            
            if len(ep_action_list) >= (10 if is_random else 5):
                ep_action_list.append(3)
                ep_done_list.append(False)
                # if len(ep_action_list) < max_length:
                #     padding_num = max_length - len(ep_action_list)
                #     ep_action_list += [3] * padding_num
                #     ep_obs_list += [ep_obs_list[-1]] * padding_num
                #     ep_done_list += [True] * padding_num
                obs_list += ep_obs_list
                action_list += ep_action_list
                done_list += ep_done_list
                goal_list.append(env.goal_pos)
                # print(goal_list[-1])
                # print(ep_obs_list[-1])

                ep_id += 1
                pbar.update(1)
            else:
                step_id -= len(ep_action_list)

    
    print(f'Number of Total Transitions: {step_id}')
    print(f'Number of Total Episodes: {ep_id}')
    
    trajs = {
        'observations': obs_list,
        'actions': action_list,
        'dones': done_list,
        'goals': goal_list
    }
    
    return trajs