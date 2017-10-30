#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
If flag "--save" is specified then data will be saved to the file with name:
    
    "expert_data/"+str(args.num_rollouts).zfill(3)+args.envname+".npy"
    
Example usage:
    python run_expert.py experts/Ant-v1.pkl Ant-v1  --save --num_rollouts 5
    
Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import os
import os.path

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        all_actions = []
        all_observations = []
        all_rewards = []
        all_steps = []
        
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            totalr = 0.
            steps = 0
            done = False
            observations = []
            actions = []
            
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break

            all_rewards.append(totalr)
            all_steps.append(steps)
            
            while steps < max_steps:
                observations.append(obs)
                actions.append(action)
                steps += 1
                
            assert len(observations) == max_steps, "{}".format(len(observations))
            assert len(actions) == max_steps, "{}".format(len(actions))
            
            all_observations.append(observations)
            all_actions.append(actions)

        expert_data = {'observations': np.array(all_observations),
                       'actions': np.squeeze(np.array(all_actions)),
                       'rewards': all_rewards,
                       'steps': all_steps}
        
        print('steps', all_steps)
        print('rewards', all_rewards)
        print('mean reward', np.mean(all_rewards))
        print('std of reward', np.std(all_rewards))
        print("obs.shape = {}".format(expert_data['observations'].shape))
        print("act.shape = {}".format(expert_data['actions'].shape))
        
        if args.save:
            root = 'expert_data'
            if not os.path.exists(root):
                os.makedirs(root)
            str_roll = str(args.num_rollouts).zfill(3)
            np.save(root+"/"+str_roll+args.envname+".npy", expert_data)

if __name__ == '__main__':
    main()