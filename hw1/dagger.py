import gym
from gym import wrappers
import load_policy
import tf_util
import tensorflow as tf
import numpy as np
import os
import os.path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split

class Policy(nn.Module):
    """
    ANN with 2 dense layers and ReLU activation function.
    It serves as a policy for the agent.
    """
    def __init__(self, obs_shape, action_shape):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(obs_shape, 64)
        self.fc2 = nn.Linear(64, action_shape)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Powerful_Policy(nn.Module):
    
    def __init__(self, obs_shape, action_shape):
        super(Powerful_Policy, self).__init__()
        self.fc1 = nn.Linear(obs_shape, 512)
        self.dr1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 512)
        self.dr2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, action_shape)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dr1(x)
        x = F.relu(self.fc2(x))
        x = self.dr2(x)
        x = self.fc3(x)
        return x

def preprocess(observation):
    """
    Description:
        This function converts data(numpy.ndarray) to the format,
        which is needed for PyTorch ANN

    Arguments:
        observation as numpy.ndarray

    Returns:
        observation as Variable(see pytorch references)
    """
    return Variable(torch.Tensor(observation).view(1, -1))

def load_dataset(args):
    """
    Desctiption:
        Loads observations and actions got from the expert policy(see run_expert.py)

    Arguments:
        args - input arguments. Must contain "envname" and "num_rollouts", because they
               define a file to load.
    Returns:
        expert_obs(numpy.ndarray) - expert observations
        expert_act(numpy.ndarray) - expert actions

    """
    str_roll = str(args.num_rollouts).zfill(3)
    file_path = "expert_data/"+str_roll+args.envname+".npy"
    expert_data = np.load(file_path)
    
    observations = expert_data[()]["observations"]
    actions = expert_data[()]["actions"]
    rewards = expert_data[()]["rewards"]
    steps = expert_data[()]["steps"]
    
    expert_obs = observations
    expert_act = actions
    
    return expert_obs, expert_act

def generate_new_data(expert_policy, policy, args):
    """
    Description:
        This function is the almost the only difference between
        DAgger and BC. It generates new data for every step of 
        DAgger and provides new dataset including new examples:
        pair of an observation got after an agent's action and
        an action from the expert for mentioned observation.

    Arguments:
        expert_policy - loaded policy from experts/
        policy - object of class Policy
        args - arguments of command line. Must include "envname", "num_rollouts"

    Returns:
        observations - numpy array of shape(N, number of observations)
        actions - numpy array of shape(N, number of actions)
    """
    with tf.Session():
        tf_util.initialize()
    
        env = gym.make(args.envname).unwrapped
        max_steps = env.spec.timestep_limit

        all_actions = []
        all_observations = []
        #all_rewards = []
        #all_steps = []

        for i in range(args.num_rollouts):
            obs_ = env.reset()
            obs  = preprocess(obs_)
            totalr = 0.
            steps = 0
            done = False
            observations = []
            actions = []

            while not done:
                output = policy(obs)
                action_= expert_policy(obs_.reshape(1, -1))
                action = output.data.numpy().reshape(1, -1)
                observations.append(obs.data.numpy())
                actions.append(action_)
                obs_, r, done, _ = env.step(action)
                obs = preprocess(obs_)
                totalr += r
                steps += 1
                if steps >= max_steps:
                    break

            #all_rewards.append(totalr)
            #all_steps.append(steps)

            while steps < max_steps:
                observations.append(obs.data.numpy())
                actions.append(action_)
                steps += 1

            assert len(observations) == max_steps, "{}".format(len(observations))
            assert len(actions) == max_steps, "{}".format(len(actions))

            all_observations.append(observations)
            all_actions.append(actions)
    
    return np.squeeze(all_observations), np.squeeze(all_actions)

def concatenate_data(old_observations, old_actions, new_observations, new_actions):
    """
    Desctiption:
        Concatenates old and new data.

    Arguments:
        old_observations, old_actions - data from the previous step
        new_observations, new_actions - newly generated data

    Returns:
        (old_observations, new_observations) - row-wise concatenated numpy.ndarrays of observations
        (old_actions, new_actions) - row-wise concatenated numpy.ndarrays of actions
    """
    assert old_observations.shape[1:] == new_observations.shape[1:]
    assert old_actions.shape[1:] == new_actions.shape[1:]
    
    return np.vstack((old_observations, new_observations)), np.vstack((old_actions, new_actions))

def train_policy(expert_obs, expert_act, args, policy=None):
    """
    Description:
        This function is responsible for ANN training.

    Arguments:
        expert_obs - array of observations
        expert_act - array of actions
        args - command line arguments
        policy - if policy is None then new policy will be initialized. Otherwise,
                 policy will continue training on a new dataset.
    Returns:
        history - statistics(epochs, iterations, loss)
        policy - trained neural network
    """
    if policy == None:
        if args.policy_type == 0:
            policy = Policy(expert_obs.shape[2], expert_act.shape[2])
        if args.policy_type == 1:
            policy = Powerful_Policy(expert_obs.shape[2], expert_act.shape[2])
    
    history = []
    
    criterion = nn.MSELoss()
    learning_rate = args.lr
    optimizer = optim.Adam(policy.parameters(), learning_rate, weight_decay=0.00001)
    #optimizer = optim.RMSprop(policy.parameters())
    
    all_observations = expert_obs.reshape(-1, expert_obs.shape[2])
    all_actions = expert_act.reshape(-1, expert_act.shape[2])
    train_obs, test_obs, train_act, test_act = train_test_split(all_observations, all_actions,
                                                                test_size=0., shuffle=True)
    #print(train_obs.shape)
    #obs_mean = all_observations.mean(axis=0)
    #obs_std  = all_observations.std(axis=0)
    
    #obs_std[np.where(obs_std == 0.)] = np.random.normal(0,0.1,1) * 10e-10
    #all_observations = (all_observations - obs_mean) / obs_std
   
    for epoch in range(args.epochs):
        
        running_loss = 0.0
        num_observations = train_obs.shape[0]
        num_batches = num_observations // args.batch_size

        for i in range(num_batches):
            observations = train_obs[i*args.batch_size:(i+1)*args.batch_size, :]
            actions = train_act[i*args.batch_size:(i+1)*args.batch_size, :]
            observations = Variable(torch.Tensor(observations).view(-1, all_observations.shape[1]))
            actions = Variable(torch.Tensor(actions).view(-1, all_actions.shape[1]))

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward -> backward -> optimize
            predicted_actions = policy(observations)
            loss = criterion(predicted_actions, actions)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.data[0]
            history.append([epoch * num_batches * args.batch_size + (i + 1) * args.batch_size, running_loss])
            running_loss = 0.0
                
    return history, policy

def validation(policy, args, root_dir="demonstrations_da", validation_rollouts=10, simplified=False):
    """
    Description:
        This function shows preformance of the agent with the provided policy,
        if "--render" and "--demo" flags are specified in the command line.

    Arguments:
        policy - policy to estimate
        args - command line arguments

    Returns:
        None
    """
    env = gym.make(args.envname)

    if not simplified and args.save_demo:
        tail = args.envname+"_"+str(args.num_rollouts).zfill(3)+"_"+str(args.dagger_steps).zfill(2)
        if args.policy_type == 1:
            root_dir = str(args.policy_type)+"_"+full_path
        full_path = root_dir+"/"+tail
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        env = wrappers.Monitor(env, full_path, force=True, video_callable=lambda x: x == 0)

    max_steps = env.spec.timestep_limit
        
    rewards = []
        
    for i in range(validation_rollouts):
        obs = preprocess(env.reset())
        done = False
        totalr = 0.
        step = 0
        while not done:
            output = policy(obs)
            action = output.data.numpy().reshape(1, -1)
            obs, r, done, _ = env.step(action)
            obs = preprocess(obs)
            totalr += r
            step += 1
            if not simplified and args.render:
                env.render()
            if step >= max_steps:
                break
        rewards.append(totalr)
    
    if not simplified:
        #print('rewards', rewards)
        print('mean reward', np.mean(rewards))
        print('std of reward', np.std(rewards))
        
    return (np.mean(rewards), np.std(rewards))

if __name__ == '__main__':
    #loading expert policy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--lr', type=float, default=0.007)
    parser.add_argument('--dagger_steps', type=int, default=5)
    parser.add_argument('--num_rollouts', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--policy_type', type=int, default=0)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--save_demo', action='store_true')
    parser.add_argument('--draw_graph', action='store_true')
    parser.add_argument('--save_graph', action='store_true')
    parser.add_argument('--draw_training_validation_curve', action='store_true')
    parser.add_argument('--save_training_validation_curve', action='store_true')
    args = parser.parse_args()
    #Setting tensorflow flags to suppress useless warning messages.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    tf.logging.set_verbosity(tf.logging.FATAL)
    #Loading expert policy
    expert_policy = load_policy.load_policy(args.expert_policy_file)
    print('Running DAgger for', args.envname)
    
    history, policy, train_obs, train_act = None, None, None, None
    #Loading prepared expert dataset
    train_obs, train_act = load_dataset(args)
    #Training on the initial dataset
    history, policy = train_policy(train_obs, train_act, args)

    rewards = []
    policies = [policy]
    histories = [history]
    new_train_obs, new_train_act = train_obs, train_act
    rewards.append(validation(policy, args, simplified=True))
    #print("Finished Dagger Step 0")
    #DAgger steps
    for i in range(args.dagger_steps):
        new_train_obs, new_train_act = concatenate_data(new_train_obs, new_train_act,
                                                        *generate_new_data(expert_policy, policies[-1], args))
        new_history, new_policy = train_policy(new_train_obs, new_train_act, args, policies[-1])
        policies.append(new_policy)
        histories.append(new_history)
        rewards.append(validation(new_policy, args, simplified=True))
        #print('Finished Dagger Step', str(i + 1))
        
    histories = list(map(np.array, histories))

    if args.draw_graph or args.save_graph:
        loss_fig = plt.figure()
        loss_ax = loss_fig.add_subplot(111)
        reward_fig = plt.figure()
        reward_ax = reward_fig.add_subplot(111)
        loss_ax.set_xlabel("DAgger step")
        loss_ax.set_ylabel("Loss")
        reward_ax.set_xlabel("DAgger step")
        reward_ax.set_ylabel("Reward")
        loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        reward_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        loss_ax.plot(range(args.dagger_steps + 1), list(map(np.mean, map(lambda x: x[:,1], histories))))
        reward_ax.errorbar(range(args.dagger_steps + 1), list(map(lambda x: x[0], rewards)),
                           yerr=list(map(lambda x: x[1], rewards)))
        
        if args.save_graph:
            root_dir = "figures_da"
            if args.policy_type == 1:
                root_dir = str(args.policy_type)+"_"+root_dir
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            str_roll = str(args.num_rollouts).zfill(3)
            loss_fig.savefig(root_dir+"/loss"+"_"+args.envname+"_"+str_roll+"_"+str(args.dagger_steps).zfill(2))
            reward_fig.savefig(root_dir+"/reward"+"_"+args.envname+"_"+str_roll+"_"+str(args.dagger_steps).zfill(2))
            
        if args.draw_graph:
            plt.show()
        
    if args.demo or args.save_demo:
        validation(policies[-1], args)