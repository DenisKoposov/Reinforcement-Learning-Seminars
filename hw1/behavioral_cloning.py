import gym
from gym import wrappers
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
        #init.xavier_uniform(self.fc1.weight)
        self.fc2 = nn.Linear(64, action_shape)
        #init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
    
    expert_train_obs = observations
    expert_train_act = actions
    
    return expert_train_obs, expert_train_act

def train_policy(expert_obs, expert_act, args):
    """
    Description:
        This function is responsible for ANN training.

    Arguments:
        expert_obs - array of observations
        expert_act - array of actions
        args - command line arguments

    Returns:
        history - statistics(epochs, iterations, loss)
        policy - trained neural network
    """
    policy = Policy(expert_obs.shape[2], expert_act.shape[2])
    histories = []
    rewards = []
    
    criterion = nn.MSELoss()
    learning_rate = args.lr
    optimizer = optim.Adam(policy.parameters(), learning_rate, weight_decay=0.0000001) 
    #optimizer = optim.RMSprop(policy.parameters())
    
    all_observations = expert_obs.reshape(-1, expert_obs.shape[2])
    all_actions = expert_act.reshape(-1, expert_act.shape[2])
    train_obs, test_obs, train_act, test_act = train_test_split(all_observations, all_actions, test_size=0., shuffle=True)
    #obs_mean = all_observations.mean(axis=0)
    #obs_std  = all_observations.std(axis=0)
    
    #obs_std[np.where(obs_std == 0.)] = np.random.normal(0,0.1,1) * 10e-10
    #all_observations = (all_observations - obs_mean) / obs_std
    
    for epoch in range(args.epochs):
        
        history = []
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
        print('Epoch', str(epoch))
        rewards.append(validation(policy, args, simplified=True))
        histories.append(history)
                

    print('Finished training')
    return histories, policy, rewards

def validation(policy, args, root_dir="demonstrations_bc", validation_rollouts=10, simplified=False):
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
        tail = args.envname+"_"+str(args.num_rollouts).zfill(3)+"_"+str(args.epochs).zfill(2)
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
            
    #print('rewards', rewards)
    print('mean reward', np.mean(rewards))
    print('std of reward', np.std(rewards))
    return (np.mean(rewards), np.std(rewards))
            
if __name__ == '__main__':
    #loading expert policy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--lr', type=float, default=0.007)
    parser.add_argument('--num_rollouts', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--save_demo', action='store_true')
    parser.add_argument('--draw_graph', action='store_true')
    parser.add_argument('--save_graph', action='store_true')
    args = parser.parse_args()
    
    history, policy, train_obs, train_act = None, None, None, None
    
    #getting dataset from the expert policy
    train_obs, train_act = load_dataset(args)
    #creating and training neural network, which represents a new policy
    history, policy, rewards = train_policy(train_obs, train_act, args)
    history = np.array(history)
        
    if args.draw_graph or args.save_graph:
        loss_fig = plt.figure()
        loss_ax = loss_fig.add_subplot(111)
        reward_fig = plt.figure()
        reward_ax = reward_fig.add_subplot(111)
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        reward_ax.set_xlabel("Epoch")
        reward_ax.set_ylabel("Reward")
        loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        reward_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        loss_ax.plot(range(args.epochs), list(map(np.mean, map(lambda x: x[:,1], history))))
        reward_ax.errorbar(range(args.epochs), list(map(lambda x: x[0], rewards)),
                           yerr=list(map(lambda x: x[1], rewards)))
        
        if args.save_graph:
            root_dir = "figures_bc"
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            str_roll = str(args.num_rollouts).zfill(3)
            loss_fig.savefig(root_dir+"/loss"+"_"+args.envname+"_"+str_roll+"_"+str(args.epochs).zfill(2))
            reward_fig.savefig(root_dir+"/reward"+"_"+args.envname+"_"+str_roll+"_"+str(args.epochs).zfill(2))
            
        if args.draw_graph:
            plt.show()
        
    if args.demo or args.save_demo:
        validation(policy, args)