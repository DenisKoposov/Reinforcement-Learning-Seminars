# HW 1: Imitation Learning

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.

## Instructions

If you want to run this code. You should follow this brief manual.

### Step 1. Expert data generation

Firtly, you need to collect dataset from the expert policy. To do this, you should run `run_expert.py` script with the following arguments:
+ **expert_policy_file**(see experts/ for available expert policies)
+ **envname**(name of environment like `Hopper-v1`)
+ **-\-render**(optional, if you want to see the video of the agent walking)
+ **-\-save**(optional, if you want to save generated data)
+ **-\-max_timesteps**(optional, by default is 1000 for every environment, excluding `Reacher-v1`)
+ **-\-num_rollouts**(optional, by default is for every environment)

Example:
`python run_expert.py experts/Hopper-v1.pkl Hopper-v1 --save --num_rollouts 50`

## Step 2. Training and collecting statistics

To train and estimate policy, you need to run `dagger.py` or `behavioral_cloning.py`. The set of arguments available is similar to the `run_expert.py`.

+ **expert_policy_file**(`only for DAgger`, see experts/ for available expert policies)
+ **envname**(name of environment like `Hopper-v1`)
+ **-\-lr**(default = 0.007, learning rate for optimizer)
+ **-\-batch_size**(default=128, neural network will get data of shape `(batch_size, observations)`)
+ **--dagger_steps**(`only for DAgger`, default=5)
+ **-\-epochs**(default=5, number of training epochs for Neural Network)
+ **-\-render**(optional, if you want to see the video of the agent walking)
+ **-\-max_timesteps**(default=1000 for every environment, excluding `Reacher-v1`)
+ **-\-num_rollouts**(default=20 for every environment, it is used to choose dataset)
+ **-\-demo**(optional, if you want to do validation step)
+ **-\-save_demo**(optional, if you want to save videos of agents moving in validation step)
+ **-\-draw_graph**(optional, if you want to see reward/loss as function of epoch for BC and of dagger_step for DAgger)
+ **-\-save_graph**(optional, if you want to save graphs as `.png` images)
+ **-\-policy_type**(optional, default=0, you can choose one of two policies, `0` or `1`)
    `0` - 2 dense layer(64 hidden neurons) with ReLU
    `1` - 3 dense layer(512 hidden neurons) with tanh and 10 % Dropout, but it works awful

## (Optional) Step 3. mp4 to gif

This step is optional. If you want to use your visual materials somewhere you may want to use `mp4_to_gif_converter.py`. This script generates `.gif` from first 8 seconds of every video recorded at a validation step.