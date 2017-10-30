# CS294-112 HW 1: Imitation Learning

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

# Results

## Summary table

|Environment   |Expert Reward Mean(std)|Behavior Cloning |DAgger             |
|--------------|-----------------------|-----------------|-------------------|
|Ant-v1        |4727.27(236.27)        |4689.20(52.14)   |**4825.16(76.79)** |
|HalfCheetah-v1|4125.13(85.51)         |3869.31(102.09)  |**4114.19(72.78)** |
|Hopper-v1     |3777.49(4.76)          |1181.05(152.07)  |**3779.39(5.24)**  |
|Humanoid-v1   |10393.68(47.56)        |**375.76(58.29)**|213.05(26.96)      |
|Reacher-v1    |-4.29(1.61)            |-8.20(2.11)      |**-4.20(1.41)**    |
|Walker2d-v1   |5510.97(46.70)         |2524.92(1758.67) |**5483.19(113.88)**|

**Hyperparameters:** batch_size = 128, 5 epochs, 5 iterations of DAgger, 50 roll-outs
**ANN:** 2 layer network with ReLU activation(64 hidden neurons)
**Optimizer**: Adam, lr = 0.007, weight_decay(L2)=0.0000001

## Visual demonstation

|Environment|Behavior Cloning|DAgger|
|-----------|----------------|------|
|Ant-v1|![Ant-v1-bc](demonstrations_bc/Ant-v1_050_05/Ant-v1_050_05.gif)|![Ant-v1-da](demonstrations_da/Ant-v1_050_05/Ant-v1_050_05.gif)|
|HalfCheetah-v1|![HalfCheetah-v1-bc](demonstrations_bc/HalfCheetah-v1_050_05/HalfCheetah-v1_050_05.gif)|![HalfCheetah-v1-da](demonstrations_da/HalfCheetah-v1_050_05/HalfCheetah-v1_050_05.gif)|
|Hopper-v1|![Hopper-v1-bc](demonstrations_bc/Hopper-v1_050_05/Hopper-v1_050_05.gif)|![Hopper-v1-da](demonstrations_da/Hopper-v1_050_05/Hopper-v1_050_05.gif)|
|Reacher-v1|![Reacher-v1-bc](demonstrations_bc/Reacher-v1_050_05/Reacher-v1_050_05.gif)|![Reacher-v1-da](demonstrations_da/Reacher-v1_050_05/Reacher-v1_050_05.gif)|
|Walker2d-v1|![Walker2d-v1-bc](demonstrations_bc/Walker2d-v1_050_05/Walker2d-v1_050_05.gif)|![Walker2d-v1-da](demonstrations_da/Walker2d-v1_050_05/Walker2d-v1_050_05.gif)|

## Summary:
As we can see, **DAgger performs much better than Behavior Cloning** in almost every case(for this model without varying paraneters), excluding Humanoid-v1 environment, because both algorithms didn't work out a good policy for this particular case. It may be a consequence of high dimensional input(~400 observations, ~100 actions), so the simple 2-layer ANN with 64 hidden neurons just can't handle this sophisticated data.

## Graphs

### Ant-v1
|Method|Reward|Loss|
|------|------|----|
|BC|![Ant-v1-bc-reward-graph](figures_bc/reward_Ant-v1_050_05.png)|![Ant-v1-bc-loss-graph](figures_bc/loss_Ant-v1_050_05.png)|
|DA|![Ant-v1-da-reward-graph](figures_da/reward_Ant-v1_050_05.png)|![Ant-v1-da-loss-graph](figures_da/loss_Ant-v1_050_05.png)|
### HalfCheetah-v1
|Method|Reward|Loss|
|------|------|----|
|BC|![HalfCheetah-v1-bc-reward-graph](figures_bc/reward_HalfCheetah-v1_050_05.png)|![HalfCheetah-v1-bc-loss-graph](figures_bc/loss_HalfCheetah-v1_050_05.png)|
|DA|![HalfCheetah-v1-da-reward-graph](figures_da/reward_HalfCheetah-v1_050_05.png)|![HalfCheetah-v1-da-loss-graph](figures_da/loss_HalfCheetah-v1_050_05.png)|
### Hopper-v1
|Method|Reward|Loss|
|------|------|----|
|BC|![Hopper-v1-bc-reward-graph](figures_bc/reward_Hopper-v1_050_05.png)|![Hopper-v1-bc-loss-graph](figures_bc/loss_Hopper-v1_050_05.png)|
|DA|![Hopper-v1-da-reward-graph](figures_da/reward_Hopper-v1_050_05.png)|![Hopper-v1-da-loss-graph](figures_da/loss_Hopper-v1_050_05.png)|
### Humanoid-v1
|Method|Reward|Loss|
|------|------|----|
|BC|![Humanoid-v1-bc-reward-graph](figures_bc/reward_Humanoid-v1_050_05.png)|![Humanoid-v1-bc-loss-graph](figures_bc/loss_Humanoid-v1_050_05.png)|
|DA|![Humanoid-v1-da-reward-graph](figures_da/reward_Humanoid-v1_050_05.png)|![Humanoid-v1-da-loss-graph](figures_da/loss_Humanoid-v1_050_05.png)|
### Reacher-v1
|Method|Reward|Loss|
|------|------|----|
|BC|![Reacher-v1-bc-reward-graph](figures_bc/reward_Reacher-v1_050_05.png)|![Reacher-v1-bc-loss-graph](figures_bc/loss_Reacher-v1_050_05.png)|
|DA|![Reacher-v1-da-reward-graph](figures_da/reward_Reacher-v1_050_05.png)|![Reacher-v1-da-loss-graph](figures_da/loss_Reacher-v1_050_05.png)| 
### Walker2d-v1
|Method|Reward|Loss|
|------|------|----|
|BC|![Walker2d-v1-bc-reward-graph](figures_bc/reward_Walker2d-v1_050_05.png)|![Walker2d-v1-bc-loss-graph](figures_bc/loss_Walker2d-v1_050_05.png)|
|DA|![Walker2d-v1-da-reward-graph](figures_da/reward_Walker2d-v1_050_05.png)|![Walker2d-v1-da-loss-graph](figures_da/loss_Walker2d-v1_050_05.png)|