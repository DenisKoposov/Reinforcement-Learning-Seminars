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

+ **Hyperparameters:** batch_size = 128, 5 epochs, 5 iterations of DAgger, 50 roll-outs
+ **ANN:** 2 layer network with ReLU activation(64 hidden neurons)
+ **Optimizer**: Adam, lr = 0.007, weight_decay(L2)=0.0000001

## Visual demonstation

|Environment|Behavior Cloning|DAgger|
|-----------|----------------|------|
|Ant-v1|![Ant-v1-bc](demonstrations_bc/Ant-v1_050_05/Ant-v1_050_05.gif)|![Ant-v1-da](demonstrations_da/Ant-v1_050_05/Ant-v1_050_05.gif)|
|HalfCheetah-v1|![HalfCheetah-v1-bc](demonstrations_bc/HalfCheetah-v1_050_05/HalfCheetah-v1_050_05.gif)|![HalfCheetah-v1-da](demonstrations_da/HalfCheetah-v1_050_05/HalfCheetah-v1_050_05.gif)|
|Hopper-v1|![Hopper-v1-bc](demonstrations_bc/Hopper-v1_050_05/Hopper-v1_050_05.gif)|![Hopper-v1-da](demonstrations_da/Hopper-v1_050_05/Hopper-v1_050_05.gif)|
|Reacher-v1|![Reacher-v1-bc](demonstrations_bc/Reacher-v1_050_05/Reacher-v1_050_05.gif)|![Reacher-v1-da](demonstrations_da/Reacher-v1_050_05/Reacher-v1_050_05.gif)|
|Walker2d-v1|![Walker2d-v1-bc](demonstrations_bc/Walker2d-v1_050_05/Walker2d-v1_050_05.gif)|![Walker2d-v1-da](demonstrations_da/Walker2d-v1_050_05/Walker2d-v1_050_05.gif)|

## Summary:
+ **Behavioral Cloning** achieves comparable results in `Ant-v1` and `HalfCheetah-v1` environments.
+ **DAgger** shows good performance in every environment, excluding `Humanoid-v1`.
+ **DAgger performs much better than Behavior Cloning** in almost every case(for this model without varying parameters), excluding `Humanoid-v1` environment, because both algorithms didn't work out a good policy for this particular case. It may be a consequence of high dimensional input(~400 observations, ~100 actions), so the simple 2-layer ANN with 64 hidden neurons just can't handle this sophisticated data.
## Graphs
From graphs below, we can see that there is **no particular need in more then 5 dagger steps(DAgger) or more then 5 epochs(BC)**, because it doesn't give better performance and takes much more computational time(especially in case of DAgger, where dataset is always growing). It makes sense only for `Humanoid-v1` and BC algorithm, where reward is growing with a number of epochs, but remains poor even after 10 itertations.
The fact is that after 5 epochs(approximately) we observe either stable(oscillating) performance or significant drop of it.
### Ant-v1
|Method|Reward|Loss|
|------|------|----|
|BC|![Ant-v1-bc-reward-graph-05](figures_bc/reward_Ant-v1_050_05.png)![Ant-v1-bc-reward-graph-10](figures_bc/reward_Ant-v1_050_10.png)|![Ant-v1-bc-loss-graph-05](figures_bc/loss_Ant-v1_050_05.png)![Ant-v1-bc-loss-graph-10](figures_bc/loss_Ant-v1_050_10.png)||
|DA|![Ant-v1-da-reward-graph-05](figures_da/reward_Ant-v1_050_05.png)![Ant-v1-da-reward-graph-10](figures_da/reward_Ant-v1_050_10.png)|![Ant-v1-da-loss-graph-05](figures_da/loss_Ant-v1_050_05.png)![Ant-v1-da-loss-graph-10](figures_da/loss_Ant-v1_050_10.png)|
### HalfCheetah-v1
|Method|Reward|Loss|
|------|------|----|
|BC|![HalfCheetah-v1-bc-reward-graph-05](figures_bc/reward_HalfCheetah-v1_050_05.png)![HalfCheetah-v1-bc-reward-graph-10](figures_bc/reward_HalfCheetah-v1_050_10.png)|![HalfCheetah-v1-bc-loss-graph-05](figures_bc/loss_HalfCheetah-v1_050_05.png)![HalfCheetah-v1-bc-loss-graph-10](figures_bc/loss_HalfCheetah-v1_050_10.png)|
|DA|![HalfCheetah-v1-da-reward-graph-05](figures_da/reward_HalfCheetah-v1_050_05.png)![HalfCheetah-v1-da-reward-graph-10](figures_da/reward_HalfCheetah-v1_050_10.png)|![HalfCheetah-v1-da-loss-graph-05](figures_da/loss_HalfCheetah-v1_050_05.png)![HalfCheetah-v1-da-loss-graph-10](figures_da/loss_HalfCheetah-v1_050_10.png)|
### Hopper-v1
|Method|Reward|Loss|
|------|------|----|
|BC|![Hopper-v1-bc-reward-graph-05](figures_bc/reward_Hopper-v1_050_05.png)![Hopper-v1-bc-reward-graph-10](figures_bc/reward_Hopper-v1_050_10.png)|![Hopper-v1-bc-loss-graph-05](figures_bc/loss_Hopper-v1_050_05.png)![Hopper-v1-bc-loss-graph-10](figures_bc/loss_Hopper-v1_050_10.png)|
|DA|![Hopper-v1-da-reward-graph-05](figures_da/reward_Hopper-v1_050_05.png)![Hopper-v1-da-reward-graph-10](figures_da/reward_Hopper-v1_050_10.png)|![Hopper-v1-da-loss-graph-05](figures_da/loss_Hopper-v1_050_05.png)![Hopper-v1-da-loss-graph-10](figures_da/loss_Hopper-v1_050_10.png)|
### Humanoid-v1
|Method|Reward|Loss|
|------|------|----|
|BC|![Humanoid-v1-bc-reward-graph-05](figures_bc/reward_Humanoid-v1_050_05.png)![Humanoid-v1-bc-reward-graph-10](figures_bc/reward_Humanoid-v1_050_10.png)|![Humanoid-v1-bc-loss-graph-05](figures_bc/loss_Humanoid-v1_050_05.png)![Humanoid-v1-bc-loss-graph-10](figures_bc/loss_Humanoid-v1_050_10.png)|
|DA|![Humanoid-v1-da-reward-graph-05](figures_da/reward_Humanoid-v1_050_05.png)![Humanoid-v1-da-reward-graph-10](figures_da/reward_Humanoid-v1_050_10.png)|![Humanoid-v1-da-loss-graph-05](figures_da/loss_Humanoid-v1_050_05.png)![Humanoid-v1-da-loss-graph-10](figures_da/loss_Humanoid-v1_050_10.png)|
### Reacher-v1
|Method|Reward|Loss|
|------|------|----|
|BC|![Reacher-v1-bc-reward-graph-05](figures_bc/reward_Reacher-v1_050_05.png)![Reacher-v1-bc-reward-graph-10](figures_bc/reward_Reacher-v1_050_10.png)|![Reacher-v1-bc-loss-graph-05](figures_bc/loss_Reacher-v1_050_05.png)![Reacher-v1-bc-loss-graph-10](figures_bc/loss_Reacher-v1_050_10.png)|
|DA|![Reacher-v1-da-reward-graph-05](figures_da/reward_Reacher-v1_050_05.png)![Reacher-v1-da-reward-graph-10](figures_da/reward_Reacher-v1_050_10.png)|![Reacher-v1-da-loss-graph-05](figures_da/loss_Reacher-v1_050_05.png)![Reacher-v1-da-loss-graph-10](figures_da/loss_Reacher-v1_050_10.png)| 
### Walker2d-v1
|Method|Reward|Loss|
|------|------|----|
|BC|![Walker2d-v1-bc-reward-graph-05](figures_bc/reward_Walker2d-v1_050_05.png)![Walker2d-v1-bc-reward-graph-10](figures_bc/reward_Walker2d-v1_050_10.png)|![Walker2d-v1-bc-loss-graph-05](figures_bc/loss_Walker2d-v1_050_05.png)![Walker2d-v1-bc-loss-graph-10](figures_bc/loss_Walker2d-v1_050_10.png)|
|DA|![Walker2d-v1-da-reward-graph-05](figures_da/reward_Walker2d-v1_050_05.png)![Walker2d-v1-da-reward-graph-10](figures_da/reward_Walker2d-v1_050_10.png)|![Walker2d-v1-da-loss-graph-05](figures_da/loss_Walker2d-v1_050_05.png)![Walker2d-v1-da-loss-graph-10](figures_da/loss_Walker2d-v1_050_10.png)|