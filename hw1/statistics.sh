flags_bc1='--save_demo --save_graph --epochs 5'
flags_da1='--save_demo --save_graph --dagger_steps 5'
flags_bc2='--save_demo --save_graph --epochs 10'
flags_da2='--save_demo --save_graph --dagger_steps 10'

for e in Hopper-v1 Humanoid-v1 Reacher-v1 HalfCheetah-v1 Ant-v1 Walker2d-v1
do
    python behavioral_cloning.py $e $flags_bc1
done

for e in Hopper-v1 Humanoid-v1 Reacher-v1 HalfCheetah-v1 Ant-v1 Walker2d-v1
do
    python dagger.py experts/$e.pkl $e $flags_da1
done

for e in Hopper-v1 Humanoid-v1 Reacher-v1 HalfCheetah-v1 Ant-v1 Walker2d-v1
do
    python behavioral_cloning.py $e $flags_bc2
done

for e in Hopper-v1 Humanoid-v1 Reacher-v1 HalfCheetah-v1 Ant-v1 Walker2d-v1
do
    python dagger.py experts/$e.pkl $e $flags_da2
done