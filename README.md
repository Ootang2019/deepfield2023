# deepfield2023

## Sec.1 PPO tutorial
1. Practice
https://colab.research.google.com/drive/1-2IUh717LBaZadyNRWJgrNp8nggYUShu?usp=sharing

2. change to GPU device
- [top right dropdown] connect to a hosted runtime
- [RAM Disk label] check if you are using GPU backend
- if not, check [Change runtime type], and change Hardware accelerator to GPU

3. Answer
https://colab.research.google.com/drive/1NPeIGPo-XkFSGTfTQyQNOjlWI7CjRmJD?usp=sharing


## Sec.2 PPO Turtlesim Installation Instructions
1. clone this repository
```
cd ~
git clone https://github.com/Ootang2019/deepfield2023.git
```

2. install dependencies (python 3.7+, PyTorch>=1.11)
```
pip install stable-baselines3 pyyaml rospkg numpy
```

3. start simulation 
- terminal1:
```
cd ~/deepfield2023
roscore
```
- terminal2:
```
cd ~/deepfield2023
roslaunch multisim_turtle.launch
```
- terminal3:
```
cd ~/deepfield2023
python rl.py
```

4. clean simulation
- ctrl+c in all terminal
- clean ros artifact
```
cd ~/deepfield2023
bash cleanup.sh
```

## Task: Can you make training faster? Possible directions:
1. Reward Engineering: 
- add different penalty terms to the reward function: in turtle_sim.py, modify reward weight *self.rew_w* array and *compute_reward()* function
- clip the reward to the range [-1,1] to reduce the reward variance
- increase reward weight, *self.rew_w* 

2. Add a penalty for hitting the wall
- wallposition: x=0, y=0, x=11.1, y=11.1
- include detection to the *observe()* function
- add a penalty in *compute_reward()* and reward weight *self.rew_w*

3. Hyper-parameter Tuning: 
- in rl.py, modify PPO hyper-parameters or NN architecture

4. Residual RL:
- add a baseline PID controller to the environment
- mix PID and RL command to control turtle

5. Curriculum learning: 
- make the goal easier to solve from the beginning, and then progressively make the task harder

6. Improve exploration:
- add exploration bonus to the reward to encourage agent discovering new states

7. Try different agent:
- in *rl.py*, import agents and replace PPO 
```
from stable_baselines3 import DDPG, SAC, TD3
```

8. Customize PPO:
- create your own PPO from the code in Colab notebook to have maximum control over the training loop

9. Try Harder Env (dynamic goal):
- in rl.py, replace TurtleEnv with TurtleEnv_hard

10. Action Smoothness:
- incorporate an accumulative action space
- penalize action changes

![](https://github.com/Ootang2019/deepfield2023/img/turtle_hard.gif)

