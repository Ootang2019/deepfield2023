# deepfield2023

## Sec.1 PPO tutorial
- Practice
https://colab.research.google.com/drive/1NPeIGPo-XkFSGTfTQyQNOjlWI7CjRmJD?usp=sharing

- Answer


## Sec.2 PPO Turtlesim Installation Instructions
1. clone this repository
```
cd ~
git clone https://github.com/Ootang2019/deepfield2023.git
```

2. install dependencies (python 3.7+, PyTorch>=1.11)
```
pip install stable-baselines3 pyyaml rospkg
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


