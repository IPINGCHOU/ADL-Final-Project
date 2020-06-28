# ADL-Final-Project

![image](https://github.com/IPINGCHOU/ADL-Final-Project/blob/master/title_image.png)

Reinforcement Learning are widely used in AI game-play learning field. Model training efficiency and performance is therefore an important study. In our study, we focus on how RL models play the bull hell game. We design numerous method, such as Proximal Policy Optimization (PPO), Generalized Advantage Estimation (GAE), Soft Actor-Critic (SAC) and Rainbow DQN, and discuss each performance in order to find the best method in the bullet hell game. By observing how the model make interaction with our game environment, we modify the game design to fit the model training with well-tuned hyper-parameters on PG and DQN-based models.



## Environment

* python3.6
* pytorch1.5.0
* pygame


## Folder structure
Our folder structure is as following:

    .
    ├── Rainbow_GPUbuffer      # Here put our rainbow dqn
    │   ├── train_rainbow.py   # Training model file
    │   ├── test_rainbow.py    # Testing model file
    │   ├── rainbow_agent.py   # Model agent file
    │   └── game.py            # Game main file
    ├── Rainbow                # Rainbow, same structure as Rainbow_GPUbuffer
    ├── PPO                    # PPO, same structure as Rainbow_GPUbuffer
    ├── GAE                    # GAE, same structure as Rainbow_GPUbuffer
    ├── DQN                    # DQN, same structure as Rainbow_GPUbuffer
    ├── game                   # Game configs
    └── README.md

## Game
The game parameters were setted in ./game/game_config.py, which includes hitbox size, plane and bullet velocity...etc. Model configuration and render were setted in ./game/game_model.py. The ./game/game_controller.py provides the whole game environment settings and function for model training.

## Rainbow
There are two Rainbow folders, one with CPU replay buffer called "Rainbow" and one with GPU replay buffer called "Rainbow_GPUbuffer", both of them could work properly by running the command below:
```bash=
cd ./Rainbow_GPUbuffer
python train_rainbow.py
```
Don't forget to generate a folder named './checkpoints' for storing the temp result game-play videos during the training, the losses and rewards will also be stored in this folder.
All of our best Rainbow model hyper-parameters were setted on the top of
```bash=
rainbow_agent.py
```
Our best model were trained by Rainbow_GPUbuffer verison, so if you want to reproduced the training process, please cd to "Rainbow_GPUbuffer" folder.
Also, test out the training result by
```bash=
python test_rainbow.py <mode> <invincible> <max_t>
```
It will generate the best game-play video in 100 episodes. With the following arguments:
+ mode
    1. test: load rainbow_checkpoints_online.cpt and rainbow_checkpoints_target.cpt for making actions
    2. random: randomly select actions for every episode
+ invincible
    1. True: for invincible mode on
    2. False: for disable invincible mode
+ max_t:
    1. if invincible mode == True, run <max_t> steps
    2. if invinvible mode == False, the max_t do not matters
    
 ## PPO & GAE
 If you want to train the PPO or GAE model, you should go to each folder and follow the command below.
 ```bash=
 python train.py
 ```
 After training, the default setting will save the best_model.ckt in where you run python, please make sure cd to each folder.
 The test command is as follows.
 ```bash=
 python test.py
 ```
 Then it will save the test.mp4.

 Also, you can draw the reward and loss line chart. You can modify the --rw and --loss as where your log you put.
 ```bash=
 python plot.py --rw="./PPO/rw.npy" --loss="./PPO/loss.npy"
 ```

## Dueling DQN
```bash
# TRAINING
python train_dqn.py

# TESTING
python test_dqn.py
```

### Soft Actor Critic
```bash
# TRAINING
python train_sac.py

# TESTING
python test_sac.py
