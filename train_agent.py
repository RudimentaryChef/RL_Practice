import gym
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import os
env = gym.make("LunarLander-v2", render_mode = 'ansi')
#day one
env.reset()
models_dir = "models/A2C"
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

model = A2C("MlpPolicy", env, verbose = 1)
TIMESTEPS = 10000
for i in range(1,30):
    #By setting it to false we are setting its learning reseting to be false
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps= False, tb_log_name= "A2C")
    #This will save the model every 10 thousand steps
    model.save(f"{models_dir}/{TIMESTEPS*i}")
#prints out a sample action
episodes = 10
'''
for ep in range(episodes):
    obs = env.reset()
    terminated = False
    while not terminated:
        #renders our thing
        env.render()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        #closes it
env.close()
'''
#Changing the hyper parameters will give you about 5-10 percent. Reward mechanism, observation space.