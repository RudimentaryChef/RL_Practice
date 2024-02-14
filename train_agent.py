import gym

env = gym.make("LunarLander-v2", render_mode = "human")

env.reset()
#prints out a sample action
for step in range(200):
    #renders our thing
    env.render()
    env.step(env.action_space.sample())
#closes it
env.close()

#Changing the hyper parameters will give you about 5-10 percent. Reward mechanism, observation space.