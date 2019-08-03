import gym

env = gym.make('Acrobot-v1')
print("action space :",env.action_space)
print("env space :",env.observation_space)
print("reward range:",env.reward_range)
print("action sample :",env.action_space.sample())
