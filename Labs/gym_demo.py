import gym.spaces

env = gym.make('FrozenLake-v0')
observation = env.reset()
n_a = env.action_space.n
n_s = env.observation_space.n

print(n_a)
print(n_s)

env.render()