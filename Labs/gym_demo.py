import gym.spaces

SEED = 1337
STEPS = 5

env = gym.make('FrozenLake-v0')
env.seed(SEED)
observation = env.reset()
n_a = env.action_space.n
n_s = env.observation_space.n

print(n_a)
print(n_s)

for _ in range(STEPS):
	env.render()
	action = env.action_space.sample()
	observation, reward, done, info = env.step(action)
	print(action, observation, reward, done, info)

env.render()
