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


def run_episodes(episodes=20):
	global observation
	for i_episode in range(episodes):
		observation = env.reset()
		for t in range(100):
			env.render()
			print(observation)
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			if done:
				print("Episode finished after {} timesteps".format(t + 1))
				break


run_episodes()
