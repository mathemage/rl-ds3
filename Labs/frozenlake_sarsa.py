import gym.spaces
import numpy as np

SEED = 1337
N_TOTAL = 10

env = gym.make('FrozenLake-v0')
env.seed(SEED)
observation = env.reset()
n_a = env.action_space.n
n_s = env.observation_space.n
gamma = 0.99


def init_q():
	return np.zeros((n_s, n_a))


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


def state_choice():
	return 42


def action_choice(q, s, a, epsilon=.5):
	# TODO epsilon-greedy
	print(q, s, a, epsilon)
	return 42


def sarsa(alpha=0.1):
	q = init_q()
	for _ in range(N_TOTAL):
		s = state_choice()
		a = action_choice(q, s, a)
		s_new, r, done, _ = env.step(a)
		while not done:
			a_new = action_choice(q, s_new, a)
			delta = r + gamma * q[s_new, a_new] - q[s, a]
			q[s, a] += alpha * delta
			s, a = s_new, a_new
	return q


if __name__ == '__main__':
	q_table = sarsa()
	# run_episodes()
	print(n_a)
	print(n_s)
	print(q_table)
