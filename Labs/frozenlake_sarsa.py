import gym.spaces
import numpy as np

SEED = 1337
EPISODES = 3
# EPISODES = 3000
GAMMA = 0.99

env = gym.make('FrozenLake-v0')
env.seed(SEED)
np.random.seed(SEED)
n_a = env.action_space.n
n_s = env.observation_space.n


def init_q():
	# return np.zeros((n_s, n_a))
	return np.random.rand(n_s, n_a)


def measure_q(q, environment, episodes=100):
	successes = 0
	for ep in range(episodes):
		print("Episode {}:".format(ep))
		s = environment.reset()
		while True:
			# env.render()
			a = action_choice(q, s, epsilon=1)
			s, r, done, _ = environment.step(a)
			if done:
				successes += r == 1
				break
	return successes


def action_choice(q, s, epsilon=.5):
	# TODO epsilon-greedy
	a = np.argmax(q[s][:])
	# print(q[s][:])
	# print("a: {}".format(a))
	return a


def sarsa(environment, alpha=0.1):
	"""
	https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html

	:param environment: OpenAI Gym environment
	:param alpha: learning rate for TD Q error
	:return: the converged Q-table
	"""
	q = init_q()
	for ep in range(EPISODES):
		print("Episode {}:".format(ep))
		s = environment.reset()
		a = action_choice(q, s)
		done = False
		while not done:
			s_new, r, done, _ = environment.step(a)
			a_new = action_choice(q, s_new)
			delta = r + GAMMA * q[s_new, a_new] - q[s, a]
			q[s, a] += alpha * delta
			s, a = s_new, a_new
			print("delta: {}".format(delta))
			print("Q-table:\n{}".format(q))
	return q


if __name__ == '__main__':
	q_table = sarsa(environment=env)
	accuracy = measure_q(environment=env, q=q_table)
	print("accuracy: {}".format(accuracy))
