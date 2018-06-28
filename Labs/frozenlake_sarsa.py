import gym.spaces
import numpy as np

SEED = 1337
# TRAINING_EPISODES = 3
TRAINING_EPISODES = 3000
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
			a = action_choice(q, s, epsilon=None)
			s, r, done, _ = environment.step(a)
			if done:
				successes += r == 1
				break
	return successes


def action_choice(q, s, epsilon=.5):
	if epsilon is None or np.random.rand() > epsilon:
		a = np.argmax(q[s][:])
	else:
		a = np.random.randint(0, n_a)
	return a


def sarsa(environment, alpha=0.1):
	"""
	https://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html

	:param environment: OpenAI Gym environment
	:param alpha: learning rate for TD Q error
	:return: the converged Q-table
	"""
	q = init_q()
	for ep in range(TRAINING_EPISODES):
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
	return q


if __name__ == '__main__':
	print("Training...")
	q_table = sarsa(environment=env)
	print("Measuring...")
	accuracy = measure_q(environment=env, q=q_table)
	print("accuracy: {}".format(accuracy))
