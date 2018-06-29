import numpy as np
import matplotlib.pyplot as plt
import gym

ENV_NAME = "CartPole-v0"
EPISODE_DURATION = 300
ALPHA_INIT = 0.1
SCORE = 195.0
TEST_TIME = 100
LEFT = 0
RIGHT = 1

VERBOSE = True


# Compute policy parameterisation
def sigmoid(x):
	return 1 / (1 + np.exp(-x))


# Return policy
def get_policy(s, theta):
	pass


# Draw an action according to current policy
def act_with_policy(s, theta):
	pass


# Generate an episode
def rollout(env, theta, max_episode_length=EPISODE_DURATION, render=False):
	pass


def test_policy(env, theta, score=SCORE, num_episodes=TEST_TIME, max_episode_length=EPISODE_DURATION, render=False):
	pass


# Returns Policy Gradient for a given episode
def compute_pg(episode_states, episode_actions, episode_rewards, theta):
	pass


# Train until average return is larger than SCORE
def train(env, theta_init, max_episode_length=EPISODE_DURATION, alpha_init=ALPHA_INIT):
	theta, i, average_returns = 0, 0, []
	return theta, i, average_returns


def main():
	env = gym.make(ENV_NAME)
	dim = env.observation_space.shape[0]
	# Init parameters to random
	theta_init = np.random.randn(1, dim)
	# Train agent
	theta, i, average_returns = train(env, theta_init)
	print("Solved after {} iterations".format(i))
	# Test final policy
	test_policy(env, theta, num_episodes=10, render=True)
	# Show training curve
	plt.plot(range(len(average_returns)), average_returns)
	plt.title("Average reward on 100 episodes")
	plt.xlabel("Training Steps")
	plt.ylabel("Reward")
	plt.show()
	env.close()


if __name__ == "__main__":
	main()
