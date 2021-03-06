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
DEFAULT_THETA = [.5, .5, .5, .5]
SEED = 1337
np.random.seed(SEED)
PG_INIT = np.zeros(4)


# Compute policy parameterisation
def sigmoid(x):
	# print(x)
	return 1.0 / (1.0 + np.exp(-x))


# Return policy
def get_policy(s, theta):
	return sigmoid(np.dot(s, theta))


# Draw an action according to current policy
def act_with_policy(s, theta):
	policy_threshold = get_policy(s, theta) > 1
	if np.random.rand() < policy_threshold:
		return 1
	else:
		return 0


# Generate an episode
def rollout(env, theta, max_episode_length=EPISODE_DURATION, render=False):
	episode_actions, episode_rewards = [], []
	s = env.reset()
	episode_states = [s]
	for _ in range(max_episode_length):
		if render:
			env.render()
		a = act_with_policy(s, theta)
		s, r, done, _ = env.step(a)
		episode_states.append(s)
		episode_actions.append(a)
		episode_rewards.append(r)
		if done:
			return episode_states, episode_actions, episode_rewards


def test_policy(env, theta, score=SCORE, num_episodes=TEST_TIME, max_episode_length=EPISODE_DURATION, render=False):
	num_success = 0
	average_return = 0
	for i_episode in range(num_episodes):
		_, _, episode_rewards = rollout(env, theta, max_episode_length, render)
		total_rewards = sum(episode_rewards)
		if total_rewards > score:
			num_success += 1
		average_return += (1.0 / num_episodes) * total_rewards
		if render:
			print(
				"Test Episode {0}: Total Reward = {1} - Success = {2}".format(i_episode, total_rewards, total_rewards > score))
	if average_return > score:
		success = True
	else:
		success = False
	return success, num_success, average_return


# Returns Policy Gradient for a given episode
def compute_pg(episode_actions, episode_rewards, episode_states, theta):
	pg = PG_INIT
	for t in range(len(episode_rewards)):
		R_t = sum(episode_rewards[t:])
		sigmoid_value = get_policy(episode_states[t], theta)
		grad_log_t = sigmoid_value * episode_states[t] * (-1 if episode_actions[t] == RIGHT else 1)
		pg += R_t * grad_log_t
	print(" theta == {}".format(theta))
	print(" pg == {}".format(pg))
	return pg


# Train until average return is larger than SCORE
def train(env, theta_init=DEFAULT_THETA, max_episode_length=EPISODE_DURATION, alpha_init=ALPHA_INIT):
	theta, alpha, n, average_returns = theta_init, alpha_init, 0, []
	while True:
		episode_states, episode_actions, episode_rewards = rollout(env, theta, max_episode_length, render=False)
		average_return = sum(episode_rewards)
		average_returns.append(average_return)
		print("Training episode {}:".format(n))
		print(" average reward == {}".format(average_return))
		if average_return > SCORE:
			return theta, n, average_returns
		pg = compute_pg(episode_actions, episode_rewards, episode_states, theta)
		theta += alpha * pg
		n += 1


def main():
	env = gym.make(ENV_NAME)
	env.seed(SEED)
	dim = env.observation_space.shape[0]
	# Init parameters to random
	theta_init = np.random.randn(dim)
	# theta_init = [.5, .5, .5, .5]

	# Train agent
	theta, i, average_returns = train(env, theta_init)
	print("Solved after {} iterations".format(i))

	# Show training curve
	plt.plot(range(len(average_returns)), average_returns)
	plt.title("Average reward on 100 episodes")
	plt.xlabel("Training Steps")
	plt.ylabel("Reward")
	plt.show()

	# Test final policy
	success, num_success, average_return = test_policy(env, theta, render=True)
	print("success == {}".format(success))
	print("num_success == {}".format(num_success))
	print("average_return == {}".format(average_return))
	env.close()


if __name__ == "__main__":
	main()
