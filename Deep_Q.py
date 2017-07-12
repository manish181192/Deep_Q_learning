from collections import defaultdict, OrderedDict
from gyp import ordered_dict

import gym
import numpy as np
import sys
import tensorflow as tf
from Q_function import nn
from gym import wrappers

model = nn(state_shape=[5], layers=1, hidden_units= 100)
class observation(object):
    def __init__(self, state):
        self.state = state
    def equal(self, other):
        return np.array_equal(self.state, other.state)

def make_epsilon_greedy_policy(epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        # best_action = np.argmax(Q[observation])
        best_action = -1
        max_reward = -1
        for action_id in range(nA):
            input = np.concatenate((observation, np.array([action_id])))
            reward = model.get_q_value(np.expand_dims(input, 0))
            if reward > max_reward:
                max_reward = reward
                best_action = action_id
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(epsilon, env.action_space.n)
    prev_state = None
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        # if i_episode % 1000 == 0:
              # , end="")
        # sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        # if prev_state is None:
        state = env.reset()
        # else:
        #     state = prev_state
        total_reward = 0
        for t in range(200):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            total_reward+=reward

            if t%3 == 1 and done == False:
                reward =1
            else:
                reward =0


            env.render()
            # episode.append([zip((state.tolist())), action, reward])
            episode.append([state, action, reward])
            if done:
                # if t == 0:
                #     state = env.reset()
                #     t =0
                #     continue
                print("Episode finished after {} timesteps".format(t + 1))
                print "Terminated"
                episode[-1][2] = -3.
                break
            prev_state = state
            state = next_state
        print"Episode"+str(i_episode)+"/" +str(num_episodes)+" reward : "+str(total_reward)

        # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = [(x[0], x[1]) for x in episode]
        for state, action in sa_in_episode:
            state_tuple = tuple(state)
            sa_pair = (state_tuple, action)
            # Find the first occurance of the (state, action) pair in the episode
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if np.array_equal(x[0],state) and x[1] == action)
            # Sum up all rewards since the first occurance
            G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            Q[state_tuple][action] = returns_sum[sa_pair] / returns_count[sa_pair]

            # The policy is improved implicitly by changing the Q dictionary

    return Q, policy

env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force= True)

for iter in range(20):
    Q, policy = mc_control_epsilon_greedy(env = env, num_episodes = 100, discount_factor=0.8, epsilon=0.1)

    #create training data
    state_list = []
    action_list = []
    reward_list = []
    # create training data
    for state in Q.keys():
        for action, reward in enumerate(Q[state]):
            # action = np.argmax(Q[state])
            if reward != 0.0:
                action_list.append(action)
                reward_list.append(reward)
                state_list.append(np.array(state))


    action_fd = np.expand_dims(np.array(action_list), 1)
    state_fd = np.array(state_list)

    inputs_fd = np.concatenate((state_fd, action_fd), axis=1)
    targets_fd = np.array(reward_list)
    model.train_model(data=[inputs_fd, targets_fd],
                      learning_rate= 0.01,
                      num_epochs= 300)

gym.upload('/tmp/cartpole-experiment-1', api_key='sk_sdaCiHFUSDKQyOjrlfYnsw')
