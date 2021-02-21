# epsilon-greedy example implementation of a multi-armed bandit
import random
import math
import os,sys,inspect
import numpy as np
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

class Bandit:
    """
    Generic epsilon-greedy bandit that you need to improve

    Author: dv18mln, ens20jeg
    """

    def __init__(self, arms, epsilon=0.1):
        """
        Initiates the bandits

        :param arms: List of arms
        :param epsilon: Epsilon value for random exploration
        """
        self.arms = arms
        self.epsilon = epsilon
        self.frequencies = [0] * len(arms)
        self.sums = [0] * len(arms)
        self.expected_values = [0] * len(arms)
        self.n = 0
        self.rewards = [[],[],[],[],[],[]]

        # Discard arms that perform badly.
        self.discarded_arms = []

        self.alpha = [1] * len(arms)
        self.beta  = [1] * len(arms)
        self.theta = [0] * len(arms)
        self.theta_max = 0

    def run(self):
        """
        Asks the bandit to recommend the next arm.

        :return: Returns the arm the bandit recommends pulling
        """

        if min(self.frequencies) == 0:
            return self.arms[self.frequencies.index(min(self.frequencies))]

        # Epsilon decay
        self.epsilon = 1/(1 + sum(self.frequencies))
        #self.epsilon = self.epsilon * (1/(1 + sum(self.frequencies)))
        #self.epsilon = self.epsilon * 0.85

        if random.random() < self.epsilon: 
            x = random.choice([i for i in range(0, len(self.arms)-1) if i not in self.discarded_arms])
            return self.arms[x]

        return self.arms[self.expected_values.index(max(self.expected_values))]


    def sliding_window(self, rewards, window_size):
        """
        Implementation of the sliding window algorithm. 

        Calculates the average of the last number of given 
        rewards.  

        :param rewards: List of the reward for a certain arm.
        :param window_size: The number of rewards to take in to 
        consideration.
        """
        lower = len(rewards) - window_size if len(rewards) - \
            window_size > 0 else 0
        window = rewards[lower:len(rewards)]
        avg = sum(window) / len(window)

        return avg

    def thompsonSampler(self, reward):
        """
        Beginning of a thompson sampler. 
        Uses a Beta bernoulli sampler.
        The thompson sampler uses "regret" to indicate
        how good or bad a move were in terms of this variable.
        The regret can be used to determine how good the best
        choice (or arm in this case) would have been. 

        The thomspsonsampler is not used in the current 
        algorithm and needs to be further implemented to
        improve the performance of the algorithm.

        The function is left as a reference to the question
        of how to further improve the bandit. 

        :param reward: The reward that was generated.
        """
        self.theta = np.random.beta(self.alpha, self.beta)
        self.k = np.argmax(self.theta)

        self.alpha[self.k] += (reward if reward > 0 else 0)
        self.beta[self.k] += 1 - (reward if reward < 1 else 1)

        if self.theta[self.k] > self.theta_max:
            self.theta_max = self.theta[self.k]

        regret = self.theta_max - self.theta[self.k]

        return self.k

    def give_feedback(self, arm, reward):
        """
        Sets the bandit's reward for the most recent arm pull

        :param arm: The arm that was pulled to generate the reward
        :param reward: The reward that was generated
        """

        arm_index = self.arms.index(arm)
        sum = self.sums[arm_index] + reward
        self.sums[arm_index] = sum
        frequency = self.frequencies[arm_index] + 1
        self.frequencies[arm_index] = frequency

        self.rewards[arm_index].append(reward)

        window_size = 25
        expected_value = self.sliding_window(self.rewards[arm_index], window_size)
        self.expected_values[arm_index] = expected_value

        # Discard arms if they perform poorly. Rare that a arm performs below 0.1.
        # The arm needs to have performed below 0.1 in the last window_size of tries.
        if expected_value < 0.1 and arm_index not in self.discarded_arms and frequency > 10:
            print("expected value", expected_value)
            self.discarded_arms.append(arm_index)