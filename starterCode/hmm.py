# File: hmm.py
# Purpose:  Starter code for building and training an HMM in CSC 246.


import os
import argparse
import numpy as np
from nlputil import *   # utility methods for working with text

# A utility class for bundling together relevant parameters - you may modify if you like.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# num_states -- this should be an integer recording the number of hidden states
#
# pi -- this should be the distribution over the first hidden state of a sequence
#
# transitions -- this should be a num_states x num_states matrix of transition probabilities
#
# emissions -- this should be a num_states x vocab_size matrix of emission probabilities
#              (i.e., the probability of generating token X when operating in state K)
#
# vocab_size -- this should be an integer recording the vocabulary size
#
# Note: You may want to add fields for expectations.


class HMM:
    __slots__ = ('pi', 'transitions', 'emissions', 'num_states', 'vocab_size')

    # The constructor should initalize all the model parameters.
    # you may want to write a helper method to initialize the emission probabilities.
    def __init__(self, num_states, vocab_size):
        # Num_states is number of hidden states
        self.num_states = num_states
        # Vocab size is the number of unique words
        self.vocab_size = vocab_size
        # Transitions are a KxK matrix where K is the number of hidden states. Initialized randomly between 0 and 1
        self.transitions = np.random.rand(self.num_states, self.num_states)
        # pi is vector of size K, also initialized between 0 and 1
        self.pi = np.random.rand(self.num_states)
        # TEMPORARY: Intializing emissions to uniform distribution
        self.emissions = np.random.uniform(
            size=(self.num_states, self.vocab_size))

    # return the loglikelihood for a complete dataset (train OR test) (list of matrices)
    # TODO: calculate mean log likelihood
    def loglikelihood(self, dataset):
        pass

    # return the loglikelihood for a single sequence (numpy matrix)
    # abdulmoid : i think this ln (p(X| model))? can use alpha beta products here i guess
    def loglikelihood_helper(self, sample):
        pass

    # given a sequence of observations, find the most probable path of hidden states it could have followed
    # here sample is the given observations. should be pre-converted to numbers
    def viterbi(self, sample):
        # Dimenstions of v are TxN where T=number of observations, N=number of hidden states
        v = np.zeros(len(sample), self.num_states)
        # Dimenstions of backpointer are TxN where T=number of observations, N=number of hidden states
        backpointer = np.zeros(len(sample), self.num_states)
        for t in range(0, len(sample)):
            for j in range(0, len(v)):
                v[t][j] = self.pi[j] * \
                    self.emissions[j][sample[0]]  # pi_j * bj(o1)
        max_v_prev = -99
        prev_state_selected = 0

        # Recursive Step
        for t in range(1, len(sample)):
            for j in range(0, self.num_states):
                # go through all states again to analyze states that pass through time t-1
                for i in range(0, self.num_states):
                    if(v[t-1][i]*self.transitions[i][j] > max_v_prev):  # we need max v_t-1 (i) *aij
                        max_v_prev = v[t-1][i]*self.transitions[i][j]
                        prev_state_selected = i
                v[t][j] = max_v_prev * self.emissions[j][sample[t]]
                backpointer[t][j] = prev_state_selected

        # Find value and indices of best v value for time T (final time)
        max_val = -99
        time = len(sample)  # final time
        best_state = 0
        for j in range(0, len(v)):
            if (v[time][j] > max_val):
                max_val = v[time][j]
                best_state = j

        # intialize path trace array
        # preparing array to build path of hidden states to output
        path_trace = np.zeros(len(sample))
        # start back trace by adding to the end, the best_state for for time T in previous state
        path_trace[len(path_trace)-1] = col

        # run backtrace
        index = len(path_trace)-2
        while(index >= 0):
            # backpointer[current_time][backpointer of the next node in path]
            path_trace[index] = backpointer[time][path_trace[index+1]]
            time -= 1
            index -= 1

        return max_val, path_trace

    # given the integer representation of a single sequence
    # return a T x num_states matrix of alpha where T is the total number of tokens in a single sequence
    def forward(self, sample):
        alpha = np.zeros((len(sample), self.num_states))
        # initialization
        for j in range(0, self.num_states):
            alpha[0][j] = self.pi[j] * self.emissions[j][sample[0]]
        # recursion
        for t in range(1, len(sample)):
            for j in range(0, self.num_states):
                for i in range(0, self.num_states):
                    alpha[t][j] += alpha[t-1][i] * \
                        self.transitions[i][j] * self.emissions[j][sample[t]]
        return alpha

    # given the integer representation of a single sequence
    # return a T x num_state matrix of beta where T is the total number of tokens in a single sequence
    def backward(self, sample):
        beta = np.zeros((len(sample), self.num_states))
        # initialization
        for i in range(0, self.num_states):
            beta[len(sample)-1][i] = 1
        # recursion
        for t in range(1, len(sample)):
            for i in range(0, self.num_states):
                for j in range(0, self.num_states):
                    beta[len(sample)-1-t][i] += self.transitions[i][j] * \
                        self.emissions[j][sample[len(
                            sample)-t]] * beta[len(sample)-t][j]
        return beta

    # Uses alpha and beta values to calculate
    # e[t][i][j] = Probability of being in state i at time t and state j at time t+1
    # y[t][j] = Probability of being in state j at time t
    def e_step(self, sample):
        alpha = self.forward(sample)
        beta = self.backward(sample)
        y = np.zeros((len(sample), self.num_states))
        e = np.zeros((len(sample), self.num_states, self.num_states))
        for t in range(1, len(sample)):
            den = 0
            for j in range(0, self.num_states):
                den += alpha[t][j] * beta[t][j]
            for j in range(0, self.num_states):
                y[t][j] = (alpha[t][j] * beta[t][j])/den
                for i in range(0, self.num_states):
                    if t != len(sample) - 1:
                        e[t][j][i] = (alpha[t][i] * self.transitions[i]
                                      [j] * self.emissions[j][t+1] * beta[t+1][j])/den
        return y, e

    # Tunes transitions
    def tune_transitions(self, sample, y, e):
        for i in range(0, self.num_states):
            for j in range(0, self.num_states):
                num = 0
                den = 0
                for t in range(1, len(sample) - 1):
                    num += e[t][i][j]
                    for k in range(0, self.num_states):
                        den += e[t][i][k]
                self.transitions[i][j] = num/den

    # Tunes emissions
    def tune_emissions(self, sample, y, e):
        for j in range(0, self.num_states):
            # Need to keep it this way since vocab values start from 1
            for vk in range(1, self.vocab_size + 1):
                den = 0
                num = 0
                for t in range(1, len(sample)):
                    den += y[t][j]
                    if vk == sample[t]:
                        num += y[t][j]
                self.emissions[j][vk-1] = num/den

    # Uses the e and y matrices from the e_step to tune transition and emission probabilities
    def m_step(self, sample, y, e):
        self.tune_transitions(sample, y, e)
        self.tune_emissions(sample, y, e)

    # apply a single step of the em algorithm to the model on all the training data,
    # which is most likely a python list of numpy matrices (one per sample).
    # Note: you may find it helpful to write helper methods for the e-step and m-step,
    def em_step(self, dataset):
        # Takes out a sample from the dataset and does e_step and m_step
        print("Before EM")
        print(self.transitions)
        print(self.emissions)
        i = 0
        for sample in dataset:
            if i == 100:
                break
            y, e = self.e_step(sample)
            self.m_step(sample, y, e)
            i += 1
        print("After EM")
        print(self.transitions)
        print(self.emissions)

    # Return a "completed" sample by additing additional steps based on model probability.
    def complete_sequence(self, sample, steps):
        pass


def main():
    parser = argparse.ArgumentParser(
        description='Program to build and train a neural network.')
    parser.add_argument('--train_path', default=None,
                        help='Path to the training data directory.')
    parser.add_argument('--dev_path', default=None,
                        help='Path to the development data directory.')
    parser.add_argument('--max_iters', type=int, default=30,
                        help='The maximum number of EM iterations (default 30)')
    parser.add_argument('--hidden_states', type=int, default=10,
                        help='The number of hidden states to use. (default 10)')
    args = parser.parse_args()

    # OVERALL PROJECT ALGORITHM:
    # 1. load training and testing data into memory
    #
    # 2. build vocabulary using training data ONLY
    #
    # 3. instantiate an HMM with given number of states -- initial parameters can
    #    be random or uniform for transitions and inital state distributions,
    #    initial emission parameters could bea uniform OR based on vocabulary
    #    frequency (you'll have to count the words/characters as they occur in
    #    the training data.)
    #
    # 4. output initial loglikelihood on training data and on testing data
    #
    # 5+. use EM to train the HMM on the training data,
    #     output loglikelihood on train and test after each iteration
    #     if it converges early, stop the loop and print a message

    # Paths for positive and negative training data
    postrain = os.path.join(args.train_path, 'pos')
    negtrain = os.path.join(args.train_path, 'neg')
    # Combine into list
    train_paths = [postrain, negtrain]

    # Create vocab and get its size. word_vocab is a dictionary from words to integers. Ex: 'painful':2070
    word_vocab = build_vocab_words(train_paths)
    vocab_size = len(word_vocab)
    dataset = load_and_convert_data_words_to_ints(train_paths, word_vocab)
    # Create model
    model = HMM(args.hidden_states, vocab_size)
    model.em_step(dataset)


if __name__ == '__main__':
    main()
