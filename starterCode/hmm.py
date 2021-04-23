# File: hmm.py
# Purpose:  Starter code for building and training an HMM in CSC 246.


import argparse
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
        pass

    # return the loglikelihood for a complete dataset (train OR test) (list of matrices)
    def loglikelihood(self, dataset):
        pass

    # return the loglikelihood for a single sequence (numpy matrix)
    def loglikelihood_helper(self, sample):
        pass

    # apply a single step of the em algorithm to the model on all the training data,
    # which is most likely a python list of numpy matrices (one per sample).
    # Note: you may find it helpful to write helper methods for the e-step and m-step,
    def em_step(self, dataset):
        pass

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


if __name__ == '__main__':
    main()
