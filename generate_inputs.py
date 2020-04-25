"""
The goal of this script is to generate input data stochastically for testing.
"""
import random


def init_distribution(k):
    """

    Parameters
    ----------
    k : int
        The number of possible actions to choose from.

    Returns
    -------
    k_distribution : list of float between 0 and 1
        A list that contains pj for action j. pj is a random float generated
        between 0 and 1.

    """
    k_distribution = []
    for j in range(k):
        pj = random.random()
        k_distribution.append(pj)
    return k_distribution

def init_payoff(k_distribution, steps):
    """
    
    Parameters
    ----------
    k_distribution : list of float between 0 and 1
        This parameter tells us the probability of drawing 1 for each of the
        action j for all possible actions k. 
    steps : int
        This parameter specifies how many steps of inputs do we like to generate.

    Returns
    -------
    payoff : nested-list
        The return is a nestesd list that contains k lists, each corresponds
        to one of the possible k actions.
    
    """
    payoff = []
    for k in k_distribution:
        k_payoff = []
        for i in range(steps):
            gauge = random.random()
            if k > gauge:
                result = 1
            else:
                result = 0
            k_payoff.append(result)
        payoff.append(k_payoff)
    return payoff


def init_inputs(k, steps):
    """

    Parameters
    ----------
    k : int
        The number of possible actions to choose from.
    steps : int
        The total number of rounds used in generating inputs stochastically.

    Returns
    -------
    k_distribution : list of float between 0 and 1
        A list that contains pj for action j. pj is a random float generated
        between 0 and 1.
    payoff : nested-list
        A nested list that contains k lists, each corresponds to one of the
        possible k actions. a.k.a. Each k's simulated payoff.
        
        
    At a minimum you should compare the algorithm's regrets on stochastic inputs 
    with the following data generating model:

    The payoff of each action j in each round is an independent draw from a 
    Bernoulli distribution, i.e., it is 1 with a given probability pj and and 0 
    with the remaining probability (1-pj).
    The probabilities of the k actions are fixed in advance as some (p1,...,pk).
    
    """
    k_distribution = init_distribution(k)
    payoff = init_payoff(k_distribution, steps)
    return k_distribution, payoff