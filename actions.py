import numpy as np

def best_in_hindsight(payoff):
    """
    Given a payoff, this function produces a list of actions that the user
    shall take following the best-in-hindsight algorithm.
    """
    k = len(payoff) # The number of actions to choose from.
    steps = len(payoff[0]) # The number of rounds.
    hindsight_payoff = []
    for i in range(k):
        p_i = sum(payoff[i])
        hindsight_payoff.append(p_i)
    opt_action = np.argmax(hindsight_payoff)
    
    actions = []
    for i in range(steps):
        actions.append(opt_action)
    return actions
    