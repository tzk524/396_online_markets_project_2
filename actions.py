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
    
# ----------------------------------------------------------------------------
def calculate_exponential_weight(action, e, step, h):
    """

    Parameters
    ----------
    action : list of int
        A list that contains the complete payoff of an action for every round.
    e : float in (0, 1)
        The learning rate used in calculating exponential weights.
    step : int
        Specifies the current round of interest.
    h : int
        Specifies the cap for the per-stage payoff.

    Returns
    -------
    expo_weight : float
        The exponential weight of choosing the current action at the input step.
        Calculated using the input learning rate e.

    """
    past = action[:step]
    past_payoff = sum(past)
    expo_weight = (1 + e) ** (past_payoff / h)
    return expo_weight


def calculate_exponential_prob(payoff, e, step, h):
    """
    
    Parameters
    ----------
    payoff : nested-list
        A nested list that contains k lists, each corresponds to one of the
        possible k actions. a.k.a. Each k's simulated payoff.
    e : float in (0, 1)
        The learning rate used in calculating exponential weights.
    step : int
        Specifies the current round of interest.
    h : int
        Specifies the cap for the per-stage payoff.

    Returns
    -------
    exponential_prob : list(float) that sums to 1
        A list that contains k elements. Specifies the probability of choosing
        action k in round(input step) according to exponential weights calculated
        using the learning rate e. 
        
    """
    expo_weights = []
    for action in payoff:
        curr_weight = calculate_exponential_weight(action, e, step, h)
        expo_weights.append(curr_weight)
    total = sum(expo_weights)
    exponential_prob = []
    for weight in expo_weights:
        curr_prob = weight / total
        exponential_prob.append(curr_prob)
    return exponential_prob


def exponential_weights(payoff, e, h=1):
    """
    Given a payoff, this funtion produces a list of actions that the user
    shall take following the exponential_weights algorithm.
    e = learning rate
    h = cap for the per-stage payoff.
    """
    actions = []
    k = len(payoff) # The number of actions to choose from.
    candidates = list(range(k)) # All actions to choose from in each round.
    steps = len(payoff[0]) # The number of rounds.
    for i in range(steps):
        if i == 0:
            curr_action = np.random.choice(candidates)
        else:
            exponential_prob = calculate_exponential_prob(payoff, e, i, h)
            curr_action = np.random.choice(candidates, 1, p=exponential_prob)[0]
        actions.append(curr_action)
    return actions
   
    
# ----------------------------------------------------------------------------
def follow_the_perturbed_leader(payoff):
    """
    Given a payoff, this funtion produces a list of actions that the user
    shall take following the follow_the_perturbed_leader algorithm.
    """
    raise NotImplemented()