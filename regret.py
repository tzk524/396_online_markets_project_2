from actions import best_in_hindsight

def check_performance(payoff, actions):
    """
    Given a set of payoffs and a set of actions (generated from some algorithm),
    check each round to see how well did the algorithm perform.
    
    return:
        performance: A list of length(steps). Shows the payoff of the algorithm
                     in each round.
    """
    performance = []
    steps = len(actions)
    for i in range(steps):
        curr_action = actions[i]
        result = payoff[curr_action][i]
        performance.append(result)
    return performance


def calculate_regret(payoff, actions):
    """
    Given a set of payoffs and a set of actions (generated from some algorithm),
    compute the regret of the algorithm by comparing to the best_in_hindsight payoff.
    """
    algo_performance = check_performance(payoff, actions)
    opt_actions = best_in_hindsight(payoff)
    opt_performance = check_performance(payoff, opt_actions)
    OPT = sum(opt_performance)
    ALG = sum(algo_performance)
    n = len(actions)
    regret = (OPT - ALG) / n
    return regret