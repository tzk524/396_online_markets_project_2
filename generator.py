import numpy as np

def draw_bids(data, rounds):
    """ Given data and rounds, randomly select a bid for each round. """
    output = np.zeros(rounds)
    for i in range(rounds):
        choice = np.random.choice(data[:, 1])
        output[i] = choice
    return output


def generate_increment_list(start, end, increment):
    """
    input:
        start(float): The starting number. The smallest of the list.
        end(float): The last number. The largest of the list.
        increment(float): The increment between every two adjecent numbers.
    output:
        output: A list of float where the first number is start and
                the last number is end, and contains all number inbetween
                start and end with increments of increment.
    """
    output = []
    current = start
    while current < end: 
        output.append(current)
        current += increment
    output.append(end)
    return output



def best_in_hindsight_helper(action, value, draws):
    payoff = 0
    for d in draws:
        if action > d:
            payoff += value - action
        elif action <= d:
            payoff += 0
    return payoff
    
def auction_best_in_hindsight(value, draws):
    action_sapce = generate_increment_list(0, 100, 0.01)
    payoff = []
    for i in action_sapce:
        curr = best_in_hindsight_helper(i, value, draws)
        payoff.append(curr)
    best_payoff = max(payoff)
    best_arg = np.argmax(payoff)
    best_action = action_sapce[best_arg]
    return best_action, best_payoff



def check_action(draws, value, action):
    output = np.zeros(len(draws))
    for i in range(len(draws)):
        d = draws[i]
        if action > d:
            output[i] = value - action
        else:
            output[i] = 0
    return output

def find_actionspace(k, value):
    base = value / k
    curr = base
    action_space = []
    while curr <= value:
        action_space.append(curr)
        curr += base
    return action_space

def gen_auction_payoff(draws, k, value):
    action_space = find_actionspace(k, value)
    payoff = []
    for action in action_space:
        row = check_action(draws, value, action)
        payoff.append(row)
    return np.array(payoff), action_space
 
    
def auction_calculate_regret(payoff, draws, value, actions):
    performance = 0
    for i in range(len(actions)):
        action = actions[i]
        performance += payoff[action][i]
    best_performance = auction_best_in_hindsight(value, draws)[1]
    regret = (best_performance - performance) / len(actions)
    return regret
     
        