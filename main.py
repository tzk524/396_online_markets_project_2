from generate_inputs import init_inputs
from actions import best_in_hindsight, exponential_weights, hallucinate, follow_the_perturbed_leader
from regret import check_performance, calculate_regret, cal_theo_e
import numpy as np
from tqdm import tqdm
from load_data import load_data
from generator import draw_bids, gen_auction_payoff, auction_calculate_regret, generate_increment_list


ACTIONS = [2, 4, 8, 16, 32]
ROUNDS = [4, 8, 16, 32, 64, 128]
ITERATIONS = 100
LEARNING_RATES = [x/100 for x in range(1, 100)]
DATA_PATH = DATA_PATH = "bid_data.csv"
data = load_data((DATA_PATH))
V0 = 35.5
V1 = 97.9
V2 = 43.1
V3 = 63.6
VALUES = [V0, V1, V2, V3]
K1 = generate_increment_list(2, 100, 1)
E1 = generate_increment_list(0.01, 0.99, 0.01)
S1 = [10, 20, 50, 100, 500]

def question1(actions, rounds, iterations, learning_rates):
    """
    t0 - theo optimal e
    t1 - empirical optimal e
    t2 - theo regret
    t3 - empirical regret
    """
    ew_result = []
    ftpl_result = []
    for i in range(4):
        ew_result.append(np.zeros((len(actions), len(rounds))))
        ftpl_result.append(np.zeros((len(actions), len(rounds))))

    
    for i in tqdm(range(len(actions))):
        K = actions[i]
        
        for j in tqdm(range(len(rounds))):
            N = rounds[j]
            theo_e = cal_theo_e(K, N)
            ew_result[0][i][j] = theo_e
            ftpl_result[0][i][j] = theo_e
            ew_theoregretsum = 0
            ftpl_theoregretsum = 0
            
            lr_performance_ew = np.zeros(len(learning_rates))
            lr_performance_ftpl = np.zeros(len(learning_rates))
                
            for iteration in range(iterations):
                payoff = init_inputs(K, N)[1]
                theo_ew = exponential_weights(payoff, theo_e)
                theo_hallucination = hallucinate(payoff, theo_e)
                theo_ftpl = follow_the_perturbed_leader(payoff, theo_hallucination)
                theo_ew_regret = calculate_regret(payoff, theo_ew)
                theo_ftpl_regret = calculate_regret(payoff, theo_ftpl)
                ew_theoregretsum += theo_ew_regret
                ftpl_theoregretsum += theo_ftpl_regret

                for ii in range(len(learning_rates)):
                    e = learning_rates[ii]
                    ew = exponential_weights(payoff, e)
                    hallucination = hallucinate(payoff, e)
                    ftpl = follow_the_perturbed_leader(payoff, hallucination)
                    ew_regret = calculate_regret(payoff, ew)
                    ftpl_regret = calculate_regret(payoff, ftpl)
                    lr_performance_ew[ii] += ew_regret
                    lr_performance_ftpl[ii] += ftpl_regret
                    
            ew_theoregretmean = ew_theoregretsum / iterations
            ftpl_theoregretmean = ftpl_theoregretsum / iterations
            ew_result[2][i][j] = ew_theoregretmean
            ftpl_result[2][i][j] = ftpl_theoregretmean
            
            lr_performance_ew = [x/iterations for x in lr_performance_ew]
            lr_performance_ftpl = [x/iterations for x in lr_performance_ftpl]
            
            ew_min = min(lr_performance_ew)
            ew_argmin = np.argmin(lr_performance_ew)
            ew_empirical = learning_rates[ew_argmin]
            ew_result[1][i][j] = ew_empirical
            ew_result[3][i][j] = ew_min
            
            ftpl_min = min(lr_performance_ftpl)
            ftpl_argmin = np.argmin(lr_performance_ftpl)
            ftpl_empirical = learning_rates[ftpl_argmin]
            ftpl_result[1][i][j] = ftpl_empirical
            ftpl_result[3][i][j] = ftpl_min
            
    return ew_result, ftpl_result

def question2_check_1(data, value, steps=100, K=K1):
    draws = draw_bids(data, steps)
    output_ew = np.zeros(len(K))
    output_ftpl = np.zeros(len(K))
    for i in tqdm(range(len(K))):
        k = K1[i]
        payoff = gen_auction_payoff(draws, k, value)[0]
        e = cal_theo_e(k, steps)
        ew = exponential_weights(payoff, e, h=value)
        hallucination = hallucinate(payoff, e, h=value)
        ftpl = follow_the_perturbed_leader(payoff, hallucination)
        ew_regret = auction_calculate_regret(payoff, draws, value, ew)
        ftpl_regret = auction_calculate_regret(payoff, draws, value, ftpl)
        output_ew[i] = ew_regret
        output_ftpl[i] = ftpl_regret
    return output_ew, output_ftpl
        

def question2_check_2(data, value, steps=100,k=50, E=E1):
    draws = draw_bids(data, steps)
    output_ew = np.zeros(len(E))
    output_ftpl = np.zeros(len(E))
    payoff = gen_auction_payoff(draws, k, value)[0]
    for i in tqdm(range(len(E))):
        e = E1[i]
        ew = exponential_weights(payoff, e, h=value)
        hallucination = hallucinate(payoff, e, h=value)
        ftpl = follow_the_perturbed_leader(payoff, hallucination)
        ew_regret = auction_calculate_regret(payoff, draws, value, ew)
        ftpl_regret = auction_calculate_regret(payoff, draws, value, ftpl)
        output_ew[i] = ew_regret
        output_ftpl[i] = ftpl_regret
    return output_ew, output_ftpl


def question_2_check_3(data, value, steps=S1, k=50):
    output_ew = np.zeros(len(steps))
    output_ftpl = np.zeros(len(steps))
    for i in tqdm(range(len(steps))):
        s = steps[i]
        draws = draw_bids(data, s)
        payoff = gen_auction_payoff(draws, k, value)[0]
        e = cal_theo_e(k, s)
        ew = exponential_weights(payoff, e, h=value)
        hallucination = hallucinate(payoff, e, h=value)
        ftpl = follow_the_perturbed_leader(payoff, hallucination)
        ew_regret = auction_calculate_regret(payoff, draws, value, ew)
        ftpl_regret = auction_calculate_regret(payoff, draws, value, ftpl)
        output_ew[i] = ew_regret
        output_ftpl[i] = ftpl_regret
    return output_ew, output_ftpl
        