import copy
import time
import numpy as np

def knapsack_multichoice_onepick(weights, values, max_weight):
    if len(weights) == 0:
        return 0

    last_array = [-1 for _ in range(max_weight + 1)]
    last_path = [[] for _ in range(max_weight + 1)]
    for i in range(len(weights[0])):
        if weights[0][i] < max_weight:
            if last_array[weights[0][i]] < values[0][i]:
                last_array[weights[0][i]] = values[0][i]
                last_path[weights[0][i]] = [(0, i)]

    for i in range(1, len(weights)):
        current_array = [-1 for _ in range(max_weight + 1)]
        current_path = [[] for _ in range(max_weight + 1)]
        for j in range(len(weights[i])):
            for k in range(weights[i][j], max_weight + 1):
                if last_array[k - weights[i][j]] > 0:
                    if current_array[k] < last_array[k - weights[i][j]] + values[i][j]:
                        current_array[k] = last_array[k - weights[i][j]] + values[i][j]
                        current_path[k] = copy.deepcopy(
                            last_path[k - weights[i][j]])
                        current_path[k].append((i, j))
        last_array = current_array
        last_path = current_path

    solution, index_path = get_onepick_solution(last_array, last_path)

    return solution, index_path


def get_onepick_solution(scores, paths):
    scores_paths = list(zip(scores, paths))
    scores_paths_by_score = sorted(scores_paths, key=lambda tup: tup[0],
                                    reverse=True)

    return scores_paths_by_score[0][0], scores_paths_by_score[0][1]


def knapsack_optimizer(Rs, K, N):
    Rs_list = Rs.reshape(K,-1)
    Rs_list = [list(a) for a in Rs_list]
    weights = []
    for i in range(K):
        weights.append(list(range(0,N+1)))
    max_value, max_action = knapsack_multichoice_onepick(weights, Rs_list, N)
    
    if (max_action == []): # when all rewards = 0
        return []
    else:
        A = np.zeros(K)
        for i in range(K):
            A[i] = i*(N+1) + max_action[i][1]
        A = A.astype(int)
        # the kth element in vector A denotes the index of the budget proportion that has been chosen
        # for example, if N=100 and we assign 0% proportion of budget to adline k=0 and adline k=1, 
        # then vector A should be (0, 101, ...).
        # \sum_x {(x)%(N+1)}/N ---- calculate the total budget proportion that has been used
        return A