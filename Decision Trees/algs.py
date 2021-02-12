import numpy as np
import math
from typing import List, Callable, Tuple

def log2(x : int) -> float:
    '''
    :param n:
        the number we are applying our log_2 function to
    :return:
        the log_2 of number
    '''
    try:
        log_num = round(np.log2(x),4)
    except RuntimeWarning as error:
        log_num = 0
    return log_num

def probability(count : int, set_length : int) -> float:
    '''
    :param count:
        the count of feature tokens present in set
    :param set_length:
        length of set feature class being observed
    :return:
        probability of feature token observation in class set
    '''
    return round(count/set_length,2)

def split_entropy(X : Callable[[np.ndarray], int], row_num: int, feature : int) -> Tuple[float,float,float,float,float]:
    '''
    alg:
        H() = ∑_c∈C -p(c) log2p(c) -> the sum of the negative probability of No times the log of the probability of No plus
        the sum of the negative probability of Yes times the log of the probability of Yes

    :param x:
        an np.array of feature inputs, where each row corresponding to a single sample
    :param y:
        a np.array of label inputs for feature input row
    :return:
        no_entropy, yes_entropy, p_yes, p_no, H

    '''

    # get feature (yes, no) counts
    feature_count = 0

    for f in X[:, row_num]:
        if f == feature:
            feature_count += 1

    #find probability
    prob = probability(feature_count, len(X))

    #calculate entropy -> H() = ∑_c∈C -p(c) log2p(c)
    entropy = (-(prob) * log2(prob))

    return prob, entropy

def set_entropy(no_entropy :float, yes_entropy: float):
    # calculate entropy -> H() = ∑_c∈C -p(c) log2p(c)
    return no_entropy + yes_entropy

def subset_array(X : Callable[[np.ndarray], float], Y : Callable[[np.ndarray], float], index : int, feature : int) -> Callable[[np.ndarray], float]:
    '''

    :param array:
        np.array
    :param index:
        column index to create subset array
    :param feature:
        feature 1 or 0 that were looking for
    :return:
        np.array that contains the subset of the feature being observed
    '''
    feature_index = np.where(X[:, index] == feature)

    X_sub = []
    Y_sub = []

    for array in feature_index:
        for feature in array:
            X_sub.append(X[feature])
            Y_sub.append(Y[feature])

    X_sub = np.array(X_sub)
    Y_sub = np.array(Y_sub)

    return X_sub, Y_sub

def info_gain(X : Callable[[np.ndarray], float], Y : Callable[[np.ndarray], float], row_num: int) -> float:
    '''
    IG = H() - ∑_t∈T -p(t)H(t) -> the entropy of the full set minus the probability that I ended
    up on a particular side of a split times the entropy of that split

    :param x:
        an np.array of feature inputs, where each row corresponding to a single sample
    :param y:
    an np.array of label inputs, where each row corresponding to a single sample
    :return:
        information gain value as a float value
    '''

    #get entropy of entire set
    no_prob, no = split_entropy(Y, row_num, 0)
    yes_prob, yes = split_entropy(Y, row_num, 1)

    H_set = set_entropy(no, yes)

    #get probability of feature across dataset
    f_no_prob, f_no_X = split_entropy(X, row_num, 0)
    f_yes_prob, f_yes_X = split_entropy(X, row_num, 1)

    #get no subset of feature
    f_no, f_no_labels = subset_array(X, Y, row_num, 0)
    #get yes subset of feature
    f_yes, f_yes_labels = subset_array(X, Y, row_num, 1)


    #get no subset split entropy
    f1_no_entropy_prob, f1_no_entropy = split_entropy(f_no_labels, row_num, 0)
    f1_yes_entropy_prob, f1_yes_entropy = split_entropy(f_no_labels, row_num, 1)

    H_no = set_entropy(f1_no_entropy, f1_yes_entropy)

    #get yes subset split entropy
    f_y_entropy_prob, f_y_entropy = split_entropy(f_yes_labels, row_num, 1)
    f_n_entropy_prob, f_n_entropy = split_entropy(f_yes_labels, row_num, 0)

    H_yes = set_entropy(f_y_entropy, f_n_entropy)

    #get info gain -> IG = H() - ∑_t∈T -p(t)H(t)

    IG = round(H_set - ((f_no_prob * H_no) + (f_yes_prob * H_yes)),4)
    print(f'{H_set} - (({f_no_prob} * {H_no}) + ({f_yes_prob} * {H_yes})) = {IG}')
    return IG

X = np.array([[1, 1, 0, 0],
              [1, 1, 1, 1],
              [1, 1, 1, 1],
              [0, 0, 0, 1],
             [0, 0, 1, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 0],
             [1, 0, 1, 0],
             [1, 1, 1, 0],
             [0, 0, 1, 1]])

Y = np.array([[0],[1],[1],[0],[0],[1],[0],[0],[1],[0]])


#print(f1_yes_entropy)
#print(f1_no_entropy)

#print(set_entropy(f1_yes_entropy, f1_no_entropy))
#print(info_gain(X,Y,0))

#H(<40min? = N) = -(1/5)*log2(1/5)-(4/5)log2(4/5) = 0.7219
#H(<40min? = Y) = -(2/5)*log2(2/5)-(3/5)log2(3/5) = 0.9709

#IG = 0.9709 - ((5/10) * 0.7219 + (5/10) * 0.9709) = 0.1245

print(info_gain(X,Y,0))
#print(X_yes_set)
#print(len(X_yes_set))