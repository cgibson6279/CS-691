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
        log_num = round(math.log2(x),4)
    except ValueError as error:
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

def entropy(X : Callable[[np.ndarray], int], row_num: int) -> Tuple[float,float,float,float,float]:
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

        no_entropy -> entropy of no branch
        yes_entropy -> entropy of yes branch
        p_yes -> probability of yes
        p_no -> probability of no
        H -> entropy of a class set in the form of a float value
    '''

    # get feature (yes, no) counts
    yes = 0
    no = 0

    for feature in X[:, row_num]:
        print(feature)
        if feature == 1:
            yes += 1
        else:
            no += 1

    #find probability
    p_yes = probability(yes, len(X))
    p_no = probability(no, len(X))

    #calculate entropy -> H() = ∑_c∈C -p(c) log2p(c)
    no_entropy = (-(p_no) * log2(p_no))
    yes_entropy = (-(p_yes) * log2(p_yes))

    H = round(no_entropy + yes_entropy,4)

    return no_entropy, yes_entropy, p_yes, p_no, H


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
    #y_entropy -> total entropy of set
    y_no_entropy, y_yes_entro, y_yes, y_no, y_entropy = entropy(Y,0)

    #get entropy of each split
    #x_no_entro -> entropy for no side of split
    #x_yes_entro -> entropy for yes side of split
    #x_yes -> yes probability
    #x_no -> no probability
    #x_entropy -> entropy of set
    x_no_entro, x_yes_entro, x_yes, x_no, x_entropy = entropy(X, row_num)

    #get info gain ->

    IG = y_entropy - ((x_no * x_no_entro) + (x_yes * x_yes_entro))
    #print(f'y_entropy: {y_entropy}\nx_no: {x_no}\nx_no_entro: {x_no_entro}\nx_yes: {x_yes}\nx_yes_entro: {x_yes_entro}')
    return IG

    #H = ((-(p_no) * log2(p_no)) + (-(p_yes) * log2(p_yes)))



'''
Test Data 

Sample      Sunny?      >90     Outside?
1           Y           Y       N
2           Y           N       Y
3           N           Y       N
4           N           N       N
5           N           Y       ?

IG = H() - ((1/2)*H(Sunny? = N) + (1/2) * H(Sunny? = Y) = 0.811 - (0 + (1/2) = 0.311

'''
X_test = np.array([[1,1],[1,0],[0,1],[0,0]])
Y_test = np.array([[0],[1],[0],[0]])

print(info_gain(X_test,Y_test,0))
#print(entropy(Y_test,0))

#X = np.array([[0,1,0,1],[1,1,1,1],[0,0,0,1]])
#Y = np.array([[1],[1],[0]])