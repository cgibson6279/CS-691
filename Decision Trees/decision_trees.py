'''
-Greedy Algorithm (how do we apply entropy and IG to build tree)
1)Calculate entropy (H) for the entire training set
2) Split training data using each feature and calculate Information Gain (IG) for each split
3) Choose to split on the feature that gives the best IG
4) Repeat steps 2-4 on each subset until IG is zero or we run out of features

-recursive process

eg) Tv Shows

Sample #    Show        <40min?     Comedy?     Female Lead?    Political   Like?
1*           The Office      Y           Y           N               N       N
2*           Veep            Y           Y           Y               Y       Y*
3*           Parks&Rec       Y           Y           Y               Y       Y*
4*           House of Cards  N           N           N               Y       N
5*           Scandal         N           N           Y               Y       N
6*           OINB            N           N           Y               N       Y*
7*           Riverdale       N           N           N               N       N
8*           Glow            Y           N           Y               N       N
9*           UKS             Y           Y           Y               N       Y*
10*          The Crown       N           N           Y               Y       N

Let's build a decision tree that will predict whether I like new shows

'''

import algs
import numpy as np
from typing import Callable, Dict, List

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

def DT_train_binary(X : Callable[[np.ndarray], int], Y : Callable[[np.ndarray], int], max_depth : int) -> Dict[str,List]:
    pass

def main():

    Y_no_entropy, Y_yes_entropy, Y_p_yes, Y_p_no, Y_H = algs.entropy(Y,0)

    set_entrop = Y_H

    #create decision tree

    decision_tree = dict()

    num_rows, num_cols = X.shape

    for i in range(num_cols):
        X_no_entropy, X_yes_entropy, X_p_yes, X_p_no, X_H = algs.entropy(X, i)
        X_IG = algs.info_gain(X,Y,i)
        decision_tree[f'Sample {i + 1}'] = [X_no_entropy, X_yes_entropy, X_H, X_IG]

    print(decision_tree)




if __name__== "__main__":
    main()