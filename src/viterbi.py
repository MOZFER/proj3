import numpy as np
import pandas as pd

'''

viterbi algorithm here

outline of algorithm:
    we want the Viterbi path of likely states

obs_list = [[sentence], ]
emission_dict = {sentiment: {feature: count}}
trans_dict = {(state1, state2): count}

'''

#class should have smoothing already
def viterbi(obs_list, e, trans_matrix):
    num_obs = len(obs_list)
    states = ["neg", "neu", "pos"]
    num_states = len(e)

    #this is the probabiliy of being in a state at time t
    viterbi_prob = pd.DataFrame(np.zeros([len(trans_matrix.index), num_obs]), index=trans_matrix.index)

    #this is the most probable 
    backpointer = pd.DataFrame(np.zeros([len(trans_matrix.index), num_obs]), index=trans_matrix.index)

    #initialize
    for state in states:
        b = obs_list[0]
        viterbi_prob.loc[state, 0] = math.log(trans_matrix.loc["<r>", state], 2) + e.return_prob(state, b)
        backpointer.loc[state, 0] = 0.

    #intermedate steps (recursion)
    for t in range(1, len(obs_list)):
        for s in states: #s_prime prev state; s current state
            #does them both in one shot
            #uses the fact that max does the max of the first element of a tuple
            #so we tack the state name as the second element of a tuple, (log-prob, state)
            viterbi_prob.loc[s,t], backpointer.loc[s,t] = \
                max( \
                [ ( viterbi_prob.loc[s_prime, t-1] + \
                math.log(trans_matrix.loc[s_prime, s], 2) + \
                e.return_prob(s, obs_list[t]), s_prime) \
                for s_prime in states ] )

    #end step
    viterbi_prob.loc["</r>", len(obs_list) - 1], backpointer.loc["</r>", len(obs_list) - 1] = \
        max( [ ( viterbi_prob.loc[s, len(obs_list) - 1] + \
        math.log(trans_matrix.loc[s, "</r>"], 2), s) for s in states])

    print(viterbi_prob)
    print(backpointer)

    return viterbi_path