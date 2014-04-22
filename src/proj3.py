from parse import *
from make_features import *
from viterbi import *
import numpy as np
import pandas as pd
import math


"""
overview
    - we'd like review level and sentence level info
    - need transitions, A, among sentence sentiments
    - need start and end review tokens
    - need start and end sentence tokens

data structure
    - see parse.py file
    - 

tuning parameters
    - capitalization
    - for laplace smoothing, the n smoothing parameter
    - good turing smoothing
    - acceptance probability for a feature


"""

A, emission_dict = parse_text(path)

e = features(emission_dict)
e.gen_prob(1000)

viterbi([["the", "cat", "went", "to", "the", "barn"]], e, A)