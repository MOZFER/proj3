import pandas as pd
import numpy as np

class features:
    def __init__(self, emission_counts_dict):
        self.emission_counts = emission_counts_dict
        self.emission_prob = {"pos": {}, "neg": {}, "neu": {}}
        self.emission_sum = {"pos": 0, "neg": 0, "neu": 0}
        self.current_smoothing = None
        self.v = None

    #call this to smooth
    def gen_prob(self, virtual_obs, smoothing = "default"):
        #will re-smooth if different smoothing specified
        if self.current_smoothing != smoothing:
            if smoothing == "default":
                if self.v == None:
                    self.v = virtual_obs
                self.laplace_prob()
                self.current_smoothing = "default"

    def laplace_prob(self):
        for key in self.emission_counts:
            self.emission_sum[key] = sum([v for k, v in self.emission_counts[key].items()])
            p = 1/self.emission_sum[key]
            for n_gram in self.emission_counts[key]:
                self.emission_prob[key][n_gram] = (self.emission_counts[key][n_gram] + p * self.v) / (self.emission_sum[key] + self.v)

    #returns the summed log-probs of a list of words
    def return_prob(self, state, words):
        if state in set(["<r>", "</r>"]): #for trivial states
            return 0
        if self.current_smoothing:
            log_prob_sum = 0
            for word in words:
                if word in self.emission_prob[state]:
                    log_prob_sum += math.log(self.emission_prob[state][word], 2)
                else:
                    log_prob_sum += (1/self.emission_sum[state] * self.v)/(self.v)
            return log_prob_sum
        else:
            return None

    def __len__(self):
        return len(self.emission_prob)