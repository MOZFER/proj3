import numpy as np
import pandas as pd
import re
from itertools import islice
from copy import deepcopy

path = "/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj3/data/training_data.txt"

#fast sliding window function
def window(iterable, size):
    it = iter(iterable)

    #this works because we knock out the first n elements
    #i.e.: you can only go through an iterator once
    result = tuple(islice(it, size))

    if len(result) == size:
        yield result
    for item in it:
        result = result[1:] + (item,)
        yield result

class Classifier:
    def __init__(self, path, n = 1, k = 5): #n can equal 1,2,3

        self.sentiments = ["neg", "neu", "pos"]

        self.A, self.sentences_by_sentiment = self.parse_text(path, n)

        self.vocab = self.make_vocab(self.sentences_by_sentiment)

        if 1 <= n:
            self.unigrams    = self.make_features(self.sentences_by_sentiment, 1)
            self.gt_unigrams = self.good_turing(self.unigrams, 1, k)
        if 2 <= n:
            self.bigrams     = self.make_features(self.sentences_by_sentiment, 2)
            self.gt_bigrams  = self.good_turing(self.bigrams,  2, k)
        if 3 <= n:
            self.trigrams    = self.make_features(self.sentences_by_sentiment, 3)
            self.gt_trigrams = self.good_turing(self.trigrams, 3, k)


    #converts text to dictionary keyed by sentiment
    #handles unknown words
    #with list of sentences as the values
    #also gets transition matrix
    def parse_text(self, path, n):

        sentence_sentiment_dict = {s: [] for s in self.sentiments}

        states = ["<r>", "neg", "neu", "pos", "</r>"]
        A = pd.DataFrame(np.zeros((5,5)), index = states, columns = states)
        A.loc["</r>", "</r>"] = 1 #just to prevent NAs

        with open(path, 'rb') as f:
            text = f.read()

        text = text.decode("utf-8")

        text = re.split(r'\n\n', text)
        text.pop()

        #no sense treating these as unknown
        seen_words = set(["!", ".", ",", "<s>", "</s>", ":", ";", "'", "\"", "/", "\\"]) 
        unk = "<unk>"

        for review in text:
            r = review.split("\n")
            review_type = r.pop(0)
            category, label, number = review_type.split("_")

            #add begin and end review tokens
            r = ["<r>\t"] + r + ["</r>\t"]

            prev_sentiment = r.pop(0).split('\t')[0] #starting sentiment, <r>

            for line in r:
                sentiment, sentence = line.split('\t')

                sentence = self.tokenize_sentence(sentence, n)

                if sentiment not in ["<r>", "</r>"]:
                    sentence_sentiment_dict[sentiment].append(sentence)

                A.loc[prev_sentiment, sentiment] += 1
                prev_sentiment = sentiment

                #for unk
                for word in range(len(sentence)):
                    if sentence[word] not in seen_words:
                        seen_words.add(sentence[word]) #add to seen words
                        sentence[word] = unk #overwrite

        s = A.sum(axis=1)

        for item in s.index:
            A.loc[item,:] = A.loc[item,:]/s.loc[item] #divide row by row sum

        return A, sentence_sentiment_dict

    #tokenizes sentence
    @staticmethod
    def tokenize_sentence(sentence, n):
        #make all lower: 
        sentence = sentence.lower()

        #puts whitespace around everything except words and whitespace
        sentence = re.sub(r'[^\w\s\')]', ' \g<0> ', sentence)
        sentence = sentence.strip()

        #for ngrams n > 1, add n-1 start tokens and an end token
        if n > 1:
            sentence = sentence + [" </s>"]
            for x in range(n-1):
                sentence = ["<s> "] + sentence

        sentence = re.split(r' +', sentence)

        return sentence

    #returns the vocab
    def make_vocab(self, parsed_text):
        vocab = set()

        for key in parsed_text:
            for sentence in parsed_text[key]:
                vocab | set(sentence)

        return vocab

    #gets unigrams if n = 1, bigrams if n = 2, etc.
    def make_features(self, parsed_text, n):
        feature_dict = {"pos": {}, "neg": {}, "neu": {}}

        for key in parsed_text:
            for sentence in parsed_text[key]:
                for n_gram in window(sentence, n):
                    if len(n_gram) == 1:
                        #need n_gram[0] because it's a tuple
                        #just pull out the underlying string
                        gram = n_gram[0]
                        if gram not in feature_dict[key]:
                            feature_dict[key][gram] = 0
                        feature_dict[key][gram] += 1

                    #for bigrams and trigrams we do the "conditional feature" dict
                    elif len(n_gram) > 1:
                        w = n_gram[-1]
                        #beginning to second-to-last word
                        n_minus_one_gram = n_gram[:-1] 

                        #dictionary here for fast lookup
                        if n_minus_one_gram not in feature_dict[key]: 
                            feature_dict[key][n_minus_one_gram] = {}
                        if w not in feature_dict[key][n_minus_one_gram]:
                            feature_dict[key][n_minus_one_gram] = 0
                        feature_dict[key][n_minus_one_gram][w] += 1

        return feature_dict

    #smoothes counts
    #returns the updated count
    def good_turing(self, feature_dict, n, cutoff):
        '''
        we don't worry about 0 counts because of katz backoff
        '''
        smoothed_feature_dict = deepcopy(feature_dict)
        freq_of_freqs = {}

        #for unigrams
        if n == 1:
            for s in self.sentiments:
                for k, v in feature_dict[s].items():
                    if v not in freq_of_freqs:
                        freq_of_freqs[v] = 0
                    freq_of_freqs[v] += 1

            #smooth feature dict
            for s in self.sentiments:
                for f, v in feature_dict[s].items():
                    smoothed_feature_dict[s][f] = self.gt_counts(v, cutoff, freq_of_freqs)

        #for bigrams/trigrams
        elif n > 1:
            for s in self.sentiments:
                for k1 in feature_dict[s]:
                    for k2, v in feature_dict[s][k1].items():
                        if v not in freq_of_freqs:
                            freq_of_freqs[v] = 0
                        freq_of_freqs[v] += 1

            #smooth feature dict
            for s in self.sentiments:
                for f1 in feature_dict[s]:
                    for f2, v in feature_dict[s][f1].items():
                        smoothed_feature_dict[s][f1][f2] = self.gt_counts(v, cutoff, freq_of_freqs)

        return smoothed_feature_dict

    @staticmethod
    def gt_counts(c, k, ffd): 
        '''
        c: MLE count
        k: cutoff
        ffd: freq of freq dict
        returns smoothed count
        '''
        if c > k:
            return c
        else:
            #c* equaiton from page 103
            return ( ( (c+1)*(ffd[c+1]/ffd[c]) ) - ( c*( (k + 1)*ffd[k+1] )/ffd[1] ) )/(1 - ( (k+1)*(ffd[k+1])/ffd[1] ) )

    def katz_backoff(self, ):
        