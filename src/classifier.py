import numpy as np
import pandas as pd
import re, math
from itertools import islice
from copy import deepcopy

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
    '''
    n is the length of n-grams to use, should be set to 1 or 2
    k is the good-turing cutoff (i.e. we stop smoothing when count is > k)
    x is the smoothing parameter for the transition matrix
    '''
    def __init__(self, std_format_text, n = 1, k = 5, x = 0): #n can equal 1,2,3

        self.n = n

        self.sentiments = ["neg", "neu", "pos"]

        #no sense treating these as unknown
        self.seen_words = set(["!", ".", ",", "<s>", "</s>", ":", ";", "'", "\"", "/", "\\"]) 
        self.unk = "<unk>"

        self.A, self.sentences_by_sentiment = self.parse_text(std_format_text, n)

        #makes the transitions less biased towards staying in same state
        #we don't change the x --> </r> probabilities
        if x > 0:  
            for s1 in self.sentiments + ["<r>"]: #go down rows
                #go across cols
                total_prob = sum([self.A.loc[s1, x] for x in self.sentiments])

                for s2 in self.sentiments:
                    self.A.loc[s1, s2] = total_prob*(self.A.loc[s1, s2] + .1*x)/(total_prob + .3*x)

        self.vocab = self.make_vocab(self.sentences_by_sentiment)

        if 1 <= n:
            self.unigrams          = self.make_features(self.sentences_by_sentiment, 1)
            self.unigram_counts    = self.sum_counts(self.unigrams)
            self.gt_unigrams       = self.good_turing(self.unigrams, 1, k)
            self.gt_unigram_counts = self.sum_counts(self.gt_unigrams)
        if 2 <= n:
            self.bigrams           = self.make_features(self.sentences_by_sentiment, 2)
            self.bigram_counts     = self.sum_counts(self.bigrams)
            self.gt_bigrams        = self.good_turing(self.bigrams,  2, k)
            self.gt_bigram_counts  = self.sum_counts(self.gt_bigrams)
        if 3 <= n:
            self.trigrams          = self.make_features(self.sentences_by_sentiment, 3)
            self.trigram_counts    = self.sum_counts(self.trigrams)
            self.gt_trigrams       = self.good_turing(self.trigrams, 3, k)
            self.gt_trigram_counts = self.sum_counts(self.gt_trigrams)

        self.admissible_features = set()






    #converts text to dictionary keyed by sentiment
    #handles unknown words
    #with list of sentences as the values
    #also gets transition matrix
    def parse_text(self, std_format_text, n):

        sentence_sentiment_dict = {s: [] for s in self.sentiments}

        states = ["<r>", "neg", "neu", "pos", "</r>"]
        A = pd.DataFrame(np.zeros((5,5)), index = states, columns = states)
        A.loc["</r>", "</r>"] = 1 #just to prevent NAs

        #with open(path, 'rb') as f:
        #    text = f.read()

        #text = text.decode("utf-8")

        #text = re.split(r'\n\n', text)
        #text.pop()

        for review in std_format_text:

            r = self.clean_review(review)

            prev_sentiment = r.pop(0).split('\t')[0] #starting sentiment, <r>

            for line in r:

                sentiment, sentence = self.tokenize_sentence(line, n)

                if sentiment not in ["<r>", "</r>"]:
                    sentence_sentiment_dict[sentiment].append(sentence)

                A.loc[prev_sentiment, sentiment] += 1
                prev_sentiment = sentiment

                #for unk
                for word in range(len(sentence)):
                    if sentence[word] not in self.seen_words:
                        self.seen_words.add(sentence[word]) #add to seen words
                        sentence[word] = self.unk #overwrite

        s = A.sum(axis=1)

        for item in s.index:
            A.loc[item,:] = A.loc[item,:]/s.loc[item] #divide row by row sum

        return A, sentence_sentiment_dict

    @staticmethod
    def clean_review(review):
        r = review.split("\n")
        review_type = r.pop(0)
        category, label, number = review_type.split("_")

        #add begin and end review tokens
        r = ["<r>\t"] + r + ["</r>\t"]

        return r

    #tokenizes sentence
    @staticmethod
    def tokenize_sentence(line, n):
        sentiment, sentence = line.split('\t')
        #make all lower: 
        sentence = sentence.lower()

        #puts whitespace around everything except words and whitespace
        sentence = re.sub(r'[^\w\s\']', ' \g<0> ', sentence)
        sentence = sentence.strip()

        #for ngrams n > 1, add n-1 start tokens and an end token
        if n > 1:
            sentence = sentence + " </s>"
            for x in range(n-1):
                sentence = "<s> " + sentence

        sentence = re.split(r' +', sentence)

        return sentiment, sentence

    #returns the vocab
    def make_vocab(self, parsed_text):
        vocab = set()

        for key in parsed_text:
            for sentence in parsed_text[key]:
                vocab | set(sentence)

        return vocab

    def sum_counts(self, count_dict):
        sum_dict = {}
        for s in self.sentiments:
            sum_dict[s] = sum([v for k, v in count_dict[s].items()])
        return sum_dict

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
                        n_minus_one_gram = n_gram[:-1][0]

                        #dictionary here for fast lookup
                        if n_minus_one_gram not in feature_dict[key]: 
                            feature_dict[key][n_minus_one_gram] = {}
                        if w not in feature_dict[key][n_minus_one_gram]:
                            feature_dict[key][n_minus_one_gram][w] = 0
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

            #treating these as different corpora here
            #smooth feature dict
            #for s in self.sentiments:
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
            #for s in self.sentiments:
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

    #give this a unigram, bigram, or trigram
    def katz_backoff(self, n_gram):
        if len(n_gram) == 2:
            #if we've seen the bigram
            w1, w2 = n_gram

            if w1 in self.bigrams and w2 in self.bigrams[w1]:
                #sum of unsmoothed occurances of prev word
                s = sum([v for k, v in self.bigrams[w1].items()])
                return self.gt_bigrams[w1][w2] / s
            else:
                beta_complement = 0
                s = sum([v for k, v in self.bigrams[w1].items()])

                for w2 in self.gt_bigrams[w1]:
                    beta_complement += self.gt_bigrams[w1][w2] / s

                #denom is 1 since we're at the last step, so alpha = beta
                alpha = 1 - beta_complement

                return alpha*self.gt_unigrams[w2]

    def return_prob(self, sentiment, words):
        if self.n == 1:
            log_prob_sum = 0
            for word in words:
                if word in self.gt_unigrams:
                    log_prob_sum += math.log(self.gt_unigrams[sentiment][word]/self.gt_unigram_counts[sentiment], 2)
                else:
                    continue
                    #log_prob_sum += math.log(1/3, 2)
            return log_prob_sum


    def viterbi(self, review):
        r = self.clean_review(review)
        r.pop()
        r.pop(0)

        ground_truth = [self.tokenize_sentence(x, self.n)[0] for x in r]

        sentences = [self.tokenize_sentence(x, self.n)[1] for x in r]
        num_sentences = len(sentences)

        #this is the probabiliy of being in a state at time t
        viterbi_prob = pd.DataFrame(np.zeros([len(self.A.index), num_sentences]), index= self.A.index)

        #this is the most probable 
        backpointer = pd.DataFrame(np.zeros([len(self.A.index), num_sentences]), index = self.A.index)

        #initialize
        for sentiment in self.sentiments:
            b = sentences[0]
            viterbi_prob.loc[sentiment, 0] = math.log(self.A.loc["<r>", sentiment], 2) + self.return_prob(sentiment, b)
            backpointer.loc[sentiment, 0] = 0.

        #intermedate steps (recursion)
        for t in range(1, num_sentences):
            for s in self.sentiments: #s_prime prev state; s current state
                #does them both in one shot
                #uses the fact that max does the max of the first element of a tuple
                #so we tack the state name as the second element of a tuple, (log-prob, state)
                viterbi_prob.loc[s,t], backpointer.loc[s,t] = \
                    max( \
                    [ ( viterbi_prob.loc[s_prime, t-1] + \
                    math.log(self.A.loc[s_prime, s], 2) + \
                    self.return_prob(s, sentences[t]), s_prime) \
                    for s_prime in self.sentiments ] )

        #end step
        viterbi_prob.loc["</r>", num_sentences - 1], backpointer.loc["</r>", num_sentences - 1] = \
            max( [ ( viterbi_prob.loc[s, num_sentences - 1] + \
            math.log(self.A.loc[s, "</r>"], 2), s) for s in self.sentiments])

        sequence = [ backpointer.loc["</r>", num_sentences-1] ]
        row_lookup = backpointer.loc["</r>", num_sentences-1]

        for col in range(num_sentences - 1, -1, -1):
            row_lookup = backpointer.loc[row_lookup, col]
            sequence.append(row_lookup)

        sequence.reverse()
        sequence.pop(0)

        return sequence, ground_truth

    def correct_share(self, reviews):

        total = 0
        correct = 0

        for review in reviews:
            predicted, ground_truth = self.viterbi(review)
            for s in range(len(predicted)):
                total += 1
                if predicted[s] == ground_truth[s]:
                    correct += 1

        return correct/total
