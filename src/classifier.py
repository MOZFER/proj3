import numpy as np
import pandas as pd
import re, math
from itertools import islice
from copy import deepcopy
from scipy import stats

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


def chi_squarify(totaltokens, totalfeatures, smoother = 0):
    '''totaltokens should be a 3-tuple of the count of all positive,neutral, 
    and negative words, totalfeatures are total counts of the feature occurring in pos,neu,neg'''
    predictedfrequencies = []
    observedfrequencies = []
    n = totaltokens[0] + totaltokens[1] + totaltokens [2] + 6*smoother
    features = totalfeatures[0] + totalfeatures[1] + totalfeatures[2] + 3*smoother
    nonfeatures = n - features
    positive = totaltokens[0] + totalfeatures[0] +2 * smoother
    neutral = totaltokens[1] + totalfeatures[1] +2* smoother
    negative = totaltokens[2] + totalfeatures[2] + 2*smoother
    predictedfrequencies.append(float(positive*features/n))
    predictedfrequencies.append(float(neutral*features/n))
    predictedfrequencies.append(float(negative*features/n))
    predictedfrequencies.append(float(positive*nonfeatures/n))
    predictedfrequencies.append(float(neutral*nonfeatures/n))
    predictedfrequencies.append(float(negative*nonfeatures/n))
    observedfrequencies.append(totalfeatures[0] + smoother)
    observedfrequencies.append(totalfeatures[1] + smoother)
    observedfrequencies.append(totalfeatures[2] + smoother)
    observedfrequencies.append(totaltokens[0] - totalfeatures[0] + smoother)
    observedfrequencies.append(totaltokens[1] - totalfeatures[1] + smoother)
    observedfrequencies.append(totaltokens[2] - totalfeatures[2] + smoother)
    chistatistic = 0  
    for x in range (0,6):
        a = predictedfrequencies[x] - observedfrequencies[x]
        b = a*a/predictedfrequencies[x]
        chistatistic = b + chistatistic
    return chistatistic


#def chi_square_filter(fetures, smoother, cutoff):
#    returnedfeatures = []
#    for x in features:
#        y = chisquarify


class Classifier:
    '''
    n is the length of n-grams to use, should be set to 1 or 2
    k is the good-turing cutoff (i.e. we stop smoothing when count is > k)
    x is the smoothing parameter for the transition matrix
    c is the chi-square stat
    t is the top # of features that should be used
    l is the log-odds ratio for admissibility
    '''
    def __init__(self, std_format_text, n = 1, k = 5, q = 1 ,x = 0.0001, c = 5, t = 3, l = 1.3, l2 = 4, z = 1, stat = "log", smoothing="gt"): #n can equal 1,2,3

        self.n = n
        self.c = c
        self.t = t
        self.l = l
        self.z = z
        self.l2 = l2
        self.q = q
        self.smoothing = smoothing
        self.c_0 = 0
        self.c_2_0 = 0
        self.priors = {
            "pos": {"count": 0, "pos": 0, "neg": 0, "neu": 0}, 
            "neg": {"count": 0, "pos": 0, "neg": 0, "neu": 0}, 
            "neu": {"count": 0, "pos": 0, "neg": 0, "neu": 0}
        } #prob of sentence being a sentiment given the review sentiment

        self.sentiments = ["neg", "neu", "pos"]

        #no sense treating these as unknown
        self.seen_words = set(["!", ".", ",", "<s>", "</s>", ":", ";", "'", "\"", "/", "\\"]) 
        self.unk = "<unk>"

        self.A, self.A_sentiments, self.sentences_by_sentiment = self.parse_text(std_format_text)

        #smooth trans matricies
        if x > 0:
            self.A = self.smooth_transitions(self.A, x)
            for sentiment in self.sentiments:
                self.A_sentiments[sentiment] = self.smooth_transitions(self.A_sentiments[sentiment], x)

        if n >= 1:
            self.unigrams          = self.make_features(self.sentences_by_sentiment, 1)
            self.unigram_counts    = self.sum_counts(self.unigrams, 1)
            self.gt_unigrams       = self.good_turing(self.unigrams, 1, k)
            self.gt_unigram_counts = self.sum_counts(self.gt_unigrams, 1)
            self.l_unigrams, self.l_unigram_default = self.laplace(self.unigrams, 1)
            self.l_unigram_counts    = self.sum_counts(self.l_unigrams, 1)
        if n >= 2:
            self.bigrams           = self.make_features(self.sentences_by_sentiment, 2)
            self.bigram_counts     = self.sum_counts(self.bigrams, 2)
            self.gt_bigrams        = self.good_turing(self.bigrams,  2, k)
            self.l_bigrams, self.bigram_default = self.laplace(self.bigrams, 2)
            self.l_bigram_counts    = self.sum_counts(self.l_bigrams, 2)
            self.seen_bigrams      = self.get_all_features(self.gt_bigrams)
            self.gt_bigram_counts  = self.sum_counts(self.gt_bigrams, 2)
        if n >= 3:
            self.trigrams          = self.make_features(self.sentences_by_sentiment, 3)
            self.trigram_counts    = self.sum_counts(self.trigrams, 3)
            self.gt_trigrams       = self.good_turing(self.trigrams, 3, k)
            self.l_trigrams, self.trigram_default = self.laplace(self.trigrams, 3)
            self.l_trigram_counts    = self.sum_counts(self.l_trigrams, 3)
            self.seen_trigrams     = self.get_all_features(self.gt_trigrams)
            self.gt_trigram_counts = self.sum_counts(self.gt_trigrams, 3)

        #this is a set of admissible features
        #provides a quick check whether a feature is discriminative enough
        self.admissible_features = set()        
        self.confidence = {} #records confidence scores for features

        if smoothing == "gt":
            self.gt_admissible(stat)
        elif smoothing == "lap":
            self.l_admissible(stat)


    def parse_text(self, std_format_text):
        '''
        converts text to: {"sentiment": [sentence1, sentence2, ...]}
        replaces the first instance of every word with <unk>
        also gets transition matrix between states for sentences
        '''

        sentence_sentiment_dict = {s: [] for s in self.sentiments}

        states = ["<r>", "neg", "neu", "pos", "</r>"]
        A = pd.DataFrame(np.zeros((5,5)), index = states, columns = states)
        A.loc["</r>", "</r>"] = 1 #just to prevent NAs
        A_sentiments = {"pos": deepcopy(A), "neg": deepcopy(A), "neu": deepcopy(A)}

        for review in std_format_text:

            review_sentiment, r = self.clean_review(review)

            prev_sentiment = r.pop(0).split('\t')[0] #starting sentiment, <r>

            for line in r:

                sentiment, sentence = self.tokenize_sentence(line, self.n)

                #for unk
                for word in range(len(sentence)):
                    if sentence[word] not in self.seen_words:
                        self.seen_words.add(sentence[word]) #add to seen words
                        sentence[word] = self.unk #overwrite

                if sentiment not in ["<r>", "</r>"]:
                    sentence_sentiment_dict[sentiment].append(sentence)
                    self.priors[review_sentiment]["count"] += 1
                    self.priors[review_sentiment][sentiment] += 1

                #increments the transition count
                A.loc[prev_sentiment, sentiment] += 1
                A_sentiments[review_sentiment].loc[prev_sentiment, sentiment] += 1

                prev_sentiment = sentiment

        try:
            #sometimes '' gets added
            self.seen_words.remove('')
        except:
            pass

        #normalizes the A matrix to be a probability
        s = A.sum(axis=1) #sums over rows

        for item in s.index:
            A.loc[item,:] = A.loc[item,:]/s.loc[item] #divide row by row sum

        for sentiment in A_sentiments:
            s_temp = A_sentiments[sentiment].sum(axis=1)
            for item in s_temp.index:
                A_sentiments[sentiment].loc[item,:] = A_sentiments[sentiment].loc[item,:]/s_temp.loc[item]

        for r_s in self.priors:
            for s in self.sentiments:
                self.priors[r_s][s] = self.priors[r_s][s] / self.priors[r_s]["count"]

        return A, A_sentiments, sentence_sentiment_dict

    @staticmethod
    def clean_review(review):
        '''
        turns a review-as-string into a list of string sentences
        '''
        r = review.split("\n")
        review_type = r.pop(0)
        category, review_sentiment, number = review_type.split("_")

        #add begin and end review tokens
        r = ["<r>\t"] + r + ["</r>\t"]

        return review_sentiment, r

    #tokenizes sentence
    @staticmethod
    def tokenize_sentence(line, n):
        '''
        takes a sentence-as-string and returns a list of tokens
        '''
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

    def sum_counts(self, count_dict, n):
        '''
        given a dictionary: {'sentiment': {'feature': count, }, }
        returns the total feature counts for each sentiment: {'sentiment': sum_features}
        '''
        sum_dict = {"pos": 0, "neg":0, "neu": 0}

        if self.smoothing == "lap":
            sum_dict = {"pos":  self.z * (len(self.seen_words)**n), "neg": self.z * (len(self.seen_words)**n), "neu": self.z * (len(self.seen_words)**n)}

        if n == 1:
            for s in self.sentiments:
                sum_dict[s] = sum([v for k, v in count_dict[s].items()])
        elif n == 2:
            for s in self.sentiments:
                for word1 in count_dict[s]:
                    sum_dict[s] += sum([v for k, v in count_dict[s][word1].items()])
        return sum_dict

    def make_features(self, parsed_text, z):
        '''
        given a dictionary with sentences keyed by sentiment,
            with tokenized sentences:
            {"sentiment": [[w, w, w, ...], [w, w, w, ...], ...], }


        gets bigramsif z = 1, bigrams if z = 2, etc.

        returns a count of unigrams or bigrams
        uses nested dictionaries, so a bigram output looks like:
        {'sentiment': {'w_n_minus_one': {'w_n': count}, }, }
        '''
        feature_dict = {"pos": {}, "neg": {}, "neu": {}}

        for sentiment in parsed_text:
            for sentence in parsed_text[sentiment]:
                for n_gram in window(sentence, z):
                    if len(n_gram) == 1:
                        #need n_gram[0] because it's a tuple
                        #just pull out the underlying string
                        gram = n_gram[0]
                        if gram not in feature_dict[sentiment]:
                            feature_dict[sentiment][gram] = 0
                        feature_dict[sentiment][gram] += 1

                    #for bigrams and trigrams we do the "conditional feature" dict
                    elif len(n_gram) == 2:
                        w = n_gram[-1]
                        #beginning to second-to-last word
                        n_minus_one_gram = n_gram[:-1][0]

                        #dictionary here for fast lookup
                        if n_minus_one_gram not in feature_dict[sentiment]: 
                            feature_dict[sentiment][n_minus_one_gram] = {}
                        if w not in feature_dict[sentiment][n_minus_one_gram]:
                            feature_dict[sentiment][n_minus_one_gram][w] = 0
                        feature_dict[sentiment][n_minus_one_gram][w] += 1

        return feature_dict

    def get_all_features(self, n_gram_dict):
        '''
        given a nested dict of ngrams
        pulls out all ngrams and returns them
        '''
        n_grams = set()

        for s in self.sentiments:
            for n_minus_one_gram in n_gram_dict[s]:
                for w in n_gram_dict[s][n_minus_one_gram]:
                    n_grams.add((n_minus_one_gram,) + (w,))

        return n_grams

    def smooth_transitions(self, trans_matrix, x):
        '''makes the transitions less biased towards staying in same state
        does something very similar to laplace smoothing
        with large x, transitions tend towards parity'''
        mat = deepcopy(trans_matrix)

        mat = mat + 0.000001

        for s1 in self.sentiments + ["<r>"]: #go down rows
            #go across cols
            total_prob = sum([mat.loc[s1, z] for z in self.sentiments])
            q = total_prob

            for s2 in self.sentiments:
                m = mat.loc[s1, s2]
                mat.loc[s1, s2] = total_prob*((m + (1.)*x)/(q + (3.)*x))

        return mat

    def good_turing(self, feature_dict, n, cutoff):
        '''
        implements good-turing smoothing

        in the case of bigrams & trigrams, does not add a probability for events seen zero times
        this is because katz backoff does not require this

        adds a probability for zero events in the unigram case

        treats all conditional features, (f|sentiment), as part of the same corpus
        an alternative approach is to treat each sentiment as a separate corpus
        does not change results substantially
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

            #smooth using the freq_of_freqs
            for s in self.sentiments:
                for f, v in feature_dict[s].items():
                    smoothed_feature_dict[s][f] = self.gt_counts(v, cutoff, freq_of_freqs)

                for word in self.seen_words:
                    if word not in smoothed_feature_dict[s]:
                        smoothed_feature_dict[s][word] = freq_of_freqs[1]/sum([v for k,v in freq_of_freqs.items()])

            self.c_0 = freq_of_freqs[1]/sum([v for k,v in freq_of_freqs.items()])

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

    def laplace(self, feature_dict, n):
        '''
        smoothes count
        '''

        smoothed_feature_dict = deepcopy(feature_dict)

        default_count = 0

        if n == 1:
            for s in self.sentiments:
                for k, v in smoothed_feature_dict[s].items():
                    smoothed_feature_dict[s][k] = feature_dict[s][k] + self.z * (1/len(self.seen_words))

            default_count = self.z * 1/len(self.seen_words)

        if n > 1:
            for s in self.sentiments:
                for f1 in smoothed_feature_dict[s]:
                    for f2 in smoothed_feature_dict[s][f1]:
                        smoothed_feature_dict[s][f1][f2] = feature_dict[s][f1][f2] + self.z * (1/len(self.seen_words)**n)

            default_count = self.z * (1/len(self.seen_words))**n

        return smoothed_feature_dict, default_count

    #do this just for unigrams
    def gt_admissible(self, stat):
        '''
        implements an admissibility criterion

        all features with more "certainty" than the criterion are added to admissible_features
        '''
        if stat == "chi": 
            if self.n >= 1:
                for word in self.seen_words:
                    word_count = [self.gt_unigrams[x].get(word, 0) for x in self.sentiments]
                    sentiment_count = [self.gt_unigram_counts[x] for x in self.sentiments]

                    chi_sqr = chi_squarify(sentiment_count, word_count)

                    if chi_sqr > self.c:
                        self.admissible_features.add(word)

                    self.confidence[word] = chi_sqr                

        elif stat == "log":
            if self.n >= 1:
                for word in self.seen_words:
                    log_odds = []

                    for s in self.sentiments:
                        num = self.gt_unigrams[s][word]/self.gt_unigram_counts[s]
                        denom = sum([self.gt_unigrams[x][word]/self.gt_unigram_counts[x] for x in self.sentiments if x != s])
                        log_odds.append(num/denom)

                    log_odds = math.log(max(log_odds), 2)

                    if log_odds > self.l:
                        self.admissible_features.add(word)

                    self.confidence[word] = log_odds

            if self.n >= 2:
                for bigram in self.seen_bigrams:

                    prob = {"pos": 0, "neg": 0, "neu": 0}

                    log_odds = []

                    for s in self.sentiments:
                        if bigram[0] in self.gt_bigrams[s]:
                            prob[s] = self.gt_bigrams[s][bigram[0]].get(bigram[1], self.c_0)/sum(self.gt_bigrams[s][bigram[0]].values())
                        else:
                            prob[s] = self.c_0

                    for s in self.sentiments:
                        log_odds.append(prob[s]/sum([prob[x] for x in self.sentiments if x != s]))

                    log_odds = math.log(max(log_odds), 2)

                    if log_odds > self.l2:
                        self.admissible_features.add(bigram)

                    self.confidence[bigram] = log_odds

    #do this just for unigrams
    def l_admissible(self, stat):
        '''
        implements an admissibility criterion

        all features with more "certainty" than the criterion are added to admissible_features
        '''
        if stat == "chi": 
            if self.n >= 1:
                for word in self.seen_words:
                    word_count = [self.l_unigrams[x].get(word, 0) for x in self.sentiments]
                    sentiment_count = [self.l_unigram_counts[x] for x in self.sentiments]

                    chi_sqr = chi_squarify(sentiment_count, word_count)

                    if chi_sqr > self.c:
                        self.admissible_features.add(word)

                    self.confidence[word] = chi_sqr                

        elif stat == "log":
            if self.n >= 1:
                for word in self.seen_words:
                    log_odds = []

                    for s in self.sentiments:
                        num = self.l_unigrams[s].get(word, self.l_unigram_default)/self.l_unigram_counts[s]
                        denom = sum([self.l_unigrams[x].get(word, self.l_unigram_default)/self.l_unigram_counts[x] for x in self.sentiments if x != s])
                        log_odds.append(num/denom)

                    log_odds = math.log(max(log_odds), 2)

                    if log_odds > self.l:
                        self.admissible_features.add(word)

                    self.confidence[word] = log_odds

            if self.n >= 2:
                for bigram in self.seen_bigrams:

                    prob = {"pos": 0, "neg": 0, "neu": 0}

                    log_odds = []

                    for s in self.sentiments:
                        if bigram[0] in self.l_bigrams[s]:
                            prob[s] = self.l_bigrams[s][bigram[0]].get(bigram[1], self.c_0)/sum(self.l_bigrams[s][bigram[0]].values())
                        else:
                            prob[s] = self.c_0

                    for s in self.sentiments:
                        log_odds.append(prob[s]/sum([prob[x] for x in self.sentiments if x != s]))

                    log_odds = math.log(max(log_odds), 2)

                    if log_odds > self.l2:
                        self.admissible_features.add(bigram)

                    self.confidence[bigram] = log_odds

    def return_prob(self, sentiment, words):
        '''
        returns the sum of logged probabilities for a sentence in the unigram case
        '''
        
        log_prob_sum = 0

        if self.n >= 1:
            for word in words:
                if word in self.admissible_features:
                    if self.smoothing == "gt":
                        log_prob_sum += math.log(self.gt_unigrams[sentiment][word]/self.gt_unigram_counts[sentiment], 2)
                    elif self.smoothing == "lap":
                        log_prob_sum += math.log(self.l_unigrams[sentiment].get(word, self.l_unigram_default)/self.l_unigram_counts[sentiment], 2)
                else:
                    continue
                    #log_prob_sum += math.log(self.gt_unigrams[sentiment]['<unk>']/self.gt_unigram_counts[sentiment], 2)

        if self.n >= 2:
            for bigram in window(words, 2):
                if bigram in self.admissible_features:
                    try:
                        log_prob_sum += math.log(self.gt_bigrams[sentiment][bigram[0]][bigram[1]]/sum(self.gt_bigrams[sentiment][bigram[0]].values()), 2)
                    except KeyError:
                        log_prob_sum += self.c_0
                else:
                    continue

        if log_prob_sum == 0:
            return self.c_0
        else:
            return log_prob_sum

    def return_prob_modified(self, sentiment, words):
        if self.n >= 1:
            log_prob_sum = 0

            candidate = []
            for word in words:
                candidate.append((self.confidence.get(word, -10000000), word))

            candidate.sort(reverse=True)

            for f in range(min((self.t), len(words))):
                if candidate[f][0] != None:
                    #if word's not seen, just give it c_0
                    log_prob_sum += math.log(self.gt_unigrams[sentiment].get(candidate[f][1],self.c_0)/self.gt_unigram_counts[sentiment], 2)

            return log_prob_sum


    def katz_backoff_prob(self, n_gram, sentiment):
        '''
        returns the sum of logged probabilities for katz backoff

        likely still has bugs
        '''
        if len(n_gram) == 2:
            #if we've seen the bigram
            w1, w2 = n_gram

            if w1 in self.bigrams[sentiment] and w2 in self.bigrams[sentiment][w1]:
                #sum of unsmoothed occurances of prev word
                if w1 in self.admissible_features or w2 in self.admissible_features:
                    s = sum([v for k, v in self.bigrams[sentiment][w1].items()])
                    return math.log(self.gt_bigrams[sentiment][w1][w2] / s, 2)
                else:
                    return self.c_0
            elif w1 in self.bigrams[sentiment]:
                #intuitively: if w1 is not in bigrams, then default to unigram with full prob
                #if w1 is in bigrams but w2 is not in bigrams[w1], weighted unigram

                beta_complement = 0

                s = sum([v for k, v in self.bigrams[sentiment][w1].items()])

                for word in self.gt_bigrams[sentiment][w1]:
                    beta_complement += self.gt_bigrams[sentiment][w1][word]/s

                #denom is 1 since we're at the last step, so alpha = beta
                alpha = 1 - beta_complement

                if w1 in self.admissible_features or w2 in self.admissible_features:
                    return math.log(alpha*self.gt_unigrams[sentiment][w2], 2)
                else:
                    return self.c_0

            elif w1 not in self.bigrams[sentiment]:
                if w2 in self.gt_unigrams[sentiment] and w2 in self.admissible_features:
                    return math.log(self.gt_unigrams[sentiment][w2], 2)
                else:
                    return self.c_0
            else:
                return self.c_0
        else:
            return math.log(self.gt_unigrams[sentiment][n_gram], 2)


    def viterbi(self, review):
        '''
        implement the viterbi algorithm 

        see ch6 of book

        ground_truth records correct sentence sentiment
        '''
        review_sentiment, r = self.clean_review(review)
        r.pop() #removes </r> token
        r.pop(0) #removes <r> token

        ground_truth = [self.tokenize_sentence(x, self.n)[0] for x in r]

        sentences = [self.tokenize_sentence(x, self.n)[1] for x in r]

        for sentence in sentences:
            for word in range(len(sentence)):
                if sentence[word] not in self.seen_words:
                    sentence[word] = self.unk

        num_sentences = len(sentences)

        #this is the probabiliy of being in a state at time t
        viterbi_prob = pd.DataFrame(np.zeros([len(self.A.index), num_sentences]), index= self.A.index)

        #this is the most probable previous state
        backpointer = pd.DataFrame(np.zeros([len(self.A.index), num_sentences]), index = self.A.index)

        #initialize
        for sentiment in self.sentiments:
            b = sentences[0]
            viterbi_prob.loc[sentiment, 0] = \
            math.log(self.A_sentiments[review_sentiment].loc["<r>", sentiment], 2) +\
            self.q*math.log(self.priors[review_sentiment][sentiment], 2) + \
            self.return_prob(sentiment, b) # + \
            #math.log(self.priors[review_sentiment][sentiment], 2)

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
                    math.log(self.A_sentiments[review_sentiment].loc[s_prime, s], 2) + \
                    self.q* math.log(self.priors[review_sentiment][sentiment], 2) + \
                    self.return_prob(s, sentences[t]), s_prime) \
                    for s_prime in self.sentiments ] )

                    #math.log(self.priors[review_sentiment][sentiment], 2), s_prime) \
                    #for s_prime in self.sentiments ] )

        #end step
        viterbi_prob.loc["</r>", num_sentences - 1], backpointer.loc["</r>", num_sentences - 1] = \
            max( [ ( viterbi_prob.loc[s, num_sentences - 1] + \
            math.log(self.A_sentiments[review_sentiment].loc[s, "</r>"], 2), s) for s in self.sentiments])

        sequence = [ backpointer.loc["</r>", num_sentences-1] ]
        row_lookup = backpointer.loc["</r>", num_sentences-1]

        for col in range(num_sentences - 1, -1, -1):
            row_lookup = backpointer.loc[row_lookup, col]
            sequence.append(row_lookup)

        sequence.reverse()
        sequence.pop(0)

        return sequence, ground_truth

    def correct_share(self, reviews):
        '''
        compares output of viterbi for all reviews to the ground truth
        '''

        total = 0
        correct = 0

        for review in reviews:
            predicted, ground_truth = self.viterbi(review)
            for s in range(len(ground_truth)):
                total += 1
                if predicted[s] == ground_truth[s]:
                    correct += 1

        return correct/total

    def print_test(self, reviews, path):
        total = 0
        header = "id,answer\n"

        conversion = {"pos": "1", "neu": "0", "neg":"-1"}

        with open(path, "w") as f:
            f.write(header)
            for review in reviews:
                predicted, ground_truth = self.viterbi(review)
                for p in predicted:
                    y = ','.join([str(total), conversion[p] + "\n"])
                    f.write(y)
                    total += 1

        return "done"

