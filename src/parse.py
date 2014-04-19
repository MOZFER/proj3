import re
import pickle
from itertools import islice

'''
data structure:

    (review features)
    emission_dict = {
        'review sentiment': {
            'feature1' : count, 'feature2': count
        }
    }

    (trans matrix)
    A = {
        (<s>, pos): count, (<s>, neg): count, ...
    }
'''

path = "/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj3/training_data.txt"

lower_first = lambda s: s[:1].lower() + s[1:] if s else ''

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


#returns the HMM transition matrix 
#returns counts for words conditional on sentiment
def parse_text(path, n = 1):

    A = {} #hmm transitions
    emission_dict = {'pos': {}, 'neg': {}, 'neu': {}}

    with open(path, 'rb') as f:
        text = f.read()

    text = text.decode("utf-8")

    t2 = re.split(r'\n\n', text)
    t2.pop()

    for review in t2:
        r = review.split('\n')
        review_type = r.pop(0)
        category, label, number = review_type.split("_")

        #begin and end review tokens
        #these are transitions states in the HMM
        r = ["<r>\t"] + r + ["</r>\t"]

        prev = r.pop(0).split('\t')[0]

        #only contains sentences now
        for line in r:
            sentiment, sentence = line.split('\t')

            if (prev, sentiment) not in A:
                A[(prev, sentiment)] = 0
            A[(prev, sentiment)] += 1
            prev = sentiment

            #make first word of every sentence lowercase
            #sentence = lower_first(sentence)

            #make all lower: 
            sentence = sentence.lower()

            #puts whitespace around everything except words and whitespace
            sentence = re.sub(r'[^\w\s]', ' \g<0> ', sentence)

            #for ngrams n > 1, add n-1 start tokens and an end token
            if n > 1:
                sentence = sentence + ["</s>"]
                for x in range(n-1):
                    sentence = ["<s>"] + sentence

            sentence = re.split(r' +', sentence)

            for gram_tuple in window(sentence, n):
                if sentiment != "</r>":
                    if gram_tuple not in emission_dict[sentiment]:
                        emission_dict[sentiment][gram_tuple] = 0
                    emission_dict[sentiment][gram_tuple] += 1

    return A, emission_dict