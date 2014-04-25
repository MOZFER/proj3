import re
import pickle
from itertools import islice

path = "training_data.txt"
reviews = "test_data_no_true_labels.txt"

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
    t2 = re.split(r'\n\n', text)
    t2.pop()

    for review in t2:
        r = review.split('\n')
        review_type = r.pop(0)
        category, label, number = review_type.split("_")
        r = ["<r>\t"] + r + ["</r>\t"]
        prev = r.pop(0).split('\t')[0]
        for line in r:
            sentiment, sentence = line.split('\t')
            if (prev, sentiment) not in A:
                A[(prev, sentiment)] = 0
            A[(prev, sentiment)] += 1
            prev = sentiment
            sentence = sentence.lower()
            sentence = re.sub(r'[^\w\s]', ' \g<0> ', sentence)
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

def parse_reviews(path):
    with open(path, 'rb') as f:
        text = f.read()
    #text = text.decode("utf-8")
    text2 = re.split(r'\n\n', text)
    reviews = []
    #for x in text2:
    #    print x
    #print len(text2)
    for line in text2:
        sentence = line.split('\n')
        for y in sentence[1:]:
            z = y.split('\t')
            newsent = z[1]
            #print newsent
            #newsentence = newsent[:-1]
            newsentence = newsent.lower()
            newsentence = re.sub(r'[^\w\s]', ' \g<0> ', newsentence)
            newsentence = re.split(r' +', newsentence)
            reviews.append(newsentence)
    #print len(reviews)
    return reviews


def baseline_bow(review,dictionary,n):
    'creates a bag of words estimate for each sentnece type using add-n smoothing'
    x = review
    pos = 1
    neu = 1
    neg = 1
    posdict = dictionary['pos']
    neudict = dictionary['neu']
    negdict = dictionary['neg']
    postotal = 0
    neutotal = 0
    negtotal = 0
    for a in posdict:
        postotal = postotal + posdict[a]
    for b in neudict:
        neutotal = neutotal + neudict[b]
    for c in negdict:
        negtotal = negtotal + negdict[c]
    #print posdict
    for y in x:
        z= (y,)
        try:
            posfreq = posdict[z]
        except: 
            posfreq = 0
        try:
            neufreq = neudict[z]
        except:
            neufreq = 0
        try: 
            negfreq = negdict[z]
        except:
            negfreq = 0
        posprob = ((posfreq)/float(postotal))+(n/float(postotal + neutotal + negtotal))
        neuprob = ((neufreq)/float(neutotal))+(n/float(postotal + neutotal + negtotal))
        negprob = ((negfreq)/float(negtotal))+(n/float(postotal + neutotal + negtotal))
        #print (posprob,neuprob,negprob)
        #posprob = (posfreq/float(posfreq+neufreq+negfreq))/(postotal + n)
        #neuprob = (neufreq/float(posfreq+neufreq+negfreq))/(neutotal + n)
        #negprob = (negfreq/float(posfreq+neufreq+negfreq))/(negtotal + n)
        pos = pos * posprob
        neu = neu * neuprob
        neg = neg * negprob
    if pos > neu and pos > neg:
        return 1
    if neg > neu:
        return -1
    else:
        return 0
    
    #return (pos,neu,neg)

if __name__ == '__main__': 
    #a = parse_reviews(path)
    #b = parse_reviews(reviews)
    #print len(a)
    #print len(b)
    '''with open(reviews, 'rb') as f:
        textreviews = f.readlines()
    textreviews2 = re.split(r'\n\n',textreviews)
    print len(textreviews2)
    print len(text)'''
    transitions,emissions = parse_text(path)
    test = parse_reviews(reviews)
    #train = parse_reviews(path)
    #print len(a)
    #print z[12]
    #print reviews[12]
    #print len(z)
    pos = 0
    neu = 0
    neg = 0
    #print a[2]
    print 'Id,answer'
    for q in range(0,1224): 
        t = baseline_bow(test[q],emissions,1)
        if t == 1:
            pos = pos +1
        if t == 0: 
            neu = neu + 1
        if t == -1:
            neg = neg + 1
        print str(q) + ',' + str(t)
    #print (pos,neu,neg)
    