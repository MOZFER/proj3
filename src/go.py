from classifier import *
import random

path = "/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj3/data/training_data.txt"

def get_reviews(path):
    with open(path, 'rb') as f:
        text = f.read()
    text = text.decode("utf-8")
    text = re.split(r'\n\n', text)
    text.pop()
    return text

def k_folds(reviews, k):
    folds = []
    l = len(reviews)
    chunk = l // k
    random.shuffle(reviews)
    for i in range(0, l, chunk):
        folds.append(reviews[i:i+chunk])
    #if uneven division
    if len(folds[-1]) < chunk:
        extra = folds.pop()
        for j in range(len(extra)):
            folds[j].append(extra[j])
    return folds

reviews = get_reviews(path)
folds = k_folds(reviews, 10)

#k fold valid!
for f in range(len(folds)):
    f1 = folds[:f] + folds[(f+1):]
    #flatten
    f1 = [item for sublist in f1 for item in sublist]
    #make classifier
    c = Classifier(f1, x = 10, c = 5)
    f2 = folds[f]
    print(c.correct_share(f2))