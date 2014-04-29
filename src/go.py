from classifier import *
import random

train_path = "/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj3/data/training_data.txt"
test_path = "/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj3/data/test_data_no_true_labels.txt"
output_path = "/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj3/results/output.csv"

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

train_reviews = get_reviews(train_path)
test_reviews  = get_reviews(test_path)

folds = k_folds(train_reviews, 10)


##test results##

c = Classifier(train_reviews, n = 2, stat="log", smoothing="gt", x = 0.0001, l = 1.1, z = 15000)

c.print_test(test_reviews, output_path)

##experimental results##


for n_gram in [1]:
    for smoothing in [1,0.1,0.001]:
        for gt_cutoff in [3,5]:
            for chi_cutoff in [1,2,5,8]:
                for prior_weight in [1,2,5,8]:
                    s = 0
                    for f in range(len(folds)):
                        f1 = folds[:f] + folds[(f+1):]
                        #flatten
                        f1 = [item for sublist in f1 for item in sublist]
                        #make classifier
                        c = Classifier(f1, n = n_gram, x = smoothing, k = gt_cutoff, c = chi_cutoff, q = prior_weight, stat="chi")
                        f2 = folds[f]
                        s += c.correct_share(f2)
                    print (n_gram,smoothing,gt_cutoff,chi_cutoff,prior_weight,s/len(folds),"chi")

for n_gram in [1,2]:
    for smoothing in [0.1,0.001,0.0001]:
        for gt_cutoff in [3,5]:
            for log_cutoff in [.5, 1.0, 1.1, 1.2, 1.5, 2]:
                for log_2_cutoff in [2, 3.5, 5]:
                    for prior_weight in [1, 4, 8]:
                        s = 0
                        for f in range(len(folds)):
                            f1 = folds[:f] + folds[(f+1):]
                            #flatten
                            f1 = [item for sublist in f1 for item in sublist]
                            #make classifier
                            c = Classifier(f1, n = n_gram, x = smoothing, k = gt_cutoff, l = log_cutoff, l2 = log_2_cutoff, q = prior_weight, stat="log")
                            f2 = folds[f]
                            s += c.correct_share(f2)
                        print (n_gram,smoothing,gt_cutoff,log_cutoff,log_2_cutoff,prior_weight,s/len(folds),"log")

s = 0

for f in range(len(folds)):
    f1 = folds[:f] + folds[(f+1):]
    #flatten
    f1 = [item for sublist in f1 for item in sublist]
    #make classifier
    c = Classifier(f1, n = 1, stat="log", c = 8, l = 1.0, l2 = 1.5, smoothing="gt", k = 3, x = 0.0001)
    f2 = folds[f]
    s += c.correct_share(f2)

print(s/len(folds))

c.print_test(test_reviews, output_path)

