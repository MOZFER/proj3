def chi_squarify(totaltokens, totalfeatures,smoother):
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
		print b
		chistatistic = b + chistatistic
	return chistatistic

def chi_square_filter(fetures, smoother, cutoff):
	returnedfeatures = []
	for x in features:
		y = chisquarify

if __name__ == '__main__': 
	a = (2,4,12)
	b = (16,18,20)
	print chi_squarify(b,a,1)


