import pickle
import sys
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

n_features = 32768
n_train_samples = 1000
n_trees = 120
stored_clf_file = '../../data/RandomForest.nf_%d.ns_%d.nt_%d.ngram23.clf' % (n_features, n_train_samples, n_trees)
print "Load Serialized classifier from %s" % (stored_clf_file)
clf = None
with open(stored_clf_file, "rb") as clf_f:
	clf = pickle.load(clf_f)


with open("../../data/test.tsv", "r") as f:
	with open('../../data/submission1.csv', 'w') as csvfile:
		hv = HashingVectorizer(n_features=n_features, non_negative=True, ngram_range=(2,3))

		csv_writer = csv.writer(csvfile, delimiter=',')
		csv_writer.writerow(['PhraseId','Sentiment'])
		i = 0
		for line in f:
			i += 1
			if i == 1:
				continue

			tokens = line.split('\t')

			topredict = hv.transform([tokens[2]])
			y_predicted = clf.predict(topredict.toarray())

			csv_writer.writerow([tokens[0],int(y_predicted[0])])

			sys.stdout.write("Progress: %d row   \r" % (i))
			sys.stdout.flush()

			i += 1


# with open('../../data/submission1.csv', 'w') as csvfile:
# 	hv = HashingVectorizer(n_features=n_features, non_negative=True, ngram_range=(2,3))
# 	topredict = hv.transform(test_corpus)

# 	y_predicted = clf.predict(topredict.toarray())

# 	csv_writer = csv.writer(csvfile, delimiter=' ',
#                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
# 	csv_writer.writerow(['PhraseId','Sentiment'])

# 	for i in range(len(test_data)):

# 		csv_writer.writerow([test_data[i],y_predicted[i]])

# 		sys.stdout.write("Gen Progress: %d row   \r" % (i))
# 		sys.stdout.flush()
