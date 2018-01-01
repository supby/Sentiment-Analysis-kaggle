import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics

dataset = None
n_features = 32768
n_train_samples = -1
# traindata_file = "../../data/binary.HashingVectorizer.fc_%d.ngram23.custom_tknz.train" % (n_features)
traindata_file = "../../data/binary.CountVectorizer.ngram15.corplim10k.custom_tknz.train"
with open(traindata_file, "rb") as f:
	dataset = pickle.load(f)

# Split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset['train_X'][:n_train_samples] if n_train_samples > -1 else dataset['train_X'],
    dataset['train_Y'][:n_train_samples] if n_train_samples > -1 else dataset['train_Y'],
    test_size=0.3)

n_features = docs_train.shape[1]
print "Train features count: ", n_features, "; Train dataset size: ", docs_train.shape[0]

n_trees = 30
clf = RandomForestClassifier(n_estimators=n_trees, min_samples_split=2, verbose=4)

print "Train RF classifier"
clf = clf.fit(docs_train.toarray(), y_train)

stored_clf_file = '../../data/CountVectorizer.nf_%d.ns_%d.nt_%d.ngram15.custom_tknz.clf' % (n_features, n_train_samples, n_trees)
print "Serialize classifier to %s" % (stored_clf_file)
with open(stored_clf_file, "wb") as clf_f:
	pickle.dump(clf, clf_f)

print "Test classifier"
print "Test features count: ", docs_test.shape[1], "; Test dataset size: ", docs_test.shape[0]
y_predicted = clf.predict(docs_test.toarray())

# Print the classification report
print(metrics.classification_report(y_test, y_predicted))