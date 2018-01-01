'''

Serialize train data to binary format with help of cPickle.

stored object looks like: dataset and dataset.train_X, dataset.train_Y

'''

import sys
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from shared.custom_tokenizer import CustomTokenizer

n_features = 32768
corpus_limit = 10000
dataset = { 'train_X': [], 'train_Y': [] }
serilize2file = "../data/binary.CountVectorizer.Stem.ngram13.corplim10k.custom_tknz.train"
with open("../data/train.tsv", "r") as f:
	with open(serilize2file, "wb") as outf:
		corpus = []
		# hv = HashingVectorizer(n_features=n_features,
		# 						non_negative=True,
		# 						ngram_range=(2,3),
		# 						tokenizer = CustomTokenizer(['NN', 'RB', 'JJ', 'VB', 'VBP']))
		# hv = TfidfVectorizer(norm='l1', ngram_range=(1,5),
		# 					tokenizer = CustomTokenizer(['RB', 'JJ', 'VB', 'VBP']))
		hv = CountVectorizer(ngram_range=(1,3), tokenizer = CustomTokenizer(['NN', 'RB', 'JJ', 'VB', 'VBP']))

		i = 0
		for line in f:
			i += 1
			if i == 1:
				continue
			if i >= corpus_limit:
				break

			tokens = line.split('\t')
			corpus.append(tokens[2])
			dataset['train_Y'].append(tokens[3])

			sys.stdout.write("Progress: %d row   \r" % (i))
			sys.stdout.flush()

		print "Extract features from docs corpus"
		# dataset['train_X'] = hv.transform(corpus)
		dataset['train_X'] = hv.fit_transform(corpus)

		print "Serialize to file %s" % (serilize2file)

		pickle.dump(dataset, outf)

		print "Processed lines count: ", i

