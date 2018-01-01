import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer

class CustomTokenizer(object):
    def __init__(self, allowed_pos):
        # self.wnl = WordNetLemmatizer()
        self.st = PorterStemmer()
        self.allowed_pos = allowed_pos
    def __call__(self, doc):
        # tokens = nltk.pos_tag([self.wnl.lemmatize(t) for t in word_tokenize(doc)])
        tokens = nltk.pos_tag([self.st.stem(t) for t in word_tokenize(doc)])
        return [t[0] for t in tokens if t[1] in self.allowed_pos]

# tests
if __name__ == '__main__':
	n_features = 5
	# hv = HashingVectorizer(n_features=n_features,
	# 						non_negative=True,
	# 						ngram_range=(2,3),
	# 						tokenizer = CustomTokenizer(['NN', 'RB', 'JJ', 'VB', 'VBP']))
	# print hv.build_analyzer()('i am the good man and wood')
	# print hv.transform(['i am the good man and wood'])

	hv = TfidfVectorizer(norm='l1', ngram_range=(1,2),
							tokenizer = CustomTokenizer(['NN', 'RB', 'JJ', 'VB', 'VBP']))
	print hv.fit_transform(['i am the good man and wood','i am the good man', 'you are good worker']).toarray()

	# print CountVectorizer(tokenizer = CustomTokenizer(['NN', 'RB', 'JJ', 'VB', 'VBP'])).build_analyzer()('i am the good man and wood')
	# print CountVectorizer().build_analyzer()('i am the good man and wood')









