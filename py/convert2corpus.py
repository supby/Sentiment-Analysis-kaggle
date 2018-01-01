import sys

with open("../data/train.tsv", "r") as f:
	with open("../data/corpus.train", "w") as outf:
		i = 0
		for line in f:
			i += 1
			if i == 1:
				continue

			outf.write("%s\n" % line.split('\t')[2])

			sys.stdout.write("Progress: %d row   \r" % (i))
			sys.stdout.flush()

			i += 1

		print "Processed lines count: ", i