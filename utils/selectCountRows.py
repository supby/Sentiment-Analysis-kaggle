import sys
import operator

select_count = 10000

with open("../data/fann_sentiments.train", "r") as f:
	with open("../data/fann_sentiments.10k.train", "w") as outf:
		i = 0		
		for line in f:
			if i == 0:
				outf.write("%s\n" % "10000 3208 5")
			else:
				outf.write("%s" % line)
				if i == select_count*2:
					break		

			sys.stdout.write("Progress: %d row   \r" % (i))
			sys.stdout.flush()

			i += 1