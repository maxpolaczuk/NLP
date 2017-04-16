import os
import numpy as np
import json

# Read in txt file of pretrained word vectors using GloVe on
# 27Billion tweets
# More casual vocabulary size of 1.2M words
# For future, try using other corpuses - larger vocabs

fname = 'glove.840B.300d.txt' # 100dimensional takes about 30s
words = []
print('starting extraction')
with open(fname) as f:
    # change directory...
    os.chdir('/home/max/Downloads/glove.twitter.27B/data/')
    print('changed saving directory for data')
    for i, line in enumerate(f):

        # remove whitespace characters:
        words.append( line.strip().split()[0])

        # update the counter and show progress...
        if (i % 1000) == 0:
            print('iteration %s done' % i)

# convert to dictionary
print('converting to dictionary format...')
words = {k: str(v).lower() for v, k in enumerate(words)}
print('%s words processed!' % len(words) )

# save as JSON file
print('saving JSON file')
with open('words.json', 'w') as f:
    #data['new_key'] = [1, 2, 3]
    json.dump(words, f)

print('file successfully saved!')
