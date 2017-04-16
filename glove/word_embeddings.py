import os
import numpy as np
import pandas as pd
import nltk

# Read in txt file of pretrained word vectors using GloVe on
# 27Billion tweets
# More casual vocabulary size of 1.2M words
# For future, try using other corpuses - larger vocabs

fname = 'glove.840B.300d.txt' # 100dimensional takes about 30s

content = []
words = []
batch = 1

# change directory...


print('starting extraction')
with open(fname) as f:
    os.chdir('/home/max/Downloads/glove.twitter.27B/data/')
    print('changed saving directory for data')
    for i, line in enumerate(f):
        # remove whitespace characters:
        # & create temporary file
        tmp = line.strip().split()
        content.append( tmp[1:])
        # add unique word
        #words.append(tmp[0])
        del tmp

        # update the counter and show progress...
        if (i % 1000) == 0:
            print('iteration %s done' % i)

        # this fucks out around 350K lines in... SO
        # every 200k iterations we will create a CSV and save that to disk
        if (i % 200000 == 0) & (i > 1):
            print('creating dataframe for batch %s...' % batch)
            content = pd.DataFrame(content)
            # save to csv:
        
            print('saving dataframe to disk ' )
            content.to_csv('word_embeddings_batch_%s.csv' % batch)
            content = []
            print('reset content list')
            batch += 1 # update the batch number

print('FINAL BATCH!')
f.close()
del f

print('creating dataframe for batch %s...' % batch)
content = pd.DataFrame(content)
# save the final amount to csv (to disk)
print('saving dataframe to disk ' )
content.to_csv('word_embeddings_batch_%s.csv' % batch)

del content # to reduce memory
print('saved final csv...')
