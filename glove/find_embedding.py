import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# open up the data:
os.chdir('/home/max/Downloads/glove.twitter.27B/data')
with open('words.json', 'r') as f:
    try:
        data = json.load(f)
    # if the file is empty the ValueError will be thrown
    except ValueError:
        data = {}
# order 66 the file
f.close()

def word_find(word):
    # user inputs the word and it is located from
    # the json file of word indexs...
    # pulls in the numbers then runs a tsne over it
    # and plots it along with a label:

    idx = int(data[str(word.lower())])

    # get the within batch index number (modular arithmetic)
    bat_id = idx % 200000

    # load in right batch and get numbers: 
    # (probably a way more elegant way of this but fuck it)
    if idx < 200000:
        batch = 1
    elif idx < 400000: 
        batch = 2
    elif idx < 600000: 
        batch = 3
    elif idx < 800000: 
        batch = 4
    elif idx < 1000000: 
        batch = 5
    elif idx < 1200000: 
        batch = 6
    elif idx < 1400000: 
        batch = 7
    elif idx < 1600000: 
        batch = 8
    elif idx < 1800000: 
        batch = 9
    elif idx < 2000000: 
        batch = 10
    else:
        batch = 11      

    # open the file
    csvdat = pd.read_csv('word_embeddings_batch_%s.csv' % batch)    
    # get the word vector
    vector = np.nan_to_num(np.array(csvdat.ix[bat_id,:].convert_objects(convert_numeric=True).fillna(0).tolist(),dtype=float))
    # delete file from memory
    del csvdat

    return vector

# now actually iterate over this:
quit = False # initialize this
word_choice = [] 
itr = 0 # set initial iteration of the words...

print('startword is Max :)')
startword = 'max'
vec = np.array(word_find(startword))
word_choice.append(startword)
print(np.shape(vec))
vec = vec.reshape(1,len(vec))

while quit == False:

    # user inputs the word
    Tword = input('Please enter word embedding: ')

    # append to the vector...
    tmp =np.array(word_find(Tword.lower()))
    print(np.shape(tmp))
    vec = np.append(vec,tmp.reshape(1,len(tmp)), axis =0)
    
    print(vec)
    # clip the high values in the array
    #vec = np.clip(vec,-5.0,100.0)
    #vec = (vec+.001) / 100.0 
    # archive the word choice
    word_choice.append(Tword)

    # now run TSNE / PCA over the vector of words & fit
    #model = TSNE(n_components = 2, random_state = 0)
    if itr > 0:
    	# reset the model...
    	del model
    model = PCA(n_components=2)

    # check for nans etc.
    print('any NANs?', np.isnan(vec).any())
    print('any INFs?', np.isinf(vec).any())
    print('max ', np.max(vec))
    print('min ', np.min(vec))

    # large maximum so need to normalize:
    # run the TSNE algo
    #tsne_words = PCA(vec)#.reshape(1,-1)

    #vec = MinMaxScaler().fit_transform(vec)
    # run the TSNE algo
    tsne_words = model.fit_transform(vec)
  
    print('tsne words look like:\n\n', tsne_words)

    # plot these words..
    print('generating graph...')
    if itr > 0:
    	# close prevous plot:
    	plt.close(fig)
    	
    fig, ax = plt.subplots()
    ax.scatter(tsne_words[:,0], tsne_words[:,1])
    for i, txt in enumerate(word_choice):
        ax.annotate(txt,(tsne_words[i,0],tsne_words[i,1]))

    fig.show()


    # next iteration:
    itr += 1
    # looping while either closing or continuing
    while True == True:
    # either continue or exit program
        cont = input('continue (Y/N): ')
        if cont.lower() == 'n':
            # close the program
            quit = True
            exit()
        elif cont.lower() == 'y':
            break
        else:
            print('I\'m sorry, Please try again with "y" or "n" in any case!')


    
