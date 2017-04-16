# NLP Stuff

## Glove word embeddings (preprocessing)
Messing around with the Glove word embeddings (Vocab of ~2Million with 300 dimensional vectors). It MAXes out RAM very quick, so my program first cuts it into smaller subsets (csv files), and the lookup table of word to word id is saved as a JSON file. Total data size becomes > 5GB. The find_embeddings.py file is a neat tool where you can enter in words to the console and it will plot them. Currently uses PCA to dimension reduce from 300d to 2d, this is becuase TSNE was not working with my setup, so need to debug that. 


