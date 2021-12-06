import os
import gensim
import spacy
from president_helper import read_file, process_speeches, merge_speeches, get_president_sentences, get_presidents_sentences, most_frequent_words

# get list of all speech files
#Getting the list of the names of all the files to be used
files = sorted([file for file in os.listdir() if file[-4:] == '.txt'])
print(files)

# read each speech file
#Getting the text from each file
speeches = [read_file(file) for file in files]

# preprocess each speech
#Breaking down each speech into sentences
processed_speeches = process_speeches(speeches)

# merge speeches
all_sentences = merge_speeches(processed_speeches)

# view most frequently used words
most_freq_words = most_frequent_words(all_sentences)
#print(most_freq_words)

# create gensim model of all speeches
all_prez_embeddings = gensim.models.Word2Vec(all_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom
similar_to_freedom = all_prez_embeddings.most_similar("freedom", topn = 20)
#print(similar_to_freedom)
# view words similar to government
similar_to_government = all_prez_embeddings.most_similar("government", topn = 20)
#print(similar_to_government)

# get President Roosevelt sentences
roosevelt_sentences = get_president_sentences("franklin-d-roosevelt")

# view most frequently used words of Roosevelt
roosevelt_most_freq_words = most_frequent_words(roosevelt_sentences)
#print(roosevelt_most_freq_words)

# create gensim model for Roosevelt
roosevelt_embeddings = gensim.models.Word2Vec(roosevelt_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom for Roosevelt
roosevelt_similar_to_freedom = roosevelt_embeddings.most_similar("freedom", topn = 20)
#print(roosevelt_similar_to_freedom)

#Increasing the corpus size by using speeches of multiple presidents because there is not enough data from President Roosevelt to produce robust word embeddings
# get sentences of multiple presidents
rushmore_prez_sentences = get_presidents_sentences(["washington","jefferson","lincoln","theodore-roosevelt"])

# view most frequently used words of presidents
rushmore_most_freq_words = most_frequent_words(rushmore_prez_sentences)
#print(rushmore_most_freq_words)

# create gensim model for the presidents
rushmore_embeddings = gensim.models.Word2Vec(rushmore_prez_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom for presidents
rushmore_similar_to_freedom = rushmore_embeddings.most_similar("freedom", topn = 20)
#print(rushmore_similar_to_freedom)

# view words similar to constitution (from rushmore_most_freq_words) for presidents
rushmore_similar_to_constitution = rushmore_embeddings.most_similar("constitution", topn = 20)
#print(rushmore_similar_to_constitution)

#Repeating steps for the 4 most current presidents
# get sentences of current presidents
current_prez_sentences = get_presidents_sentences(["obama","trump","clinton","george-w-bush"])

# view most frequently used words of current presidents
current_most_freq_words = most_frequent_words(current_prez_sentences)
#print(current_most_freq_words)

# create gensim model for the current presidents
current_embeddings = gensim.models.Word2Vec(current_prez_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom for current presidents
current_similar_to_freedom = current_embeddings.most_similar("freedom", topn = 20)
print(current_similar_to_freedom)

# view words similar to america (from current_most_freq_words) for current presidents
current_similar_to_america = current_embeddings.most_similar("america", topn = 20)
print(current_similar_to_america)