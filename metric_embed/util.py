import string
import requests
import os
import numpy as np
import gensim
from gensim.models.keyedvectors import Word2VecKeyedVectors

from biasOps import project_onto_subspace

import requests
import os

def load_legacy_w2v(w2v_file, dim=50):
    """
    Load word vectors from a legacy Word2Vec format file. Supports both local files and URLs.

    Args:
        w2v_file (str): Path to the .w2v file or a URL.
        dim (int): Expected dimensionality of word vectors.

    Returns:
        dict: A dictionary of word vectors.
        int: The dimensionality of the vectors.
    """
    # Check if the input is a URL
    if w2v_file.startswith("http://") or w2v_file.startswith("https://"):
        print(f"Downloading embeddings from {w2v_file}...")
        response = requests.get(w2v_file, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        temp_file = "temp_embedding.w2v"
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        w2v_file = temp_file  # Use the downloaded file for loading

    vectors = {}
    try:
        with open(w2v_file, 'r', encoding='utf-8') as f:
            for line in f:
                vect = line.strip().rsplit()
                word = vect[0]
                vect = np.array([float(x) for x in vect[1:]])
                if dim == len(vect):  # Ensure vector dimensionality matches
                    vectors[word] = vect
    finally:
        # Clean up the temporary file if a URL was used
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)

    return vectors, dim


def load_legacy_w2v_as_keyvecs(w2v_file, dim=50):
    vectors = None
    with open(w2v_file, 'r') as f:
        vectors = Word2VecKeyedVectors(dim)

        ws = []
        vs = []

        for line in f:
            vect = line.strip().rsplit()
            word = vect[0]
            vect = np.array([float(x) for x in vect[1:]])
            if(dim == len(vect)):
                ws.append(word)
                vs.append(vect)
        vectors.add_vectors(ws, vs, replace=True)
    return vectors

def convert_legacy_to_keyvec(legacy_w2v):
    if not legacy_w2v:  # Check if the dictionary is empty
        raise ValueError("The word vectors dictionary is empty. Please check the input file.")
    
    # Convert the keys to a list to make them indexable
    keys = list(legacy_w2v.keys())
    dim = len(legacy_w2v[keys[0]])  # Get the dimensionality of the first vector
    
    print(f"Loaded word vectors: {legacy_w2v}")
    print(f"Number of vectors: {len(legacy_w2v)}")


    vectors = Word2VecKeyedVectors(dim)
    ws = []
    vs = []

    for word, vect in legacy_w2v.items():
        ws.append(word)
        vs.append(vect)
        assert(len(vect) == dim)  # Ensure vector dimensionality is consistent
    vectors.add_vectors(ws, vs, replace=True)
    return vectors

def load_w2v(w2v_file, binary=True, limit=None):
    """
    Load Word2Vec format files using gensim and convert it to a dictionary
    """
    wv_from_bin = KeyedVectors.load_word2vec_format(w2v_file, binary=binary, limit=limit)
    dim = len(wv_from_bin[wv_from_bin.index2entity[0]])

    vectors = {w: wv_from_bin[w] for w in wv_from_bin.index2entity}

    return vectors, dim

def write_w2v(w2v_file, vectors):
    with open(w2v_file, 'w') as f:
        for word, vec in vectors.items():
            word = "".join(i for i in word if ord(i)<128)
            line = word + " " + " ".join([str(v) for v in vec]) + "\n"
            f.write(line)
        f.close()

def writeAnalogies(analogies, path):
    f = open(path, "w")
    f.write("Score,Analogy\n")
    for score, analogy, raw in analogies:
        f.write(str(score) + "," + str(analogy) + "," + str(raw) + "\n")
    f.close()

def writeGroupAnalogies(groups, path):
    f = open(path, "w")
    f.write("Score,Analogy\n")
    for analogies in groups:
        for score, analogy, raw in analogies:
            f.write(str(score) + "," + str(analogy) + "," + str(raw) + "\n")
    f.close()

def evalTerms(vocab, subspace, terms):
    for term in terms:
        vect = vocab[term]
        bias = project_onto_subspace(vect, subspace)
        print("Bias of '"+str(term)+"': {}".format(np.linalg.norm(bias)))

def pruneWordVecs(wordVecs):
    newWordVecs = {}
    for word in wordVecs.index_to_key:  # Use index_to_key to iterate over all words
        vec = wordVecs[word]           # Retrieve the vector for the current word
        valid = True
        if not isValidWord(word):      # Check if the word is valid
            valid = False
        if valid:
            newWordVecs[word] = vec    # Add the valid word and its vector to the new dictionary
    return newWordVecs

def preprocessWordVecs(wordVecs):
    """
    Following Bolukbasi:
    - only use the 50,000 most frequent words
    - only lower-case words and phrases
    - consisting of fewer than 20 lower-case characters
        (discard upper-case, digits, punctuation)
    - normalize all word vectors
    """
    newWordVecs = {}
    allowed = set(string.ascii_lowercase + ' ' + '_')

    for word, vec in wordVecs.items():
        chars = set(word)
        if chars.issubset(allowed) and len(word.replace('_', '')) < 20:
            newWordVecs[word] = vec / np.linalg.norm(vec)

    return newWordVecs

def removeWords(wordVecs, words):
    for word in words:
        if word in wordVecs:
            del wordVecs[word]
    return wordVecs

def isValidWord(word):
    return all([c.isalpha() for c in word])

def listContainsMultiple(source, target):
    for t in target:
        if(source[0] in t and source[1] in t):
            return True
    return False
