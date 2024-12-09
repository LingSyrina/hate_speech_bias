
## python bias_analysis.py <embeddingPath> <vocabPath> <mode1> <mode2> -rnsb -same -analogies

import os
import argparse
import numpy as np
from scipy.spatial.distance import cosine
from datasets import load_dataset
from util import write_w2v, load_legacy_w2v, pruneWordVecs, convert_legacy_to_keyvec
from biasOps import identify_bias_subspace, neutralize_and_equalize, equalize_and_soften
from evalBias import generateAnalogies
from loader import load_analogy_templates, load_def_sets

# Create the output directory
output_dir = "/Users/sykim/hate_speech_bias/metric_embed/output"
os.makedirs(output_dir, exist_ok=True)  # create the directory if it doesn't exist


parser = argparse.ArgumentParser()
parser.add_argument('embeddingPath', help="Hugging Face dataset name or local file path")
parser.add_argument('vocabPath')
parser.add_argument('mode1')
parser.add_argument('mode2', choices=['female', 'male', 'race'], help="Mode for target words: male, female or race")
parser.add_argument('-hard', action='store_true')
parser.add_argument('-soft', action='store_true')
parser.add_argument('-analogies', action="store_true")
parser.add_argument('-printLimit', type=int, default=500)
parser.add_argument('-rnsb', action='store_true', help="Compute RNSB metric")
parser.add_argument('-same', action='store_true', help="Compute SAME metric")
args = parser.parse_args()

def load_hf_embeddings(dataset_name):
    print(f"Loading embeddings from Hugging Face dataset: {dataset_name}")
    # Download the dataset
    dataset = load_dataset("LingSyrina/debiased_embedding", split="train")
    local_path = dataset.info.download_urls["gender_GLV1_role_biasedEmbeddingsOut.w2v"]

    # Use your existing function to load embeddings
    word_vectors, embedding_dim = load_legacy_w2v(local_path)
    return word_vectors, embedding_dim
    
defSets = load_def_sets(args.vocabPath)
analogyTemplates = load_analogy_templates(args.vocabPath, args.mode1)
#subspace = identify_bias_subspace(word_vectors, defSets, 1, embedding_dim)
neutral_words = []
for value in analogyTemplates.values():
    neutral_words.extend(value)


def load_word_list(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return [line.strip() for line in f if line.strip()]

print(f"Loading vocab...")

positive_words = load_word_list('/Users/sykim/hate_speech_bias/metric_embed/data/positive-words.txt')
negative_words = load_word_list('/Users/sykim/hate_speech_bias/metric_embed/data/negative-words.txt')

mode2_target_files = {
    'male': '/Users/sykim/hate_speech_bias/metric_embed/data/target-words-male.txt',
    'female': '/Users/sykim/hate_speech_bias/metric_embed/data/target-words-female.txt',
    'race': '/Users/sykim/hate_speech_bias/metric_embed/data/target-words-race.txt',
}

target_words_path = mode2_target_files[args.mode2]
target_words = load_word_list(target_words_path)

print(f"Target Words: {target_words[:20]}...")
print(f"Positive Words: {positive_words[:20]}...")
print(f"Negative Words: {negative_words[:20]}...")


print("Loading embeddings from {}".format(args.embeddingPath))

if args.embeddingPath.startswith("LingSyrina"):
    word_vectors, embedding_dim = load_hf_embeddings(args.embeddingPath)
else:
    word_vectors, embedding_dim = load_legacy_w2v(args.embeddingPath)
    
word_vectors = convert_legacy_to_keyvec(word_vectors)

print("Pruning Word Vectors... Starting with", len(word_vectors))
word_vectors = pruneWordVecs(word_vectors)
print("\tEnded with", len(word_vectors))


## Biased Analogies 

if args.analogies:
    print("Generating Bias Analogies...")
    biasedAnalogies, biasedAnalogyGroups = generateAnalogies(analogyTemplates, word_vectors)

    # Print Bias Analogies
    print("\nBiased Analogies:")
    for score, analogy, _ in biasedAnalogies[:args.printLimit]:
        print(f"{score:.4f}: {analogy}")

    # Save results in csv
    with open(f"{output_dir}/{args.mode1}_biased_analogies.csv", "w") as f:
        f.write("Score,Analogy\n")
        for score, analogy, _ in biasedAnalogies:
            f.write(f"{score},{analogy}\n")

    with open(f"{output_dir}/{args.mode1}_grouped_analogies.csv", "w") as f:
        f.write("Score,Analogy\n")
        for analogies in biasedAnalogyGroups:  # Iterate directly over the list
            for score, analogy, _ in analogies:
                f.write(f"{score},{analogy}\n")



# RNSB Function
def compute_rnsb(word_vectors, target_words, positive_words, negative_words):
    rnsb_scores = {}

    for target in target_words:
        if target not in word_vectors:
            continue

        target_vec = word_vectors[target]

        positive_similarities = [
            1 - cosine(target_vec, word_vectors[word]) for word in positive_words if word in word_vectors
        ]
        negative_similarities = [
            1 - cosine(target_vec, word_vectors[word]) for word in negative_words if word in word_vectors
        ]

        mean_positive = np.mean(positive_similarities) if positive_similarities else 0
        mean_negative = np.mean(negative_similarities) if negative_similarities else 0

        rnsb_scores[target] = mean_negative / mean_positive if mean_positive > 0 else float('inf')

    return rnsb_scores

# SAME Function
def compute_same(word_vectors, target_words, attribute_set1, attribute_set2):
    same_scores = {}

    for target in target_words:
        if target not in word_vectors:
            continue

        target_vec = word_vectors[target]

        similarities_set1 = [
            1 - cosine(target_vec, word_vectors[word]) for word in attribute_set1 if word in word_vectors
        ]
        similarities_set2 = [
            1 - cosine(target_vec, word_vectors[word]) for word in attribute_set2 if word in word_vectors
        ]

        mean_set1 = np.mean(similarities_set1) if similarities_set1 else 0
        mean_set2 = np.mean(similarities_set2) if similarities_set2 else 0

        same_scores[target] = mean_set1 - mean_set2

    return same_scores

# Compute RNSB
if args.rnsb:
    print("Computing RNSB...")
    rnsb_scores = compute_rnsb(word_vectors, target_words, positive_words, negative_words)

    # Print RNSB results
    print("\nRNSB Scores:")
    for target, rnsb in rnsb_scores.items():
        print(f"{target}: {rnsb}")

    # Save results in csv
    with open(f"{output_dir}/rnsb_scores.csv", "w") as f:
        f.write("Target Word,RNSB Score\n")
        for target, rnsb in rnsb_scores.items():
            f.write(f"{target},{rnsb}\n")

# Compute SAME
if args.same:
    print("Computing SAME...")
    same_scores = compute_same(word_vectors, target_words, positive_words, negative_words)

    # Print SAME results
    print("\nSAME Scores:")
    for target, same in same_scores.items():
        print(f"{target}: {same}")

    # Save results in csv
    with open(f"{output_dir}/same_scores.csv", "w") as f:
        f.write("Target Word,SAME Score\n")
        for target, same in same_scores.items():
            f.write(f"{target},{same}\n")
