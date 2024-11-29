##for RNSB & SAME, I am assuming there's a vocabpath file that contains target word, positive word, and negative word


## python bias_analysis.py <embeddingPath> <vocabPath> <mode> -rnsb -same -analogies


import argparse
import numpy as np
from scipy.spatial.distance import cosine

from util import write_w2v, load_legacy_w2v, pruneWordVecs
from biasOps import identify_bias_subspace, neutralize_and_equalize, equalize_and_soften
from evalBias import generateAnalogies
from loader import load_analogy_templates, load_eval_terms


parser = argparse.ArgumentParser()
parser.add_argument('embeddingPath')
parser.add_argument('vocabPath')
parser.add_argument('mode')
parser.add_argument('-hard', action='store_true')
parser.add_argument('-soft', action='store_true')
parser.add_argument('-analogies', action="store_true")
parser.add_argument('-printLimit', type=int, default=500)
parser.add_argument('-rnsb', action='store_true', help="Compute RNSB metric")
parser.add_argument('-same', action='store_true', help="Compute SAME metric")
args = parser.parse_args()

defSets = load_def_sets(args.vocabPath)
testTerms = load_test_terms(args.vocabPath)
analogyTemplates = load_analogy_templates(args.vocabPath, args.mode)
subspace = identify_bias_subspace(word_vectors, defSets, 1, embedding_dim)
neutral_words = []
for value in analogyTemplates.values():
    neutral_words.extend(value)

#getting wordlist for RNSB & SAME:
categories = {'target_words': [], 'positive_words': [], 'negative_words': []}
current_category = None
with open(args.vocabPath, 'r') as f:
    for line in f:
        line = line.strip()
        # Check for category header
        if line.startswith("[") and line.endswith("]"):
            current_category = line[1:-1]  # Extract category name
            if current_category not in categories:
                categories[current_category] = []  # Add new category dynamically
        elif current_category and line:
            # Append words to the current category
            categories[current_category].append(line)

print(f"Parsing vocabulary from {args.vocabPath}...")

target_words = categories.get('target_words', [])
positive_words = categories.get('positive_words', [])
negative_words = categories.get('negative_words', [])

print(f"Target Words: {target_words[:20]}...") 
print(f"Positive Words: {positive_words[:20]}...")
print(f"Negative Words: {negative_words[:20]}...")


print("Loading embeddings from {}".format(args.embeddingPath))
word_vectors, embedding_dim = load_legacy_w2v(args.embeddingPath)

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
    with open(f"output/{args.mode}_biased_analogies.csv", "w") as f:
        f.write("Score,Analogy\n")
        for score, analogy, _ in biasedAnalogies:
            f.write(f"{score},{analogy}\n")

    with open(f"output/{args.mode}_grouped_analogies.csv", "w") as f:
        f.write("Group,Analogies\n")
        for group, analogies in biasedAnalogyGroups.items():
            f.write(f"{group},{analogies}\n")



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
    with open("output/rnsb_scores.csv", "w") as f:
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
    with open("output/same_scores.csv", "w") as f:
        f.write("Target Word,SAME Score\n")
        for target, same in same_scores.items():
            f.write(f"{target},{same}\n")

# Debiasing 
if args.hard:
    print("\nPerforming Hard Debiasing...")
    hard_word_vectors = neutralize_and_equalize(
        word_vectors, analogyTemplates, subspace, embedding_dim
    )

    if args.rnsb:
        rnsb_scores_hard = compute_rnsb(hard_word_vectors, target_words, positive_words, negative_words)
        print("\nHard Debiased RNSB Scores:")
        for target, rnsb in rnsb_scores_hard.items():
            print(f"{target}: {rnsb}")

    if args.same:
        same_scores_hard = compute_same(hard_word_vectors, target_words, positive_words, negative_words)
        print("\nHard Debiased SAME Scores:")
        for target, same in same_scores_hard.items():
            print(f"{target}: {same}")

    if args.analogies:
        hardAnalogies, _ = generateAnalogies(analogyTemplates, hard_word_vectors)
        print("\nHard Debiased Analogies:")
        for score, analogy, _ in hardAnalogies[:args.printLimit]:
            print(f"{score:.4f}: {analogy}")

if args.soft:
    print("\nPerforming Soft Debiasing...")
    soft_word_vectors = equalize_and_soften(
        word_vectors, analogyTemplates, subspace, embedding_dim
    )

    if args.rnsb:
        rnsb_scores_soft = compute_rnsb(soft_word_vectors, target_words, positive_words, negative_words)
        print("\nSoft Debiased RNSB Scores:")
        for target, rnsb in rnsb_scores_soft.items():
            print(f"{target}: {rnsb}")

    if args.same:
        same_scores_soft = compute_same(soft_word_vectors, target_words, positive_words, negative_words)
        print("\nSoft Debiased SAME Scores:")
        for target, same in same_scores_soft.items():
            print(f"{target}: {same}")

    if args.analogies:
        softAnalogies, _ = generateAnalogies(analogyTemplates, soft_word_vectors)
        print("\nSoft Debiased Analogies:")
        for score, analogy, _ in softAnalogies[:args.printLimit]:
            print(f"{score:.4f}: {analogy}")
