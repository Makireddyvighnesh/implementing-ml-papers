"""
BLEU metric implementation from scratch.

Based on:
Papineni et al. (2002) – BLEU: a Method for Automatic Evaluation of Machine Translation.
"""

import math
from collections import Counter


def get_ngrams(tokens, n):
    """
    Generate n-grams from a list of tokens.

    For example:
    tokens = ["the", "cat", "sat"]
    n = 2  ->  ("the", "cat"), ("cat", "sat")

    Returns a Counter so we also keep track of how many
    times each n-gram appears.
    """
    ngrams = []

    # Slide a window of size n across the tokens
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))

    return Counter(ngrams)


def modified_precision(reference, candidate, n):
    """
    Compute the modified n-gram precision used in BLEU.

    Instead of counting all matches, BLEU uses *clipped counts*.
    This prevents a candidate from repeating the same word 
    many times to artificially inflate precision.
    """

    ref_tokens = reference.split()
    cand_tokens = candidate.split()

    ref_ngrams = get_ngrams(ref_tokens, n)
    cand_ngrams = get_ngrams(cand_tokens, n)

    clipped = 0

    # Count matches but clip them using the reference frequency
    for ngram in cand_ngrams:
        clipped += min(cand_ngrams[ngram], ref_ngrams.get(ngram, 0))

    total = sum(cand_ngrams.values())

    if total == 0:
        return 0

    return clipped / total


def brevity_penalty(reference, candidate):
    """
    Compute BLEU brevity penalty.

    Short candidates should be penalized because a model
    could cheat by producing very short outputs.

    BP = 1                      if candidate length > reference length
    BP = exp(1 - r/c)           otherwise
    """

    r = len(reference.split())
    c = len(candidate.split())

    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)


def bleu_score(reference, candidate, max_n=4):
    """
    Compute BLEU score between a reference sentence and a candidate.

    Steps:
    1. Compute modified precision for n = 1..4
    2. Take the geometric mean of the precisions
    3. Apply brevity penalty

    This implementation follows the original BLEU idea but
    keeps things simple (single reference).
    """

    precisions = []

    # Compute precision for 1-gram through 4-gram
    for n in range(1, max_n + 1):
        p = modified_precision(reference, candidate, n)
        precisions.append(p)

    # Avoid log(0) when any precision is zero
    precisions = [p if p > 0 else 1e-10 for p in precisions]

    # Geometric mean in log space
    log_precisions = sum(math.log(p) for p in precisions) / max_n

    bp = brevity_penalty(reference, candidate)

    bleu = bp * math.exp(log_precisions)

    return bleu
