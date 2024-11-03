from collections import Counter

from indicnlp import common
from indicnlp import loader
from indicnlp.tokenize import indic_tokenize


def tokenize_tamil_text(text):
    """
    Tokenizes Tamil text using Indic NLP Library.

    Args:
    text (str): Input Tamil text.

    Returns:
    str: A tokenized version of the input text.
    """
    tokens = indic_tokenize.trivial_tokenize(text, 'ta')
    return ' '.join(tokens)


def get_ngrams(tokens, n):
    """
    Get n-grams from a list of tokens.
    """
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n(ref_tokens, cand_tokens, n):
    """
    Calculate ROUGE-N score.
    """
    ref_ngrams = Counter(get_ngrams(ref_tokens, n))
    cand_ngrams = Counter(get_ngrams(cand_tokens, n))

    overlap = sum((ref_ngrams & cand_ngrams).values())
    ref_count = sum(ref_ngrams.values())
    cand_count = sum(cand_ngrams.values())

    precision = overlap / cand_count if cand_count > 0 else 0
    recall = overlap / ref_count if ref_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def lcs(X, Y):
    """
    Compute the length of the Longest Common Subsequence (LCS).
    """
    m, n = len(X), len(Y)
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]


def rouge_l(ref_tokens, cand_tokens):
    """
    Calculate ROUGE-L score.
    """
    lcs_length = lcs(ref_tokens, cand_tokens)
    ref_count = len(ref_tokens)
    cand_count = len(cand_tokens)

    precision = lcs_length / cand_count if cand_count > 0 else 0
    recall = lcs_length / ref_count if ref_count > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_rouge_for_tamil(reference_texts, candidate_texts):
    """
    Compute ROUGE scores for Tamil between reference and candidate texts.
    """
    results = []
    for i, (reference, candidate) in enumerate(zip(reference_texts, candidate_texts)):
        try:
            print(f"\nProcessing text pair {i + 1}:")
            print(f"Original Reference: {reference}")
            print(f"Original Candidate: {candidate}")

            ref_tokens = list(tokenize_tamil_text(reference))
            cand_tokens = list(tokenize_tamil_text(candidate))

            print(f"Tokenized Reference: {''.join(ref_tokens)}")
            print(f"Tokenized Candidate: {''.join(cand_tokens)}")

            rouge1 = rouge_n(ref_tokens, cand_tokens, 1)
            rouge2 = rouge_n(ref_tokens, cand_tokens, 2)
            rougel = rouge_l(ref_tokens, cand_tokens)

            print(f"Custom ROUGE-1 scores: {rouge1}")
            print(f"Custom ROUGE-2 scores: {rouge2}")
            print(f"Custom ROUGE-L scores: {rougel}")

            results.append({'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougel})
        except Exception as e:
            print(f"Error processing text pair {i + 1}: {e}")

    # Aggregate results
    aggregated = {metric: {key: sum(r[metric][key] for r in results) / len(results)
                           for key in ['precision', 'recall', 'f1']}
                  for metric in ['rouge1', 'rouge2', 'rougeL']}

    print("\nFinal aggregated scores:")
    print(aggregated)
    return aggregated


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


# Example usage:
if __name__ == "__main__":
    # Example Tamil reference and candidate translations
    reference_texts = read_file('culture_reference.txt')
    # candidate_texts = read_file('culture_claude_candidate.txt')
    # candidate_texts = read_file('culture_chatgpt_candidate.txt')
    candidate_texts = read_file('culture_gemini_candidate.txt')

    common.set_resources_path('/home/syedkhaleel/Documents/indic_nlp_resources')
    loader.load()

    try:
        # Compute ROUGE scores for Tamil
        rouge_scores = compute_rouge_for_tamil(reference_texts, candidate_texts)

        # Display the ROUGE scores
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            print(f"{rouge_type.upper()} Scores:")
            print(f"  F1 Score: {rouge_scores[rouge_type]['f1']:.4f}")
            print(f"  Precision: {rouge_scores[rouge_type]['precision']:.4f}")
            print(f"  Recall: {rouge_scores[rouge_type]['recall']:.4f}")
            print()

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
