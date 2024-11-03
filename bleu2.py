import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# Define a function to calculate BLEU score for morphologically complex languages
def calculate_bleu(reference_corpus, candidate_translation, n_gram_range=(1, 4)):
    """
    Calculate BLEU score based on n-gram precision.

    Args:
    reference_corpus (list of list of str): List of reference translations (tokenized sentences).
    candidate_translation (list of str): Tokenized candidate translation.
    n_gram_range (tuple): Range of n-grams to calculate BLEU score for. Default is (1, 4).

    Returns:
    dict: BLEU scores for n-grams in the specified range.
    """
    smooth_fn = SmoothingFunction().method1
    bleu_scores = {}

    for n in range(n_gram_range[0], n_gram_range[1] + 1):
        weights = tuple((1. / n) if i < n else 0. for i in range(4))  # Adjust weights for n-gram calculation
        score = sentence_bleu(reference_corpus, candidate_translation, weights=weights, smoothing_function=smooth_fn)
        bleu_scores[f'{n}-gram'] = score

    return bleu_scores


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


# Example usage
if __name__ == "__main__":
    # Reference translations (morphologically rich languages)
    reference_corpus = read_file("culture_reference.txt")

    # Candidate translation
    # candidate_translation = read_file("culture_claude_candidate.txt")
    # candidate_translation = read_file("culture_chatgpt_candidate.txt")
    candidate_translation = read_file("culture_gemini_candidate.txt")

    # Calculate BLEU score for n-grams (1 to 4)
    bleu_scores = calculate_bleu(reference_corpus, candidate_translation, n_gram_range=(1, 4))

    # Display BLEU scores
    for n_gram, score in bleu_scores.items():
        print(f'{n_gram} BLEU score: {score:.4f}')
