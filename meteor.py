import nltk
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize

# Download required NLTK resources (if not already downloaded)
nltk.download('punkt')


# You may also use Tamil-specific stemmers and tokenizers if required,
# though for simplicity this example assumes tokenized input.

def calculate_meteor(reference_corpus, candidate_translation):
    """
    Calculate METEOR score for a candidate translation.

    Args:
    reference_corpus (list of list of str): List of reference translations (tokenized sentences).
    candidate_translation (list of str): Tokenized candidate translation.

    Returns:
    float: METEOR score.
    """
    # Tokenize both reference corpus and candidate translation
    tokenized_references = [word_tokenize(ref) for ref in reference_corpus]
    tokenized_candidate = word_tokenize(candidate_translation)

    # METEOR score calculation for a single candidate translation
    score = meteor_score.meteor_score(tokenized_references, tokenized_candidate)
    return score


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


if __name__ == "__main__":
    # Example Tamil reference and candidate translations
    reference_texts = read_file('culture_reference.txt')
    # candidate_texts = read_file('culture_claude_candidate.txt')
    # candidate_texts = read_file('culture_chatgpt_candidate.txt')
    candidate_texts = read_file('culture_gemini_candidate.txt')

    # Calculate METEOR score
    meteor = calculate_meteor(reference_texts, candidate_texts)

    # Display METEOR score
    print(f'METEOR score: {meteor:.4f}')
