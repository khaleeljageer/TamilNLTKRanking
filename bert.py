from bert_score import score
import nltk
from indicnlp import loader, common
from indicnlp.tokenize import sentence_tokenize

# Ensure you've downloaded the NLTK tokenizer models
nltk.download('punkt_tab')


def equalize_text_lengths(reference_text, candidate_text):
    # Split both texts into sentences using indic_nlp for Tamil
    reference_sentences = sentence_tokenize.sentence_split(reference_text, lang='ta')
    candidate_sentences = sentence_tokenize.sentence_split(candidate_text, lang='ta')

    # Determine difference in length
    len_diff = len(reference_sentences) - len(candidate_sentences)

    # Adjust the shorter list to match the length of the longer one
    if len_diff > 0:  # Reference is longer
        candidate_sentences.extend([''] * len_diff)  # Add empty strings to candidate
    elif len_diff < 0:  # Candidate is longer
        reference_sentences.extend([''] * abs(len_diff))  # Add empty strings to reference

    return reference_sentences, candidate_sentences


def compute_bertscore_for_tamil(reference_texts, candidate_texts):
    """
    Compute BERTScore for Tamil between reference and candidate texts.

    Args:
    reference_texts (list of str): List of reference summaries/translations.
    candidate_texts (list of str): List of machine-generated summaries/translations.

    Returns:
    dict: A dictionary containing the BERTScore metrics.
    """
    # Compute BERTScore for the given reference and candidate texts
    P, R, F1 = score(candidate_texts, reference_texts, lang="ta", verbose=True)

    # Return the average precision, recall, and F1
    return {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


# Example usage:
if __name__ == "__main__":
    reference_texts = read_file('culture_reference.txt')
    # candidate_texts = read_file('culture_claude_candidate.txt')
    # candidate_texts = read_file('culture_chatgpt_candidate.txt')
    candidate_texts = read_file('culture_gemini_candidate.txt')

    common.set_resources_path('/home/syedkhaleel/Documents/indic_nlp_resources')
    loader.load()

    print("Before Equalize")
    print(f"Reference Text Length: {len(reference_texts)}")
    print(f"Candidate Text Length: {len(candidate_texts)}")
    # Equalize the sentence lists
    reference_sentences, candidate_sentences = equalize_text_lengths(reference_texts, candidate_texts)

    print("After Equalize")
    print(f"Reference Text Length: {len(reference_sentences)}")
    print(f"Candidate Text Length: {len(candidate_sentences)}")
    # Compute BERTScore for Tamil
    bert_scores = compute_bertscore_for_tamil(reference_sentences, candidate_sentences)

    # Display the BERTScore metrics
    print("BERT Precision:", bert_scores['Precision'])
    print("BERT Recall:", bert_scores['Recall'])
    print("BERT F1 Score:", bert_scores['F1'])
