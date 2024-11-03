import numpy as np


def edit_distance(candidate, reference):
    """
    Calculate the edit distance between candidate and reference translations using dynamic programming.

    Args:
    candidate (list of str): Tokenized candidate translation.
    reference (list of str): Tokenized reference translation.

    Returns:
    int: Edit distance between the two translations.
    """
    len_candidate = len(candidate)
    len_reference = len(reference)

    # Create a DP table to store the edit distances
    dp = np.zeros((len_candidate + 1, len_reference + 1), dtype=int)

    # Initialize the table
    for i in range(len_candidate + 1):
        dp[i][0] = i  # Cost of deletions
    for j in range(len_reference + 1):
        dp[0][j] = j  # Cost of insertions

    # Fill the DP table
    for i in range(1, len_candidate + 1):
        for j in range(1, len_reference + 1):
            if candidate[i - 1] == reference[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No change needed
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1  # Min of insert, delete, substitute

    return dp[len_candidate][len_reference]


def calculate_ter(reference_corpus, candidate_translation):
    """
    Calculate Translation Edit Rate (TER) for a candidate translation.

    Args:
    reference_corpus (list of str): List of reference translations (sentences).
    candidate_translation (str): Candidate translation (sentence).

    Returns:
    float: TER score.
    """
    # Tokenize candidate and reference translations
    tokenized_candidate = candidate_translation.split()
    ter_scores = []

    for reference in reference_corpus:
        tokenized_reference = reference.split()

        # Skip empty references to avoid divide-by-zero errors
        if len(tokenized_reference) == 0:
            continue

        # Calculate edit distance
        edit_dist = edit_distance(tokenized_candidate, tokenized_reference)

        # TER = Edit Distance / Length of Reference (Avoid divide by zero)
        ter_score = edit_dist / len(tokenized_reference)
        ter_scores.append(ter_score)

    # If all references are empty, we should handle this edge case
    if not ter_scores:
        return float('inf')  # Return infinity if no valid references are available

    # Return the minimum TER score across all references
    return min(ter_scores)


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()


if __name__ == "__main__":
    # Example reference translations (in Tamil)
    reference_corpus = read_file('culture_reference.txt')

    # Example candidate translation (in Tamil)
    # candidate_translation = read_file('culture_claude_candidate.txt')
    # candidate_translation = read_file('culture_chatgpt_candidate.txt')
    candidate_translation = read_file('culture_gemini_candidate.txt')

    # Calculate TER score
    ter_score = calculate_ter(reference_corpus, candidate_translation)

    # Display TER score
    print(f'TER score: {ter_score:.4f}')
