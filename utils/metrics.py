# metrics.py

from typing import List

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate the Word Error Rate (WER) between a reference and a hypothesis.

    Args:
        reference (str): The ground truth sentence.
        hypothesis (str): The predicted sentence.

    Returns:
        float: The WER as a percentage.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Initialize the matrix
    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    # Compute the edit distance
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    wer = dp[len(ref_words)][len(hyp_words)] / len(ref_words)
    return wer * 100

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate the Character Error Rate (CER) between a reference and a hypothesis.

    Args:
        reference (str): The ground truth sentence.
        hypothesis (str): The predicted sentence.

    Returns:
        float: The CER as a percentage.
    """
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    # Initialize the matrix
    dp = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        dp[i][0] = i
    for j in range(len(hyp_chars) + 1):
        dp[0][j] = j

    # Compute the edit distance
    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    cer = dp[len(ref_chars)][len(hyp_chars)] / len(ref_chars)
    return cer * 100

if __name__ == "__main__":
    # Example usage
    ref = "this is a test"
    hyp = "this is test"

    print(f"WER: {calculate_wer(ref, hyp):.2f}%")
    print(f"CER: {calculate_cer(ref, hyp):.2f}%")