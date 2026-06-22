from __future__ import annotations


def edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    n = len(a)
    m = len(b)

    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i

        for j in range(1, m + 1):
            old = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost,
            )
            prev = old

    return dp[m]


def cer(pred: str, target: str) -> float:
    if not target:
        return 0.0 if not pred else 1.0
    return edit_distance(pred, target) / len(target)


def wer(pred: str, target: str) -> float:
    pred_words = pred.split()
    target_words = target.split()

    if not target_words:
        return 0.0 if not pred_words else 1.0

    return edit_distance(pred_words, target_words) / len(target_words)


def exact_match(pred: str, target: str) -> float:
    return 1.0 if pred == target else 0.0