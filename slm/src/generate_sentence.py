# slm/src/generate_sentence.py
# ─────────────────────────────────────────────────────────────────────────────
# Converts a sign-language word buffer into a grammatically correct sentence.
#
# Sign language buffers are:
#   - ALL CAPS (from recognition model output)
#   - Missing articles (a, the), prepositions (to, at, in),
#     auxiliary verbs (will, is, am, have), conjunctions
#   - Often in topic-comment order, not English order
#
# Example:
#   Input : ["I", "TOMORROW", "HOSPITAL", "GO"]
#   Output: "I will go to the hospital tomorrow."
# ─────────────────────────────────────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sentence_model import generate


def words_to_sentence(words: list) -> str:
    """
    Convert a list of sign-language words into a natural English sentence.

    Args:
        words: e.g. ["I", "TOMORROW", "HOSPITAL", "GO"]

    Returns:
        A clean, grammatically correct English sentence.
    """
    if not words:
        raise ValueError("[generate_sentence] Word list is empty.")

    # Normalise: strip whitespace, title-case, drop empty tokens
    clean = [w.strip().capitalize() for w in words if w.strip()]
    if not clean:
        raise ValueError("[generate_sentence] No valid words after cleaning.")

    word_string = ", ".join(clean)

    # Instruction prompt — flan-t5-large is instruction-tuned so this works well
    prompt = (
        f"These words are from a sign language recognition system. "
        f"They may be out of order and are missing grammar words like "
        f"articles (a, the), prepositions (to, at, in, of), auxiliary verbs "
        f"(is, are, will, have), and conjunctions. "
        f"Using all of these words, write one complete, natural, grammatically "
        f"correct English sentence. Output only the sentence, nothing else. "
        f"Words: {word_string}"
    )

    print(f"[generate_sentence] Input : {words}")
    result = generate(prompt, max_length=128)

    # ── Post-processing ───────────────────────────────────────────────────────
    # Remove any accidental prompt leakage
    if "Words:" in result:
        result = result.split("Words:")[-1].strip()

    # Fallback if model returns garbage
    if not result or len(result.split()) < 2:
        print("[generate_sentence] Weak output — using direct join fallback.")
        result = " ".join(clean)

    # Ensure ends with punctuation
    if result[-1] not in ".!?":
        result += "."

    # Capitalise first character
    result = result[0].upper() + result[1:]

    print(f"[generate_sentence] Output: {result}")
    return result


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ["I", "TOMORROW", "HOSPITAL", "GO"],
        ["HUNGRY", "I", "FOOD", "WANT"],
        ["MOTHER", "SICK", "DOCTOR", "NEED"],
        ["THANK", "YOU", "HELP"],
        ["WATER", "NEED", "I"],
    ]
    print("\n" + "=" * 56)
    print("  flan-t5-large  Sign → Sentence Test")
    print("=" * 56)
    for words in tests:
        sentence = words_to_sentence(words)
        print(f"  IN : {words}")
        print(f"  OUT: {sentence}\n")
