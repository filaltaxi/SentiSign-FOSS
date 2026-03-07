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

import os
import re
import sys
sys.path.insert(0, os.path.dirname(__file__))
from buffer_clean import clean_buffer
from example_bank import select_examples
from sentence_model import generate

SYSTEM_PROMPT = (
    "You are an ASL-to-English translator. "
    "Output only the English sentence. "
    "Never repeat or echo the signs. "
    "Use simple, everyday spoken English. Never use formal or clinical language."
)
_EXACT_OVERRIDES = {
    ("THANK YOU",): "Thank you.",
    ("THANK YOU", "HELP"): "Thank you for your help.",
    ("DOCTOR", "WHERE"): "Where is the doctor?",
    ("TOILET", "WHERE"): "Where is the toilet?",
    ("EMERGENCY",): "This is an emergency!",
    ("PHONE", "PLEASE"): "Phone, please.",
}

SUBJECT_WORDS = {"i", "you", "we", "mother", "father", "child", "family"}
TIME_WORDS = {"today", "tomorrow", "now"}
PLACE_WORDS = {"hospital", "home", "toilet"}
STATE_PHRASES = {
    "hungry": "hungry",
    "sick": "sick",
    "pain": "in pain",
    "sorry": "sorry",
}
ACTION_WORDS = {"go", "come", "need", "want", "help", "sleep", "understand"}
ARTICLE_OBJECTS = {
    "doctor": "a doctor",
    "hospital": "the hospital",
    "toilet": "the toilet",
}
GREETING_TARGETS = {
    "boy": "boy",
    "girl": "girl",
    "child": "child",
    "children": "children",
    "mother": "mother",
    "father": "father",
    "grandmother": "grandmother",
    "grandfather": "grandfather",
    "family": "family",
    "you": "to you",
}


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _clean_model_text(result: str) -> str:
    result = re.sub(r"<think>.*?</think>", " ", result, flags=re.IGNORECASE | re.DOTALL)
    result = result.strip()
    if "Words:" in result:
        result = result.split("Words:")[-1].strip()
    if "Signs:" in result:
        result = result.split("Signs:")[-1].strip()
    if "Sentence:" in result:
        result = result.split("Sentence:")[-1].strip()
    if "English:" in result:
        result = result.split("English:")[-1].strip()
    result = re.sub(r"^(answer|output|assistant)\s*:\s*", "", result, flags=re.IGNORECASE)
    result = result.strip().strip('"').strip("'").strip()
    if "\n" in result:
        result = next((line.strip() for line in result.splitlines() if line.strip()), result)
    sentence_match = re.match(r"(.+?[.!?])(?:\s|$)", result)
    if sentence_match:
        result = sentence_match.group(1).strip()
    return result


def _choose_subject(tokens: list[str]) -> str | None:
    for token in tokens:
        if token in SUBJECT_WORDS:
            return token
    return None


def _subject_phrase(subject: str | None) -> str:
    if not subject:
        return "I"
    return subject.capitalize()


def _be_verb(subject: str | None) -> str:
    if subject in {"you", "we"}:
        return "are"
    if subject == "i" or subject is None:
        return "am"
    return "is"


def _main_verb(subject: str | None, infinitive: str) -> str:
    if subject in {None, "i", "you", "we"}:
        return infinitive
    if infinitive.endswith("y") and infinitive[-2:] != "ay":
        return infinitive[:-1] + "ies"
    if infinitive.endswith(("s", "sh", "ch", "x", "z", "o")):
        return infinitive + "es"
    return infinitive + "s"


def _object_phrase(token: str) -> str:
    return ARTICLE_OBJECTS.get(token, token)


def _greeting_override(cleaned: list[str]) -> str | None:
    lowered = [token.lower() for token in cleaned]
    if lowered == ["hello"]:
        return "Hello."

    if "hello" not in lowered or len(lowered) != 2:
        return None

    target = next((token for token in lowered if token != "hello"), None)
    if target not in GREETING_TARGETS:
        return None

    phrase = GREETING_TARGETS[target]
    if phrase.startswith("to "):
        return f"Hello {phrase}."
    return f"Hello, {phrase}."


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

    # Normalise: strip whitespace, uppercase, drop empty tokens
    normalized = [w.strip().upper() for w in words if w.strip()]
    if not normalized:
        raise ValueError("[generate_sentence] No valid words after cleaning.")

    cleaned = clean_buffer(normalized)
    if not cleaned:
        return ""

    greeting = _greeting_override(cleaned)
    if greeting is not None:
        print(f"[generate_sentence] Input : {words}")
        print(f"[generate_sentence] Clean : {cleaned}")
        print(f"[generate_sentence] Greeting override: {greeting}")
        return greeting

    override_key = tuple(cleaned)
    if override_key in _EXACT_OVERRIDES:
        result = _EXACT_OVERRIDES[override_key]
        print(f"[generate_sentence] Input : {words}")
        print(f"[generate_sentence] Clean : {cleaned}")
        print(f"[generate_sentence] Override: {result}")
        return result

    signs_literal = ", ".join(cleaned)
    if "PLEASE" in cleaned:
        examples = select_examples(cleaned, k=2, required_sign="PLEASE")
    else:
        examples = select_examples(cleaned, k=1)

    example_block = ""
    for example_signs, example_sentence in examples:
        example_literal = ", ".join(example_signs)
        example_block += f"Signs: {example_literal}\nEnglish: {example_sentence}\n\n"

    prompt = (
        "Translate ASL signs into one natural English sentence.\n"
        "Signs are ALL-CAPS, may be in ASL order, and may omit articles, auxiliaries, and prepositions.\n"
        "Rules: include every sign's meaning, add only the grammar needed for fluent English, and never add people, events, objects, or reasons not supported by the signs.\n"
        "If HELLO appears with a person word such as BOY or GIRL, prefer a direct greeting like 'Hello, boy.' instead of narrating who is speaking.\n\n"
        f"{example_block}"
        f"Signs: {signs_literal}\n"
        "English: "
    )

    print(f"[generate_sentence] Input : {words}")
    print(f"[generate_sentence] Clean : {cleaned}")
    print(f"[generate_sentence] Examples: {examples}")
    result = _clean_model_text(generate(prompt, max_length=40, system_prompt=SYSTEM_PROMPT))

    # Qwen-only path: surface weak generations instead of switching to a rule engine.
    if not result or len(result.split()) < 2:
        raise RuntimeError("[generate_sentence] Qwen returned unusable output.")

    # Ensure ends with punctuation
    if result[-1] not in ".!?":
        result += "."

    # Capitalise first character
    result = result[0].upper() + result[1:]

    print(f"[generate_sentence] Output: {result}")
    return result


# Backwards-compatible alias expected by `main.py`.
def generate_sentence(words: list) -> str:
    return words_to_sentence(words)


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
    print("  Sentence Model  Sign → Sentence Test")
    print("=" * 56)
    for words in tests:
        sentence = words_to_sentence(words)
        print(f"  IN : {words}")
        print(f"  OUT: {sentence}\n")
