EXAMPLES: list[tuple[list[str], str]] = [
    (["THANK YOU"], "Thank you."),
    (["THANK YOU", "HELP"], "Thank you for your help."),
    (["DOCTOR", "WHERE"], "Where is the doctor?"),
    (["TOILET", "WHERE"], "Where is the toilet?"),
    (["PHONE", "PLEASE"], "Phone, please."),
    (["WATER", "PLEASE"], "Water, please."),
    (["FOOD", "PLEASE"], "Food, please."),
    (["MEDICINE", "PLEASE"], "Medicine, please."),
    (["I", "SICK", "DOCTOR", "NEED"], "I am sick and need a doctor."),
    (["I", "PAIN"], "I am in pain."),
    (["EMERGENCY"], "This is an emergency!"),
    (["MOTHER", "SICK"], "My mother is sick."),
    (["HUNGRY", "I", "FOOD", "WANT"], "I am hungry and want food."),
    (["HOME", "GO", "TODAY"], "Go home today."),
]

_LOW_WEIGHT = {"PLEASE", "I", "YOU", "TODAY", "NOW"}
_HIGH_WEIGHT = {"THANK YOU", "DOCTOR", "TOILET", "PHONE", "EMERGENCY", "MEDICINE", "PAIN", "SICK"}


def _sign_weight(sign: str) -> float:
    if sign in _HIGH_WEIGHT:
        return 2.0
    if sign in _LOW_WEIGHT:
        return 0.5
    return 1.0


def _score(example_signs: list[str], input_signs: list[str]) -> float:
    example_set = set(example_signs)
    input_set = set(input_signs)
    if not example_set:
        return 0.0

    intersection = example_set & input_set
    union = example_set | input_set

    intersection_weight = sum(_sign_weight(sign) for sign in intersection)
    union_weight = sum(_sign_weight(sign) for sign in union)
    weighted_jaccard = intersection_weight / union_weight if union_weight else 0.0
    exact_bonus = 1.0 if example_set == input_set else 0.0
    return weighted_jaccard + exact_bonus


def select_examples(input_signs: list[str], k: int = 1) -> list[tuple[list[str], str]]:
    scored: list[tuple[float, list[str], str]] = []
    for example_signs, sentence in EXAMPLES:
        score = _score(example_signs, input_signs)
        if score > 0:
            scored.append((score, example_signs, sentence))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(signs, sentence) for _, signs, sentence in scored[:k]]
