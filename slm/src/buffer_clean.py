_FILLER_SIGNS = {"RIGHT", "MORE", "FINE", "FINISH", "OKAY", "YES", "NO"}


def clean_buffer(signs: list[str]) -> list[str]:
    """
    Remove filler signs and collapse consecutive repeats.

    The sentence model becomes unstable on long, noisy buffers, so keep only
    the earliest cleaned signs that are likely to carry semantic meaning.
    """
    filtered = [sign for sign in signs if sign not in _FILLER_SIGNS]

    deduped: list[str] = []
    for sign in filtered:
        if not deduped or sign != deduped[-1]:
            deduped.append(sign)

    return deduped[:6]
