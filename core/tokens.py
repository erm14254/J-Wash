"""Tokenizer lookup helpers for the token editor.

A word the tokenizer encodes as ONE id is a "single" candidate (editable as a
plain rule). Words that split into several ids ("Ametista" → " Amet"+"ista")
are returned as "splits": the editor builds a composite direction from the
pieces instead of rejecting the word.
"""

# at most this many distinct splits are returned (one per casing variant that
# tokenizes differently), and only for reasonably short sequences
MAX_SPLITS = 6
MAX_PIECES = 16


def lookup_variants(q):
    """Casing/spacing variants tried for a query, most useful first (the
    original spelling, then the leading-space form generation actually emits)."""
    ordered = (q, " " + q, q.lower(), " " + q.lower(),
               q.capitalize(), " " + q.capitalize(), q.upper(), " " + q.upper())
    return list(dict.fromkeys(ordered))


def token_candidates(tokenizer, q):
    """``(singles, splits)`` for a query string.

    singles: ``[{id, str}]`` — variants that encode to exactly one token.
    splits:  ``[{ids, pieces, str}]`` — multi-token variants (deduplicated by
    id sequence), for composite rules.
    """
    singles = {}
    splits = {}
    for variant in lookup_variants(q):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            if ids[0] not in singles:
                singles[ids[0]] = tokenizer.decode([ids[0]])
        elif 1 < len(ids) <= MAX_PIECES:
            key = tuple(ids)
            if key not in splits and len(splits) < MAX_SPLITS:
                splits[key] = {
                    "ids": [int(t) for t in ids],
                    "pieces": [tokenizer.decode([int(t)]) for t in ids],
                    "str": tokenizer.decode(ids),
                }
    return list({"id": tid, "str": s} for tid, s in singles.items()), list(splits.values())
