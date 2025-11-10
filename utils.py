import json
from typing import List, Dict, Tuple


# Utility to convert entity annotations into token-level BIO labels is intentionally
# simple and may need adaptation depending on your annotation format.


def entities_to_bio(text: str, entities: List[Dict], tokenizer) -> Tuple[List[str], List[int]]:
    """
    Convert character-span entities into token-level BIO labels.
    entities: list of {"start": int, "end": int, "label": str}
    Returns tokens and label ids aligned to tokenizer output (word_ids).
    """
    # Tokenize with offsets
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True)
    offsets = enc.pop('offset_mapping')
    labels = ['O'] * len(offsets)
    for ent in entities:
        s, e, lab = ent['start'], ent['end'], ent['label']
        # mark tokens that overlap
        for i, (a,b) in enumerate(offsets):
            if b <= s or a >= e:
                continue
            if labels[i] == 'O':
                labels[i] = 'B-' + lab
            else:
                labels[i] = 'I-' + lab
    return enc, labels