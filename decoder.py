from typing import List, Tuple
from torch import Tensor

class TokenDecoder:
    def __init__(self):
        self.specials_first = ('[E]',)
        self.specials_last = ('[B]', '[P]')
        self.charset = (
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
            'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!', '"', '#', '$', '%', '&', "'", '(', ')',
            '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' '
        )
        self.itos = self.specials_first + self.charset + self.specials_last
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self.itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    def filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        ids = ids.tolist()
        eos_id, bos_id, pad_id = [self.stoi[s] for s in self.specials_first + self.specials_last]

        try:
            eos_idx = ids.index(eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]
        return probs, ids

    def decode(self, token_dists: Tensor, raw: bool = False):
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self.filter(probs, ids)
            tokens = self.ids2tok(ids)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs