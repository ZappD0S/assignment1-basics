from bisect import bisect_left, insort_left
from collections.abc import Iterable, Iterator
from multiprocessing.dummy import Pool
from typing import Self

from .bpe_train import pretokenize, split_into_chunks


def parallel_pretokenize(chunks: list[str], special_tokens: set[str]) -> list[str]:

    def pretokenize_skip_special(chunk):
        if chunk in special_tokens:
            return [chunk]

        return map(lambda x: x.group(), pretokenize(chunk))

    with Pool() as pool:
        matches_iterables = pool.map(pretokenize_skip_special, chunks)

    return [match for matches in matches_iterables for match in matches]


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
        chunk_size: int = 1_000_000,
    ):
        self.merges = merges
        self.merges_set: set[tuple[bytes, bytes]] = set(merges)
        self.chunk_size = chunk_size
        special_tokens = [] if special_tokens is None else special_tokens
        self.special_tokens = set(special_tokens)

        self.vocab = vocab
        self.inverted_vocab = {tok: tok_id for tok_id, tok in vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:

        special_tokens = self.special_tokens
        if special_tokens:
            chunks = split_into_chunks(text, special_tokens)
            pretoks = parallel_pretokenize(chunks, special_tokens)
        else:
            pretoks = [match.group() for match in pretokenize(text)]

        return self._encode_from_pretoks(pretoks)

    def _encode_from_pretoks(self, pretoks: list[str]) -> list[int]:
        pretoks_bytes = []
        pair2pos: dict[tuple[bytes, bytes], dict[int, list[int]]] = {}

        special_tokens = self.special_tokens
        for i, pretok in enumerate(pretoks):
            is_special = pretok in special_tokens

            pretok = pretok.encode("utf-8")

            if is_special:
                pretoks_bytes.append([pretok])
                continue

            pretok_bytes = [pretok[i : i + 1] for i in range(len(pretok))]
            pretoks_bytes.append(pretok_bytes)

            for j in range(len(pretok) - 1):
                b1 = pretok[j : j + 1]
                b2 = pretok[j + 1 : j + 2]

                pair2pos.setdefault((b1, b2), {}).setdefault(i, []).append(j)

        for to_merge in self.merges:
            tok1, tok2 = to_merge
            merged_tok = tok1 + tok2

            if to_merge not in pair2pos:
                continue

            pair_pos_map = pair2pos[to_merge]

            for pretok_idx, pair_pos_lst in pair_pos_map.items():
                pretok_bytes = pretoks_bytes[pretok_idx]

                # TODO: for now we compute this dinamically everytime, but it could be stored and updated..
                tok_pos_lst = []
                count = 0
                for tok_bytes in pretok_bytes:
                    tok_pos_lst.append(count)
                    count += len(tok_bytes)

                # TODO: not sure if reversed is needed or not
                for tok_pos in reversed(pair_pos_lst):
                    pos_idx = bisect_left(tok_pos_lst, tok_pos)
                    if pos_idx > 0:
                        prev_tok_pos = tok_pos_lst[pos_idx - 1]
                        prev_tok = pretok_bytes[pos_idx - 1]
                        prev_pos_list = pair2pos[(prev_tok, tok1)][pretok_idx]
                        del prev_pos_list[bisect_left(prev_pos_list, prev_tok_pos)]

                        new_prev_pos_list = pair2pos.setdefault((prev_tok, merged_tok), {}).setdefault(pretok_idx, [])
                        insort_left(new_prev_pos_list, prev_tok_pos)

                    if len(pretok_bytes) >= 2 and pos_idx < len(pretok_bytes) - 2:
                        next_tok = pretok_bytes[pos_idx + 2]
                        next_pos_list = pair2pos[(tok2, next_tok)][pretok_idx]

                        tok2_pos = tok_pos_lst[pos_idx + 1]
                        del next_pos_list[bisect_left(next_pos_list, tok2_pos)]

                        new_next_pos_list = pair2pos.setdefault((merged_tok, next_tok), {}).setdefault(pretok_idx, [])
                        insort_left(new_next_pos_list, tok_pos)

                    pretok_bytes[pos_idx : pos_idx + 2] = [merged_tok]

        # map token bytes ids
        text_ids = []
        inverted_vocab = self.inverted_vocab
        for pretok_bytes in pretoks_bytes:
            pretok_ids = [inverted_vocab[tok] for tok in pretok_bytes]
            text_ids.extend(pretok_ids)

        return text_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        special_tokens = self.special_tokens

        # TODO: maybe we can make this faster by first loading batches of lines and then encoding them all at once?
        for line in iterable:
            chunks = split_into_chunks(line, special_tokens)
            pretoks = parallel_pretokenize(chunks, special_tokens)

            if pretoks:
                yield from self._encode_from_pretoks(pretoks)

    def decode(self, ids: list[int]) -> str:
        # TODO: special tokens!
        vocab = self.vocab

        toks = [vocab[tok_id] for tok_id in ids]
        text = b"".join(toks).decode("utf-8", errors="replace")

        return text
