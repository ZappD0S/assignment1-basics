from collections import defaultdict
from multiprocessing import Pool

import regex


def find_special_tokens(text: str):
    pattern = regex.compile(r"<\|\w+\|>")

    special_tokens = pattern.findall(text)
    return list(set(special_tokens))


def split_into_chunks(text: str, special_tokens: list[str]) -> list[str]:
    pattern = "|".join(regex.escape(tok) for tok in special_tokens)
    chunks = regex.split(pattern, text)
    return chunks


# here text is a single chunk
def pretokenize(text: str) -> dict[str, int]:

    pattern = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    pretok_counts = defaultdict(lambda: 0)

    for tok in pattern.finditer(text):
        pretok_counts[tok.group()] += 1

    # sorted_occurs = sorted(pretok_counts.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_occurs[:5])

    return dict(pretok_counts)


def init_positional_dicts(
    pretoks: list[bytes],
) -> tuple[dict[tuple[bytes, bytes], set[tuple[int, int]]], dict[tuple[int, int], bytes]]:
    pair2pos: dict[tuple[bytes, bytes], set[tuple[int, int]]] = {}
    pos2tok: dict[tuple[int, int], bytes] = {}

    for i, pretok in enumerate(pretoks):
        for j in range(len(pretok) - 1):
            b1 = pretok[j : j + 1]
            b2 = pretok[j + 1 : j + 2]

            pair2pos.setdefault((b1, b2), set()).add((i, j))
            pos2tok[(i, j)] = b1

        pos2tok[(i, len(pretok) - 1)] = pretok[len(pretok) - 1 :]

    return pair2pos, pos2tok


# TODO: this can me made more efficient because it's not necessary to recompute from
# scratch all the counts, just update based on last merge
def find_most_common_pairs(
    pretok_counts: dict[bytes, int],
    pretoks: list[bytes],
    pair2pos: dict[tuple[bytes, bytes], set[tuple[int, int]]],
) -> tuple[list[tuple[bytes, bytes]], int]:
    most_common: list[tuple[bytes, bytes]] = []
    highest = 0

    for pair, pos_list in pair2pos.items():
        count = 0
        for pos in pos_list:
            pretok_idx, _ = pos
            pretok = pretoks[pretok_idx]
            pretok_count = pretok_counts[pretok]

            count += pretok_count

        if count < highest:
            continue

        if count > highest:
            most_common = []
            highest = count

        most_common.append(pair)

    return most_common, highest


def bpe_tokenize(
    pretok_counts: dict[bytes, int],
    vocab_size: int = 0,
    special_tokens: list[bytes] = [],
):
    pretoks = list(pretok_counts.keys())

    pair2pos, pos2tok = init_positional_dicts(pretoks)

    merges: list[tuple[bytes, bytes]] = []

    # map token id to token bytes
    vocab: dict[int, bytes] = {i: tok for i, tok in enumerate(special_tokens)}

    for i in range(256):
        vocab[i + len(special_tokens)] = bytes([i])

    while True:
        most_common, _ = find_most_common_pairs(pretok_counts, pretoks, pair2pos)

        if not most_common:
            break

        tok1, tok2 = max(most_common)
        merged_tok = tok1 + tok2

        tok_pos_list = pair2pos[(tok1, tok2)]

        for pos in sorted(tok_pos_list):
            pretok_idx, tok_idx = pos

            # if positions of either of the pair toks is not in pos2tok, skip
            if (pretok_idx, tok_idx) not in pos2tok or (pretok_idx, tok_idx + len(tok1)) not in pos2tok:
                # this means either was already merged by  previous iteration
                continue

            # search previous token, it it exists
            prev_tok = None
            for prev_tok_idx in range(tok_idx - 1, -1, -1):
                if (pretok_idx, prev_tok_idx) in pos2tok:
                    prev_tok = pos2tok[(pretok_idx, prev_tok_idx)]
                    break

            if prev_tok is not None:
                prev_tok_pos_list = pair2pos[(prev_tok, tok1)]
                # find the matching positon and drop it
                prev_tok_pos_list.remove((pretok_idx, prev_tok_idx))
                # add it to the new entry that we add to pair2pos
                pair2pos.setdefault((prev_tok, merged_tok), set()).add((pretok_idx, prev_tok_idx))

            # no loop needed for the next token
            # only check if next is present, otherwise skip
            if (pretok_idx, tok_idx + len(merged_tok)) in pos2tok:
                next_tok = pos2tok[(pretok_idx, tok_idx + len(merged_tok))]
                next_tok_pos_list = pair2pos[(tok2, next_tok)]
                next_tok_pos_list.remove((pretok_idx, tok_idx + len(tok1)))
                pair2pos.setdefault((merged_tok, next_tok), set()).add((pretok_idx, tok_idx))

            del pos2tok[(pretok_idx, tok_idx + len(tok1))]
            pos2tok[(pretok_idx, tok_idx)] = merged_tok

        del pair2pos[(tok1, tok2)]
        merges.append((tok1, tok2))
        vocab[len(vocab)] = merged_tok

        if vocab_size > 0 and len(vocab) >= vocab_size:
            break

    return vocab, merges


def parallel_pretokenize(chunks: list[str]) -> dict[str, int]:
    pretok_counts: dict[str, int] = defaultdict(lambda: 0)
    with Pool() as pool:
        for chunk_pretok_counts in pool.imap_unordered(pretokenize, chunks):
            for pretok, count in chunk_pretok_counts.items():
                pretok_counts[pretok] += count

    return pretok_counts


if __name__ == "__main__":
    with open("../data/TinyStoriesV2-GPT4-valid.txt") as f:
        text = f.read()

    special_tokens = find_special_tokens(text)
    chunks = split_into_chunks(text, special_tokens)

    pretok_counts = parallel_pretokenize(chunks)

    pretok_counts = {pretok.encode("utf-8"): count for pretok, count in pretok_counts.items()}
    vocab, merges = bpe_tokenize(
        pretok_counts,
        special_tokens=[tok.encode("utf-8") for tok in special_tokens],
    )
    # print(vocab)
    # print(merges)
