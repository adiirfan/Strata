"""Cut the MTP draft head's token subset to the low token ids (tools/rtx5080/draft_vocab_cut.py).

    python tools/rtx5080/draft_vocab_cut.py <mtp>/draft_vocab.bin out.bin [--below 80000] [--from 247900]

Keeps, in their original order, the ids of the subset below --below (the tokenizer's earlier, more frequent merges)
plus the special tokens from --from up. Pass the result as --mtp-draft-vocab out.bin.

The draft head only proposes tokens, so the output is unchanged either way; a smaller subset makes each draft
step cheaper. Text in scripts the cut drops (most CJK, Cyrillic) drafts less.
"""
from __future__ import annotations

import argparse
from array import array


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--below", type=int, default=80000)
    ap.add_argument("--from", dest="from_id", type=int, default=247900)
    a = ap.parse_args()
    ids = array("i")
    ids.frombytes(open(a.src, "rb").read())
    keep = array("i", [i for i in ids if i < a.below or i >= a.from_id])
    open(a.out, "wb").write(keep.tobytes())
    print(f"{len(keep)} of {len(ids)} tokens kept -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
