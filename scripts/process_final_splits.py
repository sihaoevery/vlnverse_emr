"""
Process VLN-PE final_splits JSON files against R2R-preproc vocab.

Two modes (select via --vocab):
  r2r    — keep R2R preproc vocab (2504) unchanged.
           out-of-R2R words tokenize to <unk>.
           Output: vlnverse_r2r/...
           Embedding rows are copied from the R2R preprocessed embedding matrix.
           This mode keeps the official pretrained CMA/Seq2Seq token space.
           Expected raw token-level UNK rate, punctuation included: ~13% coarse / ~5.7% fine
           (by design, not a bug).

  extend — append GloVe-covered VLNverse words after the R2R prefix.
           Output: vlnverse/... with extended embeddings.
           Rows 0..2503 remain index-aligned and value-identical to R2R.
           If new words are appended, the embedding matrix is larger than the
           official 2504-row pretrained checkpoints; direct checkpoint loading 
           requires resized/partial embedding loading logic.

Shared behavior:
- Base vocab: R2R_VLNCE_v1-3_preprocessed (2504 entries). 
- Base embeddings copied VERBATIM from preproc's embeddings.json.gz for rows 0..2503.
- Guard against <pad>/<unk> literals sneaking into new_words.
- Geodesic: test must have no reference_path (asserted); non-test computes it.

Input:  data/vln_pe/raw_data/final_splits/{coarse,fine}_{train,val,val_unseen,test}.json.gz
Output: data/vln_pe/raw_data/vlnverse/{coarse,fine}/{split}/{split}.json.gz
        data/vln_pe/raw_data/vlnverse/mixed_splits/{split}/{split}.json.gz
        data/vln_pe/raw_data/vlnverse/embeddings.json.gz
        data/vln_pe/raw_data/vlnverse_r2r/{coarse,fine}/{split}/{split}.json.gz
        data/vln_pe/raw_data/vlnverse_r2r/mixed_splits/{split}/{split}.json.gz
        data/vln_pe/raw_data/vlnverse_r2r/embeddings.json.gz

Note:
- Reported UNK rates are raw token-level rates using the same tokenizer as this script.
They include punctuation tokens unless separately filtered.
"""

import argparse
import gzip
import json
import os
import sys

import numpy as np
from nltk.tokenize import word_tokenize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from internnav.utils.glove_embedding import pad_list, sentence_preprocess

FILE_MAP = {
    "coarse_train.json.gz": ("coarse", "train"),
    "coarse_val.json.gz": ("coarse", "val_seen"),
    "coarse_val_unseen.json.gz": ("coarse", "val_unseen"),
    "coarse_test.json.gz": ("coarse", "test"),
    "fine_train.json.gz": ("fine", "train"),
    "fine_val.json.gz": ("fine", "val_seen"),
    "fine_val_unseen.json.gz": ("fine", "val_unseen"),
    "fine_test.json.gz": ("fine", "test"),
}

MIXED_SPLITS = ("train", "val_seen", "val_unseen", "test")

RESERVED_TOKENS = {"<pad>", "<unk>"}

INPUT_DIR = "data/vln_pe/raw_data/final_splits"
OUTPUT_BASE_BY_VOCAB = {
    "extend": "data/vln_pe/raw_data/vlnverse",
    "r2r": "data/vln_pe/raw_data/vlnverse_r2r",
}
BASE_VOCAB_PATH = "data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz"
BASE_EMBEDDING_PATH = "data/datasets/R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz"
GLOVE_PATH = "data/glove/glove.6B.50d.txt"


def load_base_vocab(vocab_path):
    with gzip.open(vocab_path, "rt", encoding="utf-8") as f:
        vocab = json.load(f)["instruction_vocab"]
    assert vocab["word_list"][0] == "<pad>", "base vocab[0] must be <pad>"
    assert vocab["word_list"][1] == "<unk>", "base vocab[1] must be <unk>"
    assert vocab["PAD_INDEX"] == 0 and vocab["UNK_INDEX"] == 1
    assert vocab["num_vocab"] == len(vocab["word_list"])
    return vocab


def load_base_embeddings(path, expected_size):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        emb = json.load(f)
    assert len(emb) == expected_size, (
        f"base embedding rows ({len(emb)}) must match base vocab size ({expected_size})"
    )
    return emb


def load_glove_vectors(glove_path):
    vectors = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.rstrip().split(" ")
            vectors[vals[0]] = [float(x) for x in vals[1:]]
    return vectors


def compute_geodesic_distance(reference_path):
    pts = np.array(reference_path)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def flatten_instruction_texts(inst_text):
    if isinstance(inst_text, dict):
        return list(inst_text.values())
    return [inst_text]


def collect_all_instructions(input_dir):
    instructions = []
    for fname in FILE_MAP:
        input_path = os.path.join(input_dir, fname)
        if not os.path.exists(input_path):
            print(f"  [WARN] {input_path} not found, skipping for vocab collection")
            continue
        with gzip.open(input_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for episode in data["episodes"]:
            instructions.extend(
                flatten_instruction_texts(episode["instruction"]["instruction_text"])
            )
    return instructions


def build_extended_vocab(base_vocab, all_instructions, glove_vectors):
    r2r_word_list = list(base_vocab["word_list"])
    r2r_word_set = set(r2r_word_list)

    new_words = set()
    for text in all_instructions:
        tokens = word_tokenize(sentence_preprocess(text.lower()))
        for w in tokens:
            if w in RESERVED_TOKENS:
                continue
            if w not in r2r_word_set and w in glove_vectors:
                new_words.add(w)

    new_words_sorted = sorted(new_words)
    extended_word_list = r2r_word_list + new_words_sorted

    vocab = dict(base_vocab)
    vocab.update({
        "word_list": extended_word_list,
        "word2idx_dict": {w: i for i, w in enumerate(extended_word_list)},
        "itos": extended_word_list,
        "stoi": {w: i for i, w in enumerate(extended_word_list)},
        "num_vocab": len(extended_word_list),
    })
    assert vocab["PAD_INDEX"] == 0 and vocab["UNK_INDEX"] == 1
    assert vocab["word_list"][: len(r2r_word_list)] == r2r_word_list, \
        "R2R prefix must be preserved"
    return vocab, new_words_sorted


def build_vocab(vocab_mode, base_vocab, all_instructions, glove_vectors):
    """Dispatch to the right vocab-building routine based on mode.

    r2r:    return preproc vocab unchanged (no new words).
    extend: append GloVe-covered VLNverse words after the R2R prefix.
    """
    if vocab_mode == "r2r":
        return dict(base_vocab), []
    if vocab_mode == "extend":
        return build_extended_vocab(base_vocab, all_instructions, glove_vectors)
    raise ValueError(f"Unknown vocab mode: {vocab_mode!r}")


def build_embedding_matrix(vocab, base_embeddings, base_size, glove_vectors):
    """Rows [0, base_size) copied verbatim from preproc (bit-exact <pad>, <unk>, R2R).
    Rows [base_size, num_vocab) are fresh GloVe lookups for appended VLNverse words."""
    assert len(base_embeddings) == base_size
    rows = [list(r) for r in base_embeddings]
    for word in vocab["word_list"][base_size:]:
        assert word in glove_vectors, f"new word {word!r} missing from GloVe"
        rows.append(list(glove_vectors[word]))
    assert len(rows) == vocab["num_vocab"]
    # Bit-exact check on preserved prefix
    for i in range(base_size):
        assert rows[i] == base_embeddings[i], f"row {i} drifted from base embedding"
    return rows


def tokenize_instruction(text, vocab):
    tokens = word_tokenize(sentence_preprocess(text.lower()))
    token_ids = [vocab["stoi"].get(w, vocab["UNK_INDEX"]) for w in tokens]
    return pad_list(token_ids)


def process_file(input_path, output_path, vocab, is_test):
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    for episode in data["episodes"]:
        if is_test:
            if "reference_path" in episode and episode["reference_path"]:
                raise ValueError(
                    f"Test episode {episode.get('episode_id')} in {input_path} has a "
                    "reference_path; expected test splits to have none."
                )
        else:
            if "reference_path" not in episode or not episode["reference_path"]:
                raise ValueError(
                    f"Non-test episode {episode.get('episode_id')} in {input_path} "
                    "is missing reference_path."
                )
            episode.setdefault("info", {})
            episode["info"]["geodesic_distance"] = compute_geodesic_distance(
                episode["reference_path"]
            )

        inst_text = episode["instruction"]["instruction_text"]
        if isinstance(inst_text, dict):
            episode["instruction"]["instruction_tokens"] = {
                variant: tokenize_instruction(text, vocab)
                for variant, text in inst_text.items()
            }
        else:
            episode["instruction"]["instruction_tokens"] = tokenize_instruction(
                inst_text, vocab
            )

    new_data = {"episodes": data["episodes"], "instruction_vocab": vocab}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump(new_data, f)
    return len(data["episodes"])


def write_mixed_split(output_base, split, vocab):
    """Combine processed coarse + fine splits into the historical mixed_splits layout.

    Existing VLNverse mixed_splits are ordered as all coarse episodes followed by
    all fine episodes. Keep that order so downstream key filtering is stable.
    """
    mixed_episodes = []
    counts = {}
    for data_type in ("coarse", "fine"):
        input_path = os.path.join(output_base, data_type, split, f"{split}.json.gz")
        if not os.path.exists(input_path):
            print(f"  [SKIP] mixed/{split}: missing {input_path}")
            return None
        with gzip.open(input_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        mixed_episodes.extend(data["episodes"])
        counts[data_type] = len(data["episodes"])

    output_path = os.path.join(output_base, "mixed_splits", split, f"{split}.json.gz")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        json.dump({"episodes": mixed_episodes, "instruction_vocab": vocab}, f)
    return output_path, counts, len(mixed_episodes)


def report_unk_rate(input_dir, vocab):
    print("UNK rate per split (token-level, across all instruction variants):")
    for fname, (data_type, split) in FILE_MAP.items():
        path = os.path.join(input_dir, fname)
        if not os.path.exists(path):
            continue
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        total = 0
        unk = 0
        for ep in d["episodes"]:
            for text in flatten_instruction_texts(ep["instruction"]["instruction_text"]):
                for w in word_tokenize(sentence_preprocess(text.lower())):
                    total += 1
                    if w not in vocab["stoi"]:
                        unk += 1
        pct = 100 * unk / total if total else 0.0
        print(f"  {data_type}/{split}: {unk}/{total} = {pct:.3f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--vocab",
        choices=["r2r", "extend"],
        required=True,
        help="r2r: keep 2504 vocab unchanged (OOV -> <unk>). "
             "extend: append GloVe-covered VLNverse words after R2R prefix.",
    )
    args = parser.parse_args()
    vocab_mode = args.vocab
    output_base = OUTPUT_BASE_BY_VOCAB[vocab_mode]

    print(f"=== vocab mode: {vocab_mode} ===")
    print(f"Output: {output_base}")
    if vocab_mode == "r2r":
        print("[r2r mode] Vocab unchanged at 2504 entries; out-of-R2R words tokenize to <unk>.")
        print("[r2r mode] Expected UNK rate: ~13% coarse / ~5.7% fine — this is by design.")
    print()

    print("Loading base vocab + embeddings (R2R preproc)...")
    base_vocab = load_base_vocab(BASE_VOCAB_PATH)
    base_size = base_vocab["num_vocab"]
    base_embeddings = load_base_embeddings(BASE_EMBEDDING_PATH, base_size)
    print(f"  R2R preproc vocab size: {base_size}")
    print(f"  R2R preproc embedding: {len(base_embeddings)} x {len(base_embeddings[0])}")

    if vocab_mode == "extend":
        print("Loading GloVe vectors...")
        glove_vectors = load_glove_vectors(GLOVE_PATH)
        print(f"  GloVe entries: {len(glove_vectors)}")
        print("Collecting instructions from all splits (fine + coarse, all variants)...")
        instructions = collect_all_instructions(INPUT_DIR)
        print(f"  Total instructions: {len(instructions)}")
    else:
        glove_vectors = {}
        instructions = []

    print(f"Building vocab (mode={vocab_mode})...")
    vocab, new_words = build_vocab(vocab_mode, base_vocab, instructions, glove_vectors)
    print(f"  R2R words preserved: {base_size}")
    print(f"  New words appended:  {len(new_words)}")
    print(f"  Total vocab size:    {vocab['num_vocab']}")

    if new_words:
        print("Building embedding matrix (verbatim R2R prefix + GloVe for new words)...")
    else:
        print("Building embedding matrix (verbatim R2R prefix; no new rows)...")

    embeddings = build_embedding_matrix(vocab, base_embeddings, base_size, glove_vectors)

    os.makedirs(output_base, exist_ok=True)
    emb_path = os.path.join(output_base, "embeddings.json.gz")
    with gzip.open(emb_path, "wt", encoding="utf-8") as f:
        json.dump(embeddings, f)
    print(f"  Wrote {emb_path}: {len(embeddings)} x {len(embeddings[0])}")

    print("Processing splits...")
    for fname, (data_type, split) in FILE_MAP.items():
        input_path = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(input_path):
            print(f"  [SKIP] {input_path} not found")
            continue
        output_path = os.path.join(output_base, data_type, split, f"{split}.json.gz")
        count = process_file(input_path, output_path, vocab, is_test=(split == "test"))
        print(f"  {fname} -> {data_type}/{split}: {count} episodes")

    print("Writing mixed_splits (coarse + fine)...")
    for split in MIXED_SPLITS:
        result = write_mixed_split(output_base, split, vocab)
        if result is None:
            continue
        output_path, counts, total = result
        print(
            f"  mixed/{split}: coarse={counts['coarse']} + "
            f"fine={counts['fine']} -> {total} episodes ({output_path})"
        )

    print()
    report_unk_rate(INPUT_DIR, vocab)
    print("\nDone.")


if __name__ == "__main__":
    main()
