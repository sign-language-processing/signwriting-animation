import argparse
from json import dumps
from pathlib import Path


def vocab_from_tokens(tokens: list[str]):
    vocab = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    # pad to multiple of 8
    while len(vocab) % 8 != 0:
        vocab[f"<pad{len(vocab)}>"] = len(vocab)
    return vocab


def create_vocab_files(vocab_dir: str):
    vocab_path = Path(vocab_dir)
    vocab_path.mkdir(parents=True, exist_ok=True)

    sw_symbols = ["S" + hex(i)[2:] + hex(j)[2:] for i in range(0x10, 0x38 + 1) for j in range(0x0, 0xf + 1)]
    sw_symbols.remove("S38c")
    sw_symbols.remove("S38d")
    sw_symbols.remove("S38e")
    sw_symbols.remove("S38f")
    positions = ["p" + str(p) for p in range(250, 750)]
    src_tokens = [
        ["B", "L", "M", "R"] + sw_symbols,
        ["c0", "c1", "c2", "c3", "c4", "c5"],
        ["r" + hex(j)[2:] for j in range(0x0, 0xf + 1)],
        positions,
        positions
    ]

    for i in range(5):
        with open(vocab_path / f"vocab.src.{i}.json", "w", encoding="utf-8") as f:
            vocab = vocab_from_tokens(src_tokens[i])
            f.write(dumps(vocab, indent=2))

    for i in range(8):
        vocab = vocab_from_tokens([str(i) for i in range(1000)])
        with open(vocab_path / f"vocab.trg.{i}.json", "w", encoding="utf-8") as f:
            f.write(dumps(vocab, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_dir', type=str, help='Directory to store vocab files')
    args = parser.parse_args()

    create_vocab_files(args.vocab_dir)
