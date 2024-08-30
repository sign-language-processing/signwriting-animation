import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from signwriting_animation.translation.utils import factor_signwriting

csv.field_size_limit(int(1e6))


def create_splits(poses, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_dir = output_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    src_factors_files = [open(train_dir / f"source_{i}.txt", "w", encoding="utf-8") for i in range(5)]
    trg_factors_files = None

    default_source_factors = factor_signwriting("M518x518S2ff00482x483")

    with open(poses, 'r', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            codes = row["codes"].split(" ")
            length = int(row["length"])
            source_features = int(len(codes) / length)

            if trg_factors_files is None:
                trg_factors_files = [open(train_dir / f"target_{i}.txt", "w", encoding="utf-8")
                                     for i in range(source_features)]

            for i, factor_file in enumerate(trg_factors_files):
                factor_file.write(" ".join(codes[i::source_features]) + "\n")

            for i, factor_file in enumerate(src_factors_files):
                factor_file.write(default_source_factors[i] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--poses', type=str, help='Path to processed poses file')
    parser.add_argument('--output', type=str, help='Path to output directory')
    args = parser.parse_args()

    create_splits(args.poses, args.output)
