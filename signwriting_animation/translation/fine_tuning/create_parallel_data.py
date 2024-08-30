import argparse
import csv
from pathlib import Path

from signwriting.formats.swu_to_fsw import swu2fsw
from tqdm import tqdm

from signwriting_animation.translation.utils import factor_signwriting, permute_signwriting

csv.field_size_limit(int(1e6))


def load_codes(codes_path: Path):
    with open(codes_path, 'r', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(tqdm(reader))


def load_data(data_path: Path):
    with open(data_path, 'r', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(tqdm(reader))


def get_fps(row):
    if row["file"] == "19097be0e2094c4aa6b2fdc208c8231e.pose":
        return 29.97
    return float(row["fps"])


# pylint: disable=too-many-locals
def create_parallel_data(codes, data, output_dir):
    codes_example = codes[0]
    num_codes = codes_example["codes"].split(" ")
    length = int(codes_example["length"])
    source_features = int(len(num_codes) / length)

    codes_dict = {row["file"]: row for row in codes}

    for split in ["train", "dev", "test"]:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # pylint: disable=consider-using-with
        src_factors_files = [open(split_dir / f"source_{i}.txt", "w", encoding="utf-8") for i in range(5)]
        # pylint: disable=consider-using-with
        trg_factors_files = [open(split_dir / f"target_{i}.txt", "w", encoding="utf-8")
                             for i in range(source_features)]

        for datum in tqdm(data):
            if datum["split"] != split:
                continue

            file = datum["pose"]
            # data.csv may change more frequently than codes.csv
            if file not in codes_dict:
                continue
            codes_row = codes_dict[file]
            codes = codes_row["codes"].split(" ")

            fps = get_fps(codes_row)
            start_frame = int(int(datum["start"]) / 1000 * fps)
            end_frame = int(int(datum["end"]) / 1000 * fps)

            codes = codes[start_frame * source_features:end_frame * source_features]
            fsw = swu2fsw(datum["text"])
            permutations = list(permute_signwriting(fsw))
            for permuted_fsw in permutations:
                fsw_factors = factor_signwriting(permuted_fsw)

                for i, factor_file in enumerate(trg_factors_files):
                    factor_file.write(" ".join(codes[i::source_features]) + "\n")

                for i, factor_file in enumerate(src_factors_files):
                    factor_file.write(" ".join(fsw_factors[i]) + "\n")

        for factor_file in src_factors_files + trg_factors_files:
            factor_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes', type=str, help='Path to codes csv file',
                        default="codes-example.csv")
    parser.add_argument('--data', type=str, help='Path to data csv file',
                        default="../../data/data.csv")
    parser.add_argument('--output-dir', type=str, help='Path to output directory',
                        default="parallel")
    args = parser.parse_args()

    codes = load_codes(Path(args.codes))
    data = load_data(Path(args.data))

    # create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_parallel_data(codes, data, output_dir)


if __name__ == "__main__":
    main()
