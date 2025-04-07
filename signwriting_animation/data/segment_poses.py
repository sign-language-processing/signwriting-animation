# get_data.py
import os
import pandas as pd
import argparse

def create_fixed_segments(data_csv, output_csv, window_size=60):
    """
    Convert raw pose segment info (start, end) to fixed-length sliding windows.
    Args:
        data_csv (str): Path to original CSV (e.g., data.csv)
        output_csv (str): Where to save fixed segmentation CSV
        window_size (int): Number of frames per segment (fixed)
    """
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"Could not find file: {data_csv}")

    df = pd.read_csv(data_csv)
    segments = []

    for _, row in df.iterrows():
        pose = row["pose"]
        start = int(row["start"])
        end = int(row["end"])

        if end - start < window_size:
            continue

        for i in range(start, end - window_size + 1, window_size):
            segments.append({
                "pose": pose,
                "start": i,
                "end": i + window_size
            })

    out_df = pd.DataFrame(segments)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(out_df)} fixed-length segments to {output_csv}")                                             


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fixed-length sliding segments")
    parser.add_argument("--data_csv", required=True, help="Input data.csv path (contains pose, start, end)")
    parser.add_argument("--output_csv", required=True, help="Path to output fixed segment CSV")
    parser.add_argument("--window_size", type=int, default=60, help="Sliding window size in frames (default=60)")

    args = parser.parse_args()

    create_fixed_segments(
        data_csv=args.data_csv,
        output_csv=args.output_csv,
        window_size=args.window_size
    )
