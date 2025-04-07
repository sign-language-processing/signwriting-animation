import os
import csv
from pose_format import Pose
from tqdm import tqdm
import argparse

def collect_pose_metadata(pose_dir, output_csv):
    pose_files = [f for f in os.listdir(pose_dir) if f.endswith(".pose")]

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["pose", "start", "end"]) 

        for fname in tqdm(pose_files, desc="Processing poses"):
            path = os.path.join(pose_dir, fname)
            try:
                with open(path, "rb") as f:
                    pose = Pose.read(f.read())
                    frame_count = pose.body.data.shape[0]
                    writer.writerow([fname, 0, frame_count])
            except Exception as e:
                print(f" Error reading {fname}: {e}")

    print(f"\n Done! Saved metadata to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect pose metadata from .pose files")
    parser.add_argument("--pose_dir", required=True, help="Directory containing .pose files")
    parser.add_argument("--output", required=True, help="Output CSV path")

    args = parser.parse_args()
    collect_pose_metadata(args.pose_dir, args.output)
