#!/bin/bash

#SBATCH --job-name=codes-to-poses
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --output=codes-to-poses.out

#SBATCH --ntasks=1
#SBATCH --gres gpu:1

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate vq-transcription

codes_file=$1
output_dir=$2

## Install Quantization package
#pip install git+https://github.com/sign-language-processing/sign-vq.git

# un-Quantize the codes
codes_to_poses \
  --codes="$codes_file" \
  --output="$output_dir"
