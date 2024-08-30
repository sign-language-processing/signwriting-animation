#!/bin/bash

#SBATCH --job-name=poses-to-codes
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --output=poses-to-codes.out

#SBATCH --ntasks=1
#SBATCH --gres gpu:1

set -e # exit on error
set -x # echo commands

module load anaconda3
source activate vq-transcription

data_dir=$1
quantized_file=$2

## Install Quantization package
#pip install git+https://github.com/sign-language-processing/sign-vq.git

# Quantize the dataset
[ ! -f "$quantized_file" ] && \
poses_to_codes \
  --data="$data_dir" \
  --output="$quantized_file"