# Diffusion

Ideally, we would like to pretrain a diffusion model on a large number of sign language videos, 
then, fine-tune the model on a smaller dataset of sign language videos with SignWriting transcriptions as a conditional.


## Environment setup

```bash
# clone the repository
cd ~/sign-language
git clone https://github.com/sign-language-processing/signwriting-animation.git
cd signwriting-animation/

# create a virtual environment and install dependencies
conda create --name sign_diffusion_env python=3.11 -y
conda activate sign_diffusion_env
pip install .
```

## Prepare data

```bash
# clone the data repository
cd ~/sign-language/
git clone https://github.com/sign/data.git
```

Datasets required:
* [Pose sequences](https://github.com/sign/data/tree/main/signwriting-transcription/README.md#poses)
  * Download the pose sequences and unzip
  * `export POSE_SEQUENCES_FOLDER=<set to the unzipped folder>`
* SignWriting transcription data:
  * Stored in the repo at: `~/sign-language/data/signwriting-transcription/data.csv`
  * `export SIGNWRITING_TRANSCRIPTION_CSV_PATH=<set to the csv path>`
