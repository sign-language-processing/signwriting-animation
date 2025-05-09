# Diffusion

Ideally, we would like to pretrain a diffusion model on a large number of sign language videos, 
then, fine-tune the model on a smaller dataset of sign language videos with SignWriting transcriptions as a conditional.


## Set up

Install dependencies:

```
pipenv sync
```

```bash
cd ~/sign-language/signwriting-animation/signwriting_animation/diffusion

# setup the environment
conda create --name sign_diffusion_env python=3.11 -y
conda activate sign_diffusion_env
```

### Prepare data

```
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

```
cd signwriting-animation/signwriting_animation

# generate SignWriting images and corresponding embeddings
export EMBEDDINGS_FOLDER=<set to desired folder to store embeddings>
mkdir -p $EMBEDDINGS_FOLDER
python data/prepare_embeddings.py --poses $POSE_SEQUENCES_FOLDER 
                                  --transcription_csv_path $SIGNWRITING_TRANSCRIPTION_CSV_PATH 
                                  --output $EMBEDDINGS_FOLDER
```