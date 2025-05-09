# Diffusion

Ideally, we would like to pretrain a diffusion model on a large number of sign language videos, 
then, fine-tune the model on a smaller dataset of sign language videos with SignWriting transcriptions as a conditional.


## Set up

Install dependencies:

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
* [Pose sequences](https://github.com/sign/data/tree/main/signwriting-transcription/README.md)
  * Download the pose sequences and unzip   ("pose_sequences_folder")
* SignWriting transcription data:
  * Stored in the repo at: `data/signwriting-transcription/data.csv` ("signwriting_transcription_csv_path")

```
python data/prepare_embeddings.py
```