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
pip install ".[dev]"
```

## Prepare data

```bash
# Setup locations
export SIGNWRITING_TRANSCRIPTION_CSV_PATH=<set to path.csv>
export POSE_SEQUENCES_FOLDER=<set to folder/>

signwriting_animation/diffusion/download_data.sh
```
