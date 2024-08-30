# Translation

Using an encoder-decoder architecture, we can translate SignWriting into tokens from the vector quantizer.

```bash
cd ~/sign-language/signwriting-animation/signwriting_animation/translation

# 0. Setup the environment.
conda create --name vq-transcription python=3.11 -y
conda activate vq-transcription
```

## Pre-training

We pre-train the model on a large dataset of sign language poses, without any conditioning.
That means the encoder gets an empty sign (M518x518S2ff00482x483) and the decoder gets various sequences.
We can then sample from this model to generate an infinite number of signs.

```bash
# 1. Quantize a large poses dataset (508GB -> 2GB)
# This takes about 10 hours
LARGE_POSES_DIR=/shares/volk.cl.uzh/amoryo/datasets/sign-mt-poses
PROCESSED_FILES_DIR=/shares/volk.cl.uzh/amoryo/signwriting-animation
MODEL_DIR="$PROCESSED_FILES_DIR/models"
mkdir -p $PROCESSED_FILES_DIR
mkdir -p $MODEL_DIR
sbatch poses_to_codes.sh "$LARGE_POSES_DIR" "$PROCESSED_FILES_DIR/sign-mt-poses-quantized.csv"

# 2. Create source/target txt files
python pretraining/create_pretraining_data.py \
  --poses="$PROCESSED_FILES_DIR/sign-mt-poses-quantized.csv" \
  --output="$PROCESSED_FILES_DIR/pretraining"

# 2.1. Create dev set, 1000 lines from training set.
mkdir -p "$PROCESSED_FILES_DIR/pretraining/dev"
for f in "$PROCESSED_FILES_DIR"/pretraining/train/*; do head -n 1000 "$f" > "$PROCESSED_FILES_DIR"/pretraining/dev/$(basename "$f"); done

# 3. Create vocab files
python pretraining/create_vocab_files.py --vocab_dir="$PROCESSED_FILES_DIR/vocab"

# 4. Train a translation model
sbatch train_sockeye_model.sh \
  --data_dir="$PROCESSED_FILES_DIR/pretraining" \
  --model_dir="$MODEL_DIR/unconstrained" \
  --vocab_dir="$PROCESSED_FILES_DIR/vocab" \
  --optimized_metric="perplexity" \
  --use_source_factors=true \
  --use_target_factors=true \
  --partition lowprio

# 4.1 Graph perplexity over time
cat $MODEL_DIR/unconstrained/model/metrics
python pretraining/graph_ppl.py
  
# 5. Sample from the model 
# TODO this is slow on GPU - https://github.com/awslabs/sockeye/issues/1110
for i in {1..3}; do echo "M518x518S2ff00482x483" >> input.txt; done
python -m signwriting_animation.translation.translate \
  --model="$MODEL_DIR/unconstrained/model" \
  --beam-size=1 \
  --temperature=2 \
  --input="input.txt" \
  --output="$MODEL_DIR/samples.txt"

# 6. Animate translations
mkdir -p "$MODEL_DIR/samples"
sbatch codes_to_poses.sh "$MODEL_DIR/samples.txt" "$MODEL_DIR/samples"
```

## Fine-tuning

We fine tune the decoder-only model on a smaller dataset of sign language poses with 
SignWriting transcriptions as a condition, fed in the encoder.

```bash
cd ~/sign-language/signwriting-animation/

# 1. Download the SignWriting annotations from signwriting-transcription
wget https://raw.githubusercontent.com/sign-language-processing/signwriting-transcription/main/data/data.csv -O "$PROCESSED_FILES_DIR/fine_tuning_data.csv"

# 2. Create source/target txt files
# SignWriting is permuted, to make the model more robust to different permutations
# Even though those might not be valid writings
python -m signwriting_animation.translation.fine_tuning.create_parallel_data \
  --codes="$PROCESSED_FILES_DIR/sign-mt-poses-quantized.csv" \
  --data="$PROCESSED_FILES_DIR/fine_tuning_data.csv" \
  --output="$PROCESSED_FILES_DIR/fine_tuning"

# 3. Fine tune a translation model
# Ideally, I think, optimized_metric would be accuracy, but it's not implemented in Sockeye https://github.com/awslabs/sockeye/issues/1111
sbatch signwriting_animation/translation/train_sockeye_model.sh \
  --data_dir="$PROCESSED_FILES_DIR/fine_tuning" \
  --model_dir="$MODEL_DIR/constrained-permuted" \
  --vocab_dir="$PROCESSED_FILES_DIR/vocab" \
  --optimized_metric="perplexity" \
  --base_model_dir="$MODEL_DIR/unconstrained" \
  --use_source_factors=true \
  --use_target_factors=true \
  --partition lowprio

# 3.1 (Optional) See the validation metrics
tail "$MODEL_DIR/constrained-permuted/model/metrics" 

# 4. Translate "Hello"
echo "M518x529S14c20481x471S27106503x489" > input.txt
python -m signwriting_animation.translation.translate \
  --model="$MODEL_DIR/constrained-permuted/model" \
  --beam-size=5 \
  --input="input.txt" \
  --output="output.txt"
  
# 6. Animate translations
mkdir -p "$MODEL_DIR/samples"
sbatch signwriting_animation/translation/codes_to_poses.sh "output.txt" "$MODEL_DIR/samples" 
```
