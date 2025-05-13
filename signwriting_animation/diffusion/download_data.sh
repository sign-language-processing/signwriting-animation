# Download metadata file
wget -O "$SIGNWRITING_TRANSCRIPTION_CSV_PATH" https://github.com/sign/data/raw/refs/heads/main/signwriting-transcription/data.csv

# Download and unzip pose sequences
wget -O "$POSE_SEQUENCES_FOLDER.zip" https://sign-lanugage-datasets.sign-mt.cloud/poses/holistic/transcription.zip
unzip "$POSE_SEQUENCES_FOLDER.zip" -d "$POSE_SEQUENCES_FOLDER"
rm "$POSE_SEQUENCES_FOLDER.zip"
