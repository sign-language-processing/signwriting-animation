from csv import DictReader
from matplotlib import pyplot as plt
import torch
import open_clip
from PIL import Image
import os
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from signwriting.visualizer.visualize import signwriting_to_image
from signwriting.formats.swu_to_fsw import swu2fsw



def generate_embeddings(pose_data,
                        output_folder: str,
                        embeddings_file_ids_path: str,
                        embeddings_path: str) -> (pd.DataFrame, np.array):
    # Load the model
    model, preprocess, tokenizer = (
        open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    file_ids = []
    embeddings = []

    for sample in tqdm(pose_data[:1000]):
        file_id = os.path.splitext(sample['pose'])[0]
        try:
            # generate SignWriting image from SignWriting code sequence
            swu = sample["text"]
            fsw = swu2fsw(swu)
            im = signwriting_to_image(fsw)
            im.save(os.path.join(output_folder, f"{file_id}.png"))

            image = im.convert('RGB')
            image = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)

            # normalise
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            file_ids.append(file_id)
            embeddings.append(image_features.cpu().numpy())

        except Exception as e:
            print(f'Error with sample {file_id}. Skipping. Error: {e}')

    embeddings = np.vstack(embeddings)
    np.save(embeddings_path, embeddings)
    df_file_ids = pd.DataFrame(data=dict(file_ids=file_ids))  # , embeddings=embeddings))
    df_file_ids.to_csv(embeddings_file_ids_path, index=False)

    print(f'Saved {len(pose_data)} SignWriting images and embeddings to', output_folder)

    return df_file_ids, embeddings


def load_embeddings(embeddings_file_ids_path: str, embeddings_path: str) -> (pd.DataFrame, np.array):
    # load embeddings from disk
    df_file_ids = pd.read_csv(embeddings_file_ids_path)
    embeddings = np.load(embeddings_path)

    return df_file_ids, embeddings


def main():
    with open("signwriting-transcription/data.csv", "r", encoding="utf-8") as f:
        pose_data = list(DictReader(f))

    output_folder = '/home/shaun/Dropbox/sign_language/datasets/poses/generated/signwriting_images_and_embeddings'
    embeddings_path = os.path.join(output_folder, 'signwriting_image_embeddings.npy')
    embeddings_file_ids_path = os.path.join(output_folder, "embedding_file_ids.csv")

    create_embeddings = False    # True=generate embeddings, False=load embeddings

    # fsw = "AS10011S10019S2e704S2e748M525x535S2e748483x510S10011501x466S20544510x500S10019476x475"
    if create_embeddings:
        df_file_ids, embeddings = generate_embeddings(pose_data,
                                                      output_folder,
                                                      embeddings_file_ids_path,
                                                      embeddings_path)

    else:
        df_file_ids, embeddings = load_embeddings(embeddings_file_ids_path,
                                                  embeddings_path)
        print('loaded embeddings')


if __name__ == '__main__':
    main()
