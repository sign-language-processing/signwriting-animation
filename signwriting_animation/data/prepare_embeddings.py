from csv import DictReader
import argparse
import os
import torch
import open_clip
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from signwriting.visualizer.visualize import signwriting_to_image
from signwriting.formats.swu_to_fsw import swu2fsw


class EmbeddingEncoder:
    def __init__(self):
        self.model, self.preprocess, self.tokenizer = \
            open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def infer(self, im: Image) -> torch.Tensor:
        # TODO: should make this operate on tensors rather?
        im_batch = self.preprocess(im).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embeddings_batch = self.model.encode_image(im_batch)

        return embeddings_batch[0, ...]


def generate_embeddings(embedding_func: callable,
                        pose_data: list,
                        output_folder: str,
                        embeddings_file_ids_path: str,
                        embeddings_path: str) -> (pd.DataFrame, np.array):

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

            image_features = embedding_func(image)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--poses', type=str,
                        help='Path to pose sequences folder')
    parser.add_argument('--transcription_csv_path', type=str,
                        help='Path to SignWriting transcription csv')
    parser.add_argument('--output', type=str,
                        help='Path to output directory to store embeddings')
    args = parser.parse_args()


    with open(args.transcription_csv_path, "r", encoding="utf-8") as f:
        pose_data = list(DictReader(f))

    output_folder = args.output
    embeddings_path = os.path.join(output_folder, 'signwriting_image_embeddings.npy')
    embeddings_file_ids_path = os.path.join(output_folder, "embedding_file_ids.csv")

    embedding_func = EmbeddingEncoder()
    df_file_ids, embeddings = generate_embeddings(embedding_func,
                                                  pose_data,
                                                  output_folder,
                                                  embeddings_file_ids_path,
                                                  embeddings_path)

    print('Finished generating embeddings')


if __name__ == '__main__':
    main()
