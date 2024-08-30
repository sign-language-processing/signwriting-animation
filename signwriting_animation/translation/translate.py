import argparse
import time
from functools import lru_cache
from pathlib import Path
from typing import List

from sockeye.inference import TranslatorOutput

from signwriting_animation.translation.utils import factored_signwriting_str


@lru_cache(maxsize=None)
def load_sockeye_translator(model_path: str,
                            beam_size: int = 5,
                            temperature: float = 1.0,
                            log_timing: bool = False):
    if not Path(model_path).is_dir():
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(repo_id=model_path)

    from sockeye.translate import parse_translation_arguments, load_translator_from_args

    now = time.time()
    args = parse_translation_arguments([
        "-m", model_path,
        "--beam-size", str(beam_size),
        "--loglevel", "INFO",
        "--use-cpu"
        # "--softmax-temperature", str(temperature) # TODO https://github.com/awslabs/sockeye/issues/1109
    ])

    translator = load_translator_from_args(args, True)

    if log_timing:
        print("Loaded sockeye translator in", time.time() - now, "seconds")

    return translator


def process_translation_output(output: TranslatorOutput):
    all_factors = [output.tokens] + output.factor_tokens
    symbols = [" ".join(f) for f in list(zip(*all_factors))]
    return " ".join(symbols)


def translate(translator, texts: List[str], log_timing: bool = False):
    from sockeye.inference import make_input_from_factored_string

    factored_signwriting = [factored_signwriting_str(text) for text in texts]
    print("inputs", factored_signwriting[0])

    inputs = [make_input_from_factored_string(sentence_id=i,
                                              factored_string=s,
                                              translator=translator)
              for i, s in enumerate(factored_signwriting)]

    now = time.time()
    outputs = translator.translate(inputs)
    translation_time = time.time() - now
    avg_time = translation_time / len(texts)
    if log_timing:
        print("Translated", len(texts), "texts in", translation_time, "seconds", f"({avg_time:.2f} seconds per text)")
    return [process_translation_output(output) for output in outputs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--beam-size', required=True, type=int, default=5, help='beam size for beam search')
    parser.add_argument('--temperature', default=1, type=float, help='Model sampling temperature')
    parser.add_argument('--input', required=True, type=str, help='Path to input SignWriting text file')
    parser.add_argument('--output', required=True, type=str, help='path to output txt file')
    args = parser.parse_args()

    translator = load_sockeye_translator(args.model, beam_size=args.beam_size, temperature=args.temperature,
                                         log_timing=True)

    with open(args.input, 'r', encoding="utf-8") as f:
        texts = f.readlines()
        texts = [text.strip() for text in texts if text]

    with open(args.output, 'w', encoding="utf-8") as f:
        CHUNK_SIZE = 32
        for i in range(0, len(texts), CHUNK_SIZE):
            chunk = texts[i:i + CHUNK_SIZE]
            outputs = translate(translator, chunk, log_timing=True)
            f.write("\n".join(outputs) + "\n")
