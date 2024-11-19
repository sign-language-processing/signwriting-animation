# SignWriting Animation

This project aims to automatically animate SignWriting into skeletal poses.

This is the reverse
of [signwriting-transcription](https://github.com/sign-language-processing/signwriting-transcription).

We would ideally like to have to implementations:

- [diffusion](signwriting_animation/diffusion) - a diffusion-based method to animate SignWriting
- [translation](signwriting_animation/translation) - a translation-based method to animate SignWriting 
  using the [sign-vq](https://github.com/sign-language-processing/sign-vq) model.

## Usage

```bash
pip install git+https://github.com/sign-language-processing/signwriting-animation
```

To animate a SignWriting FSW sequence into a `.pose` file:

```bash
signwriting_to_pose --signwriting="M525x535S2e748483x510S10011501x466S2e704510x500S10019476x475" --pose="example.pose"
```

When generating full sentences, it is recommended to post-process the `.pose` files using
[fluent-pose-synthesis](https://github.com/sign-language-processing/fluent-pose-synthesis).

### Examples

(These examples are taken from the DSGS Vokabeltrainer)

|             |                                                                    00004                                                                     |                                                                    00007                                                                     |                                                                    00015                                                                     |
|:-----------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|
| SignWriting | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00004.png?raw=true" width="50px">  | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00007.png?raw=true" width="50px">  | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00015.png?raw=true" width="50px">  |
|    Video    | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00004.gif?raw=true" width="150px"> | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00007.gif?raw=true" width="150px"> | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00015.gif?raw=true" width="150px"> |

## Data

We use the same data as in
[signwriting-transcription](https://github.com/sign-language-processing/signwriting-transcription).

