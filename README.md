# SignWriting Animation

This project aims to automatically animate SignWriting into skeletal poses.

This is the reverse
of [signwriting-transcription](https://github.com/sign-language-processing/signwriting-transcription).

## Usage

```bash
pip install git+https://github.com/sign-language-processing/signwriting-animation
```

To animate a SignWriting FSW sequence into a `.pose` file:

```bash
signwriting_to_pose --signwriting="M525x535S2e748483x510S10011501x466S2e704510x500S10019476x475" --pose="example.pose"
```

### Examples

(These examples are taken from the DSGS Vokabeltrainer)

|             |                                                                    00004                                                                     |                                                                    00007                                                                     |                                                                    00015                                                                     |
|:-----------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------:|
| SignWriting | <img src="https://github.com/sign-language-processing/signwriting-transcription/blob/main/assets/examples/00004.png?raw=true" width="50px">  | <img src="https://github.com/sign-language-processing/signwriting-transcription/blob/main/assets/examples/00007.png?raw=true" width="50px">  | <img src="https://github.com/sign-language-processing/signwriting-transcription/blob/main/assets/examples/00015.png?raw=true" width="50px">  |
|    Video    | <img src="https://github.com/sign-language-processing/signwriting-transcription/blob/main/assets/examples/00004.gif?raw=true" width="150px"> | <img src="https://github.com/sign-language-processing/signwriting-transcription/blob/main/assets/examples/00007.gif?raw=true" width="150px"> | <img src="https://github.com/sign-language-processing/signwriting-transcription/blob/main/assets/examples/00015.gif?raw=true" width="150px"> |

## Data

We use the same data as in
[signwriting-transcription](https://github.com/sign-language-processing/signwriting-transcription).

