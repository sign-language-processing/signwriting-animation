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

### Data

The full data is available in [sign/data](https://github.com/sign/data/tree/main/signwriting-transcription). These
examples are taken from the DSGS Vokabeltrainer:

|             |                                                         00004                                                          |                                                         00007                                                          |                                                         00015                                                          |
|:-----------:|:----------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------:|
| SignWriting | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00004.png?raw=true" width="50px">  | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00007.png?raw=true" width="50px">  | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00015.png?raw=true" width="50px">  |
|    Video    | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00004.gif?raw=true" width="150px"> | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00007.gif?raw=true" width="150px"> | <img src="https://github.com/sign/data/blob/main/signwriting-transcription/examples/00015.gif?raw=true" width="150px"> |

### Background

Movement Phases in Signs and Co-speech Gestures[^movement-phases] are related, and can be characterized as
(1) resting;
(2) preparation;
(3 - optional) stroke;
(4) hold;
(5) retraction;
and (6) resting.
Ende et al. 2011[^gesture-robot] adds that in continues gestures, instead of a retraction (5),
a repetition of the stroke (3) and hold (4) can be observed,
by going from the hold back to preparation (2) or directly to a new stroke (3).

[^movement-phases]: Kita, Sotaro, Ingeborg van Gijn and Harry van der Hulst. “Movement Phase in Signs and Co-Speech
Gestures, and Their Transcriptions by Human Coders.” Gesture Workshop (1997).
[^gesture-robot]: Ende, Tobias & Haddadin, Sami & Parusel, Sven & Wüsthoff, Tilo & Hassenzahl, Marc & Albu-Schäeffer,
Alin. (2011). A Human-Centered Approach to Robot Gesture Based Communication within Collaborative Working Processes.
IEEE International Conference on Intelligent Robots and Systems. 3367-3374. 10.1109/IROS.2011.6094592. 

Automatic Gesture Phase Segmentation models such as https://github.com/srdjop/Gesture-Phase-Segmentation can be used
to understand the different phases in the data, and such to explicitly animate the SignWriting in the relevant phases.