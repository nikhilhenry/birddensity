# Bird Density Call Classifier

A PyTorch deep audio classifier to detect the number of [Cappuchin](https://en.wikipedia.org/wiki/Capuchinbird) bird calls from a given audio clip.


## Dataset

The dataset for this project is provided by the [Z by HP Unlocked Challenge](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing). The dataset consists of positive Cappuchin bird calls and negative bird calls. Inference is perfomed on variable length clips where the number of distinct Cappuchin bird calls is predicted.

The dataset is organised in the following manner from the project's root directory.

```
data/
├── Forest_Recordings
│   ├── recording_00.mp3
│   ├── ...
│   └── recording_99.mp3
├── Parsed_Capuchinbird_Clips
│   ├── XC114131-0.wav
│   ├── ...
│   └── XC9892-0.wav
└── Parsed_Not_Capuchinbird_Clips
    ├── Crickets-chirping-0.wav
    ├── ...
    └── tawny-owl-sounds-9.wav
3 directories, 910 files
```

To train the audio classifier, a random 80:20 split was performed on positive and negative clips. 

All clips were resampled to `48 kHz` and spectrograms were generated using TorchAudio's [Spectorgram Transform](https://pytorch.org/audio/main/generated/torchaudio.transforms.Spectrogram.html#spectrogram) with `64` bins and a hop length of `128`.

## Experiments

The model architecture consists of preprocessing audio clips into spectrograms and feeding the spectrogram as an input to a pretrained CNN classifier followed by a fully connected layer. All models have been trained using the Adam optimiser with BCELoss. Tensorboard logs are saved in `experiments` directory.


### EfficientNetBO

- Learning rate: 0.001
- Batch size: 32
- All layers of the CNN model were frozen
- No augmentations
- Epochs: 18
- Accuracy: 0.974 (test), 0.9866 (train)

### ResNet50

- Learning rate: 0.001
- Batch size: 32
- All layers of the CNN model were frozen
- No augmentations
- Epochs: 18
- Accuracy: 0.9583 (test), 0.9807 (train)