import torch
import torchaudio
from pathlib import Path
from itertools import groupby
import pandas as pd
from tqdm import tqdm
import datetime

NUM_FRAMES = 48000

device = "cuda" if torch.cuda.is_available() else "cpu"


gen_spectrogram = torchaudio.transforms.Spectrogram(n_fft=64, hop_length=128)
amplitude_to_db = torchaudio.transforms.AmplitudeToDB()


def load_audio_16k_mono(clip_path: Path) -> torch.Tensor:
    clip_waveform, clip_sample_rate = torchaudio.load(clip_path)
    # only get the first channel
    clip_waveform = clip_waveform[0, :].unsqueeze(0)
    # resample the clip
    resampled_clip = torchaudio.functional.resample(
        clip_waveform, clip_sample_rate, 16000
    )
    return resampled_clip


def generate_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform[:, :NUM_FRAMES]
    # pad the clip if necessary
    if waveform.shape[1] < NUM_FRAMES:
        zero_padding = torch.zeros((1, NUM_FRAMES - waveform.shape[1]))
        waveform = torch.concat((waveform, zero_padding), dim=1)
    spectrogram = amplitude_to_db(gen_spectrogram(waveform))
    spectrogram = spectrogram.expand(3, -1, -1)
    return spectrogram


def run_inference_on_chunk(x: torch.Tensor):
    sigmoid = torch.nn.Sigmoid()
    x = generate_spectrogram(x)

    with torch.inference_mode():
        x = x.to(device)
        y_logits = model(x.unsqueeze(dim=0))
        y_prob = sigmoid(y_logits)
        y_prediction = y_prob.item()
        if y_prediction >= 0.9:
            return 1
        else:
            return 0


def count_calls(y: list[int]) -> int:
    return sum([key for key, _ in groupby(y)])


# loading all the clips
DATA_FOLDER = Path("./data")
RECORDINGS_FOLDER = DATA_FOLDER / "Forest_Recordings"
if not RECORDINGS_FOLDER.exists():
    raise Exception("Recordings folder does not exist")

clip_paths = sorted([path for path in RECORDINGS_FOLDER.iterdir() if path.is_file()])
clip_names = [path.name for path in clip_paths]
clips = [load_audio_16k_mono(path) for path in tqdm(clip_paths, desc="Loading clips")]

# creating chunks from clips
chunked_clips = [torch.split(clip, NUM_FRAMES, 1) for clip in clips]

# loading the model
from models import EfficientNetB0Classifier

model = EfficientNetB0Classifier()
model.load_state_dict(torch.load("./artifacts/EfficientNetB0Classifier.pth"))
model = model.to(device)
model.eval()

# run inference on all clips
call_counts = []
for clip in tqdm(chunked_clips, desc="Inference"):
    y = [run_inference_on_chunk(chunk) for chunk in clip]
    call_counts.append(count_calls(y))
results = pd.DataFrame({"clip_names": clip_names, "call_count": call_counts})


# saving results
save_path = Path("./results")
save_path.mkdir(exist_ok=True)
timestamp = datetime.datetime.now().strftime("%m_%d_%Y_%H_%M")
save_path = save_path / str("results_" + timestamp + ".csv")
results.to_csv(save_path)

print(f"Saved results to {save_path.resolve()}.")
