import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path


class CapuchinBirdCallDataset(Dataset):
    def __init__(
        self,
        positive_folder: Path,
        negative_folder: Path,
        sample_rate: int,
        num_frames: int,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        # load all clip paths into single sorted list
        positive_clips = list(positive_folder.iterdir())
        negative_clips = list(negative_folder.iterdir())
        clips = [
            (clip_path, torch.ones(1, dtype=torch.float32))
            for clip_path in positive_clips
        ]
        clips.extend(
            [
                (clip_path, torch.zeros(1, dtype=torch.float32))
                for clip_path in negative_clips
            ]
        )
        clips.sort(key=lambda item: item[0])
        self.clips = clips

        self.gen_spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=64, hop_length=128
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def load_mono_audio(self, clip_path: Path) -> torch.Tensor:
        clip_waveform, clip_sample_rate = torchaudio.load(clip_path)
        # only get the first channel
        clip_waveform = clip_waveform[0, :].unsqueeze(0)
        # resample the clip
        resampled_clip = torchaudio.functional.resample(
            clip_waveform, clip_sample_rate, self.sample_rate
        )
        resampled_clip = resampled_clip[:, : self.num_frames]
        # pad the clip if necessary
        if resampled_clip.shape[1] < self.num_frames:
            zero_padding = torch.zeros((1, self.num_frames - resampled_clip.shape[1]))
            resampled_clip = torch.concat((resampled_clip, zero_padding), dim=1)
        return resampled_clip

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_path, label = self.clips[index]
        resampled_clip = self.load_mono_audio(clip_path)
        spectrogram = self.amplitude_to_db(self.gen_spectrogram(resampled_clip))
        # stack the spectrogram for rgb input
        spectrogram = spectrogram.expand(3, -1, -1)
        return spectrogram, label
