"""
Audio processing utilities for Eureka-Audio.
"""

import io
import re
import base64
import torch
import torchaudio
import numpy as np
from typing import Tuple, Union

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160

def convert_audio(
    wav: Union[torch.Tensor, np.ndarray],
    sr: int,
    target_sr: int,
    target_channels: int = 1
) -> torch.Tensor:
    """
    Convert audio to target sample rate and number of channels.

    Args:
        wav: Audio tensor or numpy array, shape [channels, samples]
        sr: Original sample rate
        target_sr: Target sample rate
        target_channels: Target number of channels (1 for mono)

    Returns:
        Converted audio tensor
    """
    if isinstance(wav, np.ndarray):
        wav = torch.from_numpy(wav)

    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"

    *shape, channels, length = wav.shape

    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(f"Cannot convert from {channels} to {target_channels} channels")

    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def load_audio_from_base64(b64_str: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio from base64 encoded string.

    Args:
        b64_str: Base64 encoded audio, optionally with data URI prefix

    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    match = re.match(r"data:audio/[^;]+;base64,(.*)", b64_str)
    if match:
        b64_str = match.group(1)

    audio_bytes = base64.b64decode(b64_str)
    return torchaudio.load(io.BytesIO(audio_bytes))


def load_audio_data(url: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio data from various sources.

    Args:
        url: Audio source - can be:
            - Local file path
            - HTTP/HTTPS URL
            - Base64 data URI (data:audio/xxx;base64,...)

    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    if url.startswith('http://') or url.startswith('https://'):
        import requests
        with requests.get(url, stream=True) as response:
            return torchaudio.load(io.BytesIO(response.content))
    elif re.match(r"data:audio/[^;]+;base64,(.*)", url):
        return load_audio_from_base64(url)
    else:
        return torchaudio.load(url)


def load_audio_as_base64(path: str) -> str:
    """
    Load audio file and encode as base64 string.

    Args:
        path: Path to audio file

    Returns:
        Base64 encoded string
    """
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf8')


def pad_dim_to_multiple_of_1280(arr: np.ndarray) -> np.ndarray:
    """
    Pad array's last dimension to multiple of 1280 (one encoder frame = 80ms).

    Args:
        arr: Input numpy array

    Returns:
        Padded array with last dim as multiple of 1280
    """
    last_dim = arr.shape[-1]
    remainder = last_dim % 1280

    if remainder == 0:
        return arr

    pad_width = 1280 - remainder
    pad_config = [(0, 0)] * arr.ndim
    pad_config[-1] = (0, pad_width)

    return np.pad(arr, pad_width=pad_config, mode='constant', constant_values=0)

def mel_filters_np(mel_filters_path: str, n_mels: int = 128) -> np.ndarray:
    """Load mel filter bank from npz file."""
    with np.load(mel_filters_path) as f:
        return f[f"mel_{n_mels}"]


def hann_window_np(n_fft: int = N_FFT) -> np.ndarray:
    """Create Hann window for STFT."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)


def stft_np(
    audio: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    window: np.ndarray = None
) -> np.ndarray:
    """Short-time Fourier transform."""
    if window is None:
        window = np.ones(n_fft, dtype=np.float32)

    pad = n_fft // 2
    audio = np.pad(audio, (pad, pad), mode='reflect')
    n_frames = 1 + (len(audio) - n_fft) // hop_length

    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, n_fft),
        strides=(audio.strides[0] * hop_length, audio.strides[0])
    ).copy()

    frames *= window
    fft_result = np.fft.rfft(frames, n=n_fft, axis=1)
    magnitudes = np.abs(fft_result) ** 2

    return magnitudes.T


def log_mel_spectrogram_np(
    audio: np.ndarray,
    filters: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH
) -> np.ndarray:
    """Compute log-Mel spectrogram."""
    window = hann_window_np(n_fft)
    magnitudes = stft_np(audio, n_fft=n_fft, hop_length=hop_length, window=window)

    mel_spec = np.dot(filters, magnitudes)

    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec.astype(np.float32)[:, :-1]


def audio_to_mel(audio: np.ndarray, mel_filter_path: str) -> np.ndarray:
    """
    Convert audio waveform to mel spectrogram.

    Args:
        audio: Audio waveform (1D numpy array)
        mel_filter_path: Path to mel_filters.npz

    Returns:
        Mel spectrogram, shape [128, T]
    """
    mel_filter = mel_filters_np(mel_filter_path)
    return log_mel_spectrogram_np(audio=audio, filters=mel_filter)


def split_and_pad_mel(mel_batch: np.ndarray, chunk_size: int = 3000) -> np.ndarray:
    """
    Split mel spectrogram into fixed-size chunks with padding.

    Args:
        mel_batch: Mel spectrogram, shape [1, 128, T]
        chunk_size: Size of each chunk (3000 = ~30 seconds)

    Returns:
        Chunked mel spectrograms, shape [num_chunks, 128, chunk_size]
    """
    B, C, T = mel_batch.shape
    assert B == 1, "Only batch size 1 is supported"

    num_chunks = (T + chunk_size - 1) // chunk_size
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, T)
        segment = mel_batch[0, :, start:end]

        if end - start < chunk_size:
            pad_len = chunk_size - (end - start)
            zero_pad = np.zeros((C, pad_len), dtype=segment.dtype)
            log_pad = np.log10(np.clip(zero_pad, 1e-10, None))
            log_pad = np.maximum(log_pad, log_pad.max() - 8.0)
            log_pad = (log_pad + 4.0) / 4.0
            segment = np.concatenate([segment, log_pad], axis=-1)

        chunks.append(np.expand_dims(segment, axis=0))

    return np.concatenate(chunks, axis=0)
