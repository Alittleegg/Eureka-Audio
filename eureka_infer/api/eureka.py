"""
Eureka-Audio main inference API.

This module provides the main EurekaAudio class for audio understanding inference.
"""

import os
import numpy as np
import torch
from typing import List, Dict, Optional
from loguru import logger
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, Qwen3OmniMoeProcessor

from eureka_infer.utils.audio import (
    load_audio_data,
    convert_audio,
    pad_dim_to_multiple_of_1280,
    audio_to_mel,
    split_and_pad_mel,
    SAMPLE_RATE,
)


class TokenType:
    """Token type identifiers for multimodal inputs."""
    text = 0
    audio = 3


class EurekaAudio:
    """
    Eureka-Audio inference wrapper.

    This class provides a simple interface for audio understanding inference,
    similar to KimiAudio's API design.

    Args:
        model_path: Path to model checkpoint or HuggingFace model ID
        device: Device to run inference on ("cuda" or "cpu")

    Example:
        >>> model = EurekaAudio("cslys1999/Eureka-Audio-Instruct")
        >>> messages = [
        ...     {
        ...         "role": "user",
        ...         "content": [
        ...             {"type": "audio_url", "audio_url": {"url": "path/to/audio.wav"}},
        ...             {"type": "text", "text": "What is being said in this audio?"}
        ...         ]
        ...     }
        ... ]
        >>> response = model.generate(messages)
        >>> print(response)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        logger.info(f"Loading Eureka-Audio model from {model_path}")

        # Download model if it's a HuggingFace model ID
        if os.path.exists(model_path):
            cache_path = model_path
        else:
            cache_path = snapshot_download(model_path)

        logger.info(f"Looking for resources in {cache_path}")

        # Load model with trust_remote_code for custom model classes
        logger.info("Loading Eureka-Audio model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            cache_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model = self.model.to(device)
        self.model.eval()

        # Load processor for tokenization and chat template
        logger.info("Loading processor...")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(cache_path)
        self.tokenizer = self.processor.tokenizer

        # Get special token IDs
        vocab = self.tokenizer.get_vocab()
        self.audio_placeholder_id = vocab.get('<|audio_pad|>')
        self.audio_start_id = vocab.get('<|audio_start|>')
        self.audio_end_id = vocab.get('<|audio_end|>')

        # Set mel filter path
        self.mel_filter_path = os.path.join(cache_path, "mel_filters.npz")

        self.device = device
        self.vocab_size = len(vocab)

        logger.info("Eureka-Audio model loaded successfully")

    def _process_audio_info(
        self,
        messages: List[Dict]
    ) -> tuple:
        """
        Extract and process audio information from messages.

        Args:
            messages: List of message dicts

        Returns:
            mels: List of mel spectrograms
            frames_list: List of frame counts
            total_frames: Total number of audio frames
        """
        mels = []
        frames_list = []
        total_frames = 0

        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                continue

            for item in content:
                item_type = item.get("type", "")

                if item_type == "audio_url":
                    url = item["audio_url"]["url"]

                    # Load and preprocess audio
                    wav, sr = load_audio_data(url)
                    wav = convert_audio(wav, sr, SAMPLE_RATE, 1)
                    wav = pad_dim_to_multiple_of_1280(wav.numpy())
                    wav = wav.squeeze(0)

                    # Calculate frames
                    frames = int(wav.shape[-1] / 1280)
                    frames_list.append(frames)
                    total_frames += frames

                    # Convert to mel spectrogram
                    if os.path.exists(self.mel_filter_path):
                        mel = audio_to_mel(wav, self.mel_filter_path)
                        mels.append(mel)

        return mels, frames_list, total_frames

    def _prepare_inputs(
        self,
        messages: List[Dict],
    ) -> Dict:
        """
        Prepare model inputs from messages.

        Args:
            messages: List of message dicts

        Returns:
            Dict containing input_ids, token_type_ids, and mels
        """
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        # Process audio
        mels, frames_list, _ = self._process_audio_info(messages)

        # Tokenize and build input tensors
        input_ids = []
        token_type_ids = []

        audio_placeholder = "<|audio_start|><|audio_pad|><|audio_end|>"
        text_parts = text.split(audio_placeholder)

        frame_idx = 0
        for i, part in enumerate(text_parts):
            if part:
                encoded = self.tokenizer.encode(part, add_special_tokens=False)
                if isinstance(encoded, dict):
                    curr_ids = encoded["input_ids"]
                else:
                    curr_ids = encoded
                input_ids.extend(curr_ids)
                token_type_ids.extend([TokenType.text] * len(curr_ids))

            if i < len(text_parts) - 1 and frame_idx < len(frames_list):
                frames = frames_list[frame_idx]
                curr_ids = (
                    [self.audio_start_id] +
                    [self.audio_placeholder_id] * frames +
                    [self.audio_end_id]
                )
                input_ids.extend(curr_ids)
                token_type_ids.extend(
                    [TokenType.text] +
                    [TokenType.audio] * frames +
                    [TokenType.text]
                )
                frame_idx += 1

        # Convert to tensors
        input_ids = torch.tensor(np.array(input_ids, dtype=np.int64)).unsqueeze(0)
        token_type_ids = torch.tensor(np.array(token_type_ids, dtype=np.int64)).unsqueeze(0)

        # Process mel spectrograms
        mels_tensor = None
        if mels and len(mels) > 0:
            mels_concat = np.concatenate(mels, axis=-1)
            if mels_concat.ndim == 2:
                mels_concat = np.expand_dims(mels_concat, axis=0)
            mels_tensor = split_and_pad_mel(mels_concat)
            mels_tensor = torch.tensor(mels_tensor, dtype=torch.float32)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "mels": mels_tensor,
        }

    @torch.inference_mode()
    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        do_sample: bool = True,
        repetition_penalty: float = 1.05,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate response for given messages

        Args:
            messages: List of message dicts in chat format:
                [{"role": "user", "content": [{"type": "text", "text": "..."}, ...]}]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
            repetition_penalty: Repetition penalty
            stop_sequences: List of strings that will stop generation when encountered

        Returns:
            Generated text response
        """
        # Prepare inputs
        inputs = self._prepare_inputs(messages)

        input_ids = inputs["input_ids"].to(self.device)
        token_type_ids = inputs["token_type_ids"].to(self.device)
        mels = inputs["mels"]
        if mels is not None:
            mels = mels.to(self.device).to(torch.bfloat16)

        # Build eos_token_id list
        eos_token_id = [self.tokenizer.eos_token_id]
        if stop_sequences:
            for seq in stop_sequences:
                token_ids = self.tokenizer.encode(seq, add_special_tokens=False)
                if token_ids:
                    eos_token_id.extend(token_ids)

        # Generate
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.model.generate(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                audio_input_ids=None,
                mel_batch_list=mels,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
            )

        # Decode output
        # With return_dict_in_generate=True, outputs is a GenerateOutput object
        # Access sequences via .sequences attribute
        if hasattr(outputs, 'sequences'):
            sequences = outputs.sequences
        else:
            sequences = outputs
        generated_ids = sequences[0][input_ids.shape[-1]:]
        token_ids = generated_ids.cpu().numpy().tolist()
        token_ids = [x for x in token_ids if x < self.vocab_size]
        response = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        return response

    def __call__(
        self,
        messages: List[Dict],
        **kwargs,
    ) -> str:
        """Shorthand for generate()."""
        return self.generate(messages, **kwargs)
