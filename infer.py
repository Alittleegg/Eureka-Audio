"""
Eureka-Audio Inference Script

This script demonstrates how to use Eureka-Audio for various audio understanding tasks.
"""

import argparse
import os

from eureka_infer.api import EurekaAudio


def main():
    parser = argparse.ArgumentParser(description="Eureka-Audio Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Eureka/Eureka-Audio-1.7B-Instruct",
        help="Path to the model (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default="test_audios/asr_example.wav",
        help="Path to the input audio file",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="asr",
        choices=["asr", "qa", "caption"],
        help="Task type: asr (speech recognition), qa (question answering), caption (audio captioning)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question for audio QA task",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling (default: False for greedy decoding)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = EurekaAudio(model_path=args.model_path)
    print("Model loaded successfully!")

    # Prepare messages based on task
    if args.task == "asr":
        # Automatic Speech Recognition
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an advanced ASR (Automatic Speech Recognition) AI assistant."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": args.audio_path}}
                ]
            }
        ]
        print(f"\n>>> Task: ASR (Speech Recognition)")
        print(f">>> Audio: {args.audio_path}")

    elif args.task == "qa":
        # Audio Question Answering
        question = args.question or "Based on the given audio, identify the source of the crowing.\nA. Rooster\nB. Dog\nC. Cat\nD. Cow"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": args.audio_path}},
                    {"type": "text", "text": question},
                ],
            }
        ]
        print(f"\n>>> Task: Audio Question Answering")
        print(f">>> Audio: {args.audio_path}")
        print(f">>> Question: {question}")

    elif args.task == "caption":
        # Audio Captioning
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": args.audio_path}},
                    {
                        "type": "text",
                        "text": "Please provide a comprehensive description of the input audio, covering all details.",
                    },
                ],
            }
        ]
        print(f"\n>>> Task: Audio Captioning")
        print(f">>> Audio: {args.audio_path}")

    # Generate response
    print("\nGenerating response...")
    response = model.generate(
        messages,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n>>> Output: {response}")


if __name__ == "__main__":
    main()
