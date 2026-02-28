"""
Eureka-Audio Inference Script

This script demonstrates how to use Eureka-Audio for various audio understanding tasks.

Usage:
    # 激活环境
    source env/eureka_audio_env/bin/activate

    # ASR 任务 (语音识别) - 使用专门的 ASR system prompt
    python Eureka-Audio-main/infer.py --audio_path test_wav/0.wav --task asr

    # QA 任务 (音频问答) - 无 system prompt，需要提供问题
    python Eureka-Audio-main/infer.py --audio_path test_wav/0.wav --task qa --question "这段音频说了什么？"

    # Caption 任务 (音频描述) - 无 system prompt，生成音频的详细描述
    python Eureka-Audio-main/infer.py --audio_path test_wav/0.wav --task caption

Arguments:
    --model_path      模型路径 (默认: Eureka-Audio-Instruct/)
    --audio_path      音频文件路径 (默认: test_audios/asr_example.wav)
    --task            任务类型: asr, qa, caption (默认: asr)
    --question        用户问题，仅用于 qa 任务
    --temperature     采样温度 (默认: 0.0, greedy decoding)
    --top_p           Top-p 采样参数 (默认: 0.0)
    --top_k           Top-k 采样参数 (默认: 0)
    --do_sample       是否使用采样 (默认: False)
    --max_new_tokens  最大生成 token 数 (默认: 512)

Notes:
    - 不同任务使用不同的 system prompt，这很重要:
      * ASR: 有 system prompt "You are an advanced ASR..."
      * QA/Caption: 无 system prompt
    - chat template 由 Qwen3OmniMoeProcessor 自动处理，不要手动构建
    - 音频帧数按 wav.shape[-1] / 1280 计算，每 1280 采样点 = 1 frame (80ms)
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eureka_infer.api import EurekaAudio


def main():
    parser = argparse.ArgumentParser(description="Eureka-Audio Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="../Eureka-Audio-Instruct/",
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
        # Automatic Speech Recognition - 使用专门的 ASR system prompt
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
        # Audio Question Answering - 无 system prompt
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
        # Audio Captioning - 无 system prompt
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
