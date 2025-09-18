#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vllm>=0.6.6",
#     "transformers>=4.53.0",
#     "torch",
#     "datasets",
#     "huggingface-hub[hf_transfer]",
# ]
# ///

"""Classify text columns in Hugging Face datasets using vLLM with structured outputs.

This script provides efficient GPU-based classification with guaranteed valid outputs,
optimized for running on HF Jobs.

Example:
    uv run classify-dataset.py \\
        --input-dataset imdb \\
        --column text \\
        --labels "positive,negative" \\
        --output-dataset user/imdb-classified

HF Jobs example:
    hfjobs run --flavor a10 uv run classify-dataset.py \\
        --input-dataset user/emails \\
        --column content \\
        --labels "spam,ham" \\
        --output-dataset user/emails-classified \\
        --prompt-style reasoning

"""

import argparse
import logging
import os
import sys
from typing import List

import torch
from datasets import load_dataset
from huggingface_hub import HfApi, get_token
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

# Default model - SmolLM3 for good balance of speed and quality
DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"


def parse_label_descriptions(desc_string: str) -> dict:
    """Parse label descriptions from CLI format 'label1:desc1,label2:desc2'."""
    if not desc_string:
        return {}

    descriptions = {}
    # Split by comma, but be careful about commas in descriptions
    parts = desc_string.split(",")

    current_label = None
    current_desc_parts = []

    for part in parts:
        if ":" in part and not current_label:
            # New label:description pair
            label, desc = part.split(":", 1)
            current_label = label.strip()
            current_desc_parts = [desc.strip()]
        elif ":" in part and current_label:
            # Save previous label and start new one
            descriptions[current_label] = ",".join(current_desc_parts)
            label, desc = part.split(":", 1)
            current_label = label.strip()
            current_desc_parts = [desc.strip()]
        else:
            # Continuation of previous description (had comma in it)
            current_desc_parts.append(part.strip())

    # Don't forget the last one
    if current_label:
        descriptions[current_label] = ",".join(current_desc_parts)

    return descriptions


def create_messages(
    text: str, labels: list[str], label_descriptions: dict = None, enable_reasoning: bool = False
) -> list[dict]:
    """Create messages for chat template with optional label descriptions."""
    # Build the classification prompt
    if label_descriptions:
        # Format with descriptions
        categories_text = "Categories:\n"
        for label in labels:
            desc = label_descriptions.get(label, "")
            if desc:
                categories_text += f"- {label}: {desc}\n"
            else:
                categories_text += f"- {label}\n"
    else:
        # Simple format without descriptions
        categories_text = f"Categories: {', '.join(labels)}"

    if enable_reasoning:
        # Reasoning mode: allow thinking and request JSON output
        user_content = f"""Classify this text into one of these categories:

{categories_text}

Text: {text}

Think through your classification step by step, then provide your final answer in this JSON format:
{{"label": "your_chosen_label"}}"""

        system_content = "You are a helpful classification assistant that thinks step by step."
    else:
        # Structured output mode: fast classification
        if label_descriptions:
            user_content = (
                f"Classify this text into one of these categories:\n\n{categories_text}\nText: {text}\n\nCategory:"
            )
        else:
            user_content = f"Classify this text as one of: {', '.join(labels)}\n\nText: {text}\n\nLabel:"

        system_content = "You are a helpful classification assistant. /no_think"

    return [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]


# Minimum text length for valid classification
MIN_TEXT_LENGTH = 3

# Maximum text length (in characters) to avoid context overflow
MAX_TEXT_LENGTH = 4000


def parse_reasoning_output(output: str, valid_labels: list[str]) -> tuple[str, str, bool]:
    """Parse reasoning output to extract label from JSON after </think> tag.

    Returns:
        tuple: (label or None, full reasoning text, parsing_success)

    """
    import json

    # Find the </think> tag
    think_end = output.find("</think>")

    if think_end != -1:
        # Extract everything after </think>
        json_part = output[think_end + len("</think>") :].strip()
        reasoning = output[: think_end + len("</think>")]
    else:
        # No think tags, look for JSON in the output
        # Try to find JSON by looking for {
        json_start = output.find("{")
        if json_start != -1:
            json_part = output[json_start:].strip()
            reasoning = output[:json_start].strip() if json_start > 0 else ""
        else:
            json_part = output
            reasoning = output

    # Try to parse JSON
    try:
        # Find the first complete JSON object
        if "{" in json_part:
            # Extract just the JSON object
            json_str = json_part[json_part.find("{") :]
            # Find the matching closing brace
            brace_count = 0
            end_pos = 0
            for i, char in enumerate(json_str):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break

            if end_pos > 0:
                json_str = json_str[:end_pos]
                data = json.loads(json_str)
                label = data.get("label", "")

                # Validate label
                if label in valid_labels:
                    return label, output, True
                else:
                    logger.warning(f"Parsed label '{label}' not in valid labels: {valid_labels}")
                    return None, output, False
            else:
                logger.warning("Could not find complete JSON object")
                return None, output, False
        else:
            logger.warning("No JSON found in output")
            return None, output, False

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error: {e}")
        return None, output, False
    except Exception as e:
        logger.warning(f"Unexpected error parsing output: {e}")
        return None, output, False


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify text in HuggingFace datasets using vLLM with structured outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--input-dataset",
        type=str,
        required=True,
        help="Input dataset ID on Hugging Face Hub",
    )
    parser.add_argument("--column", type=str, required=True, help="Name of the text column to classify")
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Comma-separated list of classification labels (e.g., 'positive,negative')",
    )
    parser.add_argument(
        "--output-dataset",
        type=str,
        required=True,
        help="Output dataset ID on Hugging Face Hub",
    )

    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use for classification (default: {DEFAULT_MODEL})",
    )
    # Removed --batch-size argument as vLLM handles batching internally
    parser.add_argument(
        "--label-descriptions",
        type=str,
        default=None,
        help="Descriptions for each label in format 'label1:description1,label2:description2'",
    )
    parser.add_argument(
        "--enable-reasoning",
        action="store_true",
        help="Enable reasoning mode with thinking traces (disables structured outputs)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face API token (default: auto-detect from HF_TOKEN env var or huggingface-cli login)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for generation (default: 0.1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100, automatically increased 20x for reasoning mode)",
    )
    parser.add_argument(
        "--guided-backend",
        type=str,
        default="outlines",
        help="Guided decoding backend (default: outlines)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset before selecting samples (useful with --max-samples for random sampling)",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )

    return parser.parse_args()


def preprocess_text(text: str) -> str:
    """Preprocess text for classification."""
    if not text or not isinstance(text, str):
        return ""

    # Strip whitespace
    text = text.strip()

    # Truncate if too long
    if len(text) > MAX_TEXT_LENGTH:
        text = f"{text[:MAX_TEXT_LENGTH]}..."

    return text


def validate_text(text: str) -> bool:
    """Check if text is valid for classification."""
    return bool(text and len(text) >= MIN_TEXT_LENGTH)


def prepare_prompts(
    texts: list[str],
    labels: list[str],
    tokenizer: AutoTokenizer,
    label_descriptions: dict = None,
    enable_reasoning: bool = False,
) -> tuple[list[str], list[int]]:
    """Prepare prompts using chat template for classification, filtering invalid texts."""
    prompts = []
    valid_indices = []

    for i, text in enumerate(texts):
        processed_text = preprocess_text(text)
        if validate_text(processed_text):
            # Create messages for chat template
            messages = create_messages(processed_text, labels, label_descriptions, enable_reasoning)

            # Apply chat template
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            valid_indices.append(i)

    return prompts, valid_indices


def main():
    args = parse_args()

    # Check authentication early
    logger.info("Checking authentication...")
    token = args.hf_token or (os.environ.get("HF_TOKEN") or get_token())

    if not token:
        logger.error("No authentication token found. Please either:")
        logger.error("1. Run 'huggingface-cli login'")
        logger.error("2. Set HF_TOKEN environment variable")
        logger.error("3. Pass --hf-token argument")
        sys.exit(1)

    # Validate token by checking who we are
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        logger.info(f"Authenticated as: {user_info['name']}")
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        logger.error("Please check your token is valid")
        sys.exit(1)

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU.")
        logger.error("Please run on a machine with GPU support or use HF Jobs.")
        sys.exit(1)

    logger.info(f"CUDA available. Using device: {torch.cuda.get_device_name(0)}")

    # Parse and validate labels
    labels = [label.strip() for label in args.labels.split(",")]
    if len(labels) < 2:
        logger.error("At least two labels are required for classification.")
        sys.exit(1)
    logger.info(f"Classification labels: {labels}")

    # Parse label descriptions if provided
    label_descriptions = None
    if args.label_descriptions:
        label_descriptions = parse_label_descriptions(args.label_descriptions)
        logger.info("Label descriptions provided:")
        for label, desc in label_descriptions.items():
            logger.info(f"  {label}: {desc}")

    # Load dataset
    logger.info(f"Loading dataset: {args.input_dataset}")
    try:
        dataset = load_dataset(args.input_dataset, split=args.split)
        logger.info(f"Loaded {len(dataset)} samples from split '{args.split}'")

        # Shuffle if requested
        if args.shuffle:
            logger.info(f"Shuffling dataset with seed {args.shuffle_seed}")
            dataset = dataset.shuffle(seed=args.shuffle_seed)

        # Limit samples if specified
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            logger.info(f"Limited dataset to {len(dataset)} samples")
            if args.shuffle:
                logger.info("Note: Samples were randomly selected due to shuffling")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

    # Verify column exists
    if args.column not in dataset.column_names:
        logger.error(f"Column '{args.column}' not found in dataset.")
        logger.error(f"Available columns: {dataset.column_names}")
        sys.exit(1)

    # Extract texts
    texts = dataset[args.column]

    # Load tokenizer for chat template formatting
    logger.info(f"Loading tokenizer for {args.model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)

    # Initialize vLLM
    logger.info(f"Initializing vLLM with model: {args.model}")
    logger.info(f"Using guided decoding backend: {args.guided_backend}")
    try:
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=0.95,
            guided_decoding_backend=args.guided_backend,
        )
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        sys.exit(1)

    # Set up sampling parameters based on mode
    if args.enable_reasoning:
        # Reasoning mode: no guided decoding, much more tokens for thinking
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens * 20,  # 20x more tokens for extensive reasoning
        )
        logger.info("Using reasoning mode - model will generate thinking traces with JSON output")
    else:
        # Structured output mode: guided decoding
        guided_params = GuidedDecodingParams(choice=labels)
        sampling_params = SamplingParams(
            guided_decoding=guided_params,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        logger.info("Using structured output with guided_choice - outputs guaranteed to be valid labels")

    # Prepare all prompts
    logger.info("Preparing prompts for classification...")
    all_prompts, valid_indices = prepare_prompts(texts, labels, tokenizer, label_descriptions, args.enable_reasoning)

    if not all_prompts:
        logger.error("No valid texts found for classification.")
        sys.exit(1)

    logger.info(f"Prepared {len(all_prompts)} valid prompts out of {len(texts)} texts")

    # Let vLLM handle batching internally
    logger.info("Starting classification (vLLM will handle batching internally)...")

    try:
        # Generate all classifications at once - vLLM handles batching
        outputs = llm.generate(all_prompts, sampling_params)

        # Process outputs based on mode
        if args.enable_reasoning:
            # Reasoning mode: parse JSON and extract reasoning
            all_classifications = [None] * len(texts)
            all_reasoning = [None] * len(texts)
            all_parsing_success = [False] * len(texts)

            for idx, output in enumerate(outputs):
                original_idx = valid_indices[idx]
                generated_text = output.outputs[0].text.strip()

                # Parse the reasoning output
                label, reasoning, success = parse_reasoning_output(generated_text, labels)

                all_classifications[original_idx] = label
                all_reasoning[original_idx] = reasoning
                all_parsing_success[original_idx] = success

                # Log first few examples
                if idx < 3:
                    logger.info(f"\nExample {idx + 1} output:")
                    logger.info(f"Raw output: {generated_text[:200]}...")
                    logger.info(f"Parsed label: {label}")
                    logger.info(f"Parsing success: {success}")

            # Count parsing statistics
            parsing_success_count = sum(1 for s in all_parsing_success if s)
            parsing_fail_count = sum(1 for s in all_parsing_success if s is not None and not s)
            logger.info("\nParsing statistics:")
            logger.info(
                f"  Successful: {parsing_success_count}/{len(valid_indices)} ({parsing_success_count /
                                                                               len(valid_indices) * 100:.1f}%)"
            )
            logger.info(
                f"  Failed: {parsing_fail_count}/{len(valid_indices)} ({parsing_fail_count / len(valid_indices) *
                                                                        100:.1f}%)"
            )

            valid_texts = parsing_success_count
        else:
            # Structured output mode: direct classification
            all_classifications = [None] * len(texts)
            for idx, output in enumerate(outputs):
                original_idx = valid_indices[idx]
                generated_text = output.outputs[0].text.strip()
                all_classifications[original_idx] = generated_text

            valid_texts = len(valid_indices)

        # Count statistics
        total_texts = len(texts)

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        sys.exit(1)

    # Add columns to dataset
    dataset = dataset.add_column("classification", all_classifications)

    if args.enable_reasoning:
        dataset = dataset.add_column("reasoning", all_reasoning)
        dataset = dataset.add_column("parsing_success", all_parsing_success)

    # Calculate statistics
    none_count = total_texts - valid_texts
    if none_count > 0:
        logger.warning(f"{none_count} texts were too short or invalid for classification")

    # Show classification distribution
    label_counts = {label: all_classifications.count(label) for label in labels}

    # Count None values separately
    none_classifications = all_classifications.count(None)

    logger.info("Classification distribution:")
    for label, count in label_counts.items():
        percentage = count / total_texts * 100 if total_texts > 0 else 0
        logger.info(f"  {label}: {count} ({percentage:.1f}%)")

    if none_classifications > 0:
        none_percentage = none_classifications / total_texts * 100
        if args.enable_reasoning:
            logger.info(f"  Failed to parse: {none_classifications} ({none_percentage:.1f}%)")
        else:
            logger.info(f"  Invalid/Skipped: {none_classifications} ({none_percentage:.1f}%)")

    # Log success rate
    success_rate = (valid_texts / total_texts * 100) if total_texts > 0 else 0
    logger.info(f"Classification success rate: {success_rate:.1f}%")

    # Save to Hub (token already validated at start)
    logger.info(f"Pushing dataset to Hub: {args.output_dataset}")
    try:
        dataset.push_to_hub(
            args.output_dataset,
            token=token,
            commit_message=f"Add classifications using {args.model} {'with reasoning' if args.enable_reasoning else 'with structured outputs'}",
        )
        logger.info(f"Successfully pushed to: https://huggingface.co/datasets/{args.output_dataset}")
    except Exception as e:
        logger.error(f"Failed to push to Hub: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Example commands:")
        print("\n# Simple classification:")
        print("uv run classify-dataset.py \\")
        print("  --input-dataset stanfordnlp/imdb \\")
        print("  --column text \\")
        print("  --labels 'positive,negative' \\")
        print("  --output-dataset user/imdb-classified")
        print("\n# With label descriptions:")
        print("uv run classify-dataset.py \\")
        print("  --input-dataset user/support-tickets \\")
        print("  --column content \\")
        print("  --labels 'bug,feature,question' \\")
        print(
            "  --label-descriptions 'bug:something is broken or not working,feature:request for new functionality,"
            "question:asking for help or clarification' \\"
        )
        print("  --output-dataset user/tickets-classified")
        print("\n# With reasoning mode (thinking + JSON output):")
        print("uv run classify-dataset.py \\")
        print("  --input-dataset stanfordnlp/imdb \\")
        print("  --column text \\")
        print("  --labels 'positive,negative,neutral' \\")
        print("  --enable-reasoning \\")
        print("  --output-dataset user/imdb-reasoned")
        print("\n# HF Jobs example:")
        print("hf jobs uv run \\")
        print("  --flavor l4x1 \\")
        print("  --image vllm/vllm-openai:latest \\")
        print("  classify-dataset.py \\")
        print("  --input-dataset stanfordnlp/imdb \\")
        print("  --column text \\")
        print("  --labels 'positive,negative' \\")
        print("  --output-dataset user/imdb-classified")
        sys.exit(0)

    main()
