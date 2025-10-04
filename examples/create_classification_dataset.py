#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sieves[engines]>=0.16",
#     "typer>=0.12,<1",
#     "datasets",
#     "transformers>=4.53.0,<5",
#     "huggingface-hub[hf_transfer]",
#     "torch",
# ]
# ///

"""Create a zero-shot classification dataset from any Hugging Face dataset using Sieves + Outlines.

It supports both single-label (default) and multi-label classification via a flag.

Examples
--------
  Single-label classification:
    uv run examples/create_classification_dataset_with_sieves.py \
      --input-dataset stanfordnlp/imdb \
      --column text \
      --labels "positive,negative" \
      --model HuggingFaceTB/SmolLM-360M-Instruct \
      --output-dataset your-username/imdb-classified

  With label descriptions:
    uv run examples/create_classification_dataset_with_sieves.py \
      --input-dataset user/support-tickets \
      --column content \
      --labels "bug,feature,question" \
      --label-descriptions "bug:something is broken,feature:request for new functionality,question:asking for help" \
      --model HuggingFaceTB/SmolLM-360M-Instruct \
      --output-dataset your-username/tickets-classified

  Multi-label classification (adds a multi-hot labels column):
    uv run examples/create_classification_dataset_with_sieves.py \
      --input-dataset ag_news \
      --column text \
      --labels "world,sports,business,science" \
      --multi-label \
      --model HuggingFaceTB/SmolLM-360M-Instruct \
      --output-dataset your-username/agnews-multilabel

"""

import os
from typing import Literal, Any

import outlines
import torch
import transformers
import typer
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, get_token
from loguru import logger
from outlines.models import Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import sieves

app = typer.Typer(add_completion=False, help=__doc__)


# Text constraints (simple sanity checks)
MIN_TEXT_LENGTH = 3
MAX_TEXT_LENGTH = 4000
MULTILABEL_THRESHOLD = 0.5


def _parse_label_descriptions(desc_string: str | None) -> dict[str, str]:
    """Parse a CLI description string into a mapping.

    Parses strings of the form ``"label1:desc1,label2:desc2"`` into a
    dictionary mapping labels to their descriptions. Commas inside
    descriptions are preserved by continuing the current description until
    the next ``":"`` separator is encountered.

    Args:
        desc_string: The raw CLI string to parse. If ``None`` or empty,
            returns an empty mapping.

    Returns:
        A dictionary mapping each label to its description.

    """
    if not desc_string:
        return {}

    descriptions: dict[str, str] = {}

    for label_desc in desc_string.split(","):
        label_desc_parts = label_desc.split(":")
        assert len(label_desc_parts) == 2, \
            f"Invalid label description: must be 'label1:desc1,label2:desc2', got: {label_desc}"
        descriptions[label_desc_parts[0].strip("'").strip()] = label_desc_parts[1].strip("'").strip()

    return descriptions


def _preprocess_text(text: str) -> str:
    """Normalize and truncate input text for classification.

    This function trims surrounding whitespace and truncates overly long
    inputs to ``MAX_TEXT_LENGTH`` characters, appending an ellipsis to
    signal truncation. Non-string or falsy inputs yield an empty string.

    Args:
        text: The raw input text to normalize.

    Returns:
        A cleaned string suitable for downstream classification. May be an
        empty string if the input was not a valid string.

    """
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        text = f"{text[:MAX_TEXT_LENGTH]}..."
    return text


def _is_valid_text(text: str) -> bool:
    """Validate the minimal length constraints for a text sample.

    Args:
        text: Candidate text after preprocessing.

    Returns:
        True if the text meets minimal length requirements (``MIN_TEXT_LENGTH``),
        False otherwise.

    """
    return bool(text and len(text) >= MIN_TEXT_LENGTH)


def _load_and_prepare_data(
    input_dataset: str,
    split: str,
    shuffle: bool,
    shuffle_seed: int | None,
    max_samples: int | None,
    column: str,
    labels: str,
    label_descriptions: str | None,
    hf_token: str | None,
) -> tuple[
    Dataset,
    list[str],
    list[str],
    list[int],
    list[str],
    dict[str, str],
    str | None,
]:
    """Load the dataset and prepare inputs for classification.

    This function encapsulates the data-loading and preprocessing path of the
    script: parsing labels/descriptions, detecting tokens, loading/shuffling
    the dataset, validating the target column, preprocessing texts, and
    computing valid indices.

    Args:
        input_dataset: Dataset repo ID on the Hugging Face Hub.
        split: Dataset split to load (e.g., "train").
        shuffle: Whether to shuffle the dataset.
        shuffle_seed: Seed used when shuffling is enabled.
        max_samples: Optional maximum number of samples to retain.
        column: Name of the text column to classify.
        labels: Comma-separated list of labels.
        label_descriptions: Optional mapping string of the form
            "label:desc,label2:desc2".
        hf_token: Optional Hugging Face token.

    Returns:
        A tuple containing: (dataset, raw_texts, processed_texts, valid_indices,
        labels_list, desc_map, token)

    Raises:
        typer.Exit: If labels are missing, dataset loading fails, the column is
            absent, or no valid texts remain after preprocessing.

    """
    # Parse labels and optional descriptions. Strip surrounding quotes if present.
    labels = labels.strip().strip("'\"")
    labels_list: list[str] = [label.strip().strip("'\"") for label in labels.split(",") if label.strip().strip("'\"")]
    if not labels_list:
        logger.error("No labels provided. Use --labels 'label1,label2,...'")
        raise typer.Exit(code=2)
    desc_map = _parse_label_descriptions(label_descriptions)

    # Token detection and validation (mirror legacy script behavior)
    token = hf_token or (os.environ.get("HF_TOKEN") or get_token())
    if not token:
        logger.error("No authentication token found. Please either:")
        logger.error("1. Run 'huggingface-cli login'")
        logger.error("2. Set HF_TOKEN environment variable")
        logger.error("3. Pass --hf-token argument")
        raise typer.Exit(code=1)

    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        name = user_info.get("name") or user_info.get("email") or "<unknown>"
        logger.info(f"Authenticated as: {name}")
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        logger.error("Please check your token is valid")
        raise typer.Exit(code=1)

    # Load dataset
    try:
        ds: Dataset = load_dataset(input_dataset, split=split)
    except Exception as e:
        logger.error(f"Failed to load dataset '{input_dataset}': {e}")
        raise typer.Exit(code=1)

    # Shuffle/select.
    if shuffle:
        ds = ds.shuffle(seed=shuffle_seed)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    # Validate columns.
    if column not in ds.column_names:
        logger.error(f"Column '{column}' not in dataset columns: {ds.column_names}")
        raise typer.Exit(code=1)

    # Extract and preprocess texts
    raw_texts: list[str] = list(ds[column])
    processed_texts: list[str] = []
    valid_indices: list[int] = []
    for i, t in enumerate(raw_texts):
        pt = _preprocess_text(t)
        if _is_valid_text(pt):
            processed_texts.append(pt)
            valid_indices.append(i)

    if not processed_texts:
        logger.error("No valid texts found for classification (after preprocessing).")
        raise typer.Exit(code=1)

    logger.info(f"Prepared {len(processed_texts)} valid texts out of {len(raw_texts)}")

    return ds, raw_texts, processed_texts, valid_indices, labels_list, desc_map, token


def _log_stats(
    docs: list[sieves.Doc],
    task: sieves.tasks.Classification,
    labels_list: list[str],
    multi_label: bool,
    raw_texts: list[str],
    processed_texts: list[str],
    valid_indices: list[int],
) -> None:
    """Compute and log distributions.

    Logs per-label distributions and success/skip metrics.

    Args:
        docs: Classified documents corresponding to processed_texts.
        task: The configured ``Classification`` task instance.
        labels_list: List of label names in canonical order.
        multi_label: Whether classification is multi-label.
        raw_texts: Original text column values.
        processed_texts: Preprocessed, valid texts used for inference.
        valid_indices: Indices mapping processed_texts back to raw_texts rows.

    Returns:
        None. Pushes datasets to the Hub and logs summary statistics.

    """
    if multi_label:
        # Log distribution across labels at threshold and skipped count
        label_counts = {label: 0 for label in labels_list}
        for doc in docs:
            result = doc.results[task.id]
            logger.info(result)
            if isinstance(result, list):
                for label, score in result:
                    if label in label_counts and score >= MULTILABEL_THRESHOLD:
                        label_counts[label] += 1

        total_processed = len(docs)
        skipped = len(raw_texts) - len(processed_texts)
        logger.info(f"Classification distribution (multi-label, threshold={MULTILABEL_THRESHOLD}):")

        for label in labels_list:
            count = label_counts.get(label, 0)
            pct = (count / total_processed * 100.0) if total_processed else 0.0
            logger.info(f"  {label}: {count} ({pct})")
        if skipped > 0:
            skipped_pct = (skipped / len(raw_texts) * 100.0) if raw_texts else 0.0
            logger.info(f"  Skipped/invalid: {skipped} ({skipped_pct})")

    else:
        # Map results back to original indices; invalid texts receive None
        classifications: list[str | None] = [None] * len(raw_texts)
        for idx, doc in zip(valid_indices, docs):
            result = doc.results[task.id]
            classifications[idx] = result if isinstance(result, str) else result[0]

        # Log distribution and success rate.
        total_texts = len(raw_texts)
        label_counts = {label: 0 for label in labels_list}
        for label in labels_list:
            label_counts[label] = sum(1 for c in classifications if c == label)
        none_count = sum(1 for c in classifications if c is None)

        logger.info("Classification distribution (single-label):")
        for label in labels_list:
            count = label_counts[label]
            pct = (count / total_texts * 100.0) if total_texts else 0.0
            logger.info(f"  {label}: {count} ({pct})")

        if none_count > 0:
            none_pct = (none_count / total_texts * 100.0) if total_texts else 0.0
            logger.info(f"  Invalid/Skipped: {none_count} ({none_pct})")

        success_rate = (len(valid_indices) / total_texts * 100.0) if total_texts else 0.0
        logger.info(f"Classification success rate: {success_rate}")


@app.command()  # type: ignore[misc]
def classify(
    input_dataset: str = typer.Option(..., help="Input dataset ID on Hugging Face Hub"),
    column: str = typer.Option(..., help="Name of the text column to classify"),
    labels: str = typer.Option(..., help="Comma-separated list of labels, e.g. 'positive,negative'"),
    output_dataset: str = typer.Option(..., help="Output dataset ID on Hugging Face Hub"),
    model: str = typer.Option(..., help="HF model ID to use"),
    label_descriptions: str | None = typer.Option(
        None, help="Optional descriptions per label: 'label:desc,label2:desc2'"
    ),
    max_samples: int | None = typer.Option(None, help="Max number of samples to process (for testing)"),
    hf_token: str | None = typer.Option(None, help="HF token; if omitted, uses env or cached token"),
    split: str = typer.Option("train", help="Dataset split (default: train)"),
    batch_size: int = typer.Option(64, help="Batch size"),
    max_tokens: int = typer.Option(200, help="Max tokens to generate"),
    shuffle: bool = typer.Option(False, help="Shuffle dataset before sampling"),
    shuffle_seed: int | None = typer.Option(None, help="Shuffle seed"),
    multi_label: bool = typer.Option(False, help="Enable multi-label classification (adds multi-hot 'labels')"),
) -> None:
    """Classify a Hugging Face dataset using Sieves + Outlines and push results.

    Runs zero-shot classification over a specified text column using the Sieves
    ``Classification`` task and the Outlines engine. Supports both single-label
    (default) and multi-label modes. In single-label mode, a "classification"
    column is added to the original dataset. In multi-label mode, a new dataset
    with ``text`` and multi-hot ``labels`` columns is created via
    ``Classification.to_hf_dataset``.

    Args:
        input_dataset: Dataset repo ID on the Hugging Face Hub.
        column: Name of the text column to classify.
        labels: Comma-separated list of allowed labels.
        output_dataset: Target dataset repo ID to push results to.
        model: Transformers model ID. Must be provided and non-empty.
        label_descriptions: Optional per-label descriptions in the form
            ``label:desc,label2:desc2``.
        max_samples: Optional maximum number of samples to process.
        hf_token: Optional token; if omitted, uses environment or cached login.
        split: Dataset split to load (default: ``"train"``).
        batch_size: Batch size for inference.
        max_tokens: Maximum tokens for generation per prompt.
        shuffle: Whether to shuffle the dataset before selecting samples.
        shuffle_seed: Seed used for shuffling.
        multi_label: If True, enable multi-label classification and output a
            multi-hot labels column; otherwise outputs single-label strings.

    Returns:
        None. Results are pushed to the Hugging Face Hub under ``output_dataset``.

    Raises:
        typer.Exit: If dataset loading fails, a required column is missing, or
            no valid texts are available for classification.

    """
    logger.info("Loading and preparing data.")
    (
        ds,
        raw_texts,
        processed_texts,
        valid_indices,
        labels_list,
        desc_map,
        token,
    ) = _load_and_prepare_data(
        input_dataset=input_dataset,
        split=split,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        max_samples=max_samples,
        column=column,
        labels=labels,
        label_descriptions=label_descriptions,
        hf_token=hf_token,
    )

    # Build model.
    info = HfApi().model_info(model)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    zeroshot_tag = "zero-shot-classification"
    # Explicitly designed for zero-shot classification: build directly as pipeline.
    if info.pipeline_tag == zeroshot_tag or zeroshot_tag in set(info.tags or []):
        model = transformers.pipeline(zeroshot_tag, model=model, device=device)
    # Otherwise: build Outlines model around it to enforce structured generation.
    else:
        model = outlines.models.from_transformers(
            AutoModelForCausalLM.from_pretrained(model, **({"device": device} if device else {})),
            AutoTokenizer.from_pretrained(model),
        )

    # Build task and pipeline.
    task = sieves.tasks.Classification(
        labels=labels_list,
        model=model,
        generation_settings=sieves.GenerationSettings(
            inference_kwargs={"max_new_tokens": max_tokens},
            strict_mode=False,
        ),
        batch_size=batch_size,
        label_descriptions=desc_map or None,
        multi_label=multi_label,
    )
    pipe = sieves.Pipeline([task])

    logger.info(f"Running {'multi-label ' if multi_label else ''}classification pipeline with labels {labels_list}.")
    docs = list(pipe([sieves.Doc(text=t) for t in processed_texts]))

    logger.info("Logging stats.")
    _log_stats(
        docs=docs,
        task=task,
        labels_list=labels_list,
        multi_label=multi_label,
        raw_texts=raw_texts,
        processed_texts=processed_texts,
        valid_indices=valid_indices,
    )

    logger.info("Collecting and pushing results.")
    ds = task.to_hf_dataset(docs, threshold=MULTILABEL_THRESHOLD)
    ds.push_to_hub(
        output_dataset,
        token=token,
        commit_message=f"Add classifications using Sieves + Outlines (multi-label; threshold={MULTILABEL_THRESHOLD})"
    )


@app.command("examples")  # type: ignore[misc]
def show_examples() -> None:
    """Print example commands for common use cases.

    This mirrors the examples that were previously printed when running the
    legacy script without arguments.
    """
    cmds = [
        "Example commands:",
        "\n# Simple classification:",
        "uv run examples/create_classification_dataset_with_sieves.py \\",
        "  --input-dataset stanfordnlp/imdb \\",
        "  --column text \\",
        "  --labels 'positive,negative' \\",
        "  --model MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \\",
        "  --output-dataset your-username/imdb-classified",
        "\n# With label descriptions:",
        "uv run examples/create_classification_dataset_with_sieves.py \\",
        "  --input-dataset user/support-tickets \\",
        "  --column content \\",
        "  --labels 'bug,feature,question' \\",
        "  --label-descriptions 'bug:something is broken or not working,feature:request for new functionality,"
        "question:asking for help or clarification' \\",
        "  --model MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \\",
        "  --output-dataset your-username/tickets-classified",
        "\n# Multi-label classification:",
        "uv run examples/create_classification_dataset_with_sieves.py \\",
        "  --input-dataset ag_news \\",
        "  --column text \\",
        "  --labels 'world,sports,business,science' \\",
        "  --multi-label \\",
        "  --model MoritzLaurer/deberta-v3-large-zeroshot-v2.0 \\",
        "  --output-dataset your-username/agnews-multilabel",
    ]
    for line in cmds:
        typer.echo(line)


if __name__ == "__main__":
    app()
