#!/usr/bin/env python3

import os
from pathlib import Path

# Set model caches before importing torch and open_clip.
CACHE_ROOT = Path("/net/tscratch/people/plgpiotrwojcik/.cache")

os.environ.setdefault(
    "HF_HOME",
    str(CACHE_ROOT / "huggingface"),
)
os.environ.setdefault(
    "HF_HUB_CACHE",
    str(CACHE_ROOT / "huggingface" / "hub"),
)
os.environ.setdefault(
    "TORCH_HOME",
    str(CACHE_ROOT),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(CACHE_ROOT),
)

import argparse
import csv
import json
import io
import zipfile

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image


# =============================================================================
# PATHS
# =============================================================================

VG_ROOT = Path("/net/tscratch/people/plgpiotrwojcik/vg")

VG_OBJECTS_JSON = VG_ROOT / "objects.json"

VG_IMAGE_ARCHIVES = (
    VG_ROOT / "images1.zip",
    VG_ROOT / "images2.zip",
)

DEFAULT_OBJECTS_CSV = VG_ROOT / "visual_genome_unique_objects.csv"

OPENCLIP_CACHE = CACHE_ROOT / "open_clip"


# =============================================================================
# OPENCLIP MODEL
# =============================================================================

MODEL_NAME = "ViT-H-14"
PRETRAINED_WEIGHTS = "laion2b_s32b_b79k"


# =============================================================================
# TEXT VOCABULARY
# =============================================================================

def clean_text(value: str) -> str:
    """Normalize whitespace without adding any prompt template."""
    return " ".join(str(value).strip().split())


def print_visual_genome_objects(
    image_id: int,
    objects_json: Path = VG_OBJECTS_JSON,
) -> None:
    """Print all annotated Visual Genome objects for one image."""

    with objects_json.open("r", encoding="utf-8") as f:
        images = json.load(f)

    for image in images:
        if image["image_id"] != image_id:
            continue

        print()
        print(f"Visual Genome objects for image {image_id}")
        print("=" * 80)

        for obj in image["objects"]:
            names = obj.get("names", [])

            print(
                f"id={obj['object_id']:>6} "
                f"bbox=({obj['x']}, {obj['y']}, "
                f"{obj['w']}, {obj['h']}) "
                f"names={names}"
            )

        print()
        print(f"Total objects: {len(image['objects'])}")
        return

    raise ValueError(f"Image {image_id} not found in {objects_json}")


def read_unique_objects(
    csv_path: Path,
    column: str,
) -> list[str]:
    """
    Read unique object names from a CSV file.

    Object names are encoded exactly as stored in the CSV, apart from
    surrounding and repeated whitespace normalization.

    No prompt templates are used.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Object vocabulary CSV does not exist: {csv_path}"
        )

    object_names: list[str] = []
    seen: set[str] = set()

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as source:
        reader = csv.DictReader(source)

        if reader.fieldnames is None:
            raise ValueError(
                f"The CSV file has no header: {csv_path}"
            )

        if column not in reader.fieldnames:
            raise ValueError(
                f"Column {column!r} was not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            value = row.get(column)

            if value is None:
                continue

            object_name = clean_text(value)

            if not object_name:
                continue

            deduplication_key = object_name.casefold()

            if deduplication_key in seen:
                continue

            seen.add(deduplication_key)
            object_names.append(object_name)

    if not object_names:
        raise ValueError(
            f"No object names were found in column {column!r} "
            f"of {csv_path}"
        )

    return object_names


# =============================================================================
# VISUAL GENOME IMAGE LOADING
# =============================================================================

def candidate_archive_members(image_id: int) -> tuple[str, ...]:
    """
    Return likely archive paths for a Visual Genome image.

    Visual Genome ZIP archives commonly contain:

        VG_100K/<image_id>.jpg
        VG_100K_2/<image_id>.jpg

    Some archives may include an additional top-level directory, so the
    fallback archive scan matches by filename when direct paths fail.
    """
    filenames = (
        f"{image_id}.jpg",
        f"{image_id}.jpeg",
        f"{image_id}.png",
        f"{image_id}.JPG",
        f"{image_id}.JPEG",
        f"{image_id}.PNG",
    )

    directories = (
        "",
        "VG_100K",
        "VG_100K_2",
    )

    members: list[str] = []

    for directory in directories:
        for filename in filenames:
            if directory:
                members.append(f"{directory}/{filename}")
            else:
                members.append(filename)

    return tuple(members)


def load_visual_genome_image(
    image_id: int,
) -> tuple[Image.Image, str]:
    """
    Load a Visual Genome image directly from images1.zip or images2.zip.

    Returns:
        image:
            RGB PIL image.

        image_location:
            A string describing the ZIP archive and archive member, for
            example:

                /path/images1.zip::VG_100K/123.jpg
    """
    image_id = int(image_id)
    direct_candidates = candidate_archive_members(image_id)

    archive_errors: list[str] = []

    for archive_path in VG_IMAGE_ARCHIVES:
        if not archive_path.is_file():
            archive_errors.append(
                f"Archive does not exist: {archive_path}"
            )
            continue

        try:
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive_names = archive.namelist()
                archive_name_set = set(archive_names)

                # Fast path: check standard Visual Genome paths.
                for member_name in direct_candidates:
                    if member_name not in archive_name_set:
                        continue

                    image_bytes = archive.read(member_name)

                    with Image.open(io.BytesIO(image_bytes)) as source:
                        image = source.convert("RGB")

                    image_location = (
                        f"{archive_path}::{member_name}"
                    )

                    return image, image_location

                # Fallback: match by filename regardless of directory depth.
                valid_filenames = {
                    f"{image_id}.jpg".casefold(),
                    f"{image_id}.jpeg".casefold(),
                    f"{image_id}.png".casefold(),
                }

                for member_name in archive_names:
                    member_filename = Path(member_name).name.casefold()

                    if member_filename not in valid_filenames:
                        continue

                    image_bytes = archive.read(member_name)

                    with Image.open(io.BytesIO(image_bytes)) as source:
                        image = source.convert("RGB")

                    image_location = (
                        f"{archive_path}::{member_name}"
                    )

                    return image, image_location

        except zipfile.BadZipFile as error:
            archive_errors.append(
                f"Invalid ZIP archive {archive_path}: {error}"
            )
        except OSError as error:
            archive_errors.append(
                f"Could not read {archive_path}: {error}"
            )

    details = "\n".join(
        f"  - {message}"
        for message in archive_errors
    )

    if not details:
        details = "  - Both archives were searched successfully."

    raise FileNotFoundError(
        f"Could not find Visual Genome image ID {image_id} "
        f"inside images1.zip or images2.zip.\n"
        f"{details}"
    )


# =============================================================================
# DEVICE AND MODEL
# =============================================================================

def choose_device(force_cpu: bool) -> torch.device:
    """Use CUDA when available unless CPU inference was requested."""
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
    """Select an inference dtype appropriate for the device."""
    if device.type != "cuda":
        return torch.float32

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16

    return torch.float16


def load_openclip_model(
    device: torch.device,
):
    """Load OpenCLIP ViT-H/14 and its preprocessing pipeline."""
    OPENCLIP_CACHE.mkdir(
        parents=True,
        exist_ok=True,
    )

    dtype = choose_dtype(device)

    print("Loading OpenCLIP model")
    print("=" * 80)
    print(f"Model architecture: {MODEL_NAME}")
    print(f"Pretrained weights: {PRETRAINED_WEIGHTS}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Cache directory: {OPENCLIP_CACHE}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=MODEL_NAME,
        pretrained=PRETRAINED_WEIGHTS,
        device=device,
        cache_dir=str(OPENCLIP_CACHE),
    )

    tokenizer = open_clip.get_tokenizer(
        MODEL_NAME
    )

    model.eval()

    if device.type == "cuda":
        model = model.to(dtype=dtype)

    return model, preprocess, tokenizer, dtype


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def encode_image(
    image: Image.Image,
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode the entire Visual Genome image."""
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    if device.type == "cuda":
        image_tensor = image_tensor.to(dtype=dtype)

    with torch.inference_mode():
        image_features = model.encode_image(
            image_tensor
        )

    image_features = image_features.float()

    image_features = F.normalize(
        image_features,
        p=2,
        dim=-1,
    )

    return image_features


def encode_object_names(
    object_names: list[str],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    Encode object names exactly as they appear in the CSV.

    No templates such as "a photo of a ..." are added.
    """
    if batch_size < 1:
        raise ValueError(
            "Text batch size must be at least 1."
        )

    feature_batches: list[torch.Tensor] = []
    total = len(object_names)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_names = object_names[start:end]

        tokens = tokenizer(batch_names)
        tokens = tokens.to(device)

        with torch.inference_mode():
            text_features = model.encode_text(
                tokens
            )

        text_features = text_features.float()

        text_features = F.normalize(
            text_features,
            p=2,
            dim=-1,
        )

        feature_batches.append(
            text_features.cpu()
        )

        print(
            f"Encoded object names: {end:,}/{total:,}"
        )

    return torch.cat(
        feature_batches,
        dim=0,
    )


# =============================================================================
# SIMILARITY
# =============================================================================

def compute_sorted_scores(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    object_names: list[str],
) -> list[tuple[str, float]]:
    """
    Compute cosine similarity between the image and each object name.

    Both image and text features are L2-normalized, so their dot product
    equals cosine similarity.
    """
    image_features_cpu = image_features.cpu()

    similarities = (
        image_features_cpu
        @ text_features.T
    ).squeeze(0)

    sorted_scores, sorted_indices = torch.sort(
        similarities,
        descending=True,
    )

    results: list[tuple[str, float]] = []

    for score, index in zip(
        sorted_scores.tolist(),
        sorted_indices.tolist(),
    ):
        results.append(
            (
                object_names[int(index)],
                float(score),
            )
        )

    return results


# =============================================================================
# OUTPUT
# =============================================================================

def save_scores(
    output_path: Path,
    image_id: int,
    image_location: str,
    results: list[tuple[str, float]],
    output_top_k: int | None,
) -> int:
    """
    Save CLIP scores to CSV.

    When output_top_k is None, every object score is saved.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_top_k is None:
        rows = results
    else:
        rows = results[:output_top_k]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as destination:
        writer = csv.writer(
            destination,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
            doublequote=True,
            lineterminator="\n",
        )

        writer.writerow(
            [
                "image_id",
                "image_location",
                "rank",
                "object",
                "clip_score",
            ]
        )

        for rank, (object_name, score) in enumerate(
            rows,
            start=1,
        ):
            writer.writerow(
                [
                    image_id,
                    image_location,
                    rank,
                    object_name,
                    f"{score:.8f}",
                ]
            )

    return len(rows)


def print_top_results(
    results: list[tuple[str, float]],
    top_k: int,
) -> None:
    """Print the highest-scoring object names."""
    top_k = min(
        max(top_k, 0),
        len(results),
    )

    print()
    print(f"TOP {top_k} OBJECT NAMES")
    print("=" * 80)

    for rank, (object_name, score) in enumerate(
        results[:top_k],
        start=1,
    ):
        print(
            f"{rank:4d}. "
            f"{object_name:<50} "
            f"{score:.8f}"
        )


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a Visual Genome image directly from images1.zip or "
            "images2.zip and compute OpenCLIP similarity against raw "
            "object names from a CSV file."
        )
    )

    parser.add_argument(
        "--image_id",
        type=int,
        required=True,
        help=(
            "Visual Genome image ID, for example 123."
        ),
    )

    parser.add_argument(
        "--objects_csv",
        type=Path,
        default=DEFAULT_OBJECTS_CSV,
        help=(
            "CSV containing the Visual Genome object vocabulary. "
            f"Default: {DEFAULT_OBJECTS_CSV}"
        ),
    )

    parser.add_argument(
        "--object_column",
        type=str,
        default="object",
        help=(
            "Name of the CSV column containing object names. "
            "Default: object"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. By default, the script writes "
            "clip_scores_<image_id>.csv."
        ),
    )

    parser.add_argument(
        "--print_gt_objects",
        action="store_true",
        help="Print Visual Genome ground-truth objects and exit.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help=(
            "Number of highest-scoring object names printed to stdout. "
            "Default: 20"
        ),
    )

    parser.add_argument(
        "--output_top_k",
        type=int,
        default=None,
        help=(
            "Save only the top K scores to the output CSV. "
            "By default, all scores are saved."
        ),
    )

    parser.add_argument(
        "--text_batch_size",
        type=int,
        default=256,
        help=(
            "Number of object names encoded per text batch. "
            "Default: 256"
        ),
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference instead of CUDA.",
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_arguments()

    if args.image_id < 0:
        raise ValueError(
            "--image_id must be non-negative."
        )

    if args.text_batch_size < 1:
        raise ValueError(
            "--text_batch_size must be at least 1."
        )

    if args.top_k < 0:
        raise ValueError(
            "--top_k cannot be negative."
        )

    if (
        args.output_top_k is not None
        and args.output_top_k < 1
    ):
        raise ValueError(
            "--output_top_k must be at least 1."
        )

    output_path = args.output

    if output_path is None:
        output_path = Path(
            f"clip_scores_{args.image_id}.csv"
        )

    print("VISUAL GENOME IMAGE-TEXT SIMILARITY")
    print("=" * 80)
    print(f"Image ID: {args.image_id}")
    print(f"Objects CSV: {args.objects_csv}")
    print(f"Object column: {args.object_column}")
    print(f"Output CSV: {output_path}")

    print()
    print("Loading Visual Genome image")
    print("=" * 80)

    image, image_location = load_visual_genome_image(
        image_id=args.image_id,
    )

    print(f"Image location: {image_location}")
    print(f"Image size: {image.width} x {image.height}")

    if args.print_gt_objects:
        print_visual_genome_objects(args.image_id)
        return

    print()
    print("Loading object vocabulary")
    print("=" * 80)

    object_names = read_unique_objects(
        csv_path=args.objects_csv,
        column=args.object_column,
    )

    print(
        f"Unique object names loaded: {len(object_names):,}"
    )
    print("Prompt templates: disabled")

    device = choose_device(
        force_cpu=args.cpu,
    )

    model, preprocess, tokenizer, dtype = load_openclip_model(
        device=device,
    )

    print()
    print("Encoding image")
    print("=" * 80)

    image_features = encode_image(
        image=image,
        model=model,
        preprocess=preprocess,
        device=device,
        dtype=dtype,
    )

    print(
        f"Image feature shape: "
        f"{tuple(image_features.shape)}"
    )

    print()
    print("Encoding object names")
    print("=" * 80)

    text_features = encode_object_names(
        object_names=object_names,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.text_batch_size,
    )

    print(
        f"Text feature shape: "
        f"{tuple(text_features.shape)}"
    )

    print()
    print("Computing cosine similarities")
    print("=" * 80)

    results = compute_sorted_scores(
        image_features=image_features,
        text_features=text_features,
        object_names=object_names,
    )

    rows_written = save_scores(
        output_path=output_path,
        image_id=args.image_id,
        image_location=image_location,
        results=results,
        output_top_k=args.output_top_k,
    )

    print_top_results(
        results=results,
        top_k=args.top_k,
    )

    print()
    print("FINISHED")
    print("=" * 80)
    print(f"Image ID: {args.image_id}")
    print(f"Image location: {image_location}")
    print(f"Objects compared: {len(results):,}")
    print(f"Rows written: {rows_written:,}")
    print(f"Output CSV: {output_path.resolve()}")


if __name__ == "__main__":
    main()

