from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from llm2vec import LLM2Vec
from tqdm.auto import tqdm
from typing import Any

import torch
import torch.nn.functional as F
from llm2vec import LLM2Vec
from tqdm.auto import tqdm

from visual_genome_scene_graph_dataset import (
    DEFAULT_VG_ROOT,
    clean_text,
    VisualGenomeSceneGraphDataset,
    scene_graph_collate_fn,
)


VG_ROOT = Path(
    "/net/tscratch/people/plgpiotrwojcik/vg"
)


LLM2VEC_MODEL_NAME = (
    "McGill-NLP/"
    "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
)

LLM2VEC_ADAPTER_NAME = (
    "McGill-NLP/"
    "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"
)


@torch.inference_mode()
def encode_pair_strings(
    llm2vec: LLM2Vec,
    batch: dict[str, Any],
    encode_batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """
    Encode all strings appearing in positive and negative triples.

    Each unique normalized string is passed through LLM2Vec only once.

    Returns:
        positive_anchor_embeddings: [P, D]
        positive_text_embeddings:   [P, D]
        negative_anchor_embeddings: [N, D]
        negative_text_embeddings:   [N, D]

    where:
        P = number of positive triples
        N = number of negative triples
        D = embedding dimension, normally 4096
    """

    positive_triples = batch["positive_triples"]
    negative_triples = batch["negative_triples"]

    positive_anchor_texts = [
        clean_text(triple.anchor_text)
        for triple in positive_triples
    ]

    positive_texts = [
        clean_text(triple.positive_text)
        for triple in positive_triples
    ]

    negative_anchor_texts = [
        clean_text(triple.anchor_text)
        for triple in negative_triples
    ]

    negative_texts = [
        clean_text(triple.negative_text)
        for triple in negative_triples
    ]

    all_texts = (
        positive_anchor_texts
        + positive_texts
        + negative_anchor_texts
        + negative_texts
    )

    # Preserve order while deduplicating.
    unique_texts = list(dict.fromkeys(all_texts))

    if not unique_texts:
        device = next(llm2vec.model.parameters()).device
        empty = torch.empty(
            (0, 4096),
            dtype=torch.float32,
            device=device,
        )

        return {
            "positive_anchor_embeddings": empty,
            "positive_text_embeddings": empty,
            "negative_anchor_embeddings": empty,
            "negative_text_embeddings": empty,
        }

    unique_embeddings = llm2vec.encode(
        unique_texts,
        batch_size=encode_batch_size,
        show_progress_bar=False,
    )

    if not torch.is_tensor(unique_embeddings):
        unique_embeddings = torch.as_tensor(
            unique_embeddings
        )

    model_device = next(
        llm2vec.model.parameters()
    ).device

    unique_embeddings = unique_embeddings.to(
        device=model_device,
        dtype=torch.float32,
    )

    unique_embeddings = F.normalize(
        unique_embeddings,
        p=2,
        dim=-1,
    )

    text_to_index = {
        text: index
        for index, text in enumerate(unique_texts)
    }

    def gather(texts: list[str]) -> torch.Tensor:
        if not texts:
            return unique_embeddings.new_empty(
                (0, unique_embeddings.shape[-1])
            )

        indices = torch.tensor(
            [text_to_index[text] for text in texts],
            dtype=torch.long,
            device=unique_embeddings.device,
        )

        return unique_embeddings.index_select(
            dim=0,
            index=indices,
        )

    return {
        "positive_anchor_embeddings": gather(
            positive_anchor_texts
        ),
        "positive_text_embeddings": gather(
            positive_texts
        ),
        "negative_anchor_embeddings": gather(
            negative_anchor_texts
        ),
        "negative_text_embeddings": gather(
            negative_texts
        ),
    }


def load_llm2vec(
    cache_dir: str | None = None,
) -> LLM2Vec:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for practical LLaMA-3-8B inference."
        )

    model = LLM2Vec.from_pretrained(
        LLM2VEC_MODEL_NAME,
        peft_model_name_or_path=LLM2VEC_ADAPTER_NAME,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        pooling_mode="mean",
        max_length=64,
    )

    model.eval()
    return model


if __name__ == "__main__":
    from functools import partial

    import torch
    import torch.nn.functional as F
    from llm2vec import LLM2Vec
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    dataset = VisualGenomeSceneGraphDataset(
        root=DEFAULT_VG_ROOT,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        collate_fn=partial(
            scene_graph_collate_fn,
            negatives_per_positive=4,
        ),
    )

    llm2vec = load_llm2vec(
        cache_dir=(
            "/net/tscratch/people/"
            "plgpiotrwojcik/model_cache"
        )
    )

    try:
        for batch_index, batch in enumerate(
            tqdm(
                dataloader,
                desc="Visual Genome batches",
            )
        ):
            pair_embeddings = encode_pair_strings(
                llm2vec=llm2vec,
                batch=batch,
                encode_batch_size=32,
            )

            positive_anchor_embeddings = pair_embeddings[
                "positive_anchor_embeddings"
            ]

            positive_text_embeddings = pair_embeddings[
                "positive_text_embeddings"
            ]

            negative_anchor_embeddings = pair_embeddings[
                "negative_anchor_embeddings"
            ]

            negative_text_embeddings = pair_embeddings[
                "negative_text_embeddings"
            ]

            print(f"\n{'=' * 80}")
            print(f"Batch {batch_index}")
            print(f"{'=' * 80}")

            print(
                f"Images: {len(batch['image_id'])}"
            )
            print(
                f"Positive pairs: "
                f"{len(batch['positive_triples'])}"
            )
            print(
                f"Negative pairs: "
                f"{len(batch['negative_triples'])}"
            )

            print("\nPositive pairs")
            print("-" * 80)

            for index, triple in enumerate(
                batch["positive_triples"]
            ):
                similarity = F.cosine_similarity(
                    positive_anchor_embeddings[
                        index
                    ].unsqueeze(0),
                    positive_text_embeddings[
                        index
                    ].unsqueeze(0),
                ).item()

                print(
                    f"image={triple.image_id:<8}"
                    f"{triple.anchor_text!r}"
                    f" <-> "
                    f"{triple.positive_text!r}"
                    f"  similarity={similarity:.4f}"
                )

            print("\nNegative pairs")
            print("-" * 80)

            for index, triple in enumerate(
                batch["negative_triples"]
            ):
                similarity = F.cosine_similarity(
                    negative_anchor_embeddings[
                        index
                    ].unsqueeze(0),
                    negative_text_embeddings[
                        index
                    ].unsqueeze(0),
                ).item()

                print(
                    f"image={triple.image_id:<8}"
                    f"{triple.anchor_text!r}"
                    f" <-> "
                    f"{triple.negative_text!r}"
                    f"  similarity={similarity:.4f}"
                )

            print()

    finally:
        dataset.close()