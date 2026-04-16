#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm


ALLOWED_CERTAINTY_IDS = {3, 4}


def load_id_name_file(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed line in {path} at line {line_num}: {raw!r}"
                )
            mapping[int(parts[0])] = parts[1]
    return mapping


def load_image_attribute_labels(path: Path) -> List[Tuple[int, int, int, int]]:
    rows: List[Tuple[int, int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(
                    f"Expected at least 4 columns in {path} at line {line_num}"
                )
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
    return rows


def load_bounding_boxes(path: Path) -> Dict[int, Tuple[float, float, float, float]]:
    mapping: Dict[int, Tuple[float, float, float, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Expected 5 columns in {path} at line {line_num}: {raw!r}"
                )
            mapping[int(parts[0])] = tuple(map(float, parts[1:5]))
    return mapping


def load_part_locations(path: Path) -> Dict[int, List[Tuple[int, float, float, int]]]:
    mapping: Dict[int, List[Tuple[int, float, float, int]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Expected 5 columns in {path} at line {line_num}: {raw!r}"
                )
            image_id = int(parts[0])
            part_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            visible = int(parts[4])
            mapping[image_id].append((part_id, x, y, visible))
    return mapping


def clean_token(text: str) -> str:
    text = text.replace("::", " ")
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_has_prefix(text: str) -> str:
    text = clean_token(text)
    if text.startswith("has "):
        return text[4:].strip()
    return text


def prettify_value(value: str) -> str:
    value = clean_token(value)
    value = value.replace("(up or down)", "upturned or downturned")
    value = value.replace("long-legged-like", "long-legged")
    value = value.replace("seabird", "seabird-like")
    value = value.replace("about_the_same_as_head", "about the same as head")
    value = value.replace("very_large_(32_-_72_in)", "very large (32 - 72 in)")
    value = value.replace("large_(16_-_32_in)", "large (16 - 32 in)")
    value = value.replace("small_(5_-_9_in)", "small (5 - 9 in)")
    value = value.replace("medium_(9_-_16_in)", "medium (9 - 16 in)")
    return value


def attribute_to_phrase(attr: str) -> str:
    if "::" in attr:
        left, right = attr.split("::", 1)
        left = strip_has_prefix(left)
        value = prettify_value(right)
    else:
        raw = strip_has_prefix(attr)
        parts = raw.split()
        if len(parts) < 2:
            return clean_token(attr)
        left = " ".join(parts[:-1])
        value = prettify_value(parts[-1])

    left = clean_token(left)
    parts = left.split()

    if len(parts) < 2:
        return f"{value} {left}".strip()

    feature_main = parts[-1]
    feature_mods = parts[:-1]
    target = " ".join(feature_mods)

    if feature_main == "color":
        if target == "eye":
            return f"{value} eyes"
        if target == "primary":
            return f"mainly {value}"
        if target == "wing":
            return f"{value} wings"
        if target == "leg":
            return f"{value} legs"
        return f"{value} {target}"

    if feature_main == "pattern":
        return f"{value} {target} pattern"

    if feature_main == "shape":
        if target == "bill":
            return f"{value} bill"
        if target == "head":
            return f"{value} head"
        if target == "tail":
            return f"{value} tail"
        return f"{value} {target}"

    if feature_main == "length":
        if value == "longer than head" and target == "bill":
            return "bill longer than its head"
        if value == "shorter than head" and target == "bill":
            return "bill shorter than its head"
        if value == "about the same as head" and target == "bill":
            return "bill about the same length as its head"
        return f"{target} length {value}"

    if feature_main == "size":
        return f"{value} size"

    return f"{value} {target} {feature_main}".strip()


def group_present_probable_attributes(
    attributes: Dict[int, str],
    label_rows: List[Tuple[int, int, int, int]],
) -> Dict[int, List[str]]:
    grouped: Dict[int, List[str]] = defaultdict(list)
    for image_id, attribute_id, is_present, certainty_id in label_rows:
        if is_present != 1:
            continue
        if certainty_id not in ALLOWED_CERTAINTY_IDS:
            continue
        raw_attr = attributes.get(attribute_id, f"attribute_{attribute_id}")
        grouped[image_id].append(attribute_to_phrase(raw_attr))
    return grouped


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_present_features_with_certainty(
    image_id: int,
    attributes: Dict[int, str],
    label_rows: List[Tuple[int, int, int, int]],
) -> List[Tuple[str, int]]:
    strict_features: List[Tuple[str, int]] = []
    fallback_features: List[Tuple[str, int]] = []

    for row_image_id, attribute_id, is_present, certainty_id in label_rows:
        if row_image_id != image_id:
            continue
        if is_present != 1:
            continue

        raw_attr = attributes.get(attribute_id, f"attribute_{attribute_id}")
        phrase = attribute_to_phrase(raw_attr)

        fallback_features.append((phrase, certainty_id))
        if certainty_id in ALLOWED_CERTAINTY_IDS:
            strict_features.append((phrase, certainty_id))

    chosen = strict_features if strict_features else fallback_features

    best: Dict[str, int] = {}
    for feat, cert in chosen:
        if feat not in best or cert > best[feat]:
            best[feat] = cert

    return list(best.items())


def normalize_trait_phrase(trait: str) -> str:
    trait = trait.strip(" ,.;:")

    replacements = {
        r"\bbill bill\b": "bill",
        r"\bdagger bill\b": "dagger-shaped bill",
        r"\bdagger-shaped bill bill\b": "dagger-shaped bill",
        r"\bcone bill\b": "cone-shaped bill",
        r"\bcone-shaped bill bill\b": "cone-shaped bill",
        r"\bspatulate bill bill\b": "spatulate bill",
        r"\bhooked\b$": "hooked bill",
        r"\bdagger\b$": "dagger-shaped bill",
        r"\bcone\b$": "cone-shaped bill",
        r"\bspatulate\b$": "spatulate bill",
        r"\bpointed\b$": "pointed bill",
        r"\bcurved\b$": "curved bill",
        r"\blong\b$": "long bill",
        r"\bshort\b$": "short bill",
        r"\blarge\b$": "large size",
        r"\bsmall\b$": "small size",
        r"\bmedium\b$": "medium size",
        r"\bpointed tail tail\b": "pointed tail",
        r"\bnotched tail tail\b": "notched tail",
        r"\bfan-shaped tail tail\b": "fan-shaped tail",
        r"\bforked tail tail\b": "forked tail",
        r"\blong-wings wing\b": "long wings",
        r"\bpointed-wings wing\b": "pointed wings",
        r"\bbroad-wings wing\b": "broad wings",
        r"\brounded-wings wing\b": "rounded wings",
        r"\bblack wing\b": "black wings",
        r"\bbrown wing\b": "brown wings",
        r"\bblue wing\b": "blue wings",
        r"\bgrey wing\b": "grey wings",
        r"\bwhite wing\b": "white wings",
        r"\bgrey primary\b": "grey primaries",
        r"\bblack primary\b": "black primaries",
        r"\bbrown primary\b": "brown primaries",
        r"\bblue primary\b": "blue primaries",
        r"\bwhite primary\b": "white primaries",
        r"\bbrown, and black wings\b": "brown and black wings",
        r"\bblue, and grey wings\b": "blue and grey wings",
        r"\bbrown, white wings\b": "brown and white wings",
        r"\bblack, white wings\b": "black and white wings",
        r"\bgrey, white underparts\b": "grey and white underparts",
        r"\bbrown and white upperparts\b": "brown and white upperparts",
        r"\bgrey and white underparts\b": "grey and white underparts",
        r"\bplain head pattern\b": "plain head",
        r"\bmasked head pattern\b": "masked head",
        r"\beyeline head pattern\b": "eyeline",
        r"\beyebrow head pattern\b": "eyebrow stripe",
        r"\bmalar head pattern\b": "malar pattern",
        r"\bsolid wing pattern\b": "solid wings",
        r"\bstriped wing pattern\b": "striped wings",
        r"\bspotted wing pattern\b": "spotted wings",
        r"\bmulti-colored wing pattern\b": "multi-colored wings",
        r"\bsolid belly pattern\b": "solid belly",
        r"\bmulti-colored belly pattern\b": "multi-colored belly",
        r"\bsolid breast pattern\b": "solid breast",
        r"\bstriped breast pattern\b": "striped breast",
        r"\bspotted breast pattern\b": "spotted breast",
        r"\bsolid back pattern\b": "solid back",
        r"\bmulti-colored back pattern\b": "multi-colored back",
        r"\bsolid tail pattern\b": "solid tail",
        r"\bmulti-colored tail pattern\b": "multi-colored tail",
        r"\bwhite mask around its eyes\b": "white eye mask",
        r"\baround its eyes\b": "eye mask",
        r"\bplain white\b": "white",
        r"\bgray\b": "grey",
    }

    for pattern, repl in replacements.items():
        trait = re.sub(pattern, repl, trait, flags=re.IGNORECASE)

    trait = re.sub(r"\s+", " ", trait).strip(" ,.;:")
    return trait


def is_color_word(word: str) -> bool:
    return word.lower() in {
        "black",
        "white",
        "grey",
        "brown",
        "buff",
        "red",
        "blue",
        "green",
        "yellow",
        "olive",
        "rufous",
        "pink",
        "purple",
        "orange",
        "chestnut",
        "tan",
        "cream",
        "golden",
    }


def extract_trait_key(trait: str) -> str:
    words = trait.lower().split()
    if not words:
        return trait.lower()

    direct_traits = {
        "striped wings": "wings",
        "spotted wings": "wings",
        "solid wings": "wings",
        "multi-colored wings": "wings",
        "solid tail": "tail",
        "notched tail": "tail",
        "fan-shaped tail": "tail",
        "pointed tail": "tail",
        "solid breast": "breast",
        "striped breast": "breast",
        "spotted breast": "breast",
        "solid belly": "belly",
        "striped belly": "belly",
        "solid back": "back",
        "multi-colored back": "back",
        "plain head": "head",
        "masked head": "head",
        "capped head": "head",
        "crested head": "head",
        "eyeline": "head",
        "eyebrow stripe": "head",
        "malar pattern": "head",
    }
    if trait.lower() in direct_traits:
        return direct_traits[trait.lower()]

    known_heads = {
        "bill",
        "eyes",
        "eye",
        "wing",
        "wings",
        "tail",
        "throat",
        "breast",
        "nape",
        "upperparts",
        "underparts",
        "primaries",
        "head",
        "crown",
        "belly",
        "back",
        "legs",
        "leg",
        "mask",
        "cheeks",
        "cheek",
        "forehead",
        "neck",
        "size",
    }

    for word in reversed(words):
        if word in known_heads:
            if word == "eye":
                return "eyes"
            if word == "wing":
                return "wings"
            if word == "leg":
                return "legs"
            if word == "cheek":
                return "cheeks"
            return word

    return words[-1]


def merge_similar_traits(traits: List[str]) -> List[str]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for trait in traits:
        key = extract_trait_key(trait)
        grouped[key].append(trait)

    merged: List[str] = []

    for key, vals in grouped.items():
        vals = list(dict.fromkeys(vals))

        if len(vals) == 1:
            merged.append(vals[0])
            continue

        color_prefixes: List[str] = []
        non_color_vals: List[str] = []

        for v in vals:
            words = v.split()
            if len(words) >= 2 and words[-1].lower() == key:
                prefix = " ".join(words[:-1]).strip()
                prefix_parts = [
                    p.strip() for p in re.split(r"\band\b", prefix) if p.strip()
                ]
                if prefix_parts and all(
                    is_color_word(p.split()[0]) for p in prefix_parts
                ):
                    color_prefixes.append(prefix)
                else:
                    non_color_vals.append(v)
            else:
                non_color_vals.append(v)

        if color_prefixes:
            unique_prefixes: List[str] = []
            seen_prefixes = set()
            for p in color_prefixes:
                pl = p.lower()
                if pl not in seen_prefixes:
                    seen_prefixes.add(pl)
                    unique_prefixes.append(p)

            if len(unique_prefixes) == 1:
                merged.append(f"{unique_prefixes[0]} {key}")
            elif len(unique_prefixes) == 2:
                merged.append(f"{unique_prefixes[0]} and {unique_prefixes[1]} {key}")
            else:
                merged.append(
                    f"{', '.join(unique_prefixes[:-1])}, and {unique_prefixes[-1]} {key}"
                )

        merged.extend(non_color_vals)

    return merged


def score_trait(trait: str) -> int:
    key = extract_trait_key(trait)

    direct_trait_priority = {
        "hooked bill": 4,
        "dagger-shaped bill": 5,
        "cone-shaped bill": 6,
        "spatulate bill": 7,
        "notched tail": 8,
        "fan-shaped tail": 9,
        "pointed tail": 10,
        "striped wings": 11,
        "spotted wings": 12,
        "solid wings": 13,
        "striped breast": 14,
        "spotted breast": 15,
        "plain head": 16,
        "masked head": 17,
        "eyeline": 18,
    }

    if trait.lower() in direct_trait_priority:
        return direct_trait_priority[trait.lower()]

    priority_order = {
        "bill": 20,
        "eyes": 21,
        "tail": 22,
        "wings": 23,
        "upperparts": 24,
        "underparts": 25,
        "breast": 26,
        "throat": 27,
        "head": 28,
        "nape": 29,
        "primaries": 30,
        "crown": 31,
        "back": 32,
        "belly": 33,
        "neck": 34,
        "cheeks": 35,
        "forehead": 36,
        "mask": 37,
        "legs": 38,
        "size": 45,
    }

    base = priority_order.get(key, 99)
    descriptive_bonus = -1 if len(trait.split()) >= 2 else 0
    return base + descriptive_bonus


def build_raw_caption(_: str, phrases: List[str], max_traits: int = 18) -> str:
    phrases = dedupe_keep_order(phrases)[:max_traits]
    if not phrases:
        return "A photo of a bird."
    joined = ", ".join(phrases)
    return f"A photo of a bird with {joined}."


def clean_single_caption(
    text: str,
    max_words: int = 28,
    max_traits: int = 8,
) -> str:
    if not text:
        return "A photo of a bird."

    text = text.strip()
    text = re.sub(
        r"^(assistant|human|user|system)\s*:\s*", "", text, flags=re.IGNORECASE
    )
    text = re.sub(r"^(caption|final caption)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[-*]\s*", "", text)
    text = re.sub(r"^\s*\d+[\).\s-]+", "", text)

    text = re.split(r"(?<=[.!?])\s+", text)[0].strip()
    text = re.sub(r"<\|.*?\|>", "", text).strip()

    replacements = {
        r"\bfeaturing\b": "",
        r"\bshowing\b": "",
        r"\bappearing to have\b": "with",
        r"\bappears to have\b": "with",
        r"\bwith a\b": "with",
        r"\bwith an\b": "with",
        r"\s+": " ",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,+", ",", text).strip(" ,.;:")

    if not text:
        return "A photo of a bird."

    if not text.lower().startswith("a photo of a bird"):
        if " with " in text:
            _, tail = text.split(" with ", 1)
            text = f"A photo of a bird with {tail.strip()}"
        else:
            text = f"A photo of a bird with {text}"

    if " with " in text:
        _, traits = text.split(" with ", 1)
        raw_traits = re.split(r",| and ", traits)
        normalized_traits: List[str] = []
        seen = set()

        for trait in raw_traits:
            trait = normalize_trait_phrase(trait)
            if not trait:
                continue
            key = trait.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized_traits.append(trait)

        normalized_traits = merge_similar_traits(normalized_traits)
        normalized_traits = list(dict.fromkeys(normalized_traits))
        normalized_traits.sort(key=score_trait)
        normalized_traits = normalized_traits[:max_traits]
        normalized_traits = [
            t
            for t in normalized_traits
            if not (len(t.split()) == 1 and is_color_word(t))
        ]

        def is_primary_color_trait(trait: str) -> bool:
            return trait.startswith("mainly ") or trait.startswith("primarily ")

        primary_traits = [t for t in normalized_traits if is_primary_color_trait(t)]
        other_traits = [t for t in normalized_traits if not is_primary_color_trait(t)]

        prefix = "A photo of a bird"
        parts = []

        if primary_traits:
            parts.append("that is " + " and ".join(primary_traits))

        if other_traits:
            if len(other_traits) == 1:
                parts.append(f"with {other_traits[0]}")
            elif len(other_traits) == 2:
                parts.append(f"with {other_traits[0]} and {other_traits[1]}")
            else:
                parts.append(
                    "with " + ", ".join(other_traits[:-1]) + f", and {other_traits[-1]}"
                )

        text = prefix + (" " + " ".join(parts) if parts else "")

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).strip(" ,.;:")

    return text.rstrip(".") + "."


def sample_feature_subsets_weighted(
    feature_pairs: List[Tuple[str, int]],
    num_subsets: int = 5,
    min_features: int = 3,
    max_features: int = 6,
    seed: Optional[int] = None,
) -> List[List[str]]:
    rng = random.Random(seed)

    normalized_pairs: List[Tuple[str, int]] = []
    seen = set()
    for feat, cert in feature_pairs:
        nf = normalize_trait_phrase(feat)
        if not nf:
            continue
        if nf in seen:
            continue
        seen.add(nf)
        normalized_pairs.append((nf, cert))

    if not normalized_pairs:
        return [[] for _ in range(num_subsets)]

    population = [feat for feat, _ in normalized_pairs]
    weights = [max(cert, 1) for _, cert in normalized_pairs]

    subsets: List[List[str]] = []
    for _ in range(num_subsets):
        k = min(len(population), rng.randint(min_features, max_features))

        available_feats = population[:]
        available_weights = weights[:]
        subset: List[str] = []

        for _ in range(k):
            if not available_feats:
                break
            chosen = rng.choices(available_feats, weights=available_weights, k=1)[0]
            idx = available_feats.index(chosen)
            subset.append(chosen)
            del available_feats[idx]
            del available_weights[idx]

        subset = dedupe_keep_order(subset)
        subset.sort(key=score_trait)
        subsets.append(subset)

    return subsets


def build_random_split_captions(
    feature_pairs: List[Tuple[str, int]],
    max_words: int = 28,
    num_captions: int = 5,
    seed: Optional[int] = None,
) -> List[str]:
    subsets = sample_feature_subsets_weighted(
        feature_pairs,
        num_subsets=num_captions,
        min_features=3,
        max_features=6,
        seed=seed,
    )

    captions: List[str] = []
    for subset in subsets:
        if not subset:
            cap = "A photo of a bird."
        elif len(subset) == 1:
            cap = f"A photo of a bird with {subset[0]}"
        elif len(subset) == 2:
            cap = f"A photo of a bird with {subset[0]} and {subset[1]}"
        else:
            cap = (
                "A photo of a bird with "
                + ", ".join(subset[:-1])
                + f", and {subset[-1]}"
            )
        captions.append(clean_single_caption(cap, max_words=max_words, max_traits=8))

    captions = dedupe_keep_order(captions)
    while len(captions) < num_captions:
        captions.append(captions[len(captions) % len(captions)])

    return captions[:num_captions]


def build_single_caption_messages_from_features(
    features: List[str],
    max_words: int = 28,
) -> List[Dict[str, str]]:
    if features:
        description = "A photo of a bird with " + ", ".join(features) + "."
    else:
        description = "A photo of a bird."

    user_prompt = (
        "You write bird captions.\n"
        "Return exactly one caption and nothing else.\n"
        f"Maximum {max_words} words.\n"
        'The caption must start with "A photo of a bird".\n'
        "Do not mention the species name or class name.\n"
        "Use only the visible traits from the description.\n"
        "Do not invent or guess traits.\n"
        "Do not compress too aggressively.\n"
        "Repeat the adjective for each body part when needed for clarity.\n"
        "Keep natural bird-description language.\n"
        "If the description contains no visible traits, return exactly: A photo of a bird.\n\n"
        f"Description: {description}"
    )

    return [{"role": "user", "content": user_prompt}]


def make_mlx_rewriter(
    model_name: str,
    max_tokens: int,
    max_words: int,
    num_captions: int = 5,
) -> Callable[[List[Tuple[str, int]]], List[str]]:
    from mlx_lm import load, generate

    model, tokenizer = load(model_name)

    def rewrite(feature_pairs: List[Tuple[str, int]]) -> List[str]:
        subsets = sample_feature_subsets_weighted(
            feature_pairs,
            num_subsets=num_captions,
            min_features=3,
            max_features=6,
        )

        captions: List[str] = []

        for subset in subsets:
            messages = build_single_caption_messages_from_features(
                subset,
                max_words=max_words,
            )

            merged = "\n\n".join(m["content"] for m in messages)

            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    prompt = merged + "\nCaption:"
            else:
                prompt = merged + "\nCaption:"

            output = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            captions.append(
                clean_single_caption(output, max_words=max_words, max_traits=8)
            )

        captions = dedupe_keep_order(captions)
        while len(captions) < num_captions:
            captions.append(captions[len(captions) % len(captions)])

        return captions[:num_captions]

    return rewrite


def make_hf_rewriter(
    model_name: str,
    max_tokens: int,
    max_words: int,
    num_captions: int = 5,
) -> Callable[[List[Tuple[str, int]]], List[str]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def rewrite(feature_pairs: List[Tuple[str, int]]) -> List[str]:
        subsets = sample_feature_subsets_weighted(
            feature_pairs,
            num_subsets=num_captions,
            min_features=3,
            max_features=6,
        )

        captions: List[str] = []

        for subset in subsets:
            messages = build_single_caption_messages_from_features(
                subset,
                max_words=max_words,
            )

            merged = "\n\n".join(m["content"] for m in messages)

            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    prompt = merged + "\nCaption:"
            else:
                prompt = merged + "\nCaption:"

            out = pipe(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                return_full_text=False,
            )
            captions.append(
                clean_single_caption(
                    out[0]["generated_text"],
                    max_words=max_words,
                    max_traits=8,
                )
            )

        captions = dedupe_keep_order(captions)
        while len(captions) < num_captions:
            captions.append(captions[len(captions) % len(captions)])

        return captions[:num_captions]

    return rewrite


def get_rewriter(
    use_llm: bool,
    llm_backend: str,
    llm_model: str,
    llm_max_tokens: int,
    caption_max_words: int,
    num_captions: int = 5,
) -> Optional[Callable[[List[Tuple[str, int]]], List[str]]]:
    if not use_llm:
        return None
    if llm_backend == "mlx":
        return make_mlx_rewriter(
            llm_model,
            llm_max_tokens,
            caption_max_words,
            num_captions=num_captions,
        )
    if llm_backend == "hf":
        return make_hf_rewriter(
            llm_model,
            llm_max_tokens,
            caption_max_words,
            num_captions=num_captions,
        )
    raise ValueError(f"Unsupported llm_backend: {llm_backend}")


def crop_to_bbox(
    image: Image.Image,
    bbox: Tuple[float, float, float, float],
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    x, y, w, h = bbox
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(image.width, int(round(x + w)))
    bottom = min(image.height, int(round(y + h)))

    if right <= left or bottom <= top:
        raise ValueError(f"Invalid bbox after clipping: {(left, top, right, bottom)}")

    cropped = image.crop((left, top, right, bottom))
    return cropped, (left, top, right, bottom)


def shift_parts_to_crop(
    parts_for_image: List[Tuple[int, float, float, int]],
    crop_box: Tuple[int, int, int, int],
) -> List[Tuple[int, float, float, int]]:
    left, top, right, bottom = crop_box
    shifted: List[Tuple[int, float, float, int]] = []

    for part_id, px, py, visible in parts_for_image:
        if visible != 1:
            continue
        if left <= px <= right and top <= py <= bottom:
            shifted.append((part_id, px - left, py - top, visible))

    return shifted


def draw_parts_on_crop(
    image: Image.Image,
    parts_for_image: List[Tuple[int, float, float, int]],
    part_names: Dict[int, str],
    draw_part_labels: bool = True,
) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for part_id, px, py, visible in parts_for_image:
        if visible != 1:
            continue

        cx, cy = int(round(px)), int(round(py))
        r = 4
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="lime", outline="black")

        if draw_part_labels:
            label = part_names.get(part_id, f"part_{part_id}")
            draw.text((cx + 6, cy - 6), label, fill="yellow")

    return img


def show_samples(
    output_records: List[dict],
    part_names: Dict[int, str],
    draw_part_labels: bool,
) -> None:
    if not output_records:
        return

    n = len(output_records)
    fig, axes = plt.subplots(n, 1, figsize=(15, 7 * n))
    if n == 1:
        axes = [axes]

    for ax, rec in zip(axes, output_records):
        image = Image.open(rec["cropped_path"]).convert("RGB")
        annotated = draw_parts_on_crop(
            image=image,
            parts_for_image=rec["shifted_parts"],
            part_names=part_names,
            draw_part_labels=draw_part_labels,
        )

        visible_parts = [
            part_names.get(part_id, f"part_{part_id}")
            for part_id, _, _, visible in rec["shifted_parts"]
            if visible == 1
        ]

        caption_lines = [
            f"caption {i + 1}: {cap}" for i, cap in enumerate(rec["final_captions"])
        ]
        feature_lines = [f"{feat} (c={cert})" for feat, cert in rec["feature_pairs"]]

        ax.imshow(annotated)
        ax.axis("off")
        ax.set_title(
            "\n".join(
                [
                    f"ID {rec['image_id']}: {rec['original_rel_path']}",
                    f"saved: {rec['cropped_path']}",
                    f"visible parts in crop: {', '.join(visible_parts) if visible_parts else 'none'}",
                    f"present features: {', '.join(feature_lines) if feature_lines else 'none'}",
                    *caption_lines,
                ]
            ),
            fontsize=10,
            loc="left",
        )

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop CUB images to bounding boxes, save crops, and create a CSV with multiple captions per image."
    )
    parser.add_argument("cub_root", type=Path, help="Path to CUB_200_2011 root")
    parser.add_argument("--output-dir", type=Path, default=Path("cub_crops"))
    parser.add_argument("--csv-out", type=Path, default=Path("cub_captions.csv"))
    parser.add_argument("--image-id", type=int, default=None)
    parser.add_argument("--sample-ids", type=int, nargs="*", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--display-samples", action="store_true")
    parser.add_argument("--no-part-labels", action="store_true")

    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-backend", choices=["mlx", "hf"], default="mlx")
    parser.add_argument(
        "--llm-model",
        type=str,
        default="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    )
    parser.add_argument("--llm-max-tokens", type=int, default=96)
    parser.add_argument("--caption-max-words", type=int, default=28)
    parser.add_argument("--num-captions", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--keep-raw-if-llm-fails", action="store_true")

    args = parser.parse_args()

    cub_root = args.cub_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    images_file = cub_root / "images.txt"
    attributes_file = cub_root / "attributes" / "attributes.txt"
    image_attribute_labels_file = cub_root / "attributes" / "image_attribute_labels.txt"
    bounding_boxes_file = cub_root / "bounding_boxes.txt"
    parts_file = cub_root / "parts" / "parts.txt"
    part_locs_file = cub_root / "parts" / "part_locs.txt"

    required_files = [
        images_file,
        attributes_file,
        image_attribute_labels_file,
        bounding_boxes_file,
        parts_file,
        part_locs_file,
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    images = load_id_name_file(images_file)
    attributes = load_id_name_file(attributes_file)
    label_rows = load_image_attribute_labels(image_attribute_labels_file)
    bboxes = load_bounding_boxes(bounding_boxes_file)
    part_names = load_id_name_file(parts_file)
    part_locs = load_part_locations(part_locs_file)
    grouped_attrs = group_present_probable_attributes(attributes, label_rows)

    if args.image_id is not None:
        image_ids = [args.image_id]
    elif args.sample_ids:
        image_ids = args.sample_ids
    else:
        image_ids = sorted(images.keys())

    image_ids = [iid for iid in image_ids if iid in images and iid in bboxes]

    if args.max_images is not None:
        image_ids = image_ids[: args.max_images]

    rewriter: Optional[Callable[[List[Tuple[str, int]]], List[str]]] = None
    if args.use_llm:
        rewriter = get_rewriter(
            use_llm=True,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            llm_max_tokens=args.llm_max_tokens,
            caption_max_words=args.caption_max_words,
            num_captions=args.num_captions,
        )

    sample_records: List[dict] = []

    with args.csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(
            ["filepath"] + [f"caption_{i}" for i in range(1, args.num_captions + 1)]
        )
        f.flush()

        for idx, image_id in enumerate(
            tqdm(image_ids, desc="Processing birds", unit="img")
        ):
            rel_path = images[image_id]
            src_path = cub_root / "images" / rel_path
            bbox = bboxes[image_id]

            grouped_phrases = grouped_attrs.get(image_id, [])
            feature_pairs = extract_present_features_with_certainty(
                image_id=image_id,
                attributes=attributes,
                label_rows=label_rows,
            )

            if grouped_phrases:
                grouped_set = {normalize_trait_phrase(p) for p in grouped_phrases}
                feature_pairs = [
                    (normalize_trait_phrase(feat), cert)
                    for feat, cert in feature_pairs
                    if normalize_trait_phrase(feat) in grouped_set
                ] or [
                    (normalize_trait_phrase(feat), cert) for feat, cert in feature_pairs
                ]
            else:
                feature_pairs = [
                    (normalize_trait_phrase(feat), cert) for feat, cert in feature_pairs
                ]

            best_pairs: Dict[str, int] = {}
            for feat, cert in feature_pairs:
                if feat not in best_pairs or cert > best_pairs[feat]:
                    best_pairs[feat] = cert
            feature_pairs = list(best_pairs.items())
            feature_pairs.sort(key=lambda x: score_trait(x[0]))

            phrases = [feat for feat, _ in feature_pairs]
            raw_caption = build_raw_caption(rel_path, phrases)

            if not feature_pairs:
                final_captions = ["A photo of a bird."] * args.num_captions
            else:
                local_seed = (
                    None if args.random_seed is None else args.random_seed + idx
                )
                final_captions = build_random_split_captions(
                    feature_pairs,
                    max_words=args.caption_max_words,
                    num_captions=args.num_captions,
                    seed=local_seed,
                )

                if rewriter is not None:
                    try:
                        rewritten_list = rewriter(feature_pairs)
                        if rewritten_list:
                            final_captions = rewritten_list
                    except Exception as e:
                        if args.keep_raw_if_llm_fails:
                            print(
                                f"Warning: LLM rewrite failed for image {image_id}: {e}"
                            )
                        else:
                            raise

            feature_text = (
                ", ".join(f"{feat} (c={cert})" for feat, cert in feature_pairs)
                if feature_pairs
                else "none"
            )

            print(f"present features: {feature_text}")
            for cap_i, cap in enumerate(final_captions, start=1):
                print(f"caption {cap_i}: {cap}")
            print("-" * 80)

            image = Image.open(src_path).convert("RGB")
            cropped, crop_box = crop_to_bbox(image, bbox)

            out_rel = Path(rel_path).with_suffix(".png")
            out_path = output_dir / out_rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(out_path)

            writer.writerow((str(out_path.resolve()), *final_captions))
            f.flush()

            shifted_parts = shift_parts_to_crop(part_locs.get(image_id, []), crop_box)

            sample_records.append(
                {
                    "image_id": image_id,
                    "original_rel_path": rel_path,
                    "cropped_path": str(out_path),
                    "final_captions": final_captions,
                    "feature_pairs": feature_pairs,
                    "shifted_parts": shifted_parts,
                }
            )

    print(f"Saved {len(image_ids)} cropped images to: {output_dir}")
    print(f"Saved CSV to: {args.csv_out}")

    if args.display_samples:
        show_samples(
            output_records=sample_records[: min(5, len(sample_records))],
            part_names=part_names,
            draw_part_labels=not args.no_part_labels,
        )


if __name__ == "__main__":
    main()
