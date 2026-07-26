"""
Memory-efficient Visual Genome PyTorch Dataset.

Annotations are streamed once into an on-disk SQLite database. __getitem__()
loads only one image's objects, attributes, and relationships into memory.
Images are read directly from images1.zip/images2.zip and are never saved.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    import ijson
except ImportError as exc:
    raise ImportError(
        "Install ijson first: python -m pip install ijson"
    ) from exc


DEFAULT_VG_ROOT = Path("/net/tscratch/people/plgpiotrwojcik/vg")


def build_default_image_transform(
    image_size: int = 518,
) -> Callable[[Image.Image], torch.Tensor]:
    """Return the default DINOv2-compatible image preprocessing pipeline."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def clean_text(value: Any) -> str:
    return " ".join(str(value).replace("\x00", " ").strip().split())


def clean_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]

    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            output.append(text)
    return output


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def stream_json_array(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as source:
        yield from ijson.items(source, "item")


def record_names(record: dict[str, Any]) -> list[str]:
    values = record.get("names")
    if not values and record.get("name") is not None:
        values = [record["name"]]
    return clean_list(values)


def create_annotation_database(
    root: Path,
    database_path: Path,
    rebuild: bool = False,
) -> None:
    """
    Build the SQLite annotation index.

    The temporary .building file prevents an interrupted build from being
    mistaken for a complete database.
    """
    if database_path.is_file() and not rebuild:
        connection = sqlite3.connect(database_path)
        try:
            has_descriptions = connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'descriptions'
                """
            ).fetchone() is not None
        finally:
            connection.close()

        if has_descriptions:
            return

        print(
            "Existing annotation database has no descriptions table; "
            "rebuilding it."
        )

    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = database_path.with_suffix(database_path.suffix + ".building")
    temporary_path.unlink(missing_ok=True)

    connection = sqlite3.connect(temporary_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-65536")

        connection.executescript(
            """
            CREATE TABLE objects (
                image_id INTEGER NOT NULL,
                object_id INTEGER NOT NULL,
                names_json TEXT NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                w INTEGER NOT NULL,
                h INTEGER NOT NULL,
                attributes_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (image_id, object_id)
            );

            CREATE TABLE relationships (
                image_id INTEGER NOT NULL,
                relationship_id INTEGER,
                predicate TEXT NOT NULL,
                subject_id INTEGER NOT NULL,
                object_id INTEGER NOT NULL
            );

            CREATE TABLE descriptions (
                image_id INTEGER NOT NULL,
                region_id INTEGER,
                phrase TEXT NOT NULL,
                x INTEGER,
                y INTEGER,
                w INTEGER,
                h INTEGER
            );

            CREATE INDEX objects_image_idx
                ON objects(image_id);

            CREATE INDEX relationships_image_idx
                ON relationships(image_id);

            CREATE INDEX descriptions_image_idx
                ON descriptions(image_id);
            """
        )

        print("Indexing objects.json...")
        object_rows: list[tuple[Any, ...]] = []
        for number, image_record in enumerate(
            stream_json_array(root / "objects.json"), start=1
        ):
            image_id = as_int(image_record.get("image_id"))
            if image_id is None:
                continue

            for record in image_record.get("objects", []):
                object_id = as_int(record.get("object_id"))
                if object_id is None:
                    continue

                object_rows.append(
                    (
                        image_id,
                        object_id,
                        json.dumps(record_names(record), ensure_ascii=False),
                        as_int(record.get("x"), 0) or 0,
                        as_int(record.get("y"), 0) or 0,
                        as_int(record.get("w"), 0) or 0,
                        as_int(record.get("h"), 0) or 0,
                    )
                )

            if len(object_rows) >= 50_000:
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO objects
                    (image_id, object_id, names_json, x, y, w, h)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    object_rows,
                )
                object_rows.clear()

            if number % 10_000 == 0:
                print(f"  object images indexed: {number:,}")

        if object_rows:
            connection.executemany(
                """
                INSERT OR REPLACE INTO objects
                (image_id, object_id, names_json, x, y, w, h)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                object_rows,
            )

        print("Indexing attributes.json...")
        attribute_rows: list[tuple[str, int, int]] = []
        placeholder_rows: list[tuple[Any, ...]] = []

        for number, image_record in enumerate(
            stream_json_array(root / "attributes.json"), start=1
        ):
            image_id = as_int(image_record.get("image_id"))
            if image_id is None:
                continue

            records = image_record.get("attributes", image_record.get("objects", []))
            for record in records:
                object_id = as_int(record.get("object_id"))
                if object_id is None:
                    continue

                names = record_names(record)
                placeholder_rows.append(
                    (
                        image_id,
                        object_id,
                        json.dumps(names, ensure_ascii=False),
                        as_int(record.get("x"), 0) or 0,
                        as_int(record.get("y"), 0) or 0,
                        as_int(record.get("w"), 0) or 0,
                        as_int(record.get("h"), 0) or 0,
                        json.dumps(clean_list(record.get("attributes")), ensure_ascii=False),
                    )
                )
                attribute_rows.append(
                    (
                        json.dumps(
                            clean_list(record.get("attributes")),
                            ensure_ascii=False,
                        ),
                        image_id,
                        object_id,
                    )
                )

            if len(attribute_rows) >= 50_000:
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO objects
                    (image_id, object_id, names_json, x, y, w, h, attributes_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    placeholder_rows,
                )
                connection.executemany(
                    """
                    UPDATE objects
                    SET attributes_json = ?
                    WHERE image_id = ? AND object_id = ?
                    """,
                    attribute_rows,
                )
                placeholder_rows.clear()
                attribute_rows.clear()

            if number % 10_000 == 0:
                print(f"  attribute images indexed: {number:,}")

        if attribute_rows:
            connection.executemany(
                """
                INSERT OR IGNORE INTO objects
                (image_id, object_id, names_json, x, y, w, h, attributes_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                placeholder_rows,
            )
            connection.executemany(
                """
                UPDATE objects
                SET attributes_json = ?
                WHERE image_id = ? AND object_id = ?
                """,
                attribute_rows,
            )

        print("Indexing relationships.json...")
        relationship_rows: list[tuple[Any, ...]] = []
        endpoint_rows: list[tuple[Any, ...]] = []

        for number, image_record in enumerate(
            stream_json_array(root / "relationships.json"), start=1
        ):
            image_id = as_int(image_record.get("image_id"))
            if image_id is None:
                continue

            for record in image_record.get("relationships", []):
                subject = record.get("subject") or {}
                obj = record.get("object") or {}
                subject_id = as_int(subject.get("object_id"))
                object_id = as_int(obj.get("object_id"))

                if subject_id is None or object_id is None:
                    continue

                for endpoint, endpoint_id in (
                    (subject, subject_id),
                    (obj, object_id),
                ):
                    endpoint_rows.append(
                        (
                            image_id,
                            endpoint_id,
                            json.dumps(record_names(endpoint), ensure_ascii=False),
                            as_int(endpoint.get("x"), 0) or 0,
                            as_int(endpoint.get("y"), 0) or 0,
                            as_int(endpoint.get("w"), 0) or 0,
                            as_int(endpoint.get("h"), 0) or 0,
                        )
                    )

                relationship_rows.append(
                    (
                        image_id,
                        as_int(record.get("relationship_id")),
                        clean_text(record.get("predicate", "")),
                        subject_id,
                        object_id,
                    )
                )

            if len(relationship_rows) >= 50_000:
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO objects
                    (image_id, object_id, names_json, x, y, w, h)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    endpoint_rows,
                )
                connection.executemany(
                    """
                    INSERT INTO relationships
                    (image_id, relationship_id, predicate, subject_id, object_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    relationship_rows,
                )
                endpoint_rows.clear()
                relationship_rows.clear()

            if number % 10_000 == 0:
                print(f"  relationship images indexed: {number:,}")

        if relationship_rows:
            connection.executemany(
                """
                INSERT OR IGNORE INTO objects
                (image_id, object_id, names_json, x, y, w, h)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                endpoint_rows,
            )
            connection.executemany(
                """
                INSERT INTO relationships
                (image_id, relationship_id, predicate, subject_id, object_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                relationship_rows,
            )

        descriptions_path = root / "region_descriptions.json"
        if descriptions_path.is_file():
            print("Indexing region_descriptions.json...")
            description_rows: list[tuple[Any, ...]] = []
            for number, image_record in enumerate(
                stream_json_array(descriptions_path), start=1
            ):
                image_id = as_int(
                    image_record.get("image_id", image_record.get("id"))
                )
                if image_id is None:
                    continue

                for region in image_record.get("regions", []):
                    phrase = clean_text(region.get("phrase", ""))
                    if not phrase:
                        continue
                    description_rows.append(
                        (
                            image_id,
                            as_int(region.get("region_id")),
                            phrase,
                            as_int(region.get("x")),
                            as_int(region.get("y")),
                            as_int(region.get("width", region.get("w"))),
                            as_int(region.get("height", region.get("h"))),
                        )
                    )

                if len(description_rows) >= 50_000:
                    connection.executemany(
                        """
                        INSERT INTO descriptions
                        (image_id, region_id, phrase, x, y, w, h)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        description_rows,
                    )
                    description_rows.clear()

                if number % 10_000 == 0:
                    print(f"  description images indexed: {number:,}")

            if description_rows:
                connection.executemany(
                    """
                    INSERT INTO descriptions
                    (image_id, region_id, phrase, x, y, w, h)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    description_rows,
                )
        else:
            print(f"Warning: description file not found: {descriptions_path}")

        connection.commit()
    except Exception:
        connection.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        connection.close()
        os.replace(temporary_path, database_path)
        print(f"Annotation database created: {database_path}")


def build_zip_image_index(
    archive_paths: tuple[Path, ...],
) -> dict[int, tuple[Path, str]]:
    index: dict[int, tuple[Path, str]] = {}
    valid_suffixes = {".jpg", ".jpeg", ".png"}

    for archive_path in archive_paths:
        if not archive_path.is_file():
            raise FileNotFoundError(archive_path)

        with zipfile.ZipFile(archive_path, "r") as archive:
            for member_name in archive.namelist():
                path = Path(member_name)
                if path.suffix.casefold() not in valid_suffixes:
                    continue
                try:
                    image_id = int(path.stem)
                except ValueError:
                    continue
                index.setdefault(image_id, (archive_path, member_name))

    return index


@dataclass(frozen=True)
class PositiveTriple:
    """An image, an object anchor, and its matching relationship text."""

    image: Any
    image_id: int
    object_id: int
    anchor_text: str
    positive_text: str
    relationship_id: int | None = None


@dataclass(frozen=True)
class NegativeTriple:
    """An anchor image, an anchor object, and a different object name."""

    image: Any
    image_id: int
    anchor_object_id: int
    anchor_text: str
    negative_text: str
    negative_image_id: int
    negative_object_id: int


def normalize_name(value: Any) -> str:
    """Normalize complete object names for equality checks."""

    return clean_text(value).casefold()


def build_relationship_positives(
    relationships: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build per-image positives of the form:

        subject -> subject predicate object

    Example:
        license plate -> license plate ON car

    The relationship direction is preserved exactly as stored in Visual Genome.
    """

    positives: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()

    for relationship in relationships:
        subject_name = clean_text(
            relationship.get("subject_name")
        )
        predicate = clean_text(
            relationship.get("predicate")
        )
        object_name = clean_text(
            relationship.get("object_name")
        )
        subject_id = as_int(
            relationship.get("subject_id")
        )

        if (
            subject_id is None
            or not subject_name
            or not predicate
            or not object_name
        ):
            continue

        positive_text = clean_text(
            f"{subject_name} {predicate} {object_name}"
        )

        anchor_text = subject_name

        key = (
            subject_id,
            normalize_name(anchor_text),
            positive_text.casefold(),
        )

        if key in seen:
            continue

        seen.add(key)

        positives.append(
            {
                "relationship_id": as_int(
                    relationship.get("relationship_id")
                ),
                "object_id": subject_id,
                "anchor_text": anchor_text,
                "positive_text": positive_text,
            }
        )

    return positives


class VisualGenomeSceneGraphDataset(Dataset):
    def __init__(
        self,
        root: str | Path = DEFAULT_VG_ROOT,
        transform: Callable[[Image.Image], Any] | None = None,
        image_size: int = 518,
        image_ids: list[int] | None = None,
        database_path: str | Path | None = None,
        rebuild_database: bool = False,
    ) -> None:
        self.root = Path(root)
        self.transform = (
            transform
            if transform is not None
            else build_default_image_transform(image_size)
        )
        self.archive_paths = (
            self.root / "images1.zip",
            self.root / "images2.zip",
        )

        self.database_path = Path(
            database_path or self.root / "scene_graph_annotations.sqlite"
        )

        create_annotation_database(
            root=self.root,
            database_path=self.database_path,
            rebuild=rebuild_database,
        )

        print("Indexing ZIP member names...")
        self._image_index = build_zip_image_index(self.archive_paths)

        connection = sqlite3.connect(self.database_path)
        try:
            annotated_ids = {
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT image_id FROM objects"
                )
            }
        finally:
            connection.close()

        available = set(self._image_index) & annotated_ids
        if image_ids is not None:
            available &= {int(value) for value in image_ids}

        self.image_ids = sorted(available)
        if not self.image_ids:
            raise ValueError("No matching images found.")

        self._connection: sqlite3.Connection | None = None
        self._zip_handles: dict[Path, zipfile.ZipFile] = {}

    def __len__(self) -> int:
        return len(self.image_ids)

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.database_path,
                check_same_thread=False,
            )
        return self._connection

    def _zip(self, path: Path) -> zipfile.ZipFile:
        handle = self._zip_handles.get(path)
        if handle is None:
            handle = zipfile.ZipFile(path, "r")
            self._zip_handles[path] = handle
        return handle

    def _load_image(self, image_id: int) -> tuple[Image.Image, str]:
        archive_path, member_name = self._image_index[image_id]
        raw = self._zip(archive_path).read(member_name)
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert("RGB").copy()
        return image, f"{archive_path}::{member_name}"

    def _load_scene_graph(
        self,
        image_id: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        connection = self._db()

        object_rows = connection.execute(
            """
            SELECT object_id, names_json, x, y, w, h, attributes_json
            FROM objects
            WHERE image_id = ?
            ORDER BY object_id
            """,
            (image_id,),
        ).fetchall()

        objects_by_id: dict[int, dict[str, Any]] = {}
        for object_id, names_json, x, y, w, h, attributes_json in object_rows:
            names = json.loads(names_json)
            attributes = json.loads(attributes_json)
            objects_by_id[object_id] = {
                "object_id": object_id,
                "names": names,
                "name": names[0] if names else None,
                "bbox_xywh": [x, y, w, h],
                "bbox_xyxy": [x, y, x + w, y + h],
                "attributes": attributes,
                "outgoing_relationships": [],
                "incoming_relationships": [],
            }

        relationship_rows = connection.execute(
            """
            SELECT relationship_id, predicate, subject_id, object_id
            FROM relationships
            WHERE image_id = ?
            """,
            (image_id,),
        ).fetchall()

        relationships: list[dict[str, Any]] = []
        for relationship_id, predicate, subject_id, object_id in relationship_rows:
            subject = objects_by_id.get(subject_id)
            obj = objects_by_id.get(object_id)
            subject_name = subject["name"] if subject else None
            object_name = obj["name"] if obj else None

            relationship = {
                "relationship_id": relationship_id,
                "predicate": predicate,
                "subject_id": subject_id,
                "subject_name": subject_name,
                "object_id": object_id,
                "object_name": object_name,
            }
            relationships.append(relationship)

            if subject is not None:
                subject["outgoing_relationships"].append(
                    {
                        "relationship_id": relationship_id,
                        "predicate": predicate,
                        "object_id": object_id,
                        "object_name": object_name,
                    }
                )
            #if obj is not None:
            #    obj["incoming_relationships"].append(
            #       {
            #            "relationship_id": relationship_id,
            #            "predicate": predicate,
            #            "subject_id": subject_id,
            #            "subject_name": subject_name,
            #        }
            #    )

        return list(objects_by_id.values()), relationships

    def _load_descriptions(self, image_id: int) -> list[dict[str, Any]]:
        rows = self._db().execute(
            """
            SELECT region_id, phrase, x, y, w, h
            FROM descriptions
            WHERE image_id = ?
            ORDER BY region_id
            """,
            (image_id,),
        ).fetchall()

        return [
            {
                "region_id": region_id,
                "phrase": phrase,
                "bbox_xywh": [x, y, w, h]
                if None not in (x, y, w, h)
                else None,
            }
            for region_id, phrase, x, y, w, h in rows
        ]

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_id = self.image_ids[index]
        image, image_location = self._load_image(image_id)
        if self.transform is not None:
            image = self.transform(image)

        objects, relationships = self._load_scene_graph(image_id)
        descriptions = self._load_descriptions(image_id)
        positive_records = build_relationship_positives(relationships)

        return {
            "image_id": image_id,
            "image": image,
            "image_location": image_location,
            "objects": objects,
            "relationships": relationships,
            "descriptions": descriptions,
            "positive_records": positive_records,
        }

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        for handle in self._zip_handles.values():
            handle.close()
        self._zip_handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_zip_handles"] = {}
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def scene_graph_collate_fn(
    batch: list[dict[str, Any]],
    negatives_per_positive: int | None = 4,
) -> dict[str, Any]:
    """
    Construct image/object/text training triples for a DataLoader batch.

    Positive triple:
        (anchor image, object name, matching relationship text)

    Negative triple:
        (anchor image, object name, different object name)

    Negative candidates come from all object occurrences in the current batch.
    Equal complete object names are excluded after whitespace/case normalization.
    Set ``negatives_per_positive=None`` to use every valid negative object.
    """

    if negatives_per_positive is not None and negatives_per_positive < 0:
        raise ValueError("negatives_per_positive must be non-negative or None")

    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    raw_images = [sample["image"] for sample in batch]
    if not all(torch.is_tensor(image) for image in raw_images):
        image_types = sorted({type(image).__name__ for image in raw_images})
        raise TypeError(
            "Every dataset image must be a torch.Tensor before collation. "
            f"Received element types: {image_types}."
        )

    invalid_shapes = [
        tuple(image.shape)
        for image in raw_images
        if image.ndim != 3
    ]
    if invalid_shapes:
        raise ValueError(
            "Each image must have shape [C, H, W]. "
            f"Invalid shapes: {invalid_shapes[:5]}"
        )

    shapes = {tuple(image.shape) for image in raw_images}
    if len(shapes) != 1:
        raise ValueError(
            "Images must share one shape before stacking. "
            f"Received shapes: {sorted(shapes)}"
        )

    images = torch.stack(raw_images, dim=0)

    positive_triples: list[PositiveTriple] = []
    for sample in batch:
        for record in sample["positive_records"]:
            positive_triples.append(
                PositiveTriple(
                    image=sample["image"],
                    image_id=sample["image_id"],
                    object_id=record["object_id"],
                    anchor_text=record["anchor_text"],
                    positive_text=record["positive_text"],
                    relationship_id=record["relationship_id"],
                )
            )

    # One entry per physical object occurrence. Repeated names are retained
    # because they may belong to different images or object IDs.
    object_pool: list[dict[str, Any]] = []
    seen_objects: set[tuple[int, int, str]] = set()
    for sample in batch:
        for obj in sample["objects"]:
            object_id = as_int(obj.get("object_id"))
            object_name = clean_text(obj.get("name"))
            normalized_name = normalize_name(object_name)
            if object_id is None or not normalized_name:
                continue

            key = (sample["image_id"], object_id, normalized_name)
            if key in seen_objects:
                continue
            seen_objects.add(key)
            object_pool.append(
                {
                    "image_id": sample["image_id"],
                    "object_id": object_id,
                    "text": object_name,
                    "normalized_text": normalized_name,
                }
            )

    negative_triples: list[NegativeTriple] = []
    for positive in positive_triples:
        anchor_name = normalize_name(positive.anchor_text)
        candidates = [
            candidate
            for candidate in object_pool
            if candidate["normalized_text"] != anchor_name
        ]

        # Deterministic truncation keeps DataLoader behavior reproducible.
        # Shuffle the DataLoader to vary which objects share a batch.
        if negatives_per_positive is not None:
            candidates = candidates[:negatives_per_positive]

        for candidate in candidates:
            negative_triples.append(
                NegativeTriple(
                    image=positive.image,
                    image_id=positive.image_id,
                    anchor_object_id=positive.object_id,
                    anchor_text=positive.anchor_text,
                    negative_text=candidate["text"],
                    negative_image_id=candidate["image_id"],
                    negative_object_id=candidate["object_id"],
                )
            )

    return {
        "image_id": [sample["image_id"] for sample in batch],
        "image": images,
        "image_location": [sample["image_location"] for sample in batch],
        "objects": [sample["objects"] for sample in batch],
        "relationships": [sample["relationships"] for sample in batch],
        "descriptions": [sample["descriptions"] for sample in batch],
        "positive_triples": positive_triples,
        "negative_triples": negative_triples,
        "positive_images": [triple.image for triple in positive_triples],
        "positive_anchor_texts": [
            triple.anchor_text for triple in positive_triples
        ],
        "positive_texts": [
            triple.positive_text for triple in positive_triples
        ],
        "negative_images": [triple.image for triple in negative_triples],
        "negative_anchor_texts": [
            triple.anchor_text for triple in negative_triples
        ],
        "negative_texts": [
            triple.negative_text for triple in negative_triples
        ],
    }


if __name__ == "__main__":
    from functools import partial

    from torch.utils.data import DataLoader

    dataset = VisualGenomeSceneGraphDataset(root=DEFAULT_VG_ROOT)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=partial(
            scene_graph_collate_fn,
            negatives_per_positive=4,
        ),
    )

    try:
        batch = next(iter(dataloader))
        print(f"Images: {len(batch['image_id'])}")
        print(f"Positive triples: {len(batch['positive_triples'])}")
        print(f"Negative triples: {len(batch['negative_triples'])}")

        print("\nPositive examples:")
        for triple in batch["positive_triples"][:10]:
            print(
                f"  image={triple.image_id}, "
                f"{triple.anchor_text!r} <-> {triple.positive_text!r}"
            )

        print("\nNegative examples:")
        for triple in batch["negative_triples"][:10]:
            print(
                f"  image={triple.image_id}, "
                f"{triple.anchor_text!r} <-> {triple.negative_text!r}"
            )
    finally:
        dataset.close()