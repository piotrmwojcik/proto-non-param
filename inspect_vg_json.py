#!/usr/bin/env python3

import argparse
from pathlib import Path

import ijson
import json


VG_ROOT = Path("/net/tscratch/people/plgpiotrwojcik/vg")

OBJECTS_JSON = VG_ROOT / "objects.json"
ATTRIBUTES_JSON = VG_ROOT / "attributes.json"
RELATIONSHIPS_JSON = VG_ROOT / "relationships.json"
REGION_DESCRIPTIONS_JSON = VG_ROOT / "region_descriptions.json"


def find_entry_stream(path, image_id, field_name):
    """
    Find an annotation record and return the requested field.

    Visual Genome files usually use top-level "image_id", while
    region_descriptions.json may use top-level "id".
    """
    image_id = int(image_id)

    with path.open("rb") as file:
        for item in ijson.items(file, "item"):
            raw_id = item.get("image_id", item.get("id", -1))

            try:
                current_image_id = int(raw_id)
            except (TypeError, ValueError):
                continue

            if current_image_id == image_id:
                return item.get(field_name, [])

    return []

def get_object_name(entity, object_id_to_name):
    """
    Return the best available name for an object.

    Names from objects.json are preferred. If an object is missing there,
    the name embedded in a relationship or attribute record is used.
    """
    object_id = entity.get("object_id")

    if object_id in object_id_to_name:
        return object_id_to_name[object_id]

    if entity.get("name"):
        return entity["name"]

    names = entity.get("names", [])
    if names:
        return names[0]

    return "<unknown object>"


def print_raw_attributes(attributes):
    """Print the raw attributes.json entry for the image."""
    print("\nRAW ATTRIBUTES JSON")
    print("=" * 80)

    if not attributes:
        print("No attributes found.")
        return

    print(json.dumps(attributes, indent=2, ensure_ascii=False))

def print_objects(objects, object_id_to_name):
    """Print object names, IDs, and bounding boxes."""
    print("\nOBJECTS")
    print("=" * 80)

    if not objects:
        print("No objects found in objects.json.")
        return

    for obj in objects:
        object_id = obj.get("object_id")
        names = obj.get("names", [])
        name = names[0] if names else "<no name>"

        if object_id is not None:
            object_id_to_name[object_id] = name

        print(
            f"{name} [{object_id}] "
            f"has bounding box "
            f"(x={obj.get('x')}, y={obj.get('y')}, "
            f"width={obj.get('w')}, height={obj.get('h')})."
        )


def print_attributes(attributes, object_id_to_name):
    """Print attributes as readable sentences."""
    print("\nATTRIBUTES")
    print("=" * 80)

    if not attributes:
        print("No attributes found.")
        return

    for item in attributes:
        object_id = item.get("object_id")

        names = item.get("names", [])
        fallback_name = names[0] if names else "<unknown object>"
        object_name = object_id_to_name.get(object_id, fallback_name)

        if object_id is not None and object_name != "<unknown object>":
            object_id_to_name.setdefault(object_id, object_name)

        values = item.get("attributes", [])

        if values:
            readable_values = ", ".join(str(value) for value in values)
            print(
                f"{object_name} [{object_id}] "
                f"has the following attributes: {readable_values}."
            )
        else:
            print(
                f"{object_name} [{object_id}] "
                f"has no listed attributes."
            )


def print_relationships(relationships, object_id_to_name):
    """Print every relationship as a complete readable sentence."""
    print("\nRELATIONSHIPS")
    print("=" * 80)

    if not relationships:
        print("No relationships found.")
        return

    for relationship in relationships:
        subject = relationship.get("subject", {}) or {}
        object_entity = relationship.get("object", {}) or {}

        subject_id = subject.get("object_id")
        object_id = object_entity.get("object_id")

        subject_name = get_object_name(subject, object_id_to_name)
        object_name = get_object_name(object_entity, object_id_to_name)

        predicate = relationship.get("predicate")
        relationship_id = relationship.get("relationship_id")

        print(
            f"{subject_name} [{subject_id}] "
            f"{predicate} "
            f"{object_name} [{object_id}]. "
            f"(relationship ID: {relationship_id})"
        )


def print_region_descriptions(regions):
    """
    Print all region descriptions for the image.

    Each region contains a natural-language phrase and a bounding box.
    These phrases are the closest Visual Genome provides to complete
    image descriptions, although each phrase generally describes only
    one region of the image.
    """
    print("\nREGION DESCRIPTIONS")
    print("=" * 80)

    if not regions:
        print("No region descriptions found.")
        return

    for index, region in enumerate(regions, start=1):
        region_id = region.get("region_id")
        phrase = region.get("phrase", "<no description>")

        x = region.get("x")
        y = region.get("y")
        width = region.get("width", region.get("w"))
        height = region.get("height", region.get("h"))

        print(f"\nRegion {index}")
        print(f"  Region ID: {region_id}")
        print(f"  Description: {phrase}")
        print(
            "  Bounding box: "
            f"x={x}, y={y}, width={width}, height={height}"
        )


def print_combined_description(regions):
    """
    Print all region phrases together as one image-level text block.

    This is not an official Visual Genome caption. It is a concatenation
    of all region descriptions for convenient inspection.
    """
    print("\nCOMBINED IMAGE DESCRIPTION")
    print("=" * 80)

    phrases = []

    for region in regions:
        phrase = region.get("phrase")

        if phrase:
            phrase = " ".join(str(phrase).strip().split())

            if phrase and phrase not in phrases:
                phrases.append(phrase)

    if not phrases:
        print("No descriptions are available.")
        return

    for phrase in phrases:
        if phrase[-1] not in ".!?":
            phrase += "."

        print(phrase)


def main(image_id, print_raw_attributes_json=False):
    """Load and print all available annotations for an image."""
    print(f"Searching annotations for image_id={image_id}...")

    objects = find_entry_stream(
        OBJECTS_JSON,
        image_id,
        "objects",
    )

    attributes = find_entry_stream(
        ATTRIBUTES_JSON,
        image_id,
        "attributes",
    )

    if print_raw_attributes_json:
        print_raw_attributes(attributes)

    relationships = find_entry_stream(
        RELATIONSHIPS_JSON,
        image_id,
        "relationships",
    )

    regions = find_entry_stream(
        REGION_DESCRIPTIONS_JSON,
        image_id,
        "regions",
    )

    object_id_to_name = {}

    print(f"\nIMAGE ID: {image_id}")

    print_objects(objects, object_id_to_name)

    # Attributes may provide object names when objects.json is incomplete.
    print_attributes(attributes, object_id_to_name)

    print_relationships(relationships, object_id_to_name)

    print_region_descriptions(regions)

    # Concatenate all unique region phrases into one text section.
    #print_combined_description(regions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Print Visual Genome objects, attributes, relationships, "
            "and region descriptions for an image."
        )
    )

    parser.add_argument(
        "--image_id",
        type=int,
        required=True,
        help="Visual Genome image ID, for example 123.",
    )

    parser.add_argument(
        "--print_raw_attributes",
        action="store_true",
        default=True,
        help="Print the raw attributes.json annotation for the image.",
    )

    args = parser.parse_args()
    main(args.image_id, print_raw_attributes_json=args.print_raw_attributes)
