"""
context_logic.py — VocalEyes priority and context system
"""

PRIORITY_MAP = {
    "HIGH": [
        "car", "stairs", "truck", "bus", "motorcycle",
        "bicycle", "fire hydrant", "stop sign", "traffic light"
    ],
    "MEDIUM": [
        "person", "doors", "dog", "cat", "bird"
    ],
    "LOW": [
        "chair", "table", "couch", "bottle", "cup",
        "text-sign", "book", "laptop", "keyboard"
    ],
}


def get_priority(label: str) -> str:
    """Return HIGH / MEDIUM / LOW priority for a detected label."""
    for level, items in PRIORITY_MAP.items():
        if label in items:
            return level
    return "LOW"


def is_danger(label: str, distance: str) -> bool:
    """Return True if the object poses an immediate danger."""
    return get_priority(label) == "HIGH" and distance in ("very near", "near")


def get_context_description(label: str, distance: str, direction: str) -> str:
    """Return a natural-language sentence for the detected object."""
    priority = get_priority(label)

    if priority == "HIGH":
        if distance == "very near":
            return f"DANGER! {label.capitalize()} very close on your {direction}!"
        elif distance == "near":
            return f"Warning! {label.capitalize()} nearby on your {direction}."
        elif distance == "medium":
            return f"{label.capitalize()} approaching from the {direction}."
        else:
            return f"{label.capitalize()} detected on your {direction}."

    elif priority == "MEDIUM":
        if distance in ("very near", "near"):
            return f"Watch out — {label} on your {direction}."
        else:
            return f"{label.capitalize()} spotted {direction}."

    else:
        return f"{label.capitalize()} {distance} on your {direction}."
