"""
sentence_generator.py — VocalEyes natural sentence generator with dedup + cooldown
"""

import time

# Per-label state tracking
_last_state: dict[str, str] = {}
_last_spoken: dict[str, float] = {}

# Cooldown seconds per priority level
COOLDOWN = {
    "HIGH": 2,
    "MEDIUM": 5,
    "LOW": 8,
}


def generate_sentence(obj: dict, priority: str) -> str | None:
    """
    Generate a spoken sentence for a detected object.
    Returns None if the object is on cooldown or state hasn't changed.

    obj keys: label, direction, distance
    """
    label = obj["label"]
    direction = obj["direction"]
    distance = obj["distance"]

    current_state = f"{label}-{direction}-{distance}"
    now = time.time()

    # Check cooldown
    cooldown = COOLDOWN.get(priority, 8)
    last_time = _last_spoken.get(label, 0)
    if now - last_time < cooldown:
        return None

    # Check state change (skip if same state and not HIGH)
    if priority != "HIGH" and _last_state.get(label) == current_state:
        return None

    _last_state[label] = current_state
    _last_spoken[label] = now

    # Generate sentence
    if priority == "HIGH":
        if distance == "very near":
            return f"DANGER! {label} very close on your {direction}!"
        elif distance == "near":
            return f"Warning! {label} nearby on your {direction}."
        elif distance == "medium":
            return f"{label.capitalize()} approaching from the {direction}."
        else:
            return f"{label.capitalize()} detected on your {direction}."

    elif priority == "MEDIUM":
        if distance in ("very near", "near"):
            return f"Watch out — {label} on your {direction}."
        elif label == "person":
            return "A person is in front of you."
        else:
            return f"{label.capitalize()} spotted {direction}."

    else:
        return f"{label.capitalize()} {distance} on your {direction}."


def reset_state():
    """Reset all tracking — call when restarting detection."""
    _last_state.clear()
    _last_spoken.clear()
