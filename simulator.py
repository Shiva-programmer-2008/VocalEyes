import time

def simulate_objects():
    # Simulate one object getting closer over time
    scenarios = [
        {"label": "car", "direction": "right", "distance": "far"},
        {"label": "car", "direction": "right", "distance": "medium"},
        {"label": "car", "direction": "right", "distance": "near"},
        {"label": "person", "direction": "center", "distance": "medium"},
        {"label": "stairs", "direction": "center", "distance": "near"}
    ]

    for obj in scenarios:
        time.sleep(2)  # simulate real-time delay
        yield obj