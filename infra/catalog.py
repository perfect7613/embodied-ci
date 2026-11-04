"""What the studio can pick. One entry per model / task the engine actually runs."""

MODELS = [
    {"id": "vjepa2-ac-vitg", "name": "V-JEPA 2-AC (ViT-g)", "org": "Meta", "params": "1B encoder + 300M predictor",
     "gpu": "H100", "editable": "linear probes on encoder layer + controller (~1.9k params)", "available": True},
]
TASKS = [
    {"id": "lift", "name": "Franka Panda: Lift cube", "sim": "robosuite / MuJoCo",
     "drifts": ["nominal", "low_friction", "heavy_cube", "dim_light", "cube_color", "camera_shift"], "available": True},
]
