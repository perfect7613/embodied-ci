"""Baseten Training job: GRPO post-training of a small guardrail writer.

Reward = the gate's guardrail rules, replayed offline on logged EmbodiedCI steps
(block unsafe proposals, don't block nominal steps, must compile). No sim needed.

    uv run python -m posttrain.build_dataset        # writes data/*.jsonl from the Modal Volume
    baseten train push --config posttrain/baseten_grpo/config.py
"""

from truss.base.truss_config import AcceleratorSpec
from truss_train import CacheConfig, CheckpointingConfig, Compute, Image, Runtime, TrainingJob, TrainingProject

job = TrainingJob(
    image=Image(base_image="pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime"),
    compute=Compute(accelerator=AcceleratorSpec(accelerator="H100", count=1)),
    runtime=Runtime(
        start_commands=["chmod +x ./run.sh && ./run.sh"],
        cache_config=CacheConfig(enabled=True),
        checkpointing_config=CheckpointingConfig(enabled=True),
    ),
)

training_project = TrainingProject(name="embodied-ci-guardrail-grpo", job=job)
