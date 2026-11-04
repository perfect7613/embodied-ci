"""GRPO (TRL) on a small instruct model to write EmbodiedCI guardrails. LoRA, 1x H100."""

import json
import os

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from reward import score

MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
rows = [json.loads(l) for l in open("data/train.jsonl")]
STEPS = {r["id"]: r["steps"] for r in rows}
ds = Dataset.from_list([{"prompt": [{"role": "user", "content": r["prompt"]}], "id": r["id"]} for r in rows])


def guardrail_reward(completions, id, **kw):
    return [score(c[0]["content"] if isinstance(c, list) else c, STEPS[i]) for c, i in zip(completions, id)]


cfg = GRPOConfig(
    output_dir=os.environ.get("BT_CHECKPOINT_DIR", "./ckpt"),
    num_generations=8,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_completion_length=512,
    max_prompt_length=2048,
    learning_rate=1e-5,
    max_steps=int(os.environ.get("MAX_STEPS", "120")),
    logging_steps=5,
    save_steps=40,
    bf16=True,
    report_to="none",
)
trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=[guardrail_reward],
    args=cfg,
    train_dataset=ds,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.train()
trainer.save_model(cfg.output_dir)
