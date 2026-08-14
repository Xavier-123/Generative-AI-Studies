"""Example custom reward for configs/grpo_config.yaml.

Use it as ``grpo.reward: examples.grpo_rewards:answer_match`` after adding the
project root to PYTHONPATH (which scripts/train.py does automatically).
"""

from calibra.rollout import RolloutSample


def answer_match(sample: RolloutSample) -> float:
    answer = sample.metadata.get("answer")
    if answer is None:
        return 0.0
    return float(sample.response.strip() == str(answer).strip())
