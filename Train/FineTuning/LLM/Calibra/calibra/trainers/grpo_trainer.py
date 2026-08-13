from .base_trainer import BaseTrainer


class GRPOTrainer(BaseTrainer):
    """Extension point for grouped rollout and relative-advantage optimization."""

    def train(self, *args, **kwargs):
        raise NotImplementedError("GRPO requires a rollout backend and reward function")
