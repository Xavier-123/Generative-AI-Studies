from .base_trainer import BaseTrainer


class PPOTrainer(BaseTrainer):
    """Extension point for PPO; implement rollout/value hooks in a project plugin."""

    def train(self, *args, **kwargs):
        raise NotImplementedError("PPO requires a rollout backend and value model")
