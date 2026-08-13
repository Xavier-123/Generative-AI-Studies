from .base_trainer import BaseTrainer


class AgentRLTrainer(BaseTrainer):
    """Extension point for multi-step environment interaction training."""

    def train(self, *args, **kwargs):
        raise NotImplementedError("Agent RL requires an environment and tool registry")
