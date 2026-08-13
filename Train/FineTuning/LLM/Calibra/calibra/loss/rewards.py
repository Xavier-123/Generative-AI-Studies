"""Reward-model loss extension point."""


def reward_model_loss(*args, **kwargs):
    raise NotImplementedError("Use a reward-model plugin to provide reward loss")
