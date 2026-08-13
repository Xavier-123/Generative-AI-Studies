"""Policy-loss extension point for PPO/GRPO plugins."""


def policy_loss(*args, **kwargs):
    raise NotImplementedError("Use a PPO/GRPO plugin to provide a policy loss")
