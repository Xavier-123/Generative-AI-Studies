from .base_trainer import BaseTrainer


class SFTTrainer(BaseTrainer):
    """SFT and Agent-SFT trainer; the formatter determines supervised turns."""

    def __init__(self, model, tokenizer, config, collator):
        from ..loss import compute_loss
        super().__init__(model, tokenizer, config, loss_fn=compute_loss, collator=collator)
