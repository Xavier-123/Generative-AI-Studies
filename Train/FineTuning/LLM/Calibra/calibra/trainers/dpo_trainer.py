from __future__ import annotations

from .base_trainer import BaseTrainer


class DPOTrainer(BaseTrainer):
    def __init__(self, policy_model, reference_model, tokenizer, config, collator):
        self.reference_model = reference_model
        super().__init__(policy_model, tokenizer, config, loss_fn=self._loss, collator=collator)

    def _loss(self, model, batch):
        import torch
        from ..loss import dpo_loss, model_sequence_logps
        chosen, chosen_tokens = model_sequence_logps(model, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"], self.config.dpo.normalize_logp_by_length)
        rejected, rejected_tokens = model_sequence_logps(model, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"], self.config.dpo.normalize_logp_by_length)
        reference = self.reference_model or model
        context = reference.disable_adapter() if self.reference_model is None and hasattr(reference, "disable_adapter") else torch.no_grad()
        with torch.no_grad(), context:
            ref_chosen, _ = model_sequence_logps(reference, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"], self.config.dpo.normalize_logp_by_length)
            ref_rejected, _ = model_sequence_logps(reference, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"], self.config.dpo.normalize_logp_by_length)
        metrics = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, self.config.dpo.beta, self.config.dpo.label_smoothing)
        return metrics["loss"], int(chosen_tokens.sum().item() + rejected_tokens.sum().item())
