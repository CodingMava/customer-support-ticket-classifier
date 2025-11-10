# model.py
from transformers import DistilBertModel
import torch
import torch.nn as nn

class SupportClassifierModel(nn.Module):
    def __init__(self, num_intents, num_slots):
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden_size = self.encoder.config.hidden_size

        # Intent classification head
        self.intent_classifier = nn.Linear(hidden_size, num_intents)

        # Slot tagging head
        self.slot_classifier = nn.Linear(hidden_size, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]  # [CLS] token

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        return intent_logits, slot_logits
