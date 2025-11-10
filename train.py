# train.py
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, AdamW
from model import SupportClassifierModel
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import json

class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, texts, intents, tokenizer, max_len=64):
        self.texts = texts
        self.intents = intents
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze() for k, v in item.items()}
        item['labels'] = torch.tensor(self.intents[idx])
        return item

def train_model(train_csv, output_dir, epochs=3, batch_size=8):
    df = pd.read_csv(train_csv)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    le_intent = LabelEncoder()
    df['intent_label'] = le_intent.fit_transform(df['intent'])
    num_intents = len(le_intent.classes_)
    num_slots = 10  # Simplified placeholder

    dataset = TicketDataset(df['text'].tolist(), df['intent_label'].tolist(), tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SupportClassifierModel(num_intents=num_intents, num_slots=num_slots)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids, att_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            intent_logits, slot_logits = model(input_ids, att_mask)
            loss = torch.nn.functional.cross_entropy(intent_logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "best_model.pt")
    torch.save(model.state_dict(), model_save_path)
    with open(os.path.join(output_dir, "intent_labels.json"), "w") as f:
        json.dump(le_intent.classes_.tolist(), f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train_model(args.train_csv, args.output_dir, args.epochs)
