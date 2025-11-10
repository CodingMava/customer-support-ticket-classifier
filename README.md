# 🧠 AI Project 2: Customer Support Ticket Classifier + Entity Extractor

This project performs **intent classification** and **slot extraction** on customer support queries using a multi-task transformer model.

## 🚀 Features
- Uses **DistilBERT** as shared encoder.
- Two heads: Intent + Slot tagging.
- Includes training + FastAPI inference service.
- Ready for **Docker deployment**.

## ⚙️ Setup
```bash
pip install -r requirements.txt
python train.py --train_csv data/sample_tickets.csv --output_dir outputs --epochs 3
uvicorn app:app --reload
