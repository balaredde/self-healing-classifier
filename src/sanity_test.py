"""Quick sanity test for the fine-tuned sentiment classifier.
Run after training to verify that obvious positive/negative phrases map correctly.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import numpy as np

# Get the project root (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(PROJECT_ROOT, "models", "sentiment-classifier"))

def main():
    print(f"Loading model from {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model directory not found at {MODEL_PATH}")
        print(f"Please run fine-tuning first: python src/fine_tune.py")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"Model id2label config: {model.config.id2label}")
    print(f"Model label2id config: {model.config.label2id}\n")

    examples = [
        "i love this movie",
        "this was fantastic",
        "absolutely wonderful experience",
        "i hate this",
        "this was terrible",
        "worst film ever",
        "not good",
        "really bad and disappointing",
    ]

    print("Running sanity predictions:\n")
    with torch.no_grad():
        for text in examples:
            enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=256).to(device)
            out = model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
            pred_id = int(np.argmax(probs))
            
            # Handle id2label which might be dict with int or str keys
            id2label = model.config.id2label
            if isinstance(id2label, dict):
                label = id2label.get(pred_id, id2label.get(str(pred_id), f"LABEL_{pred_id}"))
            else:
                label = id2label[pred_id] if pred_id < len(id2label) else f"LABEL_{pred_id}"
            
            # Map raw labels to human-readable format
            label_map = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}
            display_label = label_map.get(label, label)
            
            print(f"{text:40s} -> {display_label:8s} (P={probs[pred_id]:.2%})  [NEGATIVE={probs[0]:.2%}, POSITIVE={probs[1]:.2%}]")

if __name__ == "__main__":
    main()
