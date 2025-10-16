"""
Script to explore and display the IMDb dataset structure
Shows raw data samples and preprocessing
"""

from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer

def explore_imdb_data():
    """Load and display IMDb dataset samples"""

    print("🔍 Loading IMDb dataset from Hugging Face...")
    print("=" * 60)

    # Load dataset
    dataset = load_dataset("imdb")

    print(f"📊 Dataset Structure:")
    print(f"   Train split: {len(dataset['train'])} samples")
    print(f"   Test split:  {len(dataset['test'])} samples")
    print(f"   Unsupervised split: {len(dataset['unsupervised'])} samples")
    print()

    # Show a few raw samples
    print("📝 Raw Data Samples:")
    print("-" * 40)

    for i in range(3):
        sample = dataset['train'][i]
        print(f"Sample {i+1}:")
        print(f"  Label: {'POSITIVE' if sample['label'] == 1 else 'NEGATIVE'}")
        print(f"  Text length: {len(sample['text'])} characters")
        print(f"  Text preview: {sample['text'][:200]}...")
        print()

    # Show label distribution
    print("📈 Label Distribution in Training Set:")
    print("-" * 40)

    labels = [sample['label'] for sample in dataset['train'][:1000]]  # Check first 1000
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count

    print(f"  POSITIVE (1): {positive_count} samples ({positive_count/len(labels)*100:.1f}%)")
    print(f"  NEGATIVE (0): {negative_count} samples ({negative_count/len(labels)*100:.1f}%)")
    print()

    # Show tokenization example
    print("🔤 Tokenization Example:")
    print("-" * 40)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    sample_text = dataset['train'][0]['text'][:256]  # First 256 chars
    tokens = tokenizer(sample_text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")

    print(f"Original text length: {len(sample_text)} characters")
    print(f"Tokenized input_ids shape: {tokens['input_ids'].shape}")
    print(f"Tokenized attention_mask shape: {tokens['attention_mask'].shape}")
    print(f"Number of tokens: {tokens['input_ids'].numel()}")
    print()

    # Show what happens during preprocessing
    print("⚙️  Preprocessing Pipeline:")
    print("-" * 40)

    print("1. Raw text → Tokenization → Model input")
    print("2. Labels: 0=NEGATIVE, 1=POSITIVE")
    print("3. Max length: 256 tokens (truncated/padded)")
    print("4. Removed columns: ['text'] (kept: ['input_ids', 'attention_mask', 'label'])")
    print()

    # Show processed sample
    print("🔧 Processed Sample (after tokenization):")
    print("-" * 40)

    processed_sample = {
        'input_ids': tokens['input_ids'].squeeze().tolist()[:10],  # First 10 tokens
        'attention_mask': tokens['attention_mask'].squeeze().tolist()[:10],
        'label': dataset['train'][0]['label']
    }

    print(f"  input_ids (first 10): {processed_sample['input_ids']}")
    print(f"  attention_mask (first 10): {processed_sample['attention_mask']}")
    print(f"  label: {processed_sample['label']} ({'POSITIVE' if processed_sample['label'] == 1 else 'NEGATIVE'})")
    print()

    print("✅ Data exploration complete!")

if __name__ == "__main__":
    explore_imdb_data()</content>
<parameter name="filePath">e:\amazon\self-healing-classifier\explore_data.py