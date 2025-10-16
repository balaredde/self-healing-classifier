"""
Fine-tune DistilBERT with LoRA for Sentiment Classification
Dataset: IMDb Movie Reviews
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentFineTuner:
    """Fine-tune a transformer model with LoRA for sentiment classification"""
    
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        output_dir="./models/sentiment-classifier",
        use_lora=True
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.tokenizer = None
        self.model = None
        
    def load_and_prepare_data(self, max_samples=None):
        """Load and prepare IMDb dataset"""
        logger.info("Loading IMDb dataset...")
        
        # Load dataset
        dataset = load_dataset("imdb")
        
        # Optional: Limit samples for faster training
        # CRITICAL: Shuffle before selecting to avoid class imbalance (IMDb is sorted by label!)
        if max_samples:
            dataset["train"] = dataset["train"].shuffle(seed=42).select(range(max_samples))
            dataset["test"] = dataset["test"].shuffle(seed=42).select(range(max_samples // 5))
        
        logger.info(f"Train samples: {len(dataset['train'])}")
        logger.info(f"Test samples: {len(dataset['test'])}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256  # Reduced from 512 to save memory
            )
        
        logger.info("Tokenizing dataset...")
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        return tokenized_datasets
    
    def create_model(self):
        """Create model with optional LoRA configuration"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load base model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            id2label={0: "NEGATIVE", 1: "POSITIVE"},
            label2id={"NEGATIVE": 0, "POSITIVE": 1}
        )
        
        if self.use_lora:
            logger.info("Applying LoRA configuration...")
            
            # LoRA configuration - CRITICAL: use modules_to_save to train classifier head
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=16,  # Rank
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_lin", "v_lin"],  # DistilBERT attention layers
                bias="none",
                # CRITICAL: This ensures classifier head is trained alongside LoRA
                modules_to_save=["pre_classifier", "classifier"]  
            )
            
            # Apply LoRA
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        self.model = model
        return model
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(
        self,
        tokenized_datasets,
        num_epochs=3,
        batch_size=4,  # Reduced from 16 to 4 for 4GB GPU
        learning_rate=2e-4
    ):
        """Train the model"""
        logger.info("Setting up training...")
        
        # Training arguments optimized for low VRAM
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Simulate batch_size=16 (4*4)
            warmup_steps=500,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,
            eval_strategy="epoch",  # Changed from evaluation_strategy
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            push_to_hub=False,
            report_to="none",
            fp16=True,  # Enable mixed precision training to save memory
            gradient_checkpointing=True,  # Enable gradient checkpointing
            optim="adamw_torch",  # Use PyTorch's AdamW for better memory efficiency
            max_grad_norm=1.0,  # Gradient clipping
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Evaluate
        logger.info("Evaluating model...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        return trainer
    
    def save_model(self, trainer=None):
        """Save the fine-tuned model"""
        logger.info(f"Saving model to {self.output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # For LoRA, we need to merge and save the full model
        if self.use_lora:
            logger.info("Merging LoRA adapter and saving full model...")
            # Merge LoRA weights into base model
            merged_model = self.model.merge_and_unload()
            
            # CRITICAL: Restore id2label mapping that gets lost during merge
            merged_model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
            merged_model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
            
            # IMPORTANT: Save config explicitly
            merged_model.save_pretrained(self.output_dir)
            merged_model.config.save_pretrained(self.output_dir)  # Add this line!
        else:
            self.model.save_pretrained(self.output_dir)
        
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info("Model saved successfully!")

    
    def quick_sanity_check(self):
        """Run a few hand-crafted examples to sanity check polarity learning.
        This helps catch label inversion issues early."""
        self.model.eval()
        examples = [
            ("i love this movie", "POSITIVE"),
            ("this was fantastic", "POSITIVE"),
            ("i hate this", "NEGATIVE"),
            ("this was terrible", "NEGATIVE"),
        ]
        logger.info("Running quick sanity check on 4 synthetic examples...")
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for text, expected in examples:
                enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
                out = self.model(**enc)
                probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
                pred_id = int(np.argmax(probs))
                # Handle both dict with int or string keys
                id2label = self.model.config.id2label
                if isinstance(id2label, dict):
                    pred_label = id2label.get(pred_id, id2label.get(str(pred_id), f"LABEL_{pred_id}"))
                else:
                    pred_label = id2label[pred_id] if pred_id < len(id2label) else f"LABEL_{pred_id}"
                logger.info(f"Text='{text}' Pred={pred_label} (P={probs[pred_id]:.2%}) Expected={expected}")
        logger.info("Sanity check complete. If many expectations mismatch, check training or label config.")


def main():
    """Main fine-tuning pipeline"""
    print("="*80)
    print("SENTIMENT CLASSIFICATION - FINE-TUNING WITH LoRA")
    print("="*80)
    
    # Initialize fine-tuner
    finetuner = SentimentFineTuner(
        model_name="distilbert-base-uncased",
        # NOTE: Ensure this path matches the one used by the runtime (main.py).
        # main.py expects ./models/sentiment-classifier relative to project root.
        output_dir="./models/sentiment-classifier",
        use_lora=True
    )
    
    # Load and prepare data
    # Set max_samples=1000 for quick testing, or None for full dataset
    tokenized_datasets = finetuner.load_and_prepare_data(max_samples=5000)
    
    # Create model
    finetuner.create_model()
    
    # Train
    trainer = finetuner.train(
        tokenized_datasets,
        num_epochs=3,
        batch_size=4,  # Reduced for 4GB GPU
        learning_rate=2e-4
    )
    
    # Save
    finetuner.save_model(trainer)
    
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE!")
    print("="*80)
    # Run a quick post-training sanity check before announcing completion
    try:
        finetuner.quick_sanity_check()
    except Exception as e:
        logger.warning(f"Sanity check failed: {e}")

    print(f"Model saved to: {finetuner.output_dir}")
    print("\nYou can now run the CLI interface with:")
    print("  python src/main.py")


if __name__ == "__main__":
    main()
