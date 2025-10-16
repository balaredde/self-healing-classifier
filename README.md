# Self-Healing Classification DAG with Fine-Tuned Model

A robust sentiment classification system using **LangGraph** for orchestration and a **fine-tuned DistilBERT** model with **LoRA** (Low-Rank Adaptation). The pipeline features intelligent fallback mechanisms that request user clarification when confidence is low, ensuring high accuracy over blind automation.

## 🎯 Project Overview

This project implements a self-healing classification pipeline that:
- ✅ Fine-tunes DistilBERT using LoRA on the IMDb sentiment dataset
- ✅ Uses LangGraph to orchestrate a DAG workflow with confidence-based routing
- ✅ Triggers fallback mechanisms when prediction confidence is below threshold
- ✅ Integrates a backup zero-shot classifier for additional validation
- ✅ Provides interactive CLI with rich formatting and real-time feedback
- ✅ Maintains structured logging with confidence tracking and statistics
- ✅ Generates visualization of confidence curves and fallback frequencies

## 📊 Architecture

### LangGraph DAG Workflow

```
User Input
    ↓
[InferenceNode]
    ↓
[ConfidenceCheckNode]
    ↓
  (Confidence >= 70%?) ──Yes──> [END] → Accept Prediction
    ↓
   No
    ↓
[FallbackNode]
    ↓
  • Run backup zero-shot model
  • Ask user for clarification
  • Process user response
    ↓
[END] → Corrected Label
```

### Node Descriptions

1. **InferenceNode**: Runs classification using the fine-tuned DistilBERT model
   - Loads model and tokenizer
   - Processes input text
   - Returns prediction with confidence score

2. **ConfidenceCheckNode**: Evaluates prediction confidence
   - Compares confidence against threshold (default: 70%)
   - Routes to fallback if confidence is low
   - Accepts prediction if confidence is high

3. **FallbackNode**: Handles low-confidence scenarios
   - Runs backup zero-shot classifier (BART-MNLI)
   - Generates clarification question
   - Processes user's clarification response
   - Returns corrected label

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, but CPU works too)
- 8GB+ RAM

### Setup

1. **Clone or download the project**
```bash
cd E:\amazon\self-healing-classifier
```

2. **Create virtual environment**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Install dependencies**
```powershell
pip install -r requirements.txt
```

This will install:
- PyTorch and Transformers
- PEFT (for LoRA fine-tuning)
- LangGraph and LangChain
- Rich (for CLI formatting)
- Matplotlib (for visualizations)

## 📚 Dataset

The project uses the **IMDb Movie Reviews** dataset:
- **Training samples**: 25,000 (can be limited for faster training)
- **Test samples**: 25,000
- **Classes**: Binary (Positive/Negative sentiment)
- **Source**: Automatically downloaded via Hugging Face `datasets` library

## 🎓 Fine-Tuning the Model

### Quick Start (5,000 samples - ~15 minutes)

```powershell
python src/fine_tune.py
```

This will:
1. Download IMDb dataset
2. Load DistilBERT base model
3. Apply LoRA configuration (r=16, alpha=32)
4. Train for 3 epochs with early stopping
5. Save model to `./models/sentiment-classifier/`

### Configuration Options

Edit `src/fine_tune.py` to customize:

```python
# Line 229: Limit samples for quick testing
tokenized_datasets = finetuner.load_and_prepare_data(max_samples=5000)  # or None for full dataset

# Lines 237-240: Training hyperparameters
trainer = finetuner.train(
    tokenized_datasets,
    num_epochs=3,        # Number of training epochs
    batch_size=16,       # Batch size
    learning_rate=2e-4   # Learning rate
)
```

### Expected Performance

With 5,000 training samples:
- **Accuracy**: ~85-90%
- **F1 Score**: ~0.85-0.90
- **Training Time**: 10-20 minutes (GPU), 30-60 minutes (CPU)

Full dataset (25,000 samples):
- **Accuracy**: ~92-95%
- **F1 Score**: ~0.92-0.95
- **Training Time**: 45-90 minutes (GPU), 3-6 hours (CPU)

## 🖥️ Running the CLI Interface

Once the model is fine-tuned, launch the interactive CLI:

```powershell
python src/main.py
```

### Example Session

```
═══════════════════════════════════════════════════════════════════
        Self-Healing Sentiment Classifier
        Powered by LangGraph + Fine-tuned DistilBERT
═══════════════════════════════════════════════════════════════════

Configuration:
  Model: ./models/sentiment-classifier
  Confidence Threshold: 70%
  Backup Model: Enabled

Type 'quit' or 'exit' to stop

────────────────────────────────────────────────────────────────────

Enter text to classify: The movie was painfully slow and boring.

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃           [InferenceNode] Initial Prediction                  ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Predicted Label: POSITIVE                                     ┃
┃ Confidence: 54%                                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                [ConfidenceCheckNode]                          ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Confidence too low (< 70%). Triggering fallback...           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃              [FallbackNode] Backup Model                      ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Backup Model Prediction: NEGATIVE                            ┃
┃ Backup Confidence: 89%                                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

❓ The sentiment seems unclear. Was this meant to be a positive review?
Your response: No, it was definitely negative.

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃               [FallbackNode] Result                           ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Final Label: NEGATIVE                                         ┃
┃ (Corrected via user clarification)                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                 Classification Result                         ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Field              │ Value                                    ┃
┃────────────────────┼──────────────────────────────────────────┃
┃ Predicted Label    │ POSITIVE                                 ┃
┃ Confidence         │ 54%                                      ┃
┃ Fallback           │ ✓ Triggered                              ┃
┃ Backup Prediction  │ NEGATIVE (89%)                           ┃
┃ Final Label        │ NEGATIVE                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### CLI Commands
- Enter any text to classify
- Type `quit`, `exit`, or `q` to stop
- When asked for clarification, respond with:
  - "Yes" or "Correct" to confirm prediction
  - "No" or "Wrong" to reject prediction
  - "Positive" or "Negative" to specify label directly

## 📝 Logging

### Structured Logs

All predictions are logged to `./logs/` with timestamps:

**JSON Log** (`classification_log_YYYYMMDD_HHMMSS.jsonl`):
```json
{
  "timestamp": "2025-10-14T10:30:45.123456",
  "user_input": "The movie was painfully slow and boring.",
  "predicted_label": "POSITIVE",
  "confidence": 0.54,
  "raw_scores": {"NEGATIVE": 0.46, "POSITIVE": 0.54},
  "fallback_triggered": true,
  "clarification_question": "Was this meant to be a positive review?",
  "user_clarification": "No, it was definitely negative",
  "backup_prediction": "NEGATIVE",
  "backup_confidence": 0.89,
  "final_label": "NEGATIVE"
}
```

**Text Log** (`classification_log_YYYYMMDD_HHMMSS.txt`):
- Standard logging format with timestamps
- All node executions and decisions
- Error messages and warnings

### Session Summary

At the end of each session:

```
═══════════════════════════════════════════════════════════════════
SESSION SUMMARY
═══════════════════════════════════════════════════════════════════

Total Predictions: 10
Fallback Triggered: 3 (30.0%)

Confidence Statistics:
  Average: 78.5%
  Min: 54.0%
  Max: 98.2%

Log files saved:
  JSON: ./logs/classification_log_20251014_103045.jsonl
  Text: ./logs/classification_log_20251014_103045.txt

═══════════════════════════════════════════════════════════════════
FALLBACK STATISTICS (CLI Histogram)
═══════════════════════════════════════════════════════════════════

Accepted (High Confidence): ██████████████████████████████████████████ 7 (70.0%)
Fallback (Low Confidence):  ██████████████████ 3 (30.0%)

═══════════════════════════════════════════════════════════════════

Confidence plot saved to: ./logs/confidence_analysis_20251014_103045.png
```

### Visualizations

**Confidence Curve**: Line plot showing confidence scores over time
**Confidence Distribution**: Histogram of all confidence scores
**Fallback Histogram**: Bar chart of accepted vs. fallback predictions

## 🎨 Bonus Features

### 1. Backup Zero-Shot Model
- Uses `facebook/bart-large-mnli` for zero-shot classification
- Automatically activated during fallback
- Provides independent validation of predictions

### 2. Confidence Tracking
- Real-time confidence monitoring
- Statistical analysis (mean, min, max)
- Visual plots saved to logs directory

### 3. Fallback Statistics
- Tracks frequency of fallback activations
- CLI-based histogram display
- Helps identify model improvement areas

## 🛠️ Configuration

### Confidence Threshold

Adjust in `src/main.py` (line 257):
```python
classifier = SelfHealingClassifier(
    model_path=model_path,
    confidence_threshold=0.70,  # Change this (0.0 to 1.0)
    use_backup_model=True
)
```

- **Lower threshold** (e.g., 0.50): Fewer fallbacks, faster responses, potentially lower accuracy
- **Higher threshold** (e.g., 0.85): More fallbacks, slower but more accurate

### Backup Model

Disable backup model in `src/main.py` (line 259):
```python
use_backup_model=False  # Disable backup zero-shot classifier
```

## 📁 Project Structure

```
self-healing-classifier/
├── src/
│   ├── fine_tune.py      # LoRA fine-tuning script
│   ├── dag_nodes.py      # LangGraph nodes (Inference, Confidence, Fallback)
│   ├── main.py           # CLI interface and workflow orchestration
│   └── logger.py         # Structured logging and statistics
├── models/
│   └── sentiment-classifier/  # Fine-tuned model (after training)
│       ├── config.json
│       ├── model.safetensors
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── logs/
│   ├── classification_log_*.jsonl  # Structured JSON logs
│   ├── classification_log_*.txt    # Text logs
│   └── confidence_analysis_*.png   # Visualization plots
├── data/                  # Dataset cache (auto-created)
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

## 🔬 Technical Details

### Model Architecture
- **Base Model**: DistilBERT (66M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
  - Rank (r): 16
  - Alpha: 32
  - Trainable parameters: ~0.5M (~0.8% of base model)
- **Task**: Binary sentiment classification

### LoRA Advantages
- ✅ Faster training (only updates 0.8% of parameters)
- ✅ Lower memory footprint
- ✅ Better generalization
- ✅ Easier to share (small adapter files)

### Backup Model
- **Model**: `facebook/bart-large-mnli`
- **Type**: Zero-shot classification
- **Purpose**: Validation and fallback support
- **Loads**: On-demand during first fallback

## 🐛 Troubleshooting

### Model Not Found
```
Error: Model not found!
```
**Solution**: Run fine-tuning first: `python src/fine_tune.py`

### Out of Memory (GPU)
**Solution**: Reduce batch size in `fine_tune.py` (line 239):
```python
batch_size=8,  # or even 4
```

### Import Errors
**Solution**: Reinstall dependencies:
```powershell
pip install --upgrade -r requirements.txt
```

### Slow CPU Training
**Solution**: Use smaller sample size in `fine_tune.py` (line 229):
```python
max_samples=1000  # Instead of 5000
```

## 📊 Evaluation Metrics

The fine-tuning script reports:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: Coverage of actual positives
- **F1 Score**: Harmonic mean of precision and recall

The CLI tracks:
- **Confidence**: Model's certainty in its prediction
- **Fallback Rate**: Percentage of low-confidence predictions
- **User Corrections**: Manual overrides during clarification

## 🎥 Demo Video Guidelines

For the 2-4 minute demo video, cover:

1. **Introduction** (30s)
   - Project overview
   - Architecture diagram

2. **Fine-Tuning** (45s)
   - Show training command
   - Explain LoRA configuration
   - Display final metrics

3. **CLI Demonstration** (90s)
   - Example with high confidence (accepted)
   - Example with low confidence (fallback triggered)
   - Show clarification process
   - Display backup model prediction

4. **Results & Logs** (45s)
   - Session summary
   - Confidence plots
   - Fallback statistics
   - Log file walkthrough

## 📤 Submission Checklist

- [ ] Source code (all `.py` files)
- [ ] README.md (this file)
- [ ] requirements.txt
- [ ] Fine-tuned model or download link
- [ ] Sample log files (.jsonl and .txt)
- [ ] Confidence visualization plots
- [ ] Demo video (2-4 minutes)

## 🔗 Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [IMDb Dataset](https://huggingface.co/datasets/imdb)

## 📄 License

This project is for educational purposes (ATG Technical Assignment).

## 👤 Author

Machine Learning Intern Candidate
ATG Technical Assignment - October 2025

---

**Ready to start?**

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fine-tune the model (15-20 minutes)
python src/fine_tune.py

# 3. Run the CLI
python src/main.py
```

Good luck! 🚀
