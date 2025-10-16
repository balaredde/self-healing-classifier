# Quick Start Guide

## Installation and Setup

### 1. Navigate to project directory
```powershell
cd E:\amazon\self-healing-classifier
```

### 2. Create virtual environment (recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

This will take 5-10 minutes depending on your internet connection.

### 4. Fine-tune the model
```powershell
python src\fine_tune.py
```

**Options:**
- Quick test (1,000 samples, ~5 minutes): Edit line 229 in `src/fine_tune.py` to `max_samples=1000`
- Normal (5,000 samples, ~15 minutes): Keep default `max_samples=5000`
- Full dataset (25,000 samples, ~60 minutes): Change to `max_samples=None`

### 5. Run the interactive CLI
```powershell
python src\main.py
```

### 6. Quick demo (optional)
```powershell
python src\demo.py
```

## Example Usage

Once in the CLI, try these examples:

**High Confidence (Accepted):**
```
Enter text: This movie was absolutely brilliant! A masterpiece!
```

**Low Confidence (Fallback Triggered):**
```
Enter text: The movie was painfully slow and boring.
```

**Ambiguous (Fallback Triggered):**
```
Enter text: It was okay, nothing special.
```

## Configuration

### Adjust Confidence Threshold

Edit `src/main.py`, line 257:
```python
confidence_threshold=0.70  # 0.0 to 1.0
```

- Lower = Fewer fallbacks (faster, less accurate)
- Higher = More fallbacks (slower, more accurate)

### Disable Backup Model

Edit `src/main.py`, line 259:
```python
use_backup_model=False
```

## Logs

After running, check:
- `logs/classification_log_*.jsonl` - JSON structured logs
- `logs/classification_log_*.txt` - Text logs
- `logs/confidence_analysis_*.png` - Visualization plots

## Troubleshooting

**"Model not found"**
→ Run `python src\fine_tune.py` first

**Out of memory**
→ Reduce batch size in `src/fine_tune.py` line 239: `batch_size=8`

**Slow training**
→ Use fewer samples: `max_samples=1000`

## Next Steps

1. Test with various inputs
2. Review logs and statistics
3. Adjust confidence threshold
4. Record demo video
5. Package for submission

Good luck! 🚀
