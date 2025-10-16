"""
Structured logging for classification pipeline
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from collections import Counter
from rich.console import Console
from rich.table import Table

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)
console = Console()


class ClassificationLogger:
    """Structured logger for classification predictions"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"classification_{timestamp}.jsonl"
        
        # In-memory storage for statistics
        self.predictions: List[Dict] = []
        self.stats = {
            "total_predictions": 0,
            "fallback_count": 0,
            "avg_confidence": 0.0,
            "label_distribution": Counter()
        }
        
        logger.info(f"Logger initialized. Log file: {self.log_file}")
    
    def log_prediction(self, state: Dict):
        """Log a single prediction"""
        log_entry = {
            "timestamp": state.get("timestamp", datetime.now().isoformat()),
            "user_input": state["user_input"][:100],  # Truncate long inputs
            "predicted_label": state["predicted_label"],
            "confidence": state["confidence"],
            "raw_scores": state.get("raw_scores", {}),
            "fallback_triggered": state["fallback_triggered"],
            "backup_prediction": state.get("backup_prediction", "N/A"),
            "backup_confidence": state.get("backup_confidence", 0.0),
            "user_clarification": state.get("user_clarification", ""),
            "final_label": state["final_label"]
        }
        
        # Write to file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # Update stats
        self.predictions.append(log_entry)
        self.stats["total_predictions"] += 1
        if state["fallback_triggered"]:
            self.stats["fallback_count"] += 1
        self.stats["label_distribution"][state["final_label"]] += 1
        
        # Update average confidence
        total_conf = sum(p["confidence"] for p in self.predictions)
        self.stats["avg_confidence"] = total_conf / len(self.predictions)
    
    def print_summary(self):
        """Print summary statistics"""
        console.print("\n[bold cyan]═══ Session Summary ═══[/bold cyan]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green")
        
        table.add_row("Total Predictions", str(self.stats["total_predictions"]))
        
        if self.stats["total_predictions"] > 0:
            fallback_pct = (self.stats['fallback_count'] / self.stats['total_predictions']) * 100
            table.add_row("Fallback Triggered", f"{self.stats['fallback_count']} ({fallback_pct:.1f}%)")
            table.add_row("Average Confidence", f"{self.stats['avg_confidence']:.2%}")
        
        console.print(table)
        
        # Label distribution
        if self.stats["label_distribution"]:
            console.print("\n[bold]Label Distribution:[/bold]")
            for label, count in self.stats["label_distribution"].most_common():
                console.print(f"  {label}: {count}")
        
        console.print(f"\n[dim]Log file saved: {self.log_file}[/dim]\n")
    
    def print_fallback_histogram(self):
        """Print ASCII histogram of fallback frequency"""
        if self.stats["total_predictions"] == 0:
            return
        
        fallback_pct = (self.stats["fallback_count"] / self.stats["total_predictions"]) * 100
        accepted_pct = 100 - fallback_pct
        
        console.print("[bold cyan]Fallback Frequency:[/bold cyan]")
        console.print(f"  Accepted:  {'█' * int(accepted_pct/2)} {accepted_pct:.1f}%")
        console.print(f"  Fallback:  {'█' * int(fallback_pct/2)} {fallback_pct:.1f}%")
        console.print()
    
    def plot_confidence_curve(self):
        """Plot confidence scores over time"""
        if not MATPLOTLIB_AVAILABLE:
            console.print("[yellow]Matplotlib not available. Skipping plot generation.[/yellow]")
            return
        
        if not self.predictions:
            return
        
        confidences = [p["confidence"] for p in self.predictions]
        
        plt.figure(figsize=(10, 5))
        plt.plot(confidences, marker='o', linestyle='-', linewidth=2)
        plt.axhline(y=0.70, color='r', linestyle='--', label='Threshold (70%)')
        plt.xlabel('Prediction Number')
        plt.ylabel('Confidence Score')
        plt.title('Confidence Scores Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = self.log_dir / f"confidence_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=150)
        console.print(f"[green]✓ Confidence curve saved: {plot_file}[/green]")
        plt.close()
