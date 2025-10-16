"""
Main CLI Interface for Self-Healing Classification DAG
Fixed version with proper imports and error handling
"""

import os
import sys
from typing import Dict
import logging
from pathlib import Path

from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import print as rprint

from dag_nodes import (
    ClassificationState,
    InferenceNode,
    ConfidenceCheckNode,
    FallbackNode,
    route_after_confidence_check
)
from logger import ClassificationLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Rich console for pretty CLI
console = Console()


class SelfHealingClassifier:
    """Self-healing classification pipeline with LangGraph"""
    
    def __init__(
        self,
        model_path: str = "./models/sentiment-classifier",
        confidence_threshold: float = 0.70,
        use_backup_model: bool = True
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.use_backup_model = use_backup_model
        
        # Initialize logger
        self.logger = ClassificationLogger()
        
        # Initialize nodes
        console.print("[yellow]Initializing pipeline...[/yellow]")
        try:
            self.inference_node = InferenceNode(model_path, confidence_threshold)
            self.confidence_check_node = ConfidenceCheckNode(confidence_threshold)
            self.fallback_node = FallbackNode(use_backup_model)
            
            # Build workflow
            self.workflow = self.build_workflow()
            console.print("[green]✓ Pipeline ready![/green]\n")
        except Exception as e:
            console.print(f"[red]Error initializing pipeline: {e}[/red]")
            logger.exception("Initialization failed:")
            raise
    
    def build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create graph
        workflow = StateGraph(ClassificationState)
        
        # Add nodes
        workflow.add_node("inference", self.inference_node)
        workflow.add_node("confidence_check", self.confidence_check_node)
        workflow.add_node("fallback", self.fallback_node)
        
        # Define edges
        workflow.set_entry_point("inference")
        workflow.add_edge("inference", "confidence_check")
        
        # Conditional routing after confidence check
        workflow.add_conditional_edges(
            "confidence_check",
            route_after_confidence_check,
            {
                "fallback": "fallback",
                "end": END
            }
        )
        
        workflow.add_edge("fallback", END)
        
        # Compile
        compiled = workflow.compile()
        logger.info("LangGraph workflow compiled successfully!")
        return compiled
    
    def process_input(self, user_input: str) -> Dict:
        """Process a single input through the DAG"""
        
        # Initialize state
        initial_state: ClassificationState = {
            "user_input": user_input,
            "predicted_label": "",
            "confidence": 0.0,
            "raw_scores": {},
            "needs_clarification": False,
            "clarification_question": "",
            "user_clarification": "",
            "final_label": "",
            "fallback_triggered": False,
            "timestamp": "",
            "backup_prediction": "",
            "backup_confidence": 0.0
        }
        
        # Run through workflow
        try:
            result = self.workflow.invoke(initial_state)
            return result
        except Exception as e:
            logger.exception("Error processing input:")
            console.print(f"[red]Error: {e}[/red]")
            raise
    
    def display_result(self, state: Dict):
        """Display the classification result in a pretty format"""
        
        # Create result table
        table = Table(title="Classification Result", show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan", width=20)
        table.add_column("Value", style="green")
        
        table.add_row("Predicted Label", state["predicted_label"])
        table.add_row("Confidence", f"{state['confidence']:.2%}")
        
        if state["fallback_triggered"]:
            table.add_row("Fallback", "✓ Triggered", style="yellow")
            if state["backup_prediction"] != "N/A":
                table.add_row("Backup Prediction", f"{state['backup_prediction']} ({state['backup_confidence']:.2%})")
        
        table.add_row("Final Label", state["final_label"], style="bold")
        
        console.print(table)
    
    def run_interactive(self):
        """Run interactive CLI loop"""
        
        console.print(Panel.fit(
            "[bold cyan]Self-Healing Sentiment Classifier[/bold cyan]\n"
            "[dim]Powered by LangGraph + Fine-tuned DistilBERT[/dim]",
            border_style="cyan"
        ))
        
        console.print(f"\n[yellow]Configuration:[/yellow]")
        console.print(f"  Model: {self.model_path}")
        console.print(f"  Confidence Threshold: {self.confidence_threshold:.0%}")
        console.print(f"  Backup Model: {'Enabled' if self.use_backup_model else 'Disabled'}")
        console.print("\n[dim]Type 'quit' or 'exit' to stop[/dim]\n")
        
        while True:
            try:
                # Get user input
                console.print("─" * 80)
                user_input = Prompt.ask("\n[bold cyan]Enter text to classify[/bold cyan]")
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input.strip():
                    console.print("[red]Please enter some text![/red]")
                    continue
                
                console.print()  # Blank line
                
                # Process through DAG
                state = self.process_input(user_input)
                
                # Display initial prediction
                console.print(Panel(
                    f"[bold]Predicted Label:[/bold] {state['predicted_label']}\n"
                    f"[bold]Confidence:[/bold] {state['confidence']:.2%}",
                    title="[InferenceNode] Initial Prediction",
                    border_style="blue"
                ))
                
                # Handle fallback if needed
                if state["needs_clarification"]:
                    console.print(Panel(
                        f"[yellow]Confidence too low (< {self.confidence_threshold:.0%}). Triggering fallback...[/yellow]",
                        title="[ConfidenceCheckNode]",
                        border_style="yellow"
                    ))
                    
                    # Show backup prediction if available
                    if state["backup_prediction"] != "N/A":
                        console.print(Panel(
                            f"[bold]Backup Model Prediction:[/bold] {state['backup_prediction']}\n"
                            f"[bold]Backup Confidence:[/bold] {state['backup_confidence']:.2%}",
                            title="[FallbackNode] Backup Model",
                            border_style="magenta"
                        ))
                    
                    # Ask for clarification
                    console.print()
                    console.print(f"[bold yellow]❓ {state['clarification_question']}[/bold yellow]")
                    user_clarification = Prompt.ask("[cyan]Your response[/cyan]")
                    
                    # Process clarification
                    state = self.fallback_node.process_clarification(state, user_clarification)
                    
                    console.print(Panel(
                        f"[bold green]Final Label: {state['final_label']}[/bold green]\n"
                        f"[dim](Corrected via user clarification)[/dim]",
                        title="[FallbackNode] Result",
                        border_style="green"
                    ))
                else:
                    console.print(Panel(
                        f"[green]Confidence acceptable. Accepting prediction.[/green]",
                        title="[ConfidenceCheckNode]",
                        border_style="green"
                    ))
                
                # Display final result
                console.print()
                self.display_result(state)
                
                # Log the interaction
                self.logger.log_prediction(state)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
                continue
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                logger.exception("Error in interactive loop:")
                continue
        
        # Session ended
        console.print("\n[yellow]Session ended. Generating summary...[/yellow]\n")
        
        # Print summary and stats
        self.logger.print_summary()
        self.logger.print_fallback_histogram()
        
        # Generate plots if we have data
        if self.logger.stats["total_predictions"] > 0:
            try:
                self.logger.plot_confidence_curve()
            except Exception as e:
                console.print(f"[red]Could not generate plots: {e}[/red]")
        
        console.print("\n[green]Thank you for using the Self-Healing Classifier![/green]\n")


def main():
    """Main entry point"""
    
    # Check if model exists
    model_path = "./models/sentiment-classifier"
    if not Path(model_path).exists():
        console.print("[red]Error: Model not found![/red]")
        console.print(f"Expected path: {model_path}")
        console.print("\n[yellow]Please run fine-tuning first:[/yellow]")
        console.print("  python src/fine_tune.py")
        sys.exit(1)
    
    try:
        # Initialize and run classifier
        classifier = SelfHealingClassifier(
            model_path=model_path,
            confidence_threshold=0.70,
            use_backup_model=True  # Set to False to disable backup model
        )
        
        classifier.run_interactive()
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        logger.exception("Fatal error:")
        sys.exit(1)


if __name__ == "__main__":
    main()
