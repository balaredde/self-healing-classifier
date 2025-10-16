"""
LangGraph DAG Nodes for Self-Healing Classification
Fixed version with proper label mapping and error handling
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing_extensions import TypedDict
from typing import Literal
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


class ClassificationState(TypedDict):
    """State schema for the classification workflow"""
    user_input: str
    predicted_label: str
    confidence: float
    raw_scores: dict
    needs_clarification: bool
    clarification_question: str
    user_clarification: str
    final_label: str
    fallback_triggered: bool
    timestamp: str
    backup_prediction: str
    backup_confidence: float


class InferenceNode:
    """Run classification using the fine-tuned model"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.7):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Set device
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Create pipeline for easier inference
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            return_all_scores=True
        )
        
        logger.info(f"Model config id2label: {self.model.config.id2label}")
        logger.info(f"Model loaded successfully on {'GPU' if self.device == 0 else 'CPU'}!")
    
    def __call__(self, state: ClassificationState) -> ClassificationState:
        """Execute inference"""
        user_input = state["user_input"]
        
        logger.info(f"[InferenceNode] Processing input: {user_input[:50]}...")
        
        # Get predictions
        results = self.classifier(user_input)[0]  # Returns list of dicts with label and score
        
        # Find prediction with highest score
        results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
        top_prediction = results_sorted[0]
        
        # CRITICAL FIX: Force label mapping
        raw_label = top_prediction['label']
        
        # Hard-coded mapping for sentiment
        LABEL_MAP = {
            "LABEL_0": "NEGATIVE",
            "LABEL_1": "POSITIVE",
        }
        
        # Map the label
        predicted_label = LABEL_MAP.get(raw_label, raw_label)
        
        # If still LABEL_X format, try extracting number
        if predicted_label.startswith('LABEL_'):
            label_id = int(predicted_label.split('_')[1])
            predicted_label = "NEGATIVE" if label_id == 0 else "POSITIVE"
        
        confidence = top_prediction['score']
        
        # Store raw scores with proper labels
        raw_scores = {}
        for r in results:
            r_label = r['label']
            mapped_label = LABEL_MAP.get(r_label, r_label)
            
            # Double-check mapping
            if mapped_label.startswith('LABEL_'):
                label_id = int(mapped_label.split('_')[1])
                mapped_label = "NEGATIVE" if label_id == 0 else "POSITIVE"
            
            raw_scores[mapped_label] = r['score']
        
        logger.info(f"[InferenceNode] Predicted: {predicted_label} | Confidence: {confidence:.2%}")
        logger.info(f"[InferenceNode] Raw scores: {raw_scores}")
        
        # Update state
        state["predicted_label"] = predicted_label
        state["confidence"] = confidence
        state["raw_scores"] = raw_scores
        state["timestamp"] = datetime.now().isoformat()
        
        return state


class ConfidenceCheckNode:
    """Evaluate prediction confidence and decide if fallback is needed"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def __call__(self, state: ClassificationState) -> ClassificationState:
        """Check confidence and determine if clarification is needed"""
        confidence = state["confidence"]
        predicted_label = state["predicted_label"]
        
        logger.info(f"[ConfidenceCheckNode] Evaluating confidence: {confidence:.2%}")
        
        if confidence < self.confidence_threshold:
            logger.warning(f"[ConfidenceCheckNode] Confidence below threshold ({self.confidence_threshold:.2%})")
            logger.warning("[ConfidenceCheckNode] Triggering fallback mechanism...")
            
            state["needs_clarification"] = True
            state["fallback_triggered"] = True
            
            # Generate clarification question based on prediction
            if predicted_label == "POSITIVE":
                question = "The sentiment seems unclear. Was this meant to be a positive review?"
            elif predicted_label == "NEGATIVE":
                question = "The sentiment seems unclear. Was this meant to be a negative review?"
            else:
                question = f"The sentiment seems unclear. The model predicted '{predicted_label}'. Could you clarify the sentiment?"
            
            state["clarification_question"] = question
        else:
            logger.info("[ConfidenceCheckNode] Confidence acceptable. Proceeding with prediction.")
            state["needs_clarification"] = False
            state["fallback_triggered"] = False
            state["final_label"] = predicted_label
        
        return state


class FallbackNode:
    """Handle low-confidence predictions through user interaction or backup model"""
    
    def __init__(self, use_backup_model: bool = True):
        self.use_backup_model = use_backup_model
        self.backup_classifier = None
        
        if use_backup_model:
            logger.info("[FallbackNode] Initializing backup zero-shot classifier...")
            try:
                # Use a zero-shot classifier as backup
                device = 0 if torch.cuda.is_available() else -1
                self.backup_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=device,
                    model_kwargs={"low_cpu_mem_usage": True},
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                logger.info("[FallbackNode] Backup model loaded!")
            except Exception as e:
                logger.warning(f"[FallbackNode] Could not load backup model: {e}")
                logger.info("[FallbackNode] Continuing without backup model...")
                self.backup_classifier = None
                self.use_backup_model = False
    
    def get_backup_prediction(self, text: str) -> tuple:
        """Get prediction from backup zero-shot model"""
        if self.backup_classifier is None:
            return None, 0.0
        
        try:
            result = self.backup_classifier(
                text,
                candidate_labels=["negative", "positive"],
                multi_label=False
            )
            
            label = result['labels'][0].upper()
            confidence = result['scores'][0]
            
            logger.info(f"[FallbackNode] Backup model: {label} ({confidence:.2%})")
            return label, confidence
        except Exception as e:
            logger.error(f"[FallbackNode] Backup prediction failed: {e}")
            return None, 0.0
    
    def __call__(self, state: ClassificationState) -> ClassificationState:
        """Execute fallback strategy"""
        logger.info("[FallbackNode] Executing fallback strategy...")
        
        # Try backup model first if available
        if self.use_backup_model and self.backup_classifier:
            backup_label, backup_conf = self.get_backup_prediction(state["user_input"])
            state["backup_prediction"] = backup_label or "N/A"
            state["backup_confidence"] = backup_conf
        else:
            state["backup_prediction"] = "N/A"
            state["backup_confidence"] = 0.0
        
        # User clarification will be handled by CLI
        # This node just prepares the state
        logger.info("[FallbackNode] Awaiting user clarification...")
        
        return state
    
    def process_clarification(self, state: ClassificationState, user_response: str) -> ClassificationState:
        """Process user's clarification response"""
        user_response = user_response.strip().lower()
        
        # Simple yes/no parsing
        if "yes" in user_response or "correct" in user_response or "right" in user_response:
            # User confirms the prediction
            final_label = state["predicted_label"]
            logger.info(f"[FallbackNode] User confirmed: {final_label}")
        elif "no" in user_response or "wrong" in user_response or "incorrect" in user_response:
            # User rejects the prediction - flip it
            if state["predicted_label"] == "POSITIVE":
                final_label = "NEGATIVE"
            else:
                final_label = "POSITIVE"
            logger.info(f"[FallbackNode] User corrected to: {final_label}")
        elif "negative" in user_response:
            final_label = "NEGATIVE"
            logger.info(f"[FallbackNode] User specified: {final_label}")
        elif "positive" in user_response:
            final_label = "POSITIVE"
            logger.info(f"[FallbackNode] User specified: {final_label}")
        else:
            # Unclear response - use backup model if available
            if state["backup_prediction"] != "N/A" and state["backup_confidence"] > 0.5:
                final_label = state["backup_prediction"]
                logger.info(f"[FallbackNode] Using backup model: {final_label}")
            else:
                # Default to original prediction
                final_label = state["predicted_label"]
                logger.info(f"[FallbackNode] Unclear response, using original: {final_label}")
        
        state["final_label"] = final_label
        state["user_clarification"] = user_response
        
        return state


def route_after_confidence_check(state: ClassificationState) -> Literal["fallback", "end"]:
    """Routing function to determine next node after confidence check"""
    if state["needs_clarification"]:
        return "fallback"
    else:
        return "end"
