"""
Model utilities: confidence feature extraction and correctness monitoring.
Implements the core components of RC-CoT method.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.special import softmax


class ConfidenceFeatureExtractor:
    """
    Extract model-agnostic confidence features from generation scores.
    Features:
    - first_token_entropy: Entropy of the first token distribution
    - top1_top2_margin: Difference between top-1 and top-2 logits
    - mean_log_prob: Mean log probability of generated tokens (length-normalized)
    """
    
    @staticmethod
    def compute_entropy(logits: np.ndarray) -> float:
        """Compute entropy of a probability distribution."""
        probs = softmax(logits)
        # Avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    @staticmethod
    def compute_top_margin(logits: np.ndarray) -> float:
        """Compute margin between top-1 and top-2 logits."""
        sorted_logits = np.sort(logits)[::-1]  # Sort descending
        if len(sorted_logits) < 2:
            return 0.0
        return float(sorted_logits[0] - sorted_logits[1])
    
    def extract_features(
        self,
        first_token_logits: np.ndarray,
        all_token_logprobs: List[float]
    ) -> Dict[str, float]:
        """
        Extract all confidence features.
        
        Args:
            first_token_logits: Logits for the first generated token (shape: vocab_size)
            all_token_logprobs: Log probabilities of all generated tokens
            
        Returns:
            Dictionary of features
        """
        features = {
            'first_token_entropy': self.compute_entropy(first_token_logits),
            'top1_top2_margin': self.compute_top_margin(first_token_logits),
            'mean_log_prob': np.mean(all_token_logprobs) if all_token_logprobs else -10.0
        }
        return features


class CorrectnessMonitor:
    """
    Probabilistic model to predict answer correctness from confidence features.
    Supports logistic regression and isotonic regression.
    """
    
    def __init__(self, model_type: str = "logistic"):
        """
        Args:
            model_type: "logistic" or "isotonic"
        """
        self.model_type = model_type
        self.single_class_fallback = False
        self.fallback_prob = 0.5
        
        if model_type == "logistic":
            self.model = LogisticRegression(random_state=42)
        elif model_type == "isotonic":
            # Isotonic regression requires a base model first
            self.base_model = LogisticRegression(random_state=42)
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def train(self, features_list: List[Dict[str, float]], labels: List[bool]) -> None:
        """
        Train the correctness monitor.
        
        Args:
            features_list: List of feature dictionaries
            labels: List of correctness labels (True/False)
        """
        # Convert to numpy array
        X = self._features_to_array(features_list)
        y = np.array(labels, dtype=int)
        
        print(f"Training {self.model_type} monitor on {len(labels)} examples...")
        print(f"  Positive rate: {np.mean(y):.2%}")
        
        # [VALIDATOR FIX - Attempt 1]
        # [PROBLEM]: sklearn LogisticRegression fails with "ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class"
        # [CAUSE]: When GPT-2 (small model) answers GSM8K directly without CoT, all answers are likely incorrect (or all correct), resulting in only one class (all 0s or all 1s)
        # [FIX]: Check for single-class case and use a constant fallback predictor that always returns the observed class probability
        #
        # [OLD CODE]:
        # if self.model_type == "logistic":
        #     self.model.fit(X, y)
        # elif self.model_type == "isotonic":
        #     self.base_model.fit(X, y)
        #     probs = self.base_model.predict_proba(X)[:, 1]
        #     self.calibrator.fit(probs, y)
        #
        # [NEW CODE]:
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            # Single class case - use constant predictor
            self.single_class_fallback = True
            self.fallback_prob = float(np.mean(y))  # 0.0 or 1.0
            print(f"WARNING: Only one class present (class={unique_classes[0]}). Using constant predictor with prob={self.fallback_prob:.3f}")
        else:
            self.single_class_fallback = False
            if self.model_type == "logistic":
                self.model.fit(X, y)
            elif self.model_type == "isotonic":
                # Train base model
                self.base_model.fit(X, y)
                # Get probabilities for calibration
                probs = self.base_model.predict_proba(X)[:, 1]
                # Train isotonic calibrator
                self.calibrator.fit(probs, y)
    
    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        Predict probability that answer is correct.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Probability of correctness (0 to 1)
        """
        # Handle single-class fallback
        if hasattr(self, 'single_class_fallback') and self.single_class_fallback:
            return self.fallback_prob
        
        X = self._features_to_array([features])
        
        if self.model_type == "logistic":
            prob = self.model.predict_proba(X)[0, 1]
        elif self.model_type == "isotonic":
            base_prob = self.base_model.predict_proba(X)[0, 1]
            prob = self.calibrator.predict([base_prob])[0]
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        return float(prob)
    
    def _features_to_array(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        """Convert list of feature dicts to numpy array."""
        # Use fixed feature order
        feature_names = ['first_token_entropy', 'top1_top2_margin', 'mean_log_prob']
        X = np.array([[f[name] for name in feature_names] for f in features_list])
        return X


class RiskControlledThresholdSelector:
    """
    Select acceptance threshold using finite-sample risk control (RCPS-style).
    Maximizes fast-path coverage subject to risk constraint.
    """
    
    def __init__(self, alpha: float = 0.10, delta: float = 0.05):
        """
        Args:
            alpha: Maximum allowed fast-path error rate
            delta: Confidence level for finite-sample bound
        """
        self.alpha = alpha
        self.delta = delta
        self.threshold = None
    
    def select_threshold(
        self,
        predicted_probs: List[float],
        actual_correct: List[bool]
    ) -> float:
        """
        Select threshold that maximizes coverage subject to risk constraint.
        
        Args:
            predicted_probs: Predicted correctness probabilities
            actual_correct: Actual correctness labels
            
        Returns:
            Selected threshold p*
        """
        n = len(predicted_probs)
        predicted_probs = np.array(predicted_probs)
        actual_correct = np.array(actual_correct, dtype=int)
        
        # Generate candidate thresholds
        candidates = np.linspace(0.5, 0.99, 50)
        
        best_threshold = 0.5
        best_coverage = 0.0
        
        print(f"\nSelecting risk-controlled threshold (α={self.alpha}, δ={self.delta})...")
        
        for p_star in candidates:
            # Accept examples where predicted prob > p_star
            accepted = predicted_probs >= p_star
            n_accepted = np.sum(accepted)
            
            if n_accepted == 0:
                continue
            
            # Compute empirical error on accepted examples
            errors = (1 - actual_correct[accepted])
            empirical_risk = np.mean(errors)
            
            # Compute finite-sample upper bound
            K = len(candidates)
            epsilon = np.sqrt(np.log(K / self.delta) / (2 * n_accepted))
            upper_bound = empirical_risk + epsilon
            
            # Check if constraint is satisfied
            if upper_bound <= self.alpha:
                coverage = n_accepted / n
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_threshold = p_star
        
        self.threshold = best_threshold
        
        print(f"Selected threshold: {self.threshold:.3f}")
        print(f"Fast-path coverage: {best_coverage:.2%}")
        
        return self.threshold
    
    def should_use_fast_path(self, predicted_prob: float) -> bool:
        """Decide whether to use fast path (System 1) or slow path (System 2)."""
        if self.threshold is None:
            raise ValueError("Threshold not set. Call select_threshold first.")
        return predicted_prob >= self.threshold


class ShiftDetector:
    """
    Conformal OOD detection on confidence features.
    Detects when features look out-of-distribution relative to calibration set.
    """
    
    def __init__(self, alpha_shift: float = 0.10):
        """
        Args:
            alpha_shift: Conformal threshold for OOD detection
        """
        self.alpha_shift = alpha_shift
        self.calibration_centroid = None
        self.calibration_quantile = None
    
    def calibrate(self, features_list: List[Dict[str, float]]) -> None:
        """
        Compute calibration centroid and distance quantile.
        
        Args:
            features_list: List of feature dictionaries from calibration set
        """
        # Convert to array
        feature_names = ['first_token_entropy', 'top1_top2_margin', 'mean_log_prob']
        X = np.array([[f[name] for name in feature_names] for f in features_list])
        
        # Compute centroid
        self.calibration_centroid = np.mean(X, axis=0)
        
        # Compute distances
        distances = np.linalg.norm(X - self.calibration_centroid, axis=1)
        
        # Compute conformal quantile
        self.calibration_quantile = np.quantile(distances, 1 - self.alpha_shift)
        
        print(f"\nShift detector calibrated:")
        print(f"  Centroid: {self.calibration_centroid}")
        print(f"  Distance quantile (1-α={1-self.alpha_shift}): {self.calibration_quantile:.3f}")
    
    def is_in_distribution(self, features: Dict[str, float]) -> bool:
        """
        Check if features are in-distribution.
        
        Args:
            features: Feature dictionary
            
        Returns:
            True if in-distribution, False if OOD (should escalate to CoT)
        """
        if self.calibration_centroid is None:
            raise ValueError("Detector not calibrated. Call calibrate first.")
        
        # Convert to array
        feature_names = ['first_token_entropy', 'top1_top2_margin', 'mean_log_prob']
        x = np.array([features[name] for name in feature_names])
        
        # Compute distance
        distance = np.linalg.norm(x - self.calibration_centroid)
        
        return distance <= self.calibration_quantile
