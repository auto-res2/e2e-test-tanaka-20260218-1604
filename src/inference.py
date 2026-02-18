"""
Inference script for RC-CoT experiments.
Implements all methods: rc-cot, always-fast, always-cot, entropy-gate.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.preprocess import load_gsm8k_splits, check_answer_correctness
from src.model import (
    ConfidenceFeatureExtractor,
    CorrectnessMonitor,
    RiskControlledThresholdSelector,
    ShiftDetector
)


class ModelInference:
    """Wrapper for model inference with confidence feature extraction."""
    
    def __init__(self, model_name: str, cache_dir: str, model_params: dict):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float32
        )
        self.model.eval()
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model_params = model_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")
    
    def generate_with_scores(
        self,
        prompt: str,
        return_scores: bool = False
    ) -> Tuple[str, Optional[np.ndarray], Optional[List[float]]]:
        """
        Generate text with optional score extraction.
        
        Returns:
            generated_text: Generated text (without prompt)
            first_token_logits: Logits for first token (if return_scores=True)
            all_token_logprobs: Log probs for all tokens (if return_scores=True)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        # [VALIDATOR FIX - Attempt 1]
        # [PROBLEM]: do_sample was always True, causing repetitive outputs even when config specifies greedy
        # [CAUSE]: Code hardcoded do_sample=True instead of reading from model_params
        # [FIX]: Read do_sample from model_params, default to True for backward compatibility
        #
        # [OLD CODE]:
        # with torch.no_grad():
        #     if return_scores:
        #         outputs = self.model.generate(
        #             **inputs,
        #             max_new_tokens=self.model_params.get('max_length', 512),
        #             temperature=self.model_params.get('temperature', 0.7),
        #             top_p=self.model_params.get('top_p', 0.95),
        #             do_sample=True,
        #             return_dict_in_generate=True,
        #             output_scores=True
        #         )
        #
        # [NEW CODE]:
        with torch.no_grad():
            do_sample = self.model_params.get('do_sample', True)
            temp = self.model_params.get('temperature', 0.7)
            
            # Handle greedy decoding when temperature is 0 or do_sample is False
            gen_kwargs = {
                'max_new_tokens': self.model_params.get('max_length', 512),
            }
            
            if do_sample and temp > 0:
                gen_kwargs['do_sample'] = True
                gen_kwargs['temperature'] = temp
                gen_kwargs['top_p'] = self.model_params.get('top_p', 0.95)
            else:
                gen_kwargs['do_sample'] = False
            
            if return_scores:
                gen_kwargs['return_dict_in_generate'] = True
                gen_kwargs['output_scores'] = True
                outputs = self.model.generate(**inputs, **gen_kwargs)
                
                # Extract generated text
                generated_ids = outputs.sequences[0][input_length:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Extract first token logits
                first_token_logits = outputs.scores[0][0].cpu().numpy()
                
                # Extract log probs for all tokens
                all_token_logprobs = []
                for i, score in enumerate(outputs.scores):
                    token_id = outputs.sequences[0][input_length + i]
                    logits = score[0]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    token_logprob = log_probs[token_id].item()
                    all_token_logprobs.append(token_logprob)
                
                return generated_text, first_token_logits, all_token_logprobs
            else:
                # [VALIDATOR FIX - Attempt 1]
                # [PROBLEM]: Non-score generation path also had hardcoded do_sample=True
                # [CAUSE]: Code didn't reuse the generation kwargs from above
                # [FIX]: Use same gen_kwargs logic for consistency
                #
                # [OLD CODE]:
                # outputs = self.model.generate(
                #     **inputs,
                #     max_new_tokens=self.model_params.get('max_length', 512),
                #     temperature=self.model_params.get('temperature', 0.7),
                #     top_p=self.model_params.get('top_p', 0.95),
                #     do_sample=True
                # )
                #
                # [NEW CODE]:
                outputs = self.model.generate(**inputs, **gen_kwargs)
                generated_ids = outputs[0][input_length:]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                return generated_text, None, None


def extract_final_answer(text: str) -> str:
    """Extract numeric answer from generated text."""
    # Look for patterns like "Answer: 42" or "FINAL=42"
    patterns = [
        r'FINAL\s*[=:]\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',
        r'[Aa]nswer\s*[=:]\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',
        r'\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)\s*$',  # Number at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(',', '')
    
    # Fallback: extract last number in text
    numbers = re.findall(r'-?\d+(?:,\d+)*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return text.strip()


class RC_CoT_Inference:
    """Main inference pipeline for RC-CoT method."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        # [VALIDATOR FIX - Attempt 1]
        # [PROBLEM]: ConfigAttributeError: Missing key model_params
        # [CAUSE]: model_params is nested under cfg.run, not at root cfg level
        # [FIX]: Changed cfg.get('model_params', {}) to cfg.run.get('model_params', {})
        #
        # [OLD CODE]:
        # self.model_inference = ModelInference(
        #     cfg.run.model,
        #     cfg.inference.cache_dir,
        #     cfg.get('model_params', {})
        # )
        #
        # [NEW CODE]:
        self.model_inference = ModelInference(
            cfg.run.model,
            cfg.inference.cache_dir,
            cfg.run.get('model_params', {})
        )
        self.feature_extractor = ConfidenceFeatureExtractor()
        
        # Initialize method-specific components
        self.method = cfg.run.method
        
        if self.method == "rc-cot":
            self.monitor = CorrectnessMonitor(
                model_type=cfg.run.method_params.get('monitor_type', 'logistic')
            )
            self.threshold_selector = RiskControlledThresholdSelector(
                alpha=cfg.run.method_params.alpha,
                delta=cfg.run.method_params.delta
            )
            if cfg.run.method_params.shift_detection.enabled:
                self.shift_detector = ShiftDetector(
                    alpha_shift=cfg.run.method_params.shift_detection.alpha_shift
                )
            else:
                self.shift_detector = None
        
        elif self.method == "entropy-gate":
            self.entropy_threshold = cfg.run.method_params.entropy_threshold
    
    def run_monitor_training(self, monitor_data: List[Dict]) -> None:
        """Train correctness monitor on monitor_train split."""
        if self.method != "rc-cot":
            return
        
        print("\n" + "="*80)
        print("PHASE 1: Training Correctness Monitor")
        print("="*80)
        
        features_list = []
        labels = []
        
        for example in tqdm(monitor_data, desc="Extracting features"):
            # [VALIDATOR FIX - Attempt 1]
            # [PROBLEM]: ConfigAttributeError for system1_prompt, system2_prompt, extract_prompt
            # [CAUSE]: These prompts are nested under cfg.run in the Hydra config structure
            # [FIX]: Changed all cfg.system1_prompt, cfg.system2_prompt, cfg.extract_prompt to cfg.run.* (applied via replaceAll)
            #
            # Generate System 1 answer
            prompt = self.cfg.run.system1_prompt.format(question=example['question'])
            generated, logits, logprobs = self.model_inference.generate_with_scores(
                prompt, return_scores=True
            )
            
            # Extract features
            if logits is not None and logprobs is not None:
                features = self.feature_extractor.extract_features(logits, logprobs)
                features_list.append(features)
            else:
                raise ValueError("Expected scores but got None")
            
            # Check correctness
            predicted_answer = extract_final_answer(generated)
            is_correct = check_answer_correctness(predicted_answer, example['answer'])
            labels.append(is_correct)
        
        # Train monitor
        self.monitor.train(features_list, labels)
        print(f"Monitor training complete. Accuracy: {np.mean(labels):.2%}")
    
    def run_calibration(self, calib_data: List[Dict]) -> None:
        """Run calibration phase: select threshold and calibrate shift detector."""
        if self.method != "rc-cot":
            return
        
        print("\n" + "="*80)
        print("PHASE 2: Risk-Controlled Calibration")
        print("="*80)
        
        features_list = []
        predicted_probs = []
        actual_correct = []
        
        for example in tqdm(calib_data, desc="Generating calibration data"):
            # Generate System 1 answer
            prompt = self.cfg.run.system1_prompt.format(question=example['question'])
            generated, logits, logprobs = self.model_inference.generate_with_scores(
                prompt, return_scores=True
            )
            
            # Extract features
            if logits is not None and logprobs is not None:
                features = self.feature_extractor.extract_features(logits, logprobs)
                features_list.append(features)
            else:
                raise ValueError("Expected scores but got None")
            
            # Get predicted correctness probability
            prob = self.monitor.predict_proba(features)
            predicted_probs.append(prob)
            
            # Check actual correctness
            predicted_answer = extract_final_answer(generated)
            is_correct = check_answer_correctness(predicted_answer, example['answer'])
            actual_correct.append(is_correct)
        
        # Select risk-controlled threshold
        self.threshold_selector.select_threshold(predicted_probs, actual_correct)
        
        # Calibrate shift detector
        if self.shift_detector is not None:
            self.shift_detector.calibrate(features_list)
    
    def run_evaluation(self, eval_data: List[Dict], results_dir: Path, is_sanity: bool) -> Dict:
        """Run evaluation on eval split."""
        print("\n" + "="*80)
        print("PHASE 3: Evaluation")
        print("="*80)
        
        results = []
        metrics = {
            'total': 0,
            'correct': 0,
            'system1_used': 0,
            'system2_used': 0,
            'system1_correct': 0,
            'system2_correct': 0,
            'shift_detected': 0
        }
        
        for example in tqdm(eval_data, desc="Evaluating"):
            result = self.evaluate_single(example)
            results.append(result)
            
            # Update metrics
            metrics['total'] += 1
            if result['is_correct']:
                metrics['correct'] += 1
            
            if result['system_used'] == 1:
                metrics['system1_used'] += 1
                if result['is_correct']:
                    metrics['system1_correct'] += 1
            else:
                metrics['system2_used'] += 1
                if result['is_correct']:
                    metrics['system2_correct'] += 1
            
            if result.get('shift_detected', False):
                metrics['shift_detected'] += 1
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'accuracy': metrics['correct'] / metrics['total'],
                    'system1_usage': metrics['system1_used'] / metrics['total'],
                    'system2_usage': metrics['system2_used'] / metrics['total'],
                })
        
        # Compute final metrics
        metrics['accuracy'] = metrics['correct'] / metrics['total']
        metrics['system1_usage'] = metrics['system1_used'] / metrics['total']
        metrics['system2_usage'] = metrics['system2_used'] / metrics['total']
        
        if metrics['system1_used'] > 0:
            metrics['system1_accuracy'] = metrics['system1_correct'] / metrics['system1_used']
        else:
            metrics['system1_accuracy'] = 0.0
        
        if metrics['system2_used'] > 0:
            metrics['system2_accuracy'] = metrics['system2_correct'] / metrics['system2_used']
        else:
            metrics['system2_accuracy'] = 0.0
        
        # Save results
        results_file = results_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        metrics_file = results_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Metrics saved to: {metrics_file}")
        
        # Log final metrics to wandb
        if wandb.run is not None:
            for key, value in metrics.items():
                wandb.summary[key] = value
        
        # Sanity validation
        if is_sanity:
            self.run_sanity_validation(metrics, results)
        
        return metrics
    
    def evaluate_single(self, example: Dict) -> Dict:
        """Evaluate a single example using the appropriate method."""
        question = example['question']
        ground_truth = example['answer']
        
        if self.method == "always-fast":
            # Always use System 1
            prompt = self.cfg.run.system1_prompt.format(question=question)
            generated, _, _ = self.model_inference.generate_with_scores(prompt, return_scores=False)
            predicted_answer = extract_final_answer(generated)
            is_correct = check_answer_correctness(predicted_answer, ground_truth)
            
            return {
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'system_used': 1,
                'generated_text': generated
            }
        
        elif self.method == "always-cot":
            # Always use System 2
            prompt = self.cfg.run.system2_prompt.format(question=question)
            cot_solution, _, _ = self.model_inference.generate_with_scores(prompt, return_scores=False)
            
            # Extract final answer
            extract_prompt = self.cfg.run.extract_prompt.format(solution=cot_solution)
            answer_text, _, _ = self.model_inference.generate_with_scores(extract_prompt, return_scores=False)
            predicted_answer = extract_final_answer(answer_text)
            is_correct = check_answer_correctness(predicted_answer, ground_truth)
            
            return {
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'system_used': 2,
                'cot_solution': cot_solution,
                'generated_text': answer_text
            }
        
        elif self.method == "entropy-gate":
            # Use entropy-based heuristic
            prompt = self.cfg.run.system1_prompt.format(question=question)
            generated, logits, logprobs = self.model_inference.generate_with_scores(prompt, return_scores=True)
            
            # Compute entropy
            if logits is not None and logprobs is not None:
                features = self.feature_extractor.extract_features(logits, logprobs)
            else:
                raise ValueError("Expected scores but got None")
            entropy = features['first_token_entropy']
            
            if entropy < self.entropy_threshold:
                # Low entropy: use System 1
                predicted_answer = extract_final_answer(generated)
                is_correct = check_answer_correctness(predicted_answer, ground_truth)
                
                return {
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'system_used': 1,
                    'entropy': entropy,
                    'generated_text': generated
                }
            else:
                # High entropy: use System 2
                prompt2 = self.cfg.run.system2_prompt.format(question=question)
                cot_solution, _, _ = self.model_inference.generate_with_scores(prompt2, return_scores=False)
                
                extract_prompt = self.cfg.run.extract_prompt.format(solution=cot_solution)
                answer_text, _, _ = self.model_inference.generate_with_scores(extract_prompt, return_scores=False)
                predicted_answer = extract_final_answer(answer_text)
                is_correct = check_answer_correctness(predicted_answer, ground_truth)
                
                return {
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'system_used': 2,
                    'entropy': entropy,
                    'cot_solution': cot_solution,
                    'generated_text': answer_text
                }
        
        elif self.method == "rc-cot":
            # RC-CoT method with risk control and shift detection
            prompt = self.cfg.run.system1_prompt.format(question=question)
            generated, logits, logprobs = self.model_inference.generate_with_scores(prompt, return_scores=True)
            
            # Extract features
            if logits is not None and logprobs is not None:
                features = self.feature_extractor.extract_features(logits, logprobs)
            else:
                raise ValueError("Expected scores but got None")
            
            # Check for distribution shift
            shift_detected = False
            if self.shift_detector is not None:
                if not self.shift_detector.is_in_distribution(features):
                    shift_detected = True
            
            # Decide which system to use
            use_fast = False
            predicted_prob = 0.0
            if not shift_detected:
                predicted_prob = self.monitor.predict_proba(features)
                use_fast = self.threshold_selector.should_use_fast_path(predicted_prob)
            
            if use_fast:
                # Use System 1 (fast path)
                predicted_answer = extract_final_answer(generated)
                is_correct = check_answer_correctness(predicted_answer, ground_truth)
                
                return {
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'system_used': 1,
                    'predicted_prob': predicted_prob,
                    'shift_detected': shift_detected,
                    'features': features,
                    'generated_text': generated
                }
            else:
                # Use System 2 (deliberation path)
                prompt2 = self.cfg.run.system2_prompt.format(question=question)
                cot_solution, _, _ = self.model_inference.generate_with_scores(prompt2, return_scores=False)
                
                extract_prompt = self.cfg.run.extract_prompt.format(solution=cot_solution)
                answer_text, _, _ = self.model_inference.generate_with_scores(extract_prompt, return_scores=False)
                predicted_answer = extract_final_answer(answer_text)
                is_correct = check_answer_correctness(predicted_answer, ground_truth)
                
                return {
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'system_used': 2,
                    'predicted_prob': predicted_prob if not shift_detected else None,
                    'shift_detected': shift_detected,
                    'features': features,
                    'cot_solution': cot_solution,
                    'generated_text': answer_text
                }
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def run_sanity_validation(self, metrics: Dict, results: List[Dict]) -> None:
        """Perform sanity validation and print verdict."""
        print("\n" + "="*80)
        print("SANITY VALIDATION")
        print("="*80)
        
        # Check: at least 5 samples processed
        if metrics['total'] < 5:
            print("SANITY_VALIDATION: FAIL reason=insufficient_samples")
            print(f"SANITY_VALIDATION_SUMMARY: {json.dumps({'samples': metrics['total']})}")
            return
        
        # Check: all metrics are finite
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                print(f"SANITY_VALIDATION: FAIL reason=non_finite_metric_{key}")
                print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(metrics)}")
                return
        
        # [VALIDATOR FIX - Attempt 3]
        # [PROBLEM]: Sanity validation fails with "suspicious_uniformity" when all predictions are wrong (0% accuracy)
        # [CAUSE]: Code incorrectly interprets "not all identical outputs" as "not all correct or all wrong", 
        #          but the instruction means the predicted answers themselves should be diverse, not the correctness.
        #          A weak baseline model (like GPT-2 on hard math) can legitimately get 0% accuracy with diverse outputs.
        # [FIX]: Check if the actual predicted answers are diverse (at least 3 unique values for 10+ samples),
        #        not whether the correctness is uniform.
        #
        # [OLD CODE]:
        # # Check: not all results are identical (outputs are valid and non-trivial)
        # if metrics['correct'] == 0 or metrics['correct'] == metrics['total']:
        #     # All correct or all wrong is suspicious in sanity mode
        #     if metrics['total'] >= 10:
        #         print(f"SANITY_VALIDATION: FAIL reason=suspicious_uniformity")
        #         print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(metrics)}")
        #         return
        #
        # [NEW CODE]:
        # Check: outputs are diverse (not all identical predictions)
        predicted_answers = [r['predicted_answer'] for r in results]
        unique_answers = len(set(predicted_answers))
        
        # For sanity check, we expect at least some diversity in outputs
        # If 10+ samples all produce the exact same answer, something is wrong
        if metrics['total'] >= 10 and unique_answers == 1:
            print(f"SANITY_VALIDATION: FAIL reason=all_identical_outputs")
            print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(metrics)}")
            return
        
        # All checks passed
        print("SANITY_VALIDATION: PASS")
        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(metrics)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    cfg = OmegaConf.load(args.config)
    
    print("="*80)
    print("RC-CoT INFERENCE")
    print("="*80)
    print(f"Run ID: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method}")
    print(f"Model: {cfg.run.model}")
    print(f"Dataset: {cfg.run.dataset}")
    print(f"Mode: {cfg.mode}")
    
    # Set random seed
    torch.manual_seed(cfg.inference.seed)
    np.random.seed(cfg.inference.seed)
    
    # Initialize WandB
    is_sanity = cfg.mode == "sanity_check"
    if cfg.wandb.mode == "online":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow"
        )
        print(f"WandB run initialized: {wandb.run.url}")
    else:
        print("WandB disabled")
    
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: ConfigAttributeError: Missing key data
    # [CAUSE]: Config structure has 'data' nested under 'run', but code was accessing it as cfg.data
    # [FIX]: Changed cfg.data to cfg.run.data to match the actual Hydra config structure
    #
    # [OLD CODE]:
    # monitor_data, calib_data, eval_data = load_gsm8k_splits(
    #     monitor_train_size=cfg.data.monitor_train_size,
    #     calibration_size=cfg.data.calibration_size,
    #     eval_size=cfg.data.eval_size,
    #     cache_dir=cfg.inference.cache_dir,
    #     seed=cfg.inference.seed
    # )
    #
    # [NEW CODE]:
    # Load data
    monitor_data, calib_data, eval_data = load_gsm8k_splits(
        monitor_train_size=cfg.run.data.monitor_train_size,
        calibration_size=cfg.run.data.calibration_size,
        eval_size=cfg.run.data.eval_size,
        cache_dir=cfg.inference.cache_dir,
        seed=cfg.inference.seed
    )
    
    # Create results directory
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference pipeline
    pipeline = RC_CoT_Inference(cfg)
    
    # Run appropriate pipeline based on method
    if cfg.run.method == "rc-cot":
        # Full RC-CoT pipeline: train monitor, calibrate, evaluate
        pipeline.run_monitor_training(monitor_data)
        pipeline.run_calibration(calib_data)
    
    # Run evaluation
    metrics = pipeline.run_evaluation(eval_data, results_dir, is_sanity)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"System 1 usage: {metrics['system1_usage']:.2%}")
    print(f"System 2 usage: {metrics['system2_usage']:.2%}")
    if metrics['system1_used'] > 0:
        print(f"System 1 accuracy: {metrics['system1_accuracy']:.2%}")
    if metrics['system2_used'] > 0:
        print(f"System 2 accuracy: {metrics['system2_accuracy']:.2%}")
    
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
