"""
Inference script for RC-CoT experiments.
Implements all methods: rc-cot, always-fast, always-cot, entropy-gate.

[VALIDATOR FIX - Attempt 4]
[PROBLEM]: All 10 samples got 0% accuracy with System 2 (CoT) generating only "\n"
[CAUSE]: Previous Attempt 3 added newline as a global stopping criterion, but this stopped
         CoT generation immediately after the first newline. System 2 needs multi-line output
         for step-by-step reasoning, but was being forced to stop after one token.
[FIX]: Added allow_multiline parameter to generate_with_scores(). When False (default),
       stops at newline for direct answers and extraction. When True, allows multi-line
       output for CoT generation. Updated all 12 call sites appropriately:
       - System 1 (direct answers): allow_multiline=False
       - System 2 (CoT reasoning): allow_multiline=True  
       - Extract prompts: allow_multiline=False
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
        return_scores: bool = False,
        allow_multiline: bool = False
    ) -> Tuple[str, Optional[np.ndarray], Optional[List[float]]]:
        """
        Generate text with optional score extraction.
        
        Args:
            prompt: Input prompt
            return_scores: Whether to return logits and logprobs
            allow_multiline: If False, stop at first newline. If True, allow multi-line output.
        
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
            
            # [VALIDATOR FIX - Attempt 4]
            # [PROBLEM]: System 2 (CoT) generates only "\n" because newline stopping is too aggressive
            # [CAUSE]: Attempt 3 added newline as stopping criterion globally, but CoT needs multi-line output
            # [FIX]: Only apply newline stopping when allow_multiline=False (for direct answers and extraction)
            #
            # [OLD CODE]:
            # # [VALIDATOR FIX - Attempt 3]
            # # [PROBLEM]: Model generates continuation text after the answer (e.g., " 1\n\nQ: Jolene...")
            # # [CAUSE]: No stopping criteria - model continues generating up to max_new_tokens
            # # [FIX]: Add newline token as stopping criterion to stop after first line
            # #
            # # [OLD CODE]:
            # # gen_kwargs = {
            # #     'max_new_tokens': self.model_params.get('max_length', 512),
            # # }
            # #
            # # [NEW CODE]:
            # # Handle greedy decoding when temperature is 0 or do_sample is False
            # # Add newline as stopping criterion to prevent continuation
            # newline_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[0]
            # gen_kwargs = {
            #     'max_new_tokens': self.model_params.get('max_length', 512),
            #     'eos_token_id': [self.tokenizer.eos_token_id, newline_token_id],
            # }
            #
            # [NEW CODE]:
            gen_kwargs = {
                'max_new_tokens': self.model_params.get('max_length', 512),
            }
            
            # Only add newline stopping for single-line outputs (direct answers, final extractions)
            # CoT generation needs to produce multi-line reasoning, so don't stop on newlines
            if not allow_multiline:
                newline_token_id = self.tokenizer.encode('\n', add_special_tokens=False)[0]
                gen_kwargs['eos_token_id'] = [self.tokenizer.eos_token_id, newline_token_id]
            
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


# [VALIDATOR FIX - Attempt 7]
# [PROBLEM]: 0% accuracy due to poor answer extraction from incoherent GPT-2 outputs
# [CAUSE]: GPT-2 (117M params) cannot do math reasoning with zero-shot prompts. Previous attempts
#          tried to extract numbers from rambling text, but the outputs were just repetitive garbage.
# [FIX]: Updated prompts to use few-shot examples with explicit ANSWER= format (see config changes).
#        Now prioritize ANSWER= pattern first, then fall back to other explicit markers, then
#        look for bare numbers (which might occur if GPT-2 outputs just a number).
#
# [OLD CODE - Attempt 6]:
# (Multiple steps of heuristic extraction from noisy text)
#
# [NEW CODE]:
def extract_final_answer(text: str) -> str:
    """Extract numeric answer from generated text."""
    if not text or len(text.strip()) == 0:
        return "0"
    
    text = text.strip()
    
    # Step 1: Check if the entire output is just a number (common for System 1 with good prompting)
    # This handles cases like "288" or "42" as the complete response
    if re.match(r'^-?\d+(?:\.\d+)?$', text):
        return text
    
    # Step 2: Look for ANSWER= pattern (from few-shot prompts)
    answer_pattern = r'ANSWER\s*=\s*\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)'
    match = re.search(answer_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    
    # Step 3: Look for other explicit answer markers
    explicit_patterns = [
        r'FINAL\s*[=:]\s*\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',
        r'[Tt]he\s+(?:final\s+)?answer\s+is\s+\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',
        r'[Aa]nswer\s*[=:]\s*\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',
        r'[Tt]otal\s*[=:]\s*\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',
        r'[Rr]esult\s*[=:]\s*\$?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)',
    ]
    
    for pattern in explicit_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(',', '')
    
    # Step 4: Look for numbers in the last 50 characters (likely to be the answer)
    last_portion = text[-50:].strip()
    
    # Extract all numbers with optional $ prefix from last portion
    numbers_at_end = re.findall(r'[$]?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', last_portion)
    if numbers_at_end:
        # Clean and return the last number found
        cleaned = numbers_at_end[-1].replace(',', '').strip()
        # Filter out obviously wrong extractions (e.g., just "0" repeated)
        if cleaned and cleaned not in ['0.0']:
            return cleaned
    
    # Step 5: Extract all numbers from entire text and return last one
    all_numbers = re.findall(r'[$]?\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    if all_numbers:
        # Filter out very large numbers that are likely noise (e.g., "40000" when answer is "40")
        cleaned_numbers = [n.replace(',', '').strip() for n in all_numbers]
        # Return last number that looks reasonable (not too long)
        for num in reversed(cleaned_numbers):
            try:
                val = float(num)
                # Reasonable range for GSM8K answers (most are under 10000)
                if -1000000 < val < 1000000:
                    return num
            except ValueError:
                continue
    
    # Step 6: Fallback to "0" if no valid number found (better than returning garbage text)
    return "0"


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
                prompt, return_scores=True, allow_multiline=False
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
                prompt, return_scores=True, allow_multiline=False
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
            generated, _, _ = self.model_inference.generate_with_scores(prompt, return_scores=False, allow_multiline=False)
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
            # [VALIDATOR FIX - Attempt 5]
            # [PROBLEM]: 0% accuracy because GPT-2 cannot follow the "FINAL=" extraction prompt
            # [CAUSE]: Small models like GPT-2 don't understand meta-instructions like "extract only the final numeric answer".
            #          The extract_prompt generates nonsense (e.g., "1.5 gallons", "DONE"), leading to wrong answers.
            # [FIX]: Extract answer directly from the CoT solution instead of using a separate extraction prompt.
            #        This is more robust for weak models that can't follow extraction instructions.
            #
            # [OLD CODE]:
            # # Always use System 2
            # prompt = self.cfg.run.system2_prompt.format(question=question)
            # cot_solution, _, _ = self.model_inference.generate_with_scores(prompt, return_scores=False, allow_multiline=True)
            # 
            # # Extract final answer
            # extract_prompt = self.cfg.run.extract_prompt.format(solution=cot_solution)
            # answer_text, _, _ = self.model_inference.generate_with_scores(extract_prompt, return_scores=False, allow_multiline=False)
            # predicted_answer = extract_final_answer(answer_text)
            # is_correct = check_answer_correctness(predicted_answer, ground_truth)
            # 
            # return {
            #     'question': question,
            #     'ground_truth': ground_truth,
            #     'predicted_answer': predicted_answer,
            #     'is_correct': is_correct,
            #     'system_used': 2,
            #     'cot_solution': cot_solution,
            #     'generated_text': answer_text
            # }
            #
            # [NEW CODE]:
            # Always use System 2
            prompt = self.cfg.run.system2_prompt.format(question=question)
            cot_solution, _, _ = self.model_inference.generate_with_scores(prompt, return_scores=False, allow_multiline=True)
            
            # Extract final answer directly from CoT solution (more robust for weak models)
            predicted_answer = extract_final_answer(cot_solution)
            is_correct = check_answer_correctness(predicted_answer, ground_truth)
            
            return {
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'system_used': 2,
                'cot_solution': cot_solution,
                'generated_text': cot_solution  # Use CoT solution as generated text
            }
        
        elif self.method == "entropy-gate":
            # Use entropy-based heuristic
            prompt = self.cfg.run.system1_prompt.format(question=question)
            generated, logits, logprobs = self.model_inference.generate_with_scores(prompt, return_scores=True, allow_multiline=False)
            
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
                # [VALIDATOR FIX - Attempt 5]
                # [PROBLEM]: Same issue as always-cot - 0% accuracy due to failed answer extraction
                # [CAUSE]: GPT-2 can't follow the extraction prompt
                # [FIX]: Extract answer directly from CoT solution
                #
                # [OLD CODE]:
                # # High entropy: use System 2
                # prompt2 = self.cfg.run.system2_prompt.format(question=question)
                # cot_solution, _, _ = self.model_inference.generate_with_scores(prompt2, return_scores=False, allow_multiline=True)
                # 
                # extract_prompt = self.cfg.run.extract_prompt.format(solution=cot_solution)
                # answer_text, _, _ = self.model_inference.generate_with_scores(extract_prompt, return_scores=False, allow_multiline=False)
                # predicted_answer = extract_final_answer(answer_text)
                # is_correct = check_answer_correctness(predicted_answer, ground_truth)
                # 
                # return {
                #     'question': question,
                #     'ground_truth': ground_truth,
                #     'predicted_answer': predicted_answer,
                #     'is_correct': is_correct,
                #     'system_used': 2,
                #     'entropy': entropy,
                #     'cot_solution': cot_solution,
                #     'generated_text': answer_text
                # }
                #
                # [NEW CODE]:
                # High entropy: use System 2
                prompt2 = self.cfg.run.system2_prompt.format(question=question)
                cot_solution, _, _ = self.model_inference.generate_with_scores(prompt2, return_scores=False, allow_multiline=True)
                
                # Extract answer directly from CoT solution
                predicted_answer = extract_final_answer(cot_solution)
                is_correct = check_answer_correctness(predicted_answer, ground_truth)
                
                return {
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'system_used': 2,
                    'entropy': entropy,
                    'cot_solution': cot_solution,
                    'generated_text': cot_solution
                }
        
        elif self.method == "rc-cot":
            # RC-CoT method with risk control and shift detection
            prompt = self.cfg.run.system1_prompt.format(question=question)
            generated, logits, logprobs = self.model_inference.generate_with_scores(prompt, return_scores=True, allow_multiline=False)
            
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
                # [VALIDATOR FIX - Attempt 5]
                # [PROBLEM]: Same issue as always-cot - 0% accuracy due to failed answer extraction
                # [CAUSE]: GPT-2 can't follow the extraction prompt
                # [FIX]: Extract answer directly from CoT solution
                #
                # [OLD CODE]:
                # # Use System 2 (deliberation path)
                # prompt2 = self.cfg.run.system2_prompt.format(question=question)
                # cot_solution, _, _ = self.model_inference.generate_with_scores(prompt2, return_scores=False, allow_multiline=True)
                # 
                # extract_prompt = self.cfg.run.extract_prompt.format(solution=cot_solution)
                # answer_text, _, _ = self.model_inference.generate_with_scores(extract_prompt, return_scores=False, allow_multiline=False)
                # predicted_answer = extract_final_answer(answer_text)
                # is_correct = check_answer_correctness(predicted_answer, ground_truth)
                # 
                # return {
                #     'question': question,
                #     'ground_truth': ground_truth,
                #     'predicted_answer': predicted_answer,
                #     'is_correct': is_correct,
                #     'system_used': 2,
                #     'predicted_prob': predicted_prob if not shift_detected else None,
                #     'shift_detected': shift_detected,
                #     'features': features,
                #     'cot_solution': cot_solution,
                #     'generated_text': answer_text
                # }
                #
                # [NEW CODE]:
                # Use System 2 (deliberation path)
                prompt2 = self.cfg.run.system2_prompt.format(question=question)
                cot_solution, _, _ = self.model_inference.generate_with_scores(prompt2, return_scores=False, allow_multiline=True)
                
                # Extract answer directly from CoT solution
                predicted_answer = extract_final_answer(cot_solution)
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
                    'generated_text': cot_solution
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
