from unsloth import FastLanguageModel
from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



##############################################################################################
import re
from datasets import load_dataset, Dataset
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np
import psutil
import GPUtil
import threading
import time
from openai import OpenAI
from pydantic import BaseModel
from google import genai
import sys
import argparse
try:
    from ollama import chat as ollama_chat
    from ollama import ChatResponse as OllamaChatResponse
except Exception:
    ollama_chat = None
    OllamaChatResponse = None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="GRPO Training and Evaluation for HFACS Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
Examples:
  # Train the model and save LoRA to 'my_model' folder
  python test.py --train --save-lora-folder my_model

  # Train with auto-generated timestamp folder name
  python test.py --train

  # Train with custom dataset paths (synthetic data always saved automatically)
  python test.py --train --gahfacs-path /path/to/GAHFACS.xlsx --load-synthetic-data /path/to/synthetic.xlsx

  # Train and evaluate with matching synthetic data (recommended workflow)
  python test.py --train --save-lora-folder my_model --save-synthetic-data synth_my_model.xlsx
  python test.py --evaluate --load-lora-folder my_model --load-synthetic-data synth_my_model.xlsx

  # Train and then evaluate in single command (will auto-generate synthetic filename)
  python test.py --train --evaluate --save-lora-folder my_model --load-lora-folder my_model

  # Show this help
  python test.py --help
"""
    )
    
    parser.add_argument(
        '--train', 
        action='store_true', 
        help='Train the model using GRPO'
    )
    
    parser.add_argument(
        '--evaluate', 
        action='store_true', 
        help='Evaluate the model and generate comparison graphs'
    )
    
    parser.add_argument(
        '--load-lora-folder', 
        type=str, 
        default=None,
        help='Name of the LoRA folder to load from (for evaluation)'
    )
    
    parser.add_argument(
        '--save-lora-folder', 
        type=str, 
        default=None,
        help='Name of the LoRA folder to save to (default: grpo_saved_YYYYMMDD_HHMMSS)'
    )
    
    parser.add_argument(
        '--gahfacs-path',
        type=str,
        default='GAHFACS.xlsx',
        help='Path to the main GAHFACS dataset file (default: GAHFACS.xlsx)'
    )
    
    parser.add_argument(
        '--load-synthetic-data',
        type=str,
        default=None,
        help='Path to existing synthetic dataset file to load (optional)'
    )
    
    parser.add_argument(
        '--save-synthetic-data',
        type=str,
        default=None,
        help='Custom path to save generated synthetic dataset file (default: synth_GAHFACS_YYYYMMDD_HHMMSS.xlsx)'
    )
    
    args = parser.parse_args()
    
    # Generate timestamp-based folder name if save folder not provided
    if args.save_lora_folder is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_lora_folder = f"grpo_saved_{timestamp}"
    
    # For load folder, use save folder as default if not provided
    if args.load_lora_folder is None:
        args.load_lora_folder = args.save_lora_folder
    
    return args

def show_help_and_exit():
    """Show help message when no arguments provided"""
    print("="*60)
    print("GRPO Training and Evaluation for HFACS Classification")
    print("="*60)
    print()
    print("This script requires arguments to run. Use one of the following:")
    print()
    print("  --train                    Train the model using GRPO")
    print("  --evaluate                 Evaluate model and generate graphs") 
    print("  --save-lora-folder NAME    LoRA folder name to save to (default: grpo_saved_YYYYMMDD_HHMMSS)")
    print("  --load-lora-folder NAME    LoRA folder name to load from (default: same as save folder)")
    print("  --gahfacs-path PATH        Path to main GAHFACS dataset (default: GAHFACS.xlsx)")
    print("  --load-synthetic-data PATH Path to existing synthetic dataset to load (optional)")
    print("  --save-synthetic-data PATH Custom path to save synthetic data (default: auto-generated)")
    print()
    print("Examples:")
    print("  python test.py --train --save-lora-folder my_model")
    print("  python test.py --train                                  # Uses grpo_saved_YYYYMMDD_HHMMSS")
    print("  python test.py --train --gahfacs-path custom_data.xlsx --load-synthetic-data existing_synth.xlsx")
    print("  # Train and evaluate with matching synthetic data (recommended):")
    print("  python test.py --train --save-lora-folder my_model --save-synthetic-data synth_my_model.xlsx")
    print("  python test.py --evaluate --load-lora-folder my_model --load-synthetic-data synth_my_model.xlsx")
    print()
    print("For detailed help: python test.py --help")
    print()
    sys.exit(0)

# Parse command line arguments
args = parse_arguments()

# Show help if no action specified
if not args.train and not args.evaluate:
    show_help_and_exit()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.8, # Reduce if out of memory
)
print("first part!")
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

print("Everythin is loaded yayyyyy!")
# --- START: New code block for GPT-based Reward ---

# Pricing per 1M tokens (USD)
PRICING_USD_PER_MTOK = {
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
}

# Simple cost tracker
api_cost_spent_by_model = {}
api_cost_total = 0.0

def _approx_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    # rough heuristic: 4 chars/token
    return max(1, int(len(text) / 4))

def _estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING_USD_PER_MTOK.get(model_name)
    if not pricing:
        return 0.0
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]

def _check_and_log_cost(model_name: str, input_text: str, est_output_tokens: int, action_label: str) -> None:
    global api_cost_spent_by_model, api_cost_total
    input_tokens = _approx_tokens_from_text(input_text)
    cost = _estimate_cost_usd(model_name, input_tokens, est_output_tokens)
    api_cost_spent_by_model[model_name] = api_cost_spent_by_model.get(model_name, 0.0) + cost
    new_overall_total = api_cost_total + cost
    # Print only the overall total to avoid noisy per-call details
    print(f"[Cost] Total estimated spend after {action_label}: ${new_overall_total:.2f}")
    # Ask if overall cumulative crosses $20
    if new_overall_total > 20.0:
        ans = input(f"Overall estimated cost is now ${new_overall_total:.2f} (> $20). Proceed with this call? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            raise SystemExit(f"Aborting {action_label} due to cost constraints.")
    api_cost_total = new_overall_total

# Minimal pre-synthesis LLM smoke tests
def pre_synthesis_llm_smoke(openai_model_for_synth: str) -> None:
    print("[Stage] Pre-synthesis LLM readiness check...")
    # Test OpenAI synth model
    try:
        if client is None:
            raise RuntimeError("OpenAI client not initialized")
        resp = client.responses.create(
            model=openai_model_for_synth,
            input=[{"role": "user", "content": "ping"}],
        )
        ok = bool(getattr(resp, 'output_text', ''))
        print(f"[Check] OpenAI {openai_model_for_synth} OK={ok}")
        if not ok:
            raise RuntimeError("Empty output from OpenAI synth model ping")
    except Exception as e:
        raise SystemExit(f"Pre-synthesis check failed for OpenAI model {openai_model_for_synth}: {e}")

    # Test Gemini
    try:
        if genai_client is None:
            raise RuntimeError("Gemini client not initialized")
        resp = genai_client.models.generate_content(model="gemini-2.5-flash", contents="ping")
        txt = getattr(resp, 'text', None) or getattr(resp, 'content', None) or ''
        ok = bool(txt)
        print(f"[Check] Gemini 2.5 Flash OK={ok}")
        if not ok:
            raise RuntimeError("Empty output from Gemini ping")
    except Exception as e:
        raise SystemExit(f"Pre-synthesis check failed for Gemini: {e}")

    # OSS model check
    models = [
        "gpt-oss:20b", "gemma3:12b", "deepseek-r1:8b",
        "deepseek-r1:14b", "gemma3n:e4b", "qwen3:8b",
    ]
    if ollama_chat is None:
        print("[Check] Ollama unavailable; skipping OSS checks.")
    else:
        for m in models:
            try:
                r = ollama_chat(model=m, messages=[{"role": "user", "content": "ping"}])
                msg = r["message"]["content"] if isinstance(r, dict) else getattr(r, 'message', None)
                txt = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
                ok = bool(txt)
                print(f"[Check] OSS {m} OK={ok}")
                if not ok:
                    raise RuntimeError("Empty output")
            except Exception as e:
                print(f"[Check] OSS {m} FAILED: {e} (will skip this model during evaluation)")

# Pydantic model for structured response from gpt-5-nano
class GptRewardScore(BaseModel):
    score: float

# Pydantic model for structured response from Gemini
class HfacsResponse(BaseModel):
    reasoning: str
    hfacs_codes: list[str]

# Instantiate the OpenAI client
# Ensure your OPENAI_API_KEY environment variable is set
try:
    client = OpenAI()
except Exception as e:
    print(f"Failed to initialize OpenAI client. Please check API key. Error: {e}")
    client = None

# Instantiate the Google GenAI client
# Ensure your GOOGLE_API_KEY environment variable is set
try:
    genai_client = genai.Client()
except Exception as e:
    print(f"Failed to initialize Google GenAI client. Please check API key. Error: {e}")
    genai_client = None

def extract_reasoning_text(text: str) -> str:
    """Extracts the content from within the <reasoning> tags."""
    match = re.search(r'<reasoning>(.*?)</reasoning>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def gpt_reasoning_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """
    Judges the quality of the reasoning using gpt-5-nano.
    Returns a reward: 0 for bad, 0.5 for okay, 1.0 for good.
    """
    global reward_logs, client
    
    if not client:
        print("OpenAI client not available. Skipping GPT reasoning reward.")
        return [0.0] * len(completions)

    rewards = []
    
    for i, completion in enumerate(completions):
        response_text = completion[0]['content']
        reasoning_content = extract_reasoning_text(response_text)
        
        reward = 0.0  # Default reward if something fails
        
        if not reasoning_content:
            # No reasoning found, so no reward can be given.
            rewards.append(0.0)
        else:
            try:
                system_prompt = """You are an expert aviation safety analyst. Your task is to evaluate the quality of a reasoning snippet for an accident report. The reasoning should be logical, concise, directly supported by the provided narrative, and should lead to or support the correct HFACS classification.

                Context on HFACS (Human Factors Analysis and Classification System) categories used in these tasks:
                - AE100: Performance/Skill Based Errors — Behavior proceeds as intended but is inappropriate/inadequate to achieve the desired end-state
                - AE200: Judgment/Decision-making Errors — Choosing an inappropriate procedure or making a poor decision
                - AD000: Known Deviations — Willful violations or disregard of rules, regulations, instructions, or SOPs
                - PC100: Mental Awareness Conditions — Attention, concentration, vigilance, alertness issues
                - PC200: State of Mind Conditions — Complacency, false sense of security, overconfidence, stress, distraction
                - PC300: Physical Conditions — Fatigue, illness, physical impairment
                - PE100: Physical Environment — Weather, lighting, visibility, terrain
                - PE200: Technological Environment — HMI issues, equipment design problems, automation factors
                - PP100: Planning Conditions — Inadequate planning, improper briefing, insufficient preparation
                - PT100: Training Conditions — Inadequate training, lack of proficiency, skills deficits

                Score the reasoning on a three-point scale:
                - 0.5 (Good): The reasoning is clear, logical, accurately reflects the key factors of the accident, and appropriately supports the correct classification.
                - 0.25 (Okay): The reasoning is plausible and somewhat supports the correct classification but misses key details, is slightly illogical, or makes minor unsupported claims.
                - 0.0 (Bad): The reasoning is illogical, irrelevant, completely unsupported by the narrative, or leads to an incorrect understanding of the accident factors.

                Provide only a numeric score in the structured response.
                """
                
                user_prompt = f"""Evaluate the following reasoning for the accident narrative.
                
                Accident Narrative:
                {prompts[0][-1]['content']}
                
                Correct HFACS Classification:
                {answer[i]}
                
                Reasoning to Evaluate:
                {reasoning_content}
                
                Assess whether the reasoning logically supports the correct classification and accurately reflects the key factors from the narrative.
                """
                
                response = client.responses.parse(
                    model="gpt-5-nano",
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    text_format=GptRewardScore,
                )
                reward = response.output_parsed.score

            except Exception as e:
                print(f"Error calling GPT-5-nano for reward. Defaulting to 0. Error: {e}")
                reward = 0.0

            rewards.append(reward)

        # Update the corresponding log entry
        if len(reward_logs) > i:
            reward_logs[-(len(completions)-i)]['gpt_reasoning_reward'] = reward
    
    return rewards

# --- END: New code block for GPT-based Reward ---

# Load and prep dataset
# SYSTEM_PROMPT = """
# Respond in the following format:
# <reasoning>
# ...
# </reasoning>
# <answer>
# ...
# </answer>
# """

# XML_COT_FORMAT = """\
# <reasoning>
# {reasoning}
# </reasoning>
# <answer>
# {answer}
# </answer>
# """

system_prompt = """You are an expert in aviation accident analysis using the Human Factors Analysis and Classification System (HFACS). 
HFACS Categories:
- AE100: Performance/Skill Based Errors - Errors that occur when behavior proceeds as intended but the intended behavior is inappropriate or inadequate to achieve the desired end-state
- AE200: Judgment/Decision-making Errors - Errors that occur when a person chooses an inappropriate procedure or makes a poor decision
- AD000: Known Deviations - Violations that occur when an individual willingly disregards rules, regulations, instructions, or standard operating procedures
- PC100: Mental Awareness Conditions - Factors that affect mental awareness including attention, concentration, vigilance, and alertness
- PC200: State of Mind Conditions - Factors such as complacency, false sense of security, overconfidence, stress, and distraction
- PC300: Physical Conditions - Physical factors that affect performance including fatigue, illness, and physical impairment
- PE100: Physical Environment - Environmental factors such as weather, lighting, visibility, and terrain
- PE200: Technological Environment - Human-machine interface issues, equipment design problems, and automation-related factors
- PP100: Planning Conditions - Inadequate planning, improper briefing, and insufficient preparation
- PT100: Training Conditions - Inadequate training, lack of proficiency, and skills deficits

Instructions:
1. Analyze the accident narrative and identify key human factors and errors
2. Provide your reasoning ONLY within the <reasoning> tags
3. After the reasoning, provide ONLY the HFACS category codes
4. If multiple categories apply, separate them with white space (for example: "AE100 PE200")
5. Do not include any other text after the reasoning tags

Example format:
<reasoning>
The pilot reported that during takeoff from the private airstrip, the right wing of the airplane impacted bushes alongside of the narrow turf runway. As a result of the impact, the airplane to yawed to the right. The pilot attempted to pull back on the yoke, but lost control and the airplane impacted the ground and nosed over.  The pilot reported that there were no pre-accident mechanic malfunctions or failures with the airplane that would have precluded normal operation.
</reasoning>
AE200 PE100
"""

# def extract_xml_answer(text: str) -> str:
#     answer = text.split("<answer>")[-1]
#     answer = answer.split("</answer>")[0]
#     return answer.strip()


    # uncomment middle messages for 1-shot prompting
# def get_gsm8k_questions(split = "train") -> Dataset:
#     data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
#     data = data.map(lambda x: { # type: ignore
#         'prompt': [
#             {'role': 'system', 'content': SYSTEM_PROMPT},
#             {'role': 'user', 'content': x['question']}
#         ],
#         'answer': extract_hash_answer(x['answer'])
#     }) # type: ignore
#     return data # type: ignore

class GAHFACSDataLoader:
    """Load, augment, and preprocess GAHFACS dataset with strategic oversampling.

    Ensure each HFACS category appears in at least target_fraction (default 10%)
    of the original dataset size by generating synthetic narratives for the rare
    classes
    """

    def __init__(
        self,
        file_path: str = "GAHFACS.xlsx",
        sample_size: int = 1000,
        target_fraction: float = 0.10,
        max_synthetics_per_class: int = 100,
        openai_model: str = "gpt-5",
        synthetic_path: str = None,
    ):
        self.file_path = file_path
        self.sample_size = sample_size
        self.target_fraction = target_fraction
        self.max_synthetics_per_class = max_synthetics_per_class
        self.openai_model = openai_model
        self.synthetic_path = synthetic_path

        self.hfacs_categories = [
            'AE100', 'AE200', 'AD000', 'PC100', 'PC200',
            'PC300', 'PE100', 'PE200', 'PP100', 'PT100'
        ]

    def load_data(self) -> pd.DataFrame:
        """Load GAHFACS dataset without forcing balance - keep original distribution.

        The balancing will be done later during train/test split.
        """
        df_all = pd.read_excel(self.file_path)
        print(f"[Loader] Total rows in file: {len(df_all)}")
        
        # Store class counts for information
        for code in self.hfacs_categories:
            if code in df_all.columns:
                count = int(df_all[code].notna().sum())
                print(f"[Loader] Class {code}: {count} rows available")
        
        # No balancing here - return the full dataset
        # Synthesis planning will be done during split_data
        self.synth_plan = {}
        
        # Load and validate synthetic data if provided
        if self.synthetic_path:
            # Pass main dataset class counts to synthetic data loader so we don't over-generate
            main_class_counts = self._compute_class_distribution(df_all)
            synthetic_df = self._load_and_validate_synthetic_data(main_class_counts)
            if synthetic_df is not None and len(synthetic_df) > 0:
                print(f"[Loader] Loaded {len(synthetic_df)} synthetic rows from {self.synthetic_path}")
                # Merge synthetic data with main dataset
                df_all = pd.concat([df_all, synthetic_df], ignore_index=True)
                print(f"[Loader] Total rows after adding synthetic data: {len(df_all)}")
        
        return df_all.reset_index(drop=True)

    def _load_and_validate_synthetic_data(self, main_class_counts: Dict[str, int]) -> pd.DataFrame:
        """Load and validate synthetic dataset and optionally generate missing samples.

        Missing counts are computed against the COMBINED distribution of
        main dataset + provided synthetic dataset, to avoid generating for classes
        that are already sufficiently represented in the main data.
        """
        import os
        import numpy as np
        
        if not os.path.exists(self.synthetic_path):
            print(f"[Synthetic] Warning: Synthetic dataset file not found: {self.synthetic_path}")
            return pd.DataFrame()
        
        try:
            # Load synthetic dataset
            if self.synthetic_path.endswith('.xlsx'):
                synthetic_df = pd.read_excel(self.synthetic_path)
            elif self.synthetic_path.endswith('.csv'):
                synthetic_df = pd.read_csv(self.synthetic_path)
            else:
                print(f"[Synthetic] Warning: Unsupported file format: {self.synthetic_path}")
                return pd.DataFrame()
            
            print(f"[Synthetic] Loaded synthetic dataset with {len(synthetic_df)} rows")
            
            # Validate required columns exist
            missing_cols = []
            required_cols = ['ev_id', 'narr_accf'] + self.hfacs_categories
            for col in required_cols:
                if col not in synthetic_df.columns:
                    missing_cols.append(col)
            
            if missing_cols:
                print(f"[Synthetic] Warning: Missing required columns in synthetic dataset: {missing_cols}")
                # Add missing columns with NaN values
                for col in missing_cols:
                    synthetic_df[col] = np.nan
            
            # Validate class distribution
            synthetic_class_counts = self._compute_class_distribution(synthetic_df)
            print("[Synthetic] Class distribution in provided synthetic dataset:")
            for code, count in synthetic_class_counts.items():
                print(f"  {code}: {count}")
            
            # Check if we need to generate additional synthetic data
            per_class_target = 100  # Target for balanced train set
            missing_synthetic = {}
            
            for code in self.hfacs_categories:
                main_count = int(main_class_counts.get(code, 0))
                synth_count = int(synthetic_class_counts.get(code, 0))
                combined_total = main_count + synth_count
                # Only request additional synthetics if combined < target
                if combined_total < per_class_target:
                    missing_synthetic[code] = per_class_target - combined_total
                    print(f"[Synthetic] Class {code}: main={main_count}, synth={synth_count}, need {missing_synthetic[code]} more synthetic samples to reach {per_class_target}")
            
            # Generate missing synthetic data if needed
            if missing_synthetic:
                print("[Synthetic] Generating missing synthetic data...")
                additional_synthetic = self._generate_missing_synthetic_data(synthetic_df, missing_synthetic)
                if len(additional_synthetic) > 0:
                    synthetic_df = pd.concat([synthetic_df, additional_synthetic], ignore_index=True)
                    print(f"[Synthetic] Added {len(additional_synthetic)} additional synthetic samples")
            
            return synthetic_df
            
        except Exception as e:
            print(f"[Synthetic] Error loading synthetic dataset: {e}")
            return pd.DataFrame()

    def _generate_missing_synthetic_data(self, existing_synthetic_df: pd.DataFrame, missing_counts: dict[str, int]) -> pd.DataFrame:
        """Generate additional synthetic data for classes that don't meet requirements."""
        rng = np.random.RandomState(42)
        all_synthetic_parts = []
        
        for code, num_needed in missing_counts.items():
            if num_needed <= 0:
                continue
                
            print(f"[Synthetic] Generating {num_needed} additional samples for class {code}...")
            
            # Use existing synthetic data as examples if available
            existing_examples = existing_synthetic_df[existing_synthetic_df[code].notna()]
            examples = []
            if len(existing_examples) > 0 and 'narr_accf' in existing_examples.columns:
                examples = existing_examples['narr_accf'].tolist()[:5]  # Use up to 5 examples
            
            # Create a temporary dataframe for the existing synthesis method
            temp_df = pd.DataFrame([{'narr_accf': ex} for ex in examples] if examples else [])
            
            try:
                synth_df = self._generate_synthetic_samples_for_class(temp_df, code, num_needed, rng)
                if len(synth_df) > 0:
                    all_synthetic_parts.append(synth_df)
            except Exception as e:
                print(f"[Synthetic] Error generating additional data for {code}: {e}")
                continue
        
        if all_synthetic_parts:
            return pd.concat(all_synthetic_parts, ignore_index=True)
        else:
            return pd.DataFrame()

    def _compute_class_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Count non-null entries per HFACS class column. Returns { code: count }."""
        counts: Dict[str, int] = {}
        for code in self.hfacs_categories:
            counts[code] = int(df[code].notna().sum()) if code in df.columns else 0
        return counts

    def _generate_synthetic_samples_for_class(
        self,
        df: pd.DataFrame,
        code: str,
        num_needed: int,
        rng: np.random.RandomState,
    ) -> pd.DataFrame:
        """Generate up to num_needed synthetic rows for a specific HFACS class using GPT-5
        """
        if num_needed <= 0:
            return pd.DataFrame(columns=df.columns)

        # Ensure OpenAI client is available
        if client is None:
            raise RuntimeError("OpenAI client is not available. Set OPENAI_API_KEY and retry.")

        # Try to pick 10 few-shot examples from real rows for this class
        real_rows = df[df[code].notna()] if code in df.columns else pd.DataFrame()
        few_shot_examples = []
        if len(real_rows) > 0 and 'narr_accf' in real_rows.columns:
            # Sample up to 10 examples, or all available if less than 10
            n_samples = min(10, len(real_rows))
            sampled_rows = real_rows.sample(n=n_samples, random_state=42)
            few_shot_examples = sampled_rows['narr_accf'].tolist()

        synthetic_records = []
        num_to_make = int(min(num_needed, self.max_synthetics_per_class))

        # Explanatory text for the HFACS code for better guidance
        hfacs_hints = {
            'AE100': 'skill-based handling mistake during a critical phase of flight',
            'AE200': 'poor decision selection given available alternatives',
            'AD000': 'intentional deviation from established procedures',
            'PC100': 'reduced attention and situational awareness',
            'PC200': 'stress and overconfidence impacting judgment',
            'PC300': 'pilot fatigue and mild physical impairment',
            'PE100': 'adverse weather and limited visibility',
            'PE200': 'automation mode confusion and interface design issues',
            'PP100': 'inadequate preflight planning and briefing',
            'PT100': 'insufficient training and lack of proficiency',
        }
        hint = hfacs_hints.get(code, 'human factors issue')

        print(f"[Synthesis] Generating up to {num_to_make} narratives for {code}...")
        for i in range(num_to_make):
            try:
                sys_prompt = (
                    "You are a data generator for aviation safety research. "
                    "Create a realistic accident narrative (≈80–180 words) aligned with the given HFACS class. "
                    "Avoid sensationalism; keep it factual, concise, and plausible. "
                    "Do not mention HFACS explicitly; describe the situation naturally."
                )
                # Format few-shot examples for the prompt
                examples_text = "N/A"
                if few_shot_examples:
                    examples_list = [f"Example {i+1}: {example.strip()}" for i, example in enumerate(few_shot_examples)]
                    examples_text = "\n\n".join(examples_list)
                
                user_prompt = (
                    f"HFACS Code: {code}.\n"
                    f"Guidance: The scenario should reflect: {hint}.\n"
                    f"If examples are provided below, emulate their style, not their content.\n\n"
                    f"Examples (style only):\n{examples_text}\n\n"
                    "Generate only the narrative text."
                )

                if i % 5 == 0:
                    print(f"[Synthesis:{code}] Request {i+1}/{num_to_make} ...")
                # Price guard per call
                _check_and_log_cost(self.openai_model, sys_prompt + "\n" + user_prompt, est_output_tokens=220, action_label=f"Synthesis:{code}")
                resp = client.responses.create(
                    model=self.openai_model,
                    input=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                narrative_text = (getattr(resp, 'output_text', None) or "").strip()
                if not narrative_text:
                    raise RuntimeError("OpenAI returned empty narrative content.")

            except Exception as e:
                print(f"[Synthesis:{code}] Error on item {i+1}: {e}")
                raise RuntimeError(f"Synthetic generation via OpenAI failed for {code}: {e}") from e

            record = {
                'ev_id': f"SYNTH-{code}-{i}-{rng.randint(10**6)}",
                'narr_accf': narrative_text,
                code: code,  # Store the code string
            }
            synthetic_records.append(record)

        # Ensure columns align with df
        synth_df = pd.DataFrame(synthetic_records)
        for col in df.columns:
            if col not in synth_df.columns:
                synth_df[col] = np.nan
        # Keep column order consistent
        synth_df = synth_df[df.columns]
        return synth_df



    def preprocess_data(self, df: pd.DataFrame) -> list[Dict]:
        """Convert dataframe to list of dicts with chat prompts."""
        global system_prompt

        # No augmentation here - balancing will be done during train/test split
        processed_data: list[Dict] = []

        for _, row in df.iterrows():
            # Extract HFACS codes that are not null
            hfacs_codes: list[str] = []
            for cat in self.hfacs_categories:
                if cat in df.columns and pd.notna(row.get(cat, None)) and row.get(cat, None) is not None:
                    hfacs_codes.append(cat)

            user_prompt = f"""
Analyze this accident narrative:
{row.get('narr_accf', '')}"""

            entry = {
                'prompt': [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                'answer': ' '.join(sorted(hfacs_codes)),
                'ev_id': row.get('ev_id', ''),
                'narrative': row.get('narr_accf', '')
            }

            processed_data.append(entry)

        return processed_data

    def split_data(self, data: list[Dict], test_size: float = 0.2, save_synthetic_path: str = None) -> Tuple[Dataset, Dataset]:
        """Split data into train and test sets with specific requirements.
        """
        rng = np.random.RandomState(42)
        
        print("[Split] Creating test set...")
        
        # Identify REAL samples (exclude synthetic rows from test set)
        real_indices = [i for i, d in enumerate(data) if not str(d.get('ev_id', '')).startswith('SYNTH-')]
        
        n_base = min(760, len(real_indices))
        if n_base > 0:
            base_indices = rng.choice(real_indices, size=n_base, replace=False).tolist()
        else:
            base_indices = []
        test_selected = set(base_indices)

        underrep = ["AD000", "PE200", "PC200"]
        extra_pool_all = [i for i in real_indices if i not in test_selected]
        extra_pool = [i for i in extra_pool_all if any(code in data[i].get('answer', '') for code in underrep)]
        n_extra = min(75, len(extra_pool))
        if n_extra > 0:
            extra_indices = rng.choice(len(extra_pool), size=n_extra, replace=False)
            for ei in extra_indices:
                test_selected.add(extra_pool[ei])

        test_indices = sorted(list(test_selected))
        test_data = [data[i] for i in test_indices]
        
        # Now create the train set with exactly 100 rows per class
        print("[Split] Creating balanced train set (100 per class)...")
        
        # Group remaining data by class
        remaining_indices = [i for i in range(len(data)) if i not in test_selected]
        class_pools = {code: [] for code in self.hfacs_categories}
        
        for idx in remaining_indices:
            sample = data[idx]
            answer = sample.get('answer', '')
            # Add this sample to all classes it belongs to
            for code in self.hfacs_categories:
                if code in answer:
                    class_pools[code].append(idx)
        
        # Build balanced train set with synthesis if needed
        train_selected = set()
        per_class_target = 100
        synthesis_needed = {}
        
        for code in self.hfacs_categories:
            available = class_pools[code]
            available_count = len(available)
            
            if available_count >= per_class_target:
                # Sample 100 from available
                selected = rng.choice(available, size=per_class_target, replace=False)
                train_selected.update(selected)
                print(f"[Split] Class {code}: selected {per_class_target} from {available_count} available")
            else:
                # Take all available and mark for synthesis
                train_selected.update(available)
                synthesis_needed[code] = per_class_target - available_count
                print(f"[Split] Class {code}: took all {available_count}, need {synthesis_needed[code]} synthetic")
        
        # Create train data from selected indices
        train_data = [data[i] for i in sorted(train_selected)]
        
        # Generate synthetic data if needed
        if synthesis_needed:
            print("[Split] Generating synthetic data for train set...")
            synthetic_data = self._generate_synthetic_data_for_train(data, synthesis_needed, rng, save_synthetic_path)
            train_data.extend(synthetic_data)
        
        print(f"[Split] Final sizes - Train: {len(train_data)}, Test: {len(test_data)}")
        print(f"[Split] Test breakdown: {n_base} random + {len(test_data)-n_base} underrepresented")
        
        # Verify train set balance
        train_class_counts = {code: 0 for code in self.hfacs_categories}
        for sample in train_data:
            answer = sample.get('answer', '')
            for code in self.hfacs_categories:
                if code in answer:
                    train_class_counts[code] += 1
        
        print("[Split] Train set class distribution:")
        for code, count in train_class_counts.items():
            print(f"  {code}: {count}")

        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)
        return train_dataset, test_dataset

    def _generate_synthetic_data_for_train(self, data: list[Dict], synthesis_needed: dict[str, int], rng: np.random.RandomState, save_path: str = None) -> list[Dict]:
        """Generate synthetic data for training set to reach 100 per class."""
        synthetic_data = []
        synthetic_df_rows = []  # For saving to Excel
        

        
        for code, num_needed in synthesis_needed.items():
            if num_needed <= 0:
                continue
                
            print(f"[Synthesis] Generating {num_needed} samples for class {code}...")
            
            # Find examples of this class from the original data for few-shot
            examples = []
            for sample in data:
                if code in sample.get('answer', '') and 'narrative' in sample:
                    examples.append(sample['narrative'])
                    if len(examples) >= 10:  # Limit to 10 examples
                        break
            
            # Generate synthetic samples using the existing method
            try:
                # Create a temporary dataframe for the existing synthesis method
                temp_df = pd.DataFrame([{'narr_accf': ex} for ex in examples[:5]] if examples else [])
                synth_df = self._generate_synthetic_samples_for_class(temp_df, code, num_needed, rng)
                
                # Convert back to the expected format
                for _, row in synth_df.iterrows():
                    if pd.notna(row.get('narr_accf')):
                        synthetic_entry = {
                            'prompt': [
                                {'role': 'system', 'content': system_prompt},
                                {'role': 'user', 'content': f"""
Analyze this accident narrative:
{row['narr_accf']}"""}
                            ],
                            'answer': code,  # Single class for synthetic data
                            'ev_id': row.get('ev_id', f"SYNTH-{code}-{len(synthetic_data)}"),
                            'narrative': row['narr_accf']
                        }
                        synthetic_data.append(synthetic_entry)
                        
                        # Also prepare row for Excel saving
                        excel_row = {
                            'ev_id': synthetic_entry['ev_id'],
                            'narr_accf': row['narr_accf'],
                            code: code  # Use actual class code (e.g., AE100, AE200, etc.)
                        }
                        # Initialize all other HFACS categories to NaN
                        for other_code in self.hfacs_categories:
                            if other_code != code:
                                excel_row[other_code] = np.nan
                        synthetic_df_rows.append(excel_row)
                        
            except Exception as e:
                print(f"[Synthesis] Error generating data for {code}: {e}")
                # Continue with other classes even if one fails
                continue
        
        print(f"[Synthesis] Generated {len(synthetic_data)} total synthetic samples")
        
        # Save synthetic data to Excel (always save when synthetic data is generated)
        if len(synthetic_df_rows) > 0:
            try:
                # Use provided path or generate default timestamp-based filename
                if save_path:
                    final_save_path = save_path
                else:
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    final_save_path = f"synth_GAHFACS_{timestamp}.xlsx"
                
                synthetic_df = pd.DataFrame(synthetic_df_rows)
                synthetic_df.to_excel(final_save_path, index=False)
                print(f"[Synthesis] Saved {len(synthetic_df)} synthetic rows to {final_save_path}")
            except Exception as e:
                print(f"[Synthesis] Warning: failed to save synthetic data: {e}")
        
        return synthetic_data



# Smoke test all LLMs before data processing (to avoid wasting time on synthetic generation)
print("[Stage] Running pre-processing LLM smoke tests...")

def run_comprehensive_smoke_tests():
    """Run smoke tests for all LLMs before data processing"""
    print("\n[Stage] Comprehensive LLM smoke tests before data processing...")
    failures = []

    # Test OpenAI models (used for synthetic generation and evaluation)
    api_tests = [
        ("gpt-5-nano", client, [{"role": "user", "content": "ping"}]),
        ("gpt-5-mini", client, [{"role": "user", "content": "ping"}]),
        ("gpt-5", client, [{"role": "user", "content": "ping"}]),
    ]
    for model_name, api_client, messages in api_tests:
        try:
            if api_client is None:
                print(f"- {model_name} SKIPPED: Client not initialized")
                continue
            resp = api_client.responses.create(
                model=model_name,
                input=messages,
            )
            content = getattr(resp, 'output_text', '')
            ok = bool(content and isinstance(content, str))
            print(f"- {model_name} OK: {ok}")
            if not ok:
                failures.append(model_name)
        except Exception as e:
            print(f"- {model_name} FAILED: {e}")
            failures.append(model_name)

    # Test Gemini
    try:
        if genai_client is None:
            print("- gemini-2.5-flash SKIPPED: Client not initialized")
        else:
            resp = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents="ping",
            )
            content = getattr(resp, 'text', None) or getattr(resp, 'content', None) or ""
            ok = bool(content)
            print(f"- gemini-2.5-flash OK: {ok}")
            if not ok:
                failures.append("gemini-2.5-flash")
    except Exception as e:
        print(f"- gemini-2.5-flash FAILED: {e}")
        failures.append("gemini-2.5-flash")

    # Test OSS models (with robust error handling and proper unloading)
    ollama_models = [
        "gpt-oss:20b",
        "gemma3:12b", 
        "deepseek-r1:8b",
        "deepseek-r1:14b",
        "gemma3n:e4b",
        "qwen3:8b",
    ]
    if ollama_chat is None:
        print("- Ollama not available; all OSS models will be skipped during evaluation")
    else:
        import subprocess
        for m in ollama_models:
            try:
                resp = ollama_chat(model=m, messages=[{"role": "user", "content": "ping"}])
                msg = resp["message"]["content"] if isinstance(resp, dict) else getattr(resp, 'message', None)
                text = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
                ok = bool(text)
                print(f"- {m} OK: {ok}")
                if not ok:
                    print(f"  -> {m} will be skipped during evaluation")
            except Exception as e:
                print(f"- {m} FAILED: {e}")
                print(f"  -> {m} will be skipped during evaluation")
            finally:
                # Always unload the model to free VRAM for next test
                try:
                    subprocess.run(["ollama", "stop", m], capture_output=True, timeout=10)
                    print(f"  -> Unloaded {m} from VRAM")
                except Exception as unload_e:
                    print(f"  -> Warning: Could not unload {m}: {unload_e}")

    # Only fail if critical models for synthetic generation are unavailable
    critical_failures = [f for f in failures if f in ["gpt-5", "gpt-5-nano"]]
    if critical_failures:
        print(f"\nCRITICAL: Essential models for synthetic generation failed: {', '.join(critical_failures)}")
        return False
    
    if failures:
        print(f"\nWARNING: Some models failed but will be skipped: {', '.join(failures)}")
    
    print("✓ Smoke tests completed - proceeding with data processing")
    return True

# Run comprehensive smoke tests
if not run_comprehensive_smoke_tests():
    print("ABORTING: Critical LLM failures detected")
    sys.exit(1)

# Load GAHFACS dataset
print("\n[Stage] Initializing GAHFACSDataLoader...")
gahfacs_loader = GAHFACSDataLoader(file_path=args.gahfacs_path, sample_size=1000, synthetic_path=args.load_synthetic_data)

print("[Stage] Loading raw data from Excel...")
df = gahfacs_loader.load_data()
print(f"[Info] Loaded {len(df)} rows.")

print("[Stage] Preprocessing data (this may augment and call OpenAI for synthetics)...")
processed_data = gahfacs_loader.preprocess_data(df)
print(f"[Info] Preprocessing complete. Samples: {len(processed_data)}")

print("[Stage] Splitting into train/test datasets...")
train_dataset, test_dataset = gahfacs_loader.split_data(processed_data, save_synthetic_path=args.save_synthetic_data)
print(f"[Info] Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

# Use training dataset for GRPO
dataset = train_dataset
print("dataset is loaded!!")
from pprint import pprint


# Reward functions
# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     q = prompts[0][-1]['content']
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
#     return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# def int_reward_func(completions, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


# def strict_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]

# def soft_format_reward_func(completions, **kwargs) -> list[float]:
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
#     responses = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, r) for r in responses]
#     return [0.5 if match else 0.0 for match in matches]

# def count_xml(text) -> float:
#     count = 0.0
#     if text.count("<reasoning>\n") == 1:
#         count += 0.125
#     if text.count("\n</reasoning>\n") == 1:
#         count += 0.125
#     if text.count("\n<answer>\n") == 1:
#         count += 0.125
#         count -= len(text.split("\n</answer>\n")[-1])*0.001
#     if text.count("\n</answer>") == 1:
#         count += 0.125
#         count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
#     return count

# def xmlcount_reward_func(completions, **kwargs) -> list[float]:
#     contents = [completion[0]["content"] for completion in completions]
#     return [count_xml(c) for c in contents]




hfacs_pattern = re.compile(r'(AE100|AE200|AD000|PC100|PC200|PC300|PE100|PE200|PP100|PT100)')
reasoning_pattern = re.compile(r'<reasoning>.*?</reasoning>', re.DOTALL)

# Global variables for logging
reward_logs = []
log_counter = 0

# Power monitoring variables
power_data = {'cpu_percent': [], 'gpu_utilization': [], 'gpu_memory': [], 'timestamps': []}
monitoring = False
power_thread = None

def monitor_power():
    """Monitor CPU and GPU usage during training"""
    global power_data, monitoring
    while monitoring:
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # GPU usage
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0] 
                gpu_util = gpu.load * 100
                gpu_memory = gpu.memoryUtil * 100
            else:
                gpu_util = 0
                gpu_memory = 0
            
            power_data['cpu_percent'].append(cpu_percent)
            power_data['gpu_utilization'].append(gpu_util)
            power_data['gpu_memory'].append(gpu_memory)
            power_data['timestamps'].append(datetime.now())
            
        except Exception as e:
            print(f"Power monitoring error: {e}")
        
        time.sleep(2)  # Sample every 2 seconds

def start_power_monitoring():
    """Start background power monitoring if not already running"""
    global monitoring, power_thread
    if monitoring:
        return
    monitoring = True
    power_thread = threading.Thread(target=monitor_power, daemon=True)
    power_thread.start()

def stop_power_monitoring():
    """Stop background power monitoring"""
    global monitoring
    if not monitoring:
        return
    monitoring = False
    # Give the monitor thread a moment to exit cleanly
    time.sleep(3)

def evaluate_model(model, tokenizer, test_dataset, lora_request=None, model_name="Model"):
    """Evaluate model on test dataset and return metrics"""
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print(f"{'='*50}")
    
    predictions = []
    true_labels = []
    individual_rewards = []
    
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature = 1,
        top_p = 0.95,
        max_tokens = 1024,
    )
    
    for i, sample in enumerate(test_dataset):
        if i >= 100:  # Limit to first 100 samples for faster evaluation
            break
            
        # Prepare prompt
        messages = sample['prompt']
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate response
        try:
            if isinstance(text, list):
                output = model.fast_generate(
                    text,
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )[0].outputs[0].text
            else:
                output = model.fast_generate(
                    [text],
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )[0].outputs[0].text
        except Exception as e:
            print(f"Generation error for sample {i}: {e}")
            continue
        
        # Extract predictions and true labels
        predicted_codes = normalize_hfacs_codes(extract_hfacs_codes(output))
        true_codes = normalize_hfacs_codes(sample['answer'])
        
        predictions.append(predicted_codes)
        true_labels.append(true_codes)
        
        # Calculate individual rewards for this sample
        rewards = {}
        
        # Correctness reward
        is_correct = predicted_codes == true_codes
        rewards['correctness'] = 2.0 if is_correct else 0.0
        
        # Partial match reward
        if predicted_codes and true_codes:
            correct_predictions = len(set(predicted_codes) & set(true_codes))
            total_true_categories = len(true_codes)
            if correct_predictions == 0:
                rewards['partial_match'] = 0.0
            elif correct_predictions == total_true_categories:
                rewards['partial_match'] = 0.0  # Perfect match handled by correctness
            else:
                proportion_correct = correct_predictions / total_true_categories
                rewards['partial_match'] = 0.1 + 0.9 * proportion_correct
        else:
            rewards['partial_match'] = 0.0
        
        # Reasoning format reward
        rewards['reasoning_format'] = 0.5 if has_proper_format(output) else 0.0
        
        # HFACS category reward (punishment for invalid codes)
        valid_hfacs = {'AE100', 'AE200', 'AD000', 'PC100', 'PC200', 'PC300', 
                       'PE100', 'PE200', 'PP100', 'PT100'}
        potential_codes = re.findall(r'[A-Z]{2,3}\d{3}', extract_hfacs_codes(output))
        invalid_codes = [code for code in potential_codes if code not in valid_hfacs]
        rewards['hfacs_category'] = -0.25 * len(invalid_codes) if invalid_codes else 0.0
        
        # Total reward
        rewards['total'] = sum(rewards.values())
        
        individual_rewards.append(rewards)
        
        if i % 20 == 0:
            print(f"Processed {i+1} samples...")
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, true_labels, individual_rewards, model_name)
    
    return metrics, predictions, true_labels, individual_rewards

def calculate_metrics(predictions, true_labels, individual_rewards, model_name):
    """Calculate accuracy, precision, recall, F1 and reward statistics"""
    
    # Convert to binary multilabel format for sklearn metrics
    all_codes = set()
    for codes in predictions + true_labels:
        all_codes.update(codes)
    all_codes = sorted(list(all_codes))
    
    # Create binary matrices
    y_true_binary = []
    y_pred_binary = []
    
    for i in range(len(predictions)):
        true_vector = [1 if code in true_labels[i] else 0 for code in all_codes]
        pred_vector = [1 if code in predictions[i] else 0 for code in all_codes]
        y_true_binary.append(true_vector)
        y_pred_binary.append(pred_vector)
    
    y_true_binary = np.array(y_true_binary)
    y_pred_binary = np.array(y_pred_binary)
    
    # Calculate exact match accuracy (all categories must match)
    exact_match_accuracy = sum(1 for i in range(len(predictions)) 
                              if set(predictions[i]) == set(true_labels[i])) / len(predictions)
    
    # Calculate partial match accuracy (at least one category matches)
    partial_match_accuracy = sum(1 for i in range(len(predictions)) 
                                if set(predictions[i]) & set(true_labels[i])) / len(predictions)
    
    # Calculate macro-averaged metrics
    precision_macro = precision_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    recall_macro = recall_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    
    # Calculate micro-averaged metrics
    precision_micro = precision_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    recall_micro = recall_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    f1_micro = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    
    # Reward statistics
    reward_stats = {}
    for reward_type in ['correctness', 'partial_match', 'reasoning_format', 'hfacs_category', 'total']:
        values = [r[reward_type] for r in individual_rewards]
        reward_stats[reward_type] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values)
        }
    
    metrics = {
        'model_name': model_name,
        'exact_match_accuracy': exact_match_accuracy,
        'partial_match_accuracy': partial_match_accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'reward_stats': reward_stats,
        'sample_count': len(predictions)
    }
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Sample Count: {metrics['sample_count']}")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")
    print(f"Partial Match Accuracy: {partial_match_accuracy:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    print(f"Micro F1: {f1_micro:.4f}")
    print(f"Micro Precision: {precision_micro:.4f}")
    print(f"Micro Recall: {recall_micro:.4f}")
    

    
    return metrics

def evaluate_external_model(model_name: str, test_dataset, system_prompt: str) -> Dict:
    """Evaluate external models (GPT-5-mini, Gemini-2.5-flash) on test dataset"""
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print(f"{'='*50}")
    
    predictions = []
    true_labels = []
    individual_rewards = []
    
    # Check which model and client to use
    if "gpt" in model_name.lower():
        if client is None:
            raise RuntimeError(f"OpenAI client not available for {model_name}")
        api_client = client
        use_openai = True
    elif "gemini" in model_name.lower():
        if genai_client is None:
            raise RuntimeError(f"Google GenAI client not available for {model_name}")
        api_client = genai_client
        use_openai = False
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    for i, sample in enumerate(test_dataset):
        if i >= 100:  # Limit to first 100 samples for faster evaluation
            break
            
        # Prepare prompt - use the exact same format as GRPO model
        messages = sample['prompt']
        
        # Extract user prompt content
        user_content = messages[1]['content'] if len(messages) > 1 else ""
        
        try:
            if use_openai:
                # GPT-5-mini via OpenAI Responses API
                _check_and_log_cost(model_name, system_prompt + "\n" + user_content, est_output_tokens=1024, action_label=f"ExternalEval:{model_name}")
                response = api_client.responses.create(
                    model=model_name,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                )
                output = getattr(response, 'output_text', '')
                
                # Parse output using existing function
                predicted_codes = normalize_hfacs_codes(extract_hfacs_codes(output))
                
            else:
                # Gemini-2.5-flash via Google GenAI API with structured output
                response = api_client.models.generate_content(
                    model=model_name,
                    contents=f"{system_prompt}\n\n{user_content}",
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": HfacsResponse,
                    }
                )
                
                # Extract codes from structured response
                parsed_response = response.parsed
                if parsed_response and hasattr(parsed_response, 'hfacs_codes'):
                    predicted_codes = sorted([code for code in parsed_response.hfacs_codes if code])
                else:
                    predicted_codes = []
                
        except Exception as e:
            print(f"API call failed for sample {i} with {model_name}: {e}")
            predicted_codes = []  # Empty prediction on failure
            
        # Get true labels
        true_codes = normalize_hfacs_codes(sample['answer'])
        
        predictions.append(predicted_codes)
        true_labels.append(true_codes)
        
        # Calculate individual rewards for this sample (same as local model)
        rewards = {}
        
        # Correctness reward
        is_correct = predicted_codes == true_codes
        rewards['correctness'] = 2.0 if is_correct else 0.0
        
        # Partial match reward
        if predicted_codes and true_codes:
            correct_predictions = len(set(predicted_codes) & set(true_codes))
            total_true_categories = len(true_codes)
            if correct_predictions == 0:
                rewards['partial_match'] = 0.0
            elif correct_predictions == total_true_categories:
                rewards['partial_match'] = 0.0  # Perfect match handled by correctness
            else:
                proportion_correct = correct_predictions / total_true_categories
                rewards['partial_match'] = 0.1 + 0.9 * proportion_correct
        else:
            rewards['partial_match'] = 0.0
        
        # Reasoning format reward (always 0 for external models as we don't control format)
        rewards['reasoning_format'] = 0.0
        
        # HFACS category reward (punishment for invalid codes)
        valid_hfacs = {'AE100', 'AE200', 'AD000', 'PC100', 'PC200', 'PC300', 
                       'PE100', 'PE200', 'PP100', 'PT100'}
        invalid_codes = [code for code in predicted_codes if code not in valid_hfacs]
        rewards['hfacs_category'] = -0.25 * len(invalid_codes) if invalid_codes else 0.0
        
        # Total reward
        rewards['total'] = sum(rewards.values())
        
        individual_rewards.append(rewards)
        
        if i % 20 == 0:
            print(f"Processed {i+1} samples...")
    
    # Calculate metrics using existing function
    metrics = calculate_metrics(predictions, true_labels, individual_rewards, model_name)
    
    return metrics

def evaluate_oss_model(model_name: str, test_dataset, system_prompt: str) -> Dict:
    """Evaluate OSS models via Ollama on test dataset"""
    print(f"\n{'='*50}")
    print(f"Evaluating {model_name}")
    print(f"{'='*50}")
    
    predictions = []
    true_labels = []
    individual_rewards = []
    
    if ollama_chat is None:
        raise RuntimeError(f"Ollama client not available for {model_name}")
    
    for i, sample in enumerate(test_dataset):
        if i >= 100:  # Limit to first 100 samples for faster evaluation
            break
            
        # Prepare prompt - use the exact same format as other models
        messages = sample['prompt']
        
        # Extract user prompt content
        user_content = messages[1]['content'] if len(messages) > 1 else ""
        
        try:
            # Combine system and user prompts for Ollama
            full_prompt = f"{system_prompt}\n\n{user_content}"
            
            response = ollama_chat(model=model_name, messages=[
                {"role": "user", "content": full_prompt}
            ])
            
            # Extract response text
            if isinstance(response, dict):
                output = response.get("message", {}).get("content", "")
            else:
                output = getattr(response, 'message', {}).get('content', '') if hasattr(response, 'message') else str(response)
            
            # Parse output using existing function
            predicted_codes = normalize_hfacs_codes(extract_hfacs_codes(output))
            
        except Exception as e:
            print(f"Ollama call failed for sample {i} with {model_name}: {e}")
            predicted_codes = []  # Empty prediction on failure
            
        # Get true labels
        true_codes = normalize_hfacs_codes(sample['answer'])
        
        predictions.append(predicted_codes)
        true_labels.append(true_codes)
        
        # Calculate individual rewards for this sample (same as other models)
        rewards = {}
        
        # Correctness reward
        is_correct = predicted_codes == true_codes
        rewards['correctness'] = 2.0 if is_correct else 0.0
        
        # Partial match reward
        if predicted_codes and true_codes:
            correct_predictions = len(set(predicted_codes) & set(true_codes))
            total_true_categories = len(true_codes)
            if correct_predictions == 0:
                rewards['partial_match'] = 0.0
            elif correct_predictions == total_true_categories:
                rewards['partial_match'] = 0.0  # Perfect match handled by correctness
            else:
                proportion_correct = correct_predictions / total_true_categories
                rewards['partial_match'] = 0.1 + 0.9 * proportion_correct
        else:
            rewards['partial_match'] = 0.0
        
        # Reasoning format reward (always 0 for external models)
        rewards['reasoning_format'] = 0.0
        
        # HFACS category reward (punishment for invalid codes)
        valid_hfacs = {'AE100', 'AE200', 'AD000', 'PC100', 'PC200', 'PC300', 
                       'PE100', 'PE200', 'PP100', 'PT100'}
        invalid_codes = [code for code in predicted_codes if code not in valid_hfacs]
        rewards['hfacs_category'] = -0.25 * len(invalid_codes) if invalid_codes else 0.0
        
        # Total reward
        rewards['total'] = sum(rewards.values())
        
        individual_rewards.append(rewards)
        
        if i % 20 == 0:
            print(f"Processed {i+1} samples...")
    
    # Calculate metrics using existing function
    metrics = calculate_metrics(predictions, true_labels, individual_rewards, model_name)
    
    # Unload the model to free VRAM
    try:
        import subprocess
        subprocess.run(["ollama", "stop", model_name], capture_output=True, timeout=10)
        print(f"  -> Unloaded {model_name} from VRAM after evaluation")
    except Exception as unload_e:
        print(f"  -> Warning: Could not unload {model_name} after evaluation: {unload_e}")
    
    return metrics



def extract_hfacs_codes(text: str) -> str:
    """Extract everything after </reasoning> tag from model output"""
    reasoning_end = text.rfind('</reasoning>')
    if reasoning_end == -1:
        return ""  # No reasoning tag found, return empty
    
    # Get everything after </reasoning> tag and strip whitespace
    codes_section = text[reasoning_end + len('</reasoning>'):].strip()
    return codes_section

def normalize_hfacs_codes(codes_str: str) -> list:
    """Convert space-separated HFACS codes to sorted list for comparison"""
    if not codes_str or not str(codes_str).strip():
        return []
    codes = str(codes_str).strip().split()
    # Extract only valid HFACS codes and sort them
    valid_codes = [code for code in codes if hfacs_pattern.match(code)]
    return sorted(valid_codes)



def save_reward_logs_to_excel():
    """Persist reward logs in multiple formats to avoid truncation/data loss.

    - JSONL: Full-fidelity, no truncation, easiest to parse later
    - Parquet: Columnar, efficient, preserves types (if pyarrow is available)
    - Excel: Human-readable; strings sanitized to avoid Excel cell limits
    """
    global reward_logs, log_counter
    if not reward_logs:
        return
    
    # Calculate total rewards for each entry
    for log_entry in reward_logs:
        log_entry['total_reward'] = (
            log_entry['correctness_reward'] + 
            log_entry['reasoning_format_reward'] + 
            log_entry['hfacs_category_reward'] + 
            log_entry['partial_match_reward'] +
            log_entry.get('gpt_reasoning_reward', 0)
        )

    from datetime import datetime
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"reward_logs_{timestamp}"

    # 1) JSONL (no truncation)
    jsonl_path = f"{base}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in reward_logs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Reward logs saved (JSONL): {jsonl_path}")

    # 2) Parquet (optional)
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        table = pa.Table.from_pylist(reward_logs)
        parquet_path = f"{base}.parquet"
        pq.write_table(table, parquet_path)
        print(f"Reward logs saved (Parquet): {parquet_path}")
    except Exception as e:
        print(f"Parquet save skipped: {e}")

    # 3) Excel with sanitized strings for very long cells
    def _sanitize_for_excel(value):
        if isinstance(value, str) and len(value) > 32767:
            return value[:32760] + "... [TRUNCATED]"
        return value

    try:
        cleaned_logs = []
        for log_entry in reward_logs:
            cleaned_entry = {k: _sanitize_for_excel(v) for k, v in log_entry.items()}
            cleaned_logs.append(cleaned_entry)
        df = pd.DataFrame(cleaned_logs)
        xlsx_path = f"{base}.xlsx"
        df.to_excel(xlsx_path, index=False)
        print(f"Reward logs saved (Excel): {xlsx_path}")
    except Exception as e:
        print(f"Excel save failed: {e}")
    
    # Clear logs after saving
    reward_logs = []
    log_counter = 0

def has_proper_format(text: str) -> bool:
    """Check if output has proper reasoning format and HFACS codes after"""
    has_reasoning = bool(reasoning_pattern.search(text))
    if not has_reasoning:
        return False
    
    # Check if there's content after </reasoning> tag
    content_after_reasoning = extract_hfacs_codes(text)
    has_codes_after = bool(content_after_reasoning.strip())
    
    return has_reasoning and has_codes_after

def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward +2.0 for correct classification, 0 otherwise"""
    global reward_logs, log_counter
    
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_hfacs_codes(r) for r in responses]
    
    rewards = []
    for i, (r, a) in enumerate(zip(extracted_responses, answer)):
        # Normalize codes for comparison (order-independent)
        predicted_codes = normalize_hfacs_codes(r)
        true_codes = normalize_hfacs_codes(a)
        
        # Check if sets are equal (order-independent comparison)
        is_correct = predicted_codes == true_codes
        reward = 2.0 if is_correct else 0.0
        rewards.append(reward)
        
        # Log the details
        log_entry = {
            'log_id': log_counter,
            'question': q,
            'true_answer': a,
            'true_codes_array': str(true_codes),
            'model_response': responses[i],
            'extracted_response': r,
            'predicted_codes_array': str(predicted_codes),
            'correctness_reward': reward,
            'reasoning_format_reward': 0,  # Will be filled by other functions
            'hfacs_category_reward': 0,
            'partial_match_reward': 0,
            'gpt_reasoning_reward': 0,
            'total_reward': 0  # Will be calculated later
        }
        reward_logs.append(log_entry)
        log_counter += 1
    
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return rewards

def reasoning_format_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward +0.25 for proper reasoning tag format (reduced for stability)"""
    global reward_logs
    
    rewards = []
    
    for i, completion in enumerate(completions):
        response = completion[0]['content']
        
        if has_proper_format(response):
            reward = 0.25  # Reduced reward for numerical stability
        else:
            reward = 0.0  # No penalty, just no reward
        
        rewards.append(reward)
        
        # Update the corresponding log entry
        if len(reward_logs) > i:
            reward_logs[-(len(completions)-i)]['reasoning_format_reward'] = reward
    
    return rewards

def hfacs_category_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Punish -0.25 for outputting invalid HFACS categories, 0 for valid only"""
    global reward_logs
    
    # Valid HFACS categories
    valid_hfacs = {'AE100', 'AE200', 'AD000', 'PC100', 'PC200', 'PC300', 
                   'PE100', 'PE200', 'PP100', 'PT100'}
    
    rewards = []
    
    for i, completion in enumerate(completions):
        response = completion[0]['content']
        predicted_output = extract_hfacs_codes(response)
        
        # Extract all tokens that look like HFACS codes (pattern: 2-3 letters + 3 digits)
        import re
        potential_codes = re.findall(r'[A-Z]{2,3}\d{3}', predicted_output)
        
        # Check for invalid HFACS codes
        invalid_codes = []
        valid_codes_found = []
        
        for code in potential_codes:
            if code in valid_hfacs:
                valid_codes_found.append(code)
            else:
                invalid_codes.append(code)
        
        # Punishment logic: -0.25 for each invalid code found
        if invalid_codes:
            reward = -0.25 * len(invalid_codes)  # Punish for each invalid code
        else:
            reward = 0.0  # No reward or punishment for valid codes only
        
        rewards.append(reward)
        
        # Update the corresponding log entry
        if len(reward_logs) > i:
            reward_logs[-(len(completions)-i)]['hfacs_category_reward'] = reward
    
    return rewards

def partial_match_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward based on partial matches - scales from 0.1 to 1.0 based on how many categories are correctly predicted"""
    global reward_logs
    
    rewards = []
    
    for i, (completion, true_answer) in enumerate(zip(completions, answer)):
        response = completion[0]['content']
        predicted_output = extract_hfacs_codes(response)
        
        # Normalize codes for comparison (order-independent)
        predicted_codes = set(normalize_hfacs_codes(predicted_output))
        true_codes = set(normalize_hfacs_codes(true_answer))
        
        if not predicted_codes or not true_codes:
            reward = 0.0  # No prediction or no ground truth
            rewards.append(reward)
        else:
            # Calculate how many categories were correctly predicted
            correct_predictions = len(predicted_codes & true_codes)
            total_true_categories = len(true_codes)
            
            if correct_predictions == 0:
                reward = 0.0  # No correct predictions
            elif correct_predictions == total_true_categories:
                # Perfect match - but this should be handled by correctness_reward
                # Give partial credit to avoid double rewarding
                reward = 0.0  
            else:
                # Partial match - scale reward based on proportion of correct predictions
                # Scale from 0.1 (minimum for at least one correct) to 1.0 (maximum for most correct)
                proportion_correct = correct_predictions / total_true_categories
                # Scale: 0.1 + 0.9 * proportion gives range [0.1, 1.0]
                reward = 0.1 + 0.9 * proportion_correct
            
            rewards.append(reward)
        
        # Update the corresponding log entry
        if len(reward_logs) > i:
            reward_logs[-(len(completions)-i)]['partial_match_reward'] = reward
    
    return rewards


# def spacing_format_reward(prompts, completions, answer, **kwargs) -> list[float]:
#     """Reward +0.1 for proper spacing between multiple HFACS codes, 0 for improper spacing"""
#     global reward_logs
    
#     rewards = []
    
#     for i, (completion, true_answer) in enumerate(zip(completions, answer)):
#         response = completion[0]['content']
#         true_codes = true_answer.split()
        
#         # Only check spacing if there should be multiple codes
#         if len(true_codes) <= 1:
#             reward = 0.0  # No reward/penalty for single code cases
#             rewards.append(reward)
#         else:
#             # Extract output after reasoning tag
#             predicted_output = extract_hfacs_codes(response)
            
#             # Extract HFACS codes from the output
#             found_codes = hfacs_pattern.findall(predicted_output)
            
#             if len(found_codes) <= 1:
#                 reward = 0.0  # No multiple codes to check spacing for
#                 rewards.append(reward)
#             else:
#                 # Check for improper spacing (consecutive codes without space)
#                 has_improper_spacing = False
#                 for j in range(len(found_codes)-1):
#                     consecutive_pattern = found_codes[j] + found_codes[j+1]
#                     if consecutive_pattern in predicted_output:
#                         has_improper_spacing = True
#                         break
                
#                 if has_improper_spacing:
#                     reward = 0.0  # No reward for improper spacing
#                 else:
#                     reward = 0.1   # Small reward for proper spacing
                
#                 rewards.append(reward)
        
#         # Update the corresponding log entry
#         if len(reward_logs) > i:
#             reward_logs[-(len(completions)-i)]['spacing_format_reward'] = reward
    
#     return rewards


max_prompt_length = 1024

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 1000,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir = "outputs",
)

if args.train:
    # Start power monitoring
    print("[Stage] Starting power monitoring...")
    start_power_monitoring()

    print("[Stage] Power monitor running. Preparing GRPO trainer...")

    print("[Stage] Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            correctness_reward,
            partial_match_reward,
            hfacs_category_reward,
            reasoning_format_reward,
            gpt_reasoning_reward
        ],
        args = training_args,
        train_dataset = dataset,
    )
    
    def estimate_api_costs_before_training():
        print("\nEstimating API costs before training/evaluation...")
        # Very rough, configurable estimates of tokens per call and counts
        # Adjust as necessary or wire to actual counters if available
        est = {
            "gpt-5-nano": {"label": "Reward (during GRPO)", "num_calls": training_args.max_steps * training_args.num_generations, "input_tokens_per_call": 1500, "output_tokens_per_call": 200},
            "gpt-5-mini": {"label": "External eval + CoT/CoT+", "num_calls": 2 * min(100, len(test_dataset)) + 2 * min(100, len(test_dataset)), "input_tokens_per_call": 1800, "output_tokens_per_call": 400},
            "gpt-5": {"label": "Sampling (data augmentation)", "num_calls": 200, "input_tokens_per_call": 800, "output_tokens_per_call": 180},
            "gemini-2.5-flash": {"label": "External eval", "num_calls": min(100, len(test_dataset)), "input_tokens_per_call": 1800, "output_tokens_per_call": 400},
        }

        over_budget = []
        for model_name, cfg in est.items():
            pricing = PRICING_USD_PER_MTOK.get(model_name)
            if not pricing:
                continue
            total_input_mtok = (cfg["num_calls"] * cfg["input_tokens_per_call"]) / 1_000_000
            total_output_mtok = (cfg["num_calls"] * cfg["output_tokens_per_call"]) / 1_000_000
            cost = total_input_mtok * pricing["input"] + total_output_mtok * pricing["output"]
            print(f"- {model_name} ({cfg['label']}): calls={cfg['num_calls']}, est_cost=${cost:.2f}")
            if cost > 20.0:
                over_budget.append((model_name, cost))

        if over_budget:
            names = ", ".join([f"{n} (${c:.2f})" for n, c in over_budget])
            print(f"\nThe following estimated costs exceed $20: {names}")
            ans = input("Proceed anyway? [y/N]: ").strip().lower()
            if ans not in ("y", "yes"):
                raise SystemExit("Aborting due to cost constraints.")

    print("[Stage] Estimating API costs...")
    estimate_api_costs_before_training()

    # Smoke test all requested models (API and local) before training
    def smoke_test_models():
        print("\n[Stage] Running smoke tests for all models (API first, OSS optional)...")
        failures = []

        # API models
        api_tests = [
            ("gpt-5-nano", client, [{"role": "user", "content": "ping"}]),
            ("gpt-5-mini", client, [{"role": "user", "content": "ping"}]),
        ]
        for model_name, api_client, messages in api_tests:
            try:
                if api_client is None:
                    raise RuntimeError("Client not initialized")
                resp = api_client.responses.create(
                    model=model_name,
                    input=messages,
                )
                content = getattr(resp, 'output_text', '')
                ok = bool(content and isinstance(content, str))
                print(f"- {model_name} OK: {ok}")
                if not ok:
                    failures.append(model_name)
            except Exception as e:
                print(f"- {model_name} FAILED: {e}")
                failures.append(model_name)

        # Gemini
        try:
            if genai_client is None:
                raise RuntimeError("GenAI client not initialized")
            resp = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents="ping",
            )
            content = getattr(resp, 'text', None) or getattr(resp, 'content', None) or ""
            ok = bool(content)
            print(f"- gemini-2.5-flash OK: {ok}")
            if not ok:
                failures.append("gemini-2.5-flash")
        except Exception as e:
            print(f"- gemini-2.5-flash FAILED: {e}")
            failures.append("gemini-2.5-flash")

        # Local OLLAMA models (with robust VRAM handling and proper unloading)
        ollama_models = [
            "gpt-oss:20b",
            "gemma3:12b", 
            "deepseek-r1:8b",
            "deepseek-r1:14b",
            "gemma3n:e4b",
            "qwen3:8b",
        ]
        if ollama_chat is None:
            print("Ollama not available; skipping local model smoke tests.")
        else:
            import subprocess
            for m in ollama_models:
                try:
                    resp = ollama_chat(model=m, messages=[{"role": "user", "content": "ping"}])
                    msg = resp["message"]["content"] if isinstance(resp, dict) else getattr(resp, 'message', None)
                    text = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
                    ok = bool(text)
                    print(f"- {m} OK: {ok}")
                    if not ok:
                        print(f"  -> {m} will be skipped during evaluation")
                        # Don't add to failures - just skip during evaluation
                except Exception as e:
                    error_msg = str(e).lower()
                    if any(keyword in error_msg for keyword in ['memory', 'vram', 'cuda', 'out of memory', 'resource']):
                        print(f"- {m} FAILED (VRAM/Memory): {e}")
                        print(f"  -> {m} will be skipped during evaluation")
                    else:
                        print(f"- {m} FAILED: {e}")
                        print(f"  -> {m} will be skipped during evaluation")
                    # Don't add OSS model failures to critical failures list
                finally:
                    # Always unload the model to free VRAM for next test
                    try:
                        subprocess.run(["ollama", "stop", m], capture_output=True, timeout=10)
                        print(f"  -> Unloaded {m} from VRAM")
                    except Exception as unload_e:
                        print(f"  -> Warning: Could not unload {m}: {unload_e}")

        if failures:
            raise SystemExit(f"Smoke tests failed for: {', '.join(failures)}")

    print("[Stage] Starting smoke tests...")
    smoke_test_models()
    print("[Stage] Smoke tests complete. Starting training...")

    trainer.train()

    # Save reward logs to Excel
    save_reward_logs_to_excel()

    # Stop power monitoring
    stop_power_monitoring()

    print(f"Saving GRPO model to '{args.save_lora_folder}'...")
    model.save_lora(args.save_lora_folder)

if args.evaluate:
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)

    # Create timestamped graphs directory
    import os
    from datetime import datetime
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_dir = os.path.join("graphs", run_ts)
    os.makedirs(graph_dir, exist_ok=True)
    print(f"Using graphs directory: {graph_dir}")

    # Start power monitoring for evaluation as well
    start_power_monitoring()

    # Import visualization function
    from visualization_utils import create_visualizations

    # Evaluate model WITHOUT GRPO
    print("\nEvaluating model WITHOUT GRPO...")
    metrics_without_grpo, pred_without, true_without, rewards_without = evaluate_model(
        model, tokenizer, test_dataset, lora_request=None, model_name="Without GRPO"
    )

    # Evaluate model WITH GRPO
    print(f"\nEvaluating model WITH GRPO (loading from '{args.load_lora_folder}')...")
    try:
        lora_request = model.load_lora(args.load_lora_folder)
        metrics_with_grpo, pred_with, true_with, rewards_with = evaluate_model(
            model, tokenizer, test_dataset, lora_request=lora_request, model_name="With GRPO"
        )
    except Exception as e:
        print(f"Error loading LoRA model '{args.load_lora_folder}': {e}")
        print("Make sure you have trained a model first with --train")
        sys.exit(1)

    # Evaluate external models for comparison (pricing-guarded)
    print("\nEvaluating external models for benchmarking...")

    # Evaluate GPT-5-mini
    try:
        pricing = PRICING_USD_PER_MTOK["gpt-5-mini"]
        est_calls = min(100, len(test_dataset))
        est_cost = (est_calls * 1800 / 1_000_000) * pricing['input'] + (est_calls * 400 / 1_000_000) * pricing['output']
        if est_cost > 20.0:
            ans = input(f"Estimated cost for gpt-5-mini eval is ${est_cost:.2f} (> $20). Proceed? [y/N]: ").strip().lower()
            if ans not in ("y", "yes"):
                raise SystemExit("Aborting gpt-5-mini eval due to cost constraints.")
        metrics_gpt5_mini = evaluate_external_model("gpt-5-mini", test_dataset, system_prompt)
    except Exception as e:
        print(f"Failed to evaluate GPT-5-mini: {e}")
        metrics_gpt5_mini = None

    # Evaluate Gemini-2.5-flash
    try:
        pricing = PRICING_USD_PER_MTOK["gemini-2.5-flash"]
        est_calls = min(100, len(test_dataset))
        est_cost = (est_calls * 1800 / 1_000_000) * pricing['input'] + (est_calls * 400 / 1_000_000) * pricing['output']
        if est_cost > 20.0:
            ans = input(f"Estimated cost for gemini-2.5-flash eval is ${est_cost:.2f} (> $20). Proceed? [y/N]: ").strip().lower()
            if ans not in ("y", "yes"):
                raise SystemExit("Aborting gemini-2.5-flash eval due to cost constraints.")
        metrics_gemini_flash = evaluate_external_model("gemini-2.5-flash", test_dataset, system_prompt)
    except Exception as e:
        print(f"Failed to evaluate Gemini-2.5-flash: {e}")
        metrics_gemini_flash = None

    # Evaluate OSS models (with robust VRAM error handling and proper unloading)
    print("\nEvaluating OSS models...")
    oss_metrics = {}
    oss_models = ["gpt-oss:20b", "gemma3:12b", "deepseek-r1:8b", "deepseek-r1:14b", "gemma3n:e4b", "qwen3:8b"]
    
    if ollama_chat is None:
        print("Ollama not available; skipping OSS model evaluation.")
    else:
        import subprocess
        successful_models = 0
        for model_name in oss_models:
            try:
                print(f"\nAttempting to evaluate {model_name}...")
                
                # Quick test to see if model is available and has sufficient VRAM
                test_resp = ollama_chat(model=model_name, messages=[{"role": "user", "content": "test"}])
                if not test_resp or not test_resp.get("message", {}).get("content"):
                    print(f"⚠️  {model_name}: Model test failed, skipping...")
                    oss_metrics[model_name] = None
                    # Still try to unload in case it was partially loaded
                    try:
                        subprocess.run(["ollama", "stop", model_name], capture_output=True, timeout=10)
                    except:
                        pass
                    continue
                
                # If test passes, proceed with full evaluation
                print(f"✓ {model_name}: Model test passed, proceeding with evaluation...")
                oss_metrics[model_name] = evaluate_oss_model(model_name, test_dataset, system_prompt)
                successful_models += 1
                print(f"✅ {model_name}: Evaluation completed successfully")
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['memory', 'vram', 'cuda', 'out of memory', 'resource']):
                    print(f"🚫 {model_name}: VRAM/Memory issue detected - {e}")
                    print(f"   Skipping {model_name} and continuing with next model...")
                else:
                    print(f"❌ {model_name}: Failed with error - {e}")
                    print(f"   Skipping {model_name} and continuing...")
                
                oss_metrics[model_name] = None
            finally:
                # Always unload the model to free VRAM for next evaluation
                try:
                    subprocess.run(["ollama", "stop", model_name], capture_output=True, timeout=10)
                    print(f"  -> Unloaded {model_name} from VRAM")
                except Exception as unload_e:
                    print(f"  -> Warning: Could not unload {model_name}: {unload_e}")
        
        print(f"\nOSS Model Evaluation Summary: {successful_models}/{len(oss_models)} models evaluated successfully")

    # Stop power monitoring before visualization/saving
    stop_power_monitoring()

    # Create comprehensive visualizations
    print("\nGenerating visualizations...")
    viz_filename = create_visualizations(metrics_without_grpo, metrics_with_grpo, power_data)
    # Move visualization to graphs directory
    import shutil
    if os.path.exists(viz_filename):
        base = os.path.basename(viz_filename)
        new_viz_path = os.path.join(graph_dir, base)
        shutil.move(viz_filename, new_viz_path)
        viz_filename = new_viz_path



    # Save detailed results to Excel and Markdown in graphs directory
    results_data = []
    for i in range(len(pred_without)):
        results_data.append({
            'sample_id': i,
            'true_codes': str(true_without[i]),
            'pred_without_grpo': str(pred_without[i]),
            'pred_with_grpo': str(pred_with[i]),
            'exact_match_without': set(pred_without[i]) == set(true_without[i]),
            'exact_match_with': set(pred_with[i]) == set(true_with[i]),
            'partial_match_without': bool(set(pred_without[i]) & set(true_without[i])),
            'partial_match_with': bool(set(pred_with[i]) & set(true_with[i])),
            'total_reward_without': rewards_without[i]['total'],
            'total_reward_with': rewards_with[i]['total']
        })

    results_df = pd.DataFrame(results_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join(graph_dir, f"detailed_evaluation_results_{timestamp}.xlsx")
    results_df.to_excel(results_filename, index=False)

    # Save summary metrics to Markdown for paper reporting
    md_lines = []
    md_lines.append(f"# Evaluation Summary ({timestamp})\n")
    def fmt_overall(m):
        return f"Exact Match: {m['exact_match_accuracy']:.4f}, Partial Match: {m['partial_match_accuracy']:.4f}, Macro F1: {m['f1_macro']:.4f}"
    # GRPO overall
    md_lines.append("\n## GRPO Overall\n")
    md_lines.append(f"- Without GRPO: {fmt_overall(metrics_without_grpo)}\n")
    md_lines.append(f"- With GRPO: {fmt_overall(metrics_with_grpo)}\n")

    md_path = os.path.join(graph_dir, f"evaluation_summary_{timestamp}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(md_lines)

    # Save power consumption data in graphs directory
    power_df = pd.DataFrame(power_data)
    power_filename = os.path.join(graph_dir, f"power_consumption_{timestamp}.xlsx")
    power_df.to_excel(power_filename, index=False)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print(f"Detailed results saved to: {results_filename}")
    print(f"Power consumption data saved to: {power_filename}")
    print(f"Visualizations saved as: {viz_filename}")

    # Print final summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE MODEL COMPARISON")
    print(f"{'='*60}")

    # Collect all available metrics for comparison
    all_models = [
        ("Without GRPO", metrics_without_grpo),
        ("With GRPO", metrics_with_grpo),
    ]

    if metrics_gpt5_mini is not None:
        all_models.append(("GPT-5-mini", metrics_gpt5_mini))

    if metrics_gemini_flash is not None:
        all_models.append(("Gemini-2.5-flash", metrics_gemini_flash))

    # Add OSS models to comparison
    for model_name, metrics in oss_metrics.items():
        if metrics is not None:
            all_models.append((model_name, metrics))

    # Print comparison table
    print(f"\n{'Model':<20} {'Exact Match':<12} {'Partial Match':<14} {'Macro F1':<10}")
    print("-" * 60)

    for model_name, metrics in all_models:
        exact_match = metrics['exact_match_accuracy']
        partial_match = metrics['partial_match_accuracy'] 
        macro_f1 = metrics['f1_macro']
        
        print(f"{model_name:<20} {exact_match:<12.4f} {partial_match:<14.4f} {macro_f1:<10.4f}")

    print("\n" + "="*60)
    print("GRPO IMPROVEMENT ANALYSIS")
    print("="*60)
    print(f"Exact Match Accuracy Improvement: {metrics_with_grpo['exact_match_accuracy'] - metrics_without_grpo['exact_match_accuracy']:.4f}")
    print(f"Partial Match Accuracy Improvement: {metrics_with_grpo['partial_match_accuracy'] - metrics_without_grpo['partial_match_accuracy']:.4f}")
    print(f"Macro F1 Improvement: {metrics_with_grpo['f1_macro'] - metrics_without_grpo['f1_macro']:.4f}")
    print(f"Total Reward Improvement: {metrics_with_grpo['reward_stats']['total']['mean'] - metrics_without_grpo['reward_stats']['total']['mean']:.4f}")

    # Compare with external models if available
    if metrics_gpt5_mini is not None:
        print(f"\nGRPO vs GPT-5-mini:")
        print(f"  Exact Match Advantage: {metrics_with_grpo['exact_match_accuracy'] - metrics_gpt5_mini['exact_match_accuracy']:.4f}")
        print(f"  Macro F1 Advantage: {metrics_with_grpo['f1_macro'] - metrics_gpt5_mini['f1_macro']:.4f}")

    if metrics_gemini_flash is not None:
        print(f"\nGRPO vs Gemini-2.5-flash:")
        print(f"  Exact Match Advantage: {metrics_with_grpo['exact_match_accuracy'] - metrics_gemini_flash['exact_match_accuracy']:.4f}")
        print(f"  Macro F1 Advantage: {metrics_with_grpo['f1_macro'] - metrics_gemini_flash['f1_macro']:.4f}")

    print("="*60)



print("\nScript completed successfully!")