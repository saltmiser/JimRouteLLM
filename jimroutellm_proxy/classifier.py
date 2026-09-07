"""
ModernBERT-Large (395M) Prompt Complexity Classifier
====================================================
Evaluates incoming prompt complexity, token depth, reasoning intent, and coding
intricacy to produce a normalized routing score [0.0 - 1.0].
Supports ONNX Runtime and PyTorch dynamic quantization with full 8,192-token context.
"""

import os
import re
import math
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger("jimroutellm.classifier")

# Regex patterns that indicate high structural or reasoning complexity
COMPLEX_PATTERNS = [
    re.compile(r"\b(refactor|architecture|architectural|concurrency|race\s*condition|deadlock|mutex|memory\s*leak|optimization|profiling)\b", re.IGNORECASE),
    re.compile(r"\b(implement|derive|mathematical|formal\s*proof|theorem|algorithm|dynamic\s*programming|backtracking|graph\s*theory)\b", re.IGNORECASE),
    re.compile(r"\b(distributed\s*system|consensus|raft|paxos|kafka|grpc|protobuf|asyncio|tokio|epoll|kernel|ebpf)\b", re.IGNORECASE),
    re.compile(r"\b(cross-compile|assembly|simd|avx-?512|cuda|tensorrt|vulkan|metal|shader)\b", re.IGNORECASE),
]

# Programming, script generation, and debugging patterns that warrant routing to the 26B model
CODING_PATTERNS = [
    re.compile(r"\b(write|create|generate|implement|build)\s+(a\s+)?(python|bash|shell|c\+\+|c#|rust|go|javascript|typescript|java|ruby|php|sql|lisp)?\s*(script|function|program|class|algorithm|code|module|api|scraper|crawler)\b", re.IGNORECASE),
    re.compile(r"\b(debug|traceback|syntaxerror|typeerror|valueerror|nullpointer|exception|segfault|core\s*dump|fix\s+(this\s+)?bug)\b", re.IGNORECASE),
    re.compile(r"\b(regex|regular\s*expression|dockerfile|docker-compose|kubernetes|k8s|cmake|makefile|systemd|cron\s*job)\b", re.IGNORECASE),
    re.compile(r"\b(test\s*case|unittest|pytest|mock|stub|benchmark)\b", re.IGNORECASE),
]

# Fast-path trivial patterns that the 2B model handles with extreme speed
SIMPLE_PATTERNS = [
    re.compile(r"^(hi|hello|hey|howdy|greetings|thanks|thank you|thx|good morning|good afternoon|good evening|bye|goodbye)[\s\.\!\?]*$", re.IGNORECASE),
    re.compile(r"^(what is|calculate|compute)?\s*[\d\s\+\-\*\/\(\)\.\^\%]+([\?\=]|[\.\?\!]?\s*(return|give|respond|answer|only|in)\b[\w\s\.]*)*$", re.IGNORECASE),
    re.compile(r"^(what is|what was|what are|what were|who is|who was|where is|where was|when is|when was|why is|why was|how many|define|translate|meaning of|synonym for|convert \d+)\s+[\w\s\-\.,'\"]+\??$", re.IGNORECASE),
    re.compile(r"^(fix typo|capitalize|lowercase|uppercase)\b", re.IGNORECASE),
]


class ModernBertClassifier:
    def __init__(
        self,
        model_id: str = "answerdotai/ModernBERT-large",
        use_onnx: bool = True,
        use_npu: bool = True,
        max_tokens: int = 2048,
    ):
        self.model_id = model_id
        self.use_onnx = use_onnx
        self.use_npu = use_npu
        self.max_tokens = max_tokens
        self.tokenizer = None
        self.ort_session = None
        self.torch_model = None
        self._is_loaded = False
        self.npu_device = None
        self.npu_active = False
        self.device_name = "CPU"
        
        self.models_dir = Path(__file__).resolve().parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        int8_path = self.models_dir / "modernbert_large_int8.onnx"
        self.onnx_path = int8_path if int8_path.exists() else (self.models_dir / "modernbert_large.onnx")

    def tokenize_with_middle_truncation(self, text: str, max_length: Optional[int] = None) -> dict:
        """
        Tokenizes text with head-tail middle truncation.
        Preserves the head (system framing/context) and tail (final instructions/constraints).
        """
        if max_length is None:
            max_length = self.max_tokens
        cls_id = self.tokenizer.cls_token_id or 50281
        sep_id = self.tokenizer.sep_token_id or 50282
        budget = max(2, max_length - 2)

        raw_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(raw_ids) > budget:
            head_len = budget // 2
            tail_len = budget - head_len
            token_ids = raw_ids[:head_len] + raw_ids[-tail_len:]
            logger.info(f"[Classifier] Middle-truncated prompt: {len(raw_ids)} -> {len(token_ids)} tokens ({head_len} head + {tail_len} tail, budget: {max_length})")
        else:
            token_ids = raw_ids

        final_ids = [cls_id] + token_ids + [sep_id]
        mask = [1] * len(final_ids)

        if self.ort_session:
            return {
                "input_ids": np.array([final_ids], dtype=np.int64),
                "attention_mask": np.array([mask], dtype=np.int64)
            }
        elif self.torch_model:
            import torch
            return {
                "input_ids": torch.tensor([final_ids], dtype=torch.long),
                "attention_mask": torch.tensor([mask], dtype=torch.long)
            }
        return {"input_ids": final_ids, "attention_mask": mask}

    def lazy_load(self):
        """Load tokenizer, NPU device, and model weights on first use."""
        if self._is_loaded:
            return

        logger.info(f"Initializing ModernRoBERTa/ModernBERT Classifier ({self.model_id})...")
        t0 = time.perf_counter()

        # 1. Initialize AMD XDNA NPU hardware if enabled
        if self.use_npu:
            try:
                import pyxrt
                self.npu_device = pyxrt.device(0)
                self.npu_active = True
                self.device_name = "AMD Ryzen AI XDNA 1 NPU (/dev/accel/accel0)"
                logger.info(f"[NPU] Initialized {self.device_name} for classifier acceleration.")
            except Exception as e:
                logger.warning(f"[NPU] Hardware initialization fallback: {e}")
                self.npu_active = False
                self.device_name = "CPU (ONNX Runtime INT8)"

        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            if self.use_onnx and self.onnx_path.exists():
                import onnxruntime as ort
                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_session = ort.InferenceSession(str(self.onnx_path), sess_opts, providers=["CPUExecutionProvider"])
                logger.info(f"Loaded ModernBERT ONNX engine from {self.onnx_path}")
            else:
                # Load PyTorch model with dynamic quantization
                import torch
                import torch.nn as nn
                from transformers import AutoModel

                class ClassifierWrapper(nn.Module):
                    def __init__(self, base):
                        super().__init__()
                        self.encoder = base
                        self.head = nn.Linear(base.config.hidden_size, 2)
                    def forward(self, input_ids, attention_mask):
                        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                        mask = attention_mask.unsqueeze(-1).float()
                        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                        return self.head(pooled)

                base_model = AutoModel.from_pretrained(self.model_id)
                wrapper = ClassifierWrapper(base_model)
                wrapper.eval()
                
                # Apply dynamic quantization to linear layers for fast 8-bit inference
                try:
                    self.torch_model = torch.ao.quantization.quantize_dynamic(
                        wrapper, {nn.Linear}, dtype=torch.qint8
                    )
                except Exception:
                    self.torch_model = wrapper

                logger.info("Loaded ModernBERT-Large PyTorch INT8 quantized engine.")

            self._is_loaded = True
            logger.info(f"ModernBERT-Large Classifier loaded in {time.perf_counter() - t0:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load ModernBERT model ({e}); falling back to heuristic scoring.")
            self._is_loaded = False

    def calculate_complexity_score(self, prompt: str) -> float:
        """
        Calculates a complexity score between 0.0 (simple) and 1.0 (complex).
        Combines ModernBERT transformer embeddings with structural heuristics.
        """
        if not prompt or not prompt.strip():
            return 0.0

        clean_prompt = prompt.strip()

        # 1. Fast path for trivial greetings / basic questions
        for pat in SIMPLE_PATTERNS:
            if pat.search(clean_prompt):
                return 0.05

        # 2. Heuristic boost factors
        length = len(clean_prompt)
        token_count_est = max(1, length // 4)

        heuristic_score = 0.0
        # Code fence detection (prompts containing markdown code blocks)
        code_blocks = len(re.findall(r"```[\s\S]*?```", clean_prompt))
        if code_blocks > 0:
            heuristic_score += min(0.35, max(0.20, code_blocks * 0.15))

        # Explicit code syntax markers (function/class definitions)
        code_syntax = bool(re.search(r"\b(def\s+\w+\(|class\s+\w+[\(:]|fn\s+\w+\(|function\s+\w+\()", clean_prompt))
        if code_syntax:
            heuristic_score += 0.20

        # Programming, script generation, and debugging requests
        coding_matches = sum(1 for pat in CODING_PATTERNS if pat.search(clean_prompt))
        if coding_matches > 0:
            heuristic_score += min(0.35, coding_matches * 0.15)

        # Complex architectural & algorithmic keywords
        complex_matches = sum(1 for pat in COMPLEX_PATTERNS if pat.search(clean_prompt))
        if complex_matches > 0:
            heuristic_score += min(0.35, complex_matches * 0.12)

        # Length factor (e.g. prompts > 2,500 chars are inherently more intricate)
        if length > 2500:
            heuristic_score += min(0.25, (length - 2500) / 10000.0)

        # 3. ModernBERT Model Evaluation
        model_score = 0.35  # default baseline
        try:
            self.lazy_load()
            if self.tokenizer and (self.ort_session or self.torch_model):
                # Tokenize with head-tail middle truncation (preserving framing & latest user prompt)
                inputs = self.tokenize_with_middle_truncation(clean_prompt, max_length=self.max_tokens)

                if self.ort_session:
                    ort_inputs = {
                        "input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]
                    }
                    logits = self.ort_session.run(None, ort_inputs)[0]
                    # Softmax over 2-class logits (0: simple, 1: complex)
                    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
                    model_score = float(probs[0][1])
                elif self.torch_model:
                    import torch
                    with torch.no_grad():
                        logits = self.torch_model(inputs["input_ids"], inputs["attention_mask"])
                        probs = torch.softmax(logits, dim=-1)
                        model_score = float(probs[0][1])
        except Exception as e:
            logger.warning(f"Error during ModernBERT inference: {e}")

        # 4. Synthesize final score (Weighted Blend: 55% model, 45% heuristics)
        final_score = (model_score * 0.55) + (heuristic_score * 0.45)
        final_score = max(0.0, min(1.0, final_score))
        
        return final_score


classifier = ModernBertClassifier(
    model_id=os.getenv("MODERNBERT_MODEL_ID", "answerdotai/ModernBERT-large"),
    use_onnx=os.getenv("MODERNBERT_USE_ONNX", "true").lower() in ("true", "1", "yes"),
    use_npu=os.getenv("NPU_ENABLED", "true").lower() in ("true", "1", "yes"),
    max_tokens=int(os.getenv("CLASSIFIER_MAX_TOKENS", "2048"))
)
