# -*- coding: utf-8 -*-
"""
HuggingFace Transformers adapter for sentence_importance.GenerationModel.

Requires: pip install transformers accelerate torch (or equivalent).
This is a scaffold; you may need to tailor generate parameters and device placement.
"""

from typing import List, Optional
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_importance import GenerationModel, extract_boxed

class HFGenModel(GenerationModel):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", device: Optional[str]=None, dtype: str="bfloat16"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, dtype) if hasattr(torch, dtype) else torch.float16,
            device_map="auto" if device is None else None
        )
        if device:
            self.model.to(device)
        self.model.eval()

    def generate_once(self, ctx_text: str, seed: Optional[int]=None) -> str:
        g = torch.Generator(device=self.model.device)
        if seed is not None:
            g.manual_seed(seed)
        inputs = self.tokenizer(ctx_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                generator=g
            )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # Return only the last \\boxed{...} if present
        boxed = extract_boxed(text)
        return boxed or text.strip()

    def tokenize_answer(self, answer_text: str) -> List[int]:
        return self.tokenizer.encode(answer_text, add_special_tokens=False)

    def logprob_answer(self, ctx_text: str, answer_tokens: List[int]) -> float:
        # Compute sum of log-probs of answer tokens conditioned on the context + prefix of answer
        ctx_ids = self.tokenizer.encode(ctx_text, add_special_tokens=False)
        ans_ids = answer_tokens
        input_ids = ctx_ids + ans_ids[:-1]  # up to last-1 as context for next token
        target_ids = ans_ids  # tokens to score
        import torch
        with torch.no_grad():
            inputs = torch.tensor([input_ids], device=self.model.device)
            outputs = self.model(inputs)
            logits = outputs.logits  # [1, seq_len, vocab]
            # We need the logits aligned to next token positions
            # For simplicity, only take the last len(target_ids) positions
            L = len(target_ids)
            last_logits = logits[:, -L:, :]
            logprobs = torch.log_softmax(last_logits, dim=-1)
            target = torch.tensor([target_ids], device=self.model.device)
            sel = logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, L]
            return float(sel.sum().item())
