"""YaYan_Reasoner / YaYan_Translator LLM 用戶端。

支援兩個後端：
- transformers + bitsandbytes 4-bit（預設，Turing 穩定）
- vLLM in-process（可選，較快但 Turing 需 enforce_eager）
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional

import torch

from ..config import CONFIG, model_path, load_prompt

logger = logging.getLogger("YaYan.LLM")


class LlmClient:
    def __init__(self, alias: str = "YaYan_Reasoner"):
        self.alias = alias
        self.local_dir: Path = model_path(alias)
        if not self.local_dir.exists():
            raise FileNotFoundError(f"{alias} 不存在: {self.local_dir}")

        self.device = CONFIG["devices"]["llm_gpu"]
        self.backend = CONFIG["llm"]["backend"].lower()
        self._tokenizer = None
        self._model = None
        self._vllm = None
        self._load()

    def _load(self) -> None:
        if self.backend == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_transformers(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(f"載入 {self.alias}（transformers + 4-bit）…")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.local_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.local_dir),
            quantization_config=bnb,
            device_map=self.device,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16,
        )

    def _load_vllm(self) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM 未安裝。請改 backend: transformers 或安裝 vllm。"
            ) from e

        logger.info(f"載入 {self.alias}（vLLM）…")
        cfg = CONFIG["llm"]
        self._vllm = LLM(
            model=str(self.local_dir),
            quantization=cfg.get("quantization", None) if cfg.get("quantization") in ("awq", "gptq", "fp8") else None,
            dtype=cfg["dtype"],
            max_model_len=cfg["max_model_len"],
            gpu_memory_utilization=cfg["gpu_memory_utilization"],
            enforce_eager=cfg["enforce_eager"],
            tensor_parallel_size=1,
            trust_remote_code=True,
        )
        self._sampling_default = SamplingParams(
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            max_tokens=cfg["max_new_tokens"],
        )
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.local_dir), trust_remote_code=True, local_files_only=True
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        cfg = CONFIG["llm"]
        max_new_tokens = max_new_tokens or cfg["max_new_tokens"]
        temperature = temperature if temperature is not None else cfg["temperature"]

        text_input = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if self.backend == "vllm":
            return self._chat_vllm(text_input, max_new_tokens, temperature)
        return self._chat_transformers(text_input, max_new_tokens, temperature)

    def _chat_transformers(self, text_input: str, max_new_tokens: int, temperature: float) -> str:
        inputs = self._tokenizer([text_input], return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _chat_vllm(self, text_input: str, max_new_tokens: int, temperature: float) -> str:
        from vllm import SamplingParams
        params = SamplingParams(
            temperature=temperature,
            top_p=CONFIG["llm"]["top_p"],
            max_tokens=max_new_tokens,
        )
        outputs = self._vllm.generate([text_input], params)
        return outputs[0].outputs[0].text.strip()

    def translate(self, raw_text: str, source_language: str) -> str:
        prompt = load_prompt("translate").replace("{source_language}", source_language)
        return self.chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": raw_text},
        ])

    def refine(self, raw_text: str, user_edit: str, source_language: str) -> str:
        prompt = (
            load_prompt("refine")
            .replace("{raw_text}", raw_text)
            .replace("{user_edit}", user_edit)
            .replace("{source_language}", source_language)
        )
        return self.chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_edit},
        ])
