from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union
import hashlib
import json
import os

import numpy as np


TextInput = Union[str, List[str]]


def _normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


class BaseTextEncoder:
    dim: int

    def encode(self, texts: TextInput) -> np.ndarray:
        raise NotImplementedError

    def warm_cache(self, texts: List[str]) -> None:
        self.encode(texts)

    def save_cache(self, output_prefix: str) -> None:
        return None


class HashingTextEncoder(BaseTextEncoder):
    """Deterministic local encoder for smoke tests and offline runs."""

    def __init__(self, dim: int = 256, normalize: bool = True):
        self.dim = dim
        self.normalize = normalize

    def encode(self, texts: TextInput) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re_tokenize(str(text))
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                idx = int(digest[:8], 16) % self.dim
                sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
                arr[row, idx] += sign
        if self.normalize:
            arr = _normalize(arr)
        return arr[0] if single else arr


def re_tokenize(text: str) -> List[str]:
    return [t for t in text.lower().replace("\n", " ").split() if t]


class BertTextEncoder(BaseTextEncoder):
    """HuggingFace BERT-style contextual text encoder."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 128,
        pooling: str = "mean",
        normalize: bool = True,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("BERT encoding requires torch and transformers. Use --encoder hashing for offline smoke tests.") from exc

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self.normalize = normalize
        self.cache: Dict[str, np.ndarray] = {}

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if self.device == "cpu":
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            torch.set_num_threads(1)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=local_files_only).to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    def _encode_uncached(self, texts: List[str]) -> np.ndarray:
        outputs = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**encoded).last_hidden_state
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                elif self.pooling == "mean":
                    mask = encoded["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                else:
                    raise ValueError(f"Unsupported pooling: {self.pooling}")
                outputs.append(pooled.detach().cpu().numpy())
        arr = np.concatenate(outputs, axis=0).astype(np.float32)
        return _normalize(arr) if self.normalize else arr

    def encode(self, texts: TextInput) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        texts = ["" if t is None else str(t) for t in texts]
        missing = []
        seen = set()
        for text in texts:
            if text not in self.cache and text not in seen:
                missing.append(text)
                seen.add(text)
        if missing:
            vecs = self._encode_uncached(missing)
            for text, vec in zip(missing, vecs):
                self.cache[text] = vec.astype(np.float32)
        arr = np.stack([self.cache[text] for text in texts], axis=0).astype(np.float32)
        return arr[0] if single else arr

    def save_cache(self, output_prefix: str) -> None:
        if not self.cache:
            return
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        texts = list(self.cache.keys())
        np.save(str(prefix) + ".npy", np.stack([self.cache[t] for t in texts], axis=0).astype(np.float32))
        with open(str(prefix) + ".texts.json", "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)


class BertEmbeddingLayerTextEncoder(BaseTextEncoder):
    """Fast CPU fallback using the BERT word embedding matrix and mean pooling."""

    def __init__(self, model_name: str, batch_size: int = 512, max_length: int = 64, normalize: bool = True):
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("BERT embedding-layer encoding requires torch and transformers.") from exc

        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.cache: Dict[str, np.ndarray] = {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

        model_dir = Path(model_name)
        safetensor_path = model_dir / "model.safetensors"
        bin_path = model_dir / "pytorch_model.bin"
        if safetensor_path.exists():
            from safetensors.torch import load_file

            state = load_file(str(safetensor_path))
        elif bin_path.exists():
            state = torch.load(str(bin_path), map_location="cpu")
        else:
            raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {model_name}")

        key = next((k for k in state if k.endswith("word_embeddings.weight")), None)
        if key is None:
            raise KeyError("Cannot find BERT word embedding weights.")
        self.word_embeddings = state[key].float().cpu().numpy().astype(np.float32)
        self.dim = int(self.word_embeddings.shape[1])

    def _encode_uncached(self, texts: List[str]) -> np.ndarray:
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="np")
            input_ids = encoded["input_ids"]
            mask = encoded["attention_mask"].astype(np.float32)
            if input_ids.shape[1] >= 2:
                special = np.zeros_like(mask)
                special[:, 0] = 1.0
                for row, length in enumerate(mask.sum(axis=1).astype(int)):
                    if length > 1:
                        special[row, length - 1] = 1.0
                mask = np.maximum(mask - special, 0.0)
            emb = self.word_embeddings[input_ids]
            pooled = (emb * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1, keepdims=True).clip(min=1.0)
            outputs.append(pooled.astype(np.float32))
        arr = np.concatenate(outputs, axis=0).astype(np.float32)
        return _normalize(arr) if self.normalize else arr

    def encode(self, texts: TextInput) -> np.ndarray:
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        texts = ["" if t is None else str(t) for t in texts]
        missing = []
        seen = set()
        for text in texts:
            if text not in self.cache and text not in seen:
                missing.append(text)
                seen.add(text)
        if missing:
            vecs = self._encode_uncached(missing)
            for text, vec in zip(missing, vecs):
                self.cache[text] = vec.astype(np.float32)
        arr = np.stack([self.cache[text] for text in texts], axis=0).astype(np.float32)
        return arr[0] if single else arr

    def save_cache(self, output_prefix: str) -> None:
        if not self.cache:
            return
        prefix = Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        texts = list(self.cache.keys())
        np.save(str(prefix) + ".npy", np.stack([self.cache[t] for t in texts], axis=0).astype(np.float32))
        with open(str(prefix) + ".texts.json", "w", encoding="utf-8") as f:
            json.dump(texts, f, ensure_ascii=False)


def build_text_encoder(
    encoder: str = "hashing",
    bert_model: str = "bert-base-uncased",
    device: str = "auto",
    batch_size: int = 32,
    max_length: int = 128,
    pooling: str = "mean",
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    bert_encoder_mode: str = "contextual",
) -> BaseTextEncoder:
    if encoder == "hashing":
        return HashingTextEncoder()
    if encoder == "bert" and bert_encoder_mode == "embedding_layer":
        return BertEmbeddingLayerTextEncoder(bert_model, batch_size=max(batch_size, 512), max_length=max_length)
    if encoder == "bert":
        return BertTextEncoder(
            bert_model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            pooling=pooling,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    raise ValueError(f"Unknown encoder: {encoder}")
