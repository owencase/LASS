"""Minimal inference runtime for bigdefence/Bigvox-Exaone4-Audio.

The multimodal projection and prompt layout follow BigDefence's Apache-2.0
``bigvox-exaone`` reference implementation.  Training-only OmniSpeech modules
are intentionally not included here.

Portions adapted from https://github.com/bigdefence/bigvox-exaone and released
under the Apache License 2.0 by their respective authors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    Exaone4Config,
    Exaone4ForCausalLM,
    WhisperConfig,
    WhisperFeatureExtractor,
)
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from src.runtime import resolve_device


SPEECH_TOKEN = "<speech>"
SPEECH_TOKEN_INDEX = -200
SYSTEM_PROMPT = (
    "You are a helpful language and speech assistant. You are able to "
    "understand the speech content that the user provides, and assist the "
    "user with a variety of tasks using natural language and speech."
)


class EncoderProjectorConcat(nn.Module):
    """Downsample and project Whisper encoder frames into EXAONE embeddings."""

    def __init__(self, config: Exaone4Config) -> None:
        super().__init__()
        self.k = int(getattr(config, "speech_encoder_ds_rate", 5))
        encoder_dim = int(getattr(config, "speech_encoder_hidden_size", 1280))
        hidden_size = int(config.hidden_size)
        self.linear1 = nn.Linear(encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, hidden_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, dimension = values.size()
        discard = sequence_length % self.k
        if discard:
            values = values[:, :-discard, :]
        values = values.contiguous().view(
            batch_size, values.size(1) // self.k, dimension * self.k
        )
        return self.linear2(self.relu(self.linear1(values)))


class BigvoxForCausalLM(Exaone4ForCausalLM):
    """EXAONE language model with the projector expected by Bigvox weights."""

    def __init__(self, config: Exaone4Config) -> None:
        super().__init__(config)
        encoder_config = getattr(config, "_bigvox_speech_encoder_config", None)
        if encoder_config is None:
            raise RuntimeError("Bigvox Whisper encoder configuration is missing.")
        # The published Bigvox safetensors include the encoder parameters under
        # ``model.speech_encoder.*``.  Constructing the module before loading the
        # checkpoint makes Transformers restore those trained parameters rather
        # than discarding them as unexpected keys.
        self.model.speech_encoder = WhisperEncoder(encoder_config)
        self.model.speech_projector = EncoderProjectorConcat(config)


def _torch_dtype(compute_type: str) -> torch.dtype:
    normalized = compute_type.strip().casefold()
    if normalized in {"float16", "fp16", "auto"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Bigvox가 지원하지 않는 연산 정밀도입니다: {compute_type}")


def _speech_encoder_location(model_location: str, configured: str, fallback: str) -> str:
    """Prefer a separately downloaded local encoder, then use the HF model ID."""
    configured_path = Path(configured).expanduser()
    candidates = []
    if configured_path.is_absolute():
        candidates.append(configured_path)
    else:
        model_path = Path(model_location).expanduser()
        if model_path.is_dir():
            candidates.extend(
                (
                    model_path / configured_path,
                    model_path.parent / configured_path,
                    model_path.parent / "speech_encoder" / configured_path.name,
                    model_path.parent / configured_path.name,
                )
            )
        candidates.append(Path.cwd() / configured_path)
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return str(candidate.resolve())
    return fallback


def _prompt_ids(tokenizer: Any) -> torch.Tensor:
    tokenizer.add_tokens([SPEECH_TOKEN], special_tokens=True)
    speech_token_id = tokenizer.convert_tokens_to_ids(SPEECH_TOKEN)
    ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SPEECH_TOKEN},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = [SPEECH_TOKEN_INDEX if item == speech_token_id else item for item in ids]
    if ids.count(SPEECH_TOKEN_INDEX) != 1:
        raise RuntimeError("Bigvox 프롬프트에 음성 토큰을 구성하지 못했습니다.")
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


@dataclass
class BigvoxRuntime:
    tokenizer: Any
    model: BigvoxForCausalLM
    feature_extractor: WhisperFeatureExtractor
    device: str
    dtype: torch.dtype

    def _speech_features(
        self, samples: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.feature_extractor(
            samples,
            sampling_rate=16_000,
            return_tensors="pt",
        )
        values = inputs.input_features.to(device=self.device, dtype=self.dtype)
        raw_frames = min(len(samples), 30 * 16_000)
        feature_frames = round(
            raw_frames / (30 * 16_000) * values.shape[-1] + 0.5
        )
        lengths = torch.tensor(
            [(feature_frames + 1) // 2], device=self.device, dtype=torch.long
        )
        encoder = self.model.model.speech_encoder
        encoded = encoder(values).last_hidden_state
        projected = self.model.model.speech_projector(encoded)
        lengths = lengths // self.model.model.speech_projector.k
        return projected, lengths

    def _input_embeddings(
        self, input_ids: torch.Tensor, speech: torch.Tensor, length: torch.Tensor
    ) -> torch.Tensor:
        token_ids = input_ids[0]
        speech_positions = torch.where(token_ids == SPEECH_TOKEN_INDEX)[0]
        if speech_positions.numel() != 1:
            raise RuntimeError("Bigvox 입력에는 음성 토큰이 정확히 하나 필요합니다.")
        position = int(speech_positions[0].item())
        embed_tokens = self.model.get_input_embeddings()
        before = embed_tokens(token_ids[:position])
        after = embed_tokens(token_ids[position + 1 :])
        speech_length = max(1, int(length[0].item()))
        return torch.cat(
            (before, speech[0, :speech_length], after), dim=0
        ).unsqueeze(0)

    def transcribe(self, samples: Any) -> str:
        input_ids = _prompt_ids(self.tokenizer).to(self.device)
        with torch.inference_mode():
            speech, length = self._speech_features(samples)
            embeddings = self._input_embeddings(input_ids, speech, length)
            attention_mask = torch.ones(
                embeddings.shape[:2], device=self.device, dtype=torch.long
            )
            outputs = self.model.generate(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=0,
                eos_token_id=361,
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()


def load_bigvox_runtime(
    model_location: str,
    *,
    device: str,
    compute_type: str,
    speech_encoder_id: str,
) -> BigvoxRuntime:
    """Load Bigvox plus the separately published Whisper encoder metadata."""
    selected_device = resolve_device(torch, device)
    dtype = _torch_dtype(compute_type)
    config_values, _ = Exaone4Config.get_config_dict(model_location)
    # Bigvox uses the regular EXAONE 4 backbone under a custom upstream model
    # type.  We instantiate that known local architecture without remote code.
    config_values["model_type"] = "exaone4"
    config = Exaone4Config.from_dict(config_values)
    configured_encoder = str(
        getattr(config, "speech_encoder", "./models/speech_encoder/whisper-large-v3")
    )
    encoder_location = _speech_encoder_location(
        model_location, configured_encoder, speech_encoder_id
    )
    config._bigvox_speech_encoder_config = WhisperConfig.from_pretrained(
        encoder_location
    )

    tokenizer = AutoTokenizer.from_pretrained(model_location, use_fast=False)
    model = BigvoxForCausalLM.from_pretrained(
        model_location,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    model.to(selected_device).eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(encoder_location)
    return BigvoxRuntime(tokenizer, model, feature_extractor, selected_device, dtype)
