"""Persistent user-managed STT and LLM profile registry."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from uuid import uuid4

from src.config import Settings


MODEL_KINDS = ("stt", "llm")
MODEL_REGISTRY_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class ModelProfile:
    key: str
    kind: str
    name: str
    identifier: str
    path: str = ""
    device: str = "auto"
    compute_type: str = "int8"
    backend: str = ""

    @property
    def runtime_location(self) -> str:
        return self.path.strip() or self.identifier.strip()

    @property
    def runtime_backend(self) -> str:
        if self.backend.strip():
            return self.backend.strip()
        return "faster-whisper" if self.kind == "stt" else "transformers"


class ModelRegistry:
    """Store selectable model profiles without deleting model assets."""

    def __init__(self, path: Path, defaults: Iterable[ModelProfile]) -> None:
        self.path = path.expanduser().resolve()
        self.defaults = tuple(defaults)
        if {profile.kind for profile in self.defaults} != set(MODEL_KINDS):
            raise ValueError("Model registry defaults must include STT and LLM profiles")

    @classmethod
    def from_settings(cls, settings: Settings) -> "ModelRegistry":
        defaults = [
            ModelProfile(
                key="sensevoice-small",
                kind="stt",
                name="SenseVoice Small",
                identifier="FunAudioLLM/SenseVoiceSmall",
                device="auto",
                compute_type="float32",
                backend="sensevoice",
            ),
            ModelProfile(
                key="bigvox-exaone4-audio",
                kind="stt",
                name="Bigvox · EXAONE 4 Audio",
                identifier="bigdefence/Bigvox-Exaone4-Audio",
                device="auto",
                compute_type="float16",
                backend="bigvox",
            ),
            ModelProfile(
                key="default-stt",
                kind="stt",
                name=f"Whisper · {settings.whisper_model}",
                identifier=settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                backend="faster-whisper",
            ),
        ]
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            defaults.append(
                ModelProfile(
                    key="default-llm-mlx",
                    kind="llm",
                    name="EXAONE 4.0 · 1.2B · MLX 4-bit",
                    identifier=settings.exaone_mlx_model_id,
                    path=str(settings.exaone_mlx_model_dir),
                    device="mps",
                    compute_type="int4",
                    backend="mlx",
                )
            )
        defaults.append(
            ModelProfile(
                key="default-llm",
                kind="llm",
                name="EXAONE 4.0 · 1.2B",
                identifier=settings.exaone_model_id,
                path=str(settings.exaone_model_dir),
                device=settings.exaone_device,
                compute_type="auto",
                backend="transformers",
            ),
        )
        return cls(settings.model_registry_path, defaults)

    @staticmethod
    def _profile_ready(profile: ModelProfile) -> bool:
        if not profile.path.strip():
            return False
        location = Path(profile.path).expanduser()
        return (
            (location / "config.json").is_file()
            and (location / "tokenizer_config.json").is_file()
            and any(location.glob("*.safetensors"))
        )

    def _default_payload(self) -> Dict[str, object]:
        active: Dict[str, str] = {}
        for profile in self.defaults:
            active.setdefault(profile.kind, profile.key)
        memory_profile = next(
            (
                profile
                for profile in self.defaults
                if profile.kind == "llm"
                and profile.runtime_backend == "mlx"
                and self._profile_ready(profile)
            ),
            None,
        )
        if memory_profile is not None:
            active["llm"] = memory_profile.key
        return {
            "schema_version": MODEL_REGISTRY_SCHEMA_VERSION,
            "active": active,
            "profiles": [asdict(profile) for profile in self.defaults],
        }

    def _load(self) -> Dict[str, object]:
        if not self.path.is_file():
            return self._default_payload()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"모델 목록을 읽을 수 없습니다: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("모델 목록 형식이 올바르지 않습니다.")
        profiles = payload.get("profiles")
        active = payload.get("active")
        if not isinstance(profiles, list) or not isinstance(active, dict):
            raise RuntimeError("모델 목록 형식이 올바르지 않습니다.")
        try:
            schema_version = int(payload.get("schema_version", 1))
        except (TypeError, ValueError):
            schema_version = 1
        if schema_version < MODEL_REGISTRY_SCHEMA_VERSION:
            existing_keys = {
                str(item.get("key", ""))
                for item in profiles
                if isinstance(item, dict)
            }
            for profile in self.defaults:
                if profile.key not in existing_keys:
                    profiles.append(asdict(profile))
            memory_profile = next(
                (
                    profile
                    for profile in self.defaults
                    if profile.kind == "llm"
                    and profile.runtime_backend == "mlx"
                    and self._profile_ready(profile)
                ),
                None,
            )
            if memory_profile is not None and active.get("llm") == "default-llm":
                active["llm"] = memory_profile.key
            payload["schema_version"] = MODEL_REGISTRY_SCHEMA_VERSION
            self._save(payload)
        return payload

    def _save(self, payload: Dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def profiles(self, kind: Optional[str] = None) -> List[ModelProfile]:
        if kind is not None and kind not in MODEL_KINDS:
            raise ValueError(f"Unknown model kind: {kind}")
        result = []
        for item in self._load()["profiles"]:
            if not isinstance(item, dict):
                continue
            try:
                profile = ModelProfile(**item)
            except TypeError:
                continue
            if profile.kind in MODEL_KINDS and (kind is None or profile.kind == kind):
                result.append(profile)
        return result

    def active(self, kind: str) -> ModelProfile:
        profiles = self.profiles(kind)
        if not profiles:
            raise RuntimeError(f"사용 가능한 {kind.upper()} 모델이 없습니다.")
        active_key = str(self._load()["active"].get(kind, ""))
        return next((item for item in profiles if item.key == active_key), profiles[0])

    def set_active(self, kind: str, key: str) -> ModelProfile:
        profile = next((item for item in self.profiles(kind) if item.key == key), None)
        if profile is None:
            raise KeyError(f"Unknown {kind} model profile: {key}")
        payload = self._load()
        payload["active"][kind] = key
        self._save(payload)
        return profile

    def add(
        self,
        *,
        kind: str,
        name: str,
        identifier: str,
        path: str = "",
        device: str = "auto",
        compute_type: str = "int8",
        backend: str = "",
    ) -> ModelProfile:
        if kind not in MODEL_KINDS:
            raise ValueError(f"Unknown model kind: {kind}")
        clean_name = name.strip()
        clean_identifier = identifier.strip()
        if not clean_name or not clean_identifier:
            raise ValueError("모델 이름과 식별자를 모두 입력해 주세요.")
        payload = self._load()
        if any(
            isinstance(item, dict)
            and item.get("kind") == kind
            and (
                str(item.get("name", "")).casefold() == clean_name.casefold()
                or (
                    str(item.get("identifier", "")) == clean_identifier
                    and str(item.get("path", "")) == path.strip()
                )
            )
            for item in payload["profiles"]
        ):
            raise ValueError("같은 모델이 이미 목록에 있습니다.")
        profile = ModelProfile(
            key=f"{kind}-{uuid4().hex}",
            kind=kind,
            name=clean_name,
            identifier=clean_identifier,
            path=path.strip(),
            device=device.strip() or "auto",
            compute_type=compute_type.strip() or "int8",
            backend=backend.strip(),
        )
        payload["profiles"].append(asdict(profile))
        self._save(payload)
        return profile

    def update_device(self, key: str, device: str) -> ModelProfile:
        profile = next((item for item in self.profiles() if item.key == key), None)
        if profile is None:
            raise KeyError(f"Unknown model profile: {key}")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"지원하지 않는 장치: {device}")
        if profile.runtime_backend == "faster-whisper" and device == "mps":
            raise ValueError("faster-whisper는 Apple GPU를 지원하지 않습니다. SenseVoice를 선택해 주세요.")
        updated = replace(profile, device=device)
        payload = self._load()
        payload["profiles"] = [asdict(updated) if item.get("key") == key else item
                               for item in payload["profiles"]]
        self._save(payload)
        return updated

    def remove(self, key: str) -> ModelProfile:
        payload = self._load()
        profiles = self.profiles()
        target = next((item for item in profiles if item.key == key), None)
        if target is None:
            raise KeyError(f"Unknown model profile: {key}")
        same_kind = [item for item in profiles if item.kind == target.kind]
        if len(same_kind) == 1:
            raise ValueError("각 종류마다 모델을 하나 이상 남겨야 합니다.")
        payload["profiles"] = [
            item
            for item in payload["profiles"]
            if not isinstance(item, dict) or item.get("key") != key
        ]
        if payload["active"].get(target.kind) == key:
            payload["active"][target.kind] = next(
                item.key for item in same_kind if item.key != key
            )
        self._save(payload)
        return target
