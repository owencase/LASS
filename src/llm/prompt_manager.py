"""Load and validate summary prompts stored as YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class PromptTemplate:
    key: str
    name: str
    description: str
    system_prompt: str
    user_prompt: str

    def render(self, context: str) -> str:
        if "{context}" not in self.user_prompt:
            raise ValueError(f"Prompt {self.key!r} must contain a {{context}} placeholder")
        return self.user_prompt.replace("{context}", context)


class PromptManager:
    def __init__(self, prompt_directory: Path) -> None:
        self.prompt_directory = prompt_directory.expanduser().resolve()

    @staticmethod
    def _yaml() -> object:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required for prompt files. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        return yaml

    def list_templates(self) -> List[PromptTemplate]:
        if not self.prompt_directory.is_dir():
            raise FileNotFoundError(f"Prompt directory not found: {self.prompt_directory}")
        return [self.load(path.stem) for path in sorted(self.prompt_directory.glob("*.yaml"))]

    def load(self, key: str) -> PromptTemplate:
        if not key or Path(key).name != key:
            raise ValueError("Prompt key must be a plain file name without a path")
        path = self.prompt_directory / f"{key}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Prompt template not found: {path}")

        yaml = self._yaml()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Prompt template must be a mapping: {path}")
        required = ("name", "description", "system_prompt", "user_prompt")
        missing = [field for field in required if not str(data.get(field, "")).strip()]
        if missing:
            raise ValueError(f"Prompt template {key!r} is missing: {', '.join(missing)}")

        template = PromptTemplate(
            key=key,
            name=str(data["name"]).strip(),
            description=str(data["description"]).strip(),
            system_prompt=str(data["system_prompt"]).strip(),
            user_prompt=str(data["user_prompt"]).strip(),
        )
        template.render("")
        return template

    def as_options(self) -> Dict[str, str]:
        return {template.key: template.name for template in self.list_templates()}
