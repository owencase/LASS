"""Lazy, offline-only loader for the project-local EXAONE model."""

from __future__ import annotations

import gc
import os
import subprocess
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Iterator, Mapping, Sequence, Tuple


from .model_manager import LocalModelManager, GGUF_FILENAME
from src.runtime import release_memory
from .text_input import text_input, text_messages


class LocalEXAONEClient:
    """Run local EXAONE IQ4_XS GGUF through llama.cpp."""

    def __init__(
        self,
        model_dir: Path,
        model_id: str = "mradermacher/EXAONE-4.0-1.2B-i1-GGUF",
        device: str = "auto",
        temperature: float = 0.1,
        max_input_tokens: int = 2048,
        isolated: bool = False,
    ) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.model_id = model_id
        self.requested_device = device
        self.temperature = temperature
        self.max_input_tokens = max_input_tokens
        self.isolated = isolated
        self.keep_loaded = False
        self._process = None
        self.manager = LocalModelManager(model_id, self.model_dir)
        self._model: Any = None
        self._device = "cpu"
        self._lock = RLock()

    def _runtime(self):
        with self._lock:
            if self._model is not None:
                return self._model
            self.manager.require_ready()
            from .native_runtime import prepare_native, gpu_available
            prepare_native()
            from llama_cpp import Llama, llama_supports_gpu_offload
            device = self.requested_device.strip().lower()
            if device not in {'auto', 'cpu', 'cuda', 'cuda:0'}:
                raise ValueError(f'지원하지 않는 장치: {device}')
            gpu = bool(llama_supports_gpu_offload()) and gpu_available()
            if device.startswith('cuda') and not gpu:
                raise RuntimeError('CUDA GGUF 런타임을 사용할 수 없습니다.')
            self._device = 'cuda' if gpu and device != 'cpu' else 'cpu'
            options = dict(model_path=str(self.model_dir / GGUF_FILENAME),
                           n_ctx=max(8192, self.max_input_tokens + 4096),
                           n_gpu_layers=-1 if self._device == 'cuda' else 0,
                           verbose=False)
            try:
                self._model = Llama(**options)
            except (RuntimeError, ValueError):
                if device != 'auto' or self._device != 'cuda':
                    raise
                self._device = 'cpu'
                options['n_gpu_layers'] = 0
                self._model = Llama(**options)
            if self._model.metadata.get('general.file_type') != '30':
                self._model.close()
                self._model = None
                raise ValueError('IQ4_XS GGUF 모델만 지원합니다.')
            return self._model

    def unload(self) -> None:
        with self._lock:
            if self._process is not None:
                process, self._process = self._process, None
                if process.poll() is None:
                    # EOF makes the worker's watchdog exit, even during generation.
                    process.stdin.close()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW)
                        else:
                            process.kill()
                        process.wait()
                if not process.stdin.closed:
                    process.stdin.close()
                process.stdout.close()
            if self._model is not None:
                self._model.close()
            self._model = None
            gc.collect()
            release_memory()

    @staticmethod
    def _prompt(conversation):
        # EXAONE 4.0 official text-only template, enable_thinking=False.
        parts = []
        for item in conversation:
            content = item['content']
            if item['role'] == 'assistant':
                content = '<think>\n\n</think>\n\n' + content.split('</think>')[-1].strip()
            parts.append('[|' + item['role'] + '|]\n' + content + '[|endofturn|]\n')
        return ''.join(parts) + '[|assistant|]\n<think>\n\n</think>\n\n'

    def _bounded_input(self, model, conversation):
        conversation = text_messages(conversation)
        while True:
            tokens = model.tokenize(self._prompt(conversation).encode('utf-8'), add_bos=False, special=True)
            excess = len(tokens) - self.max_input_tokens
            if excess <= 0:
                return tokens
            if len(conversation) > 2:
                del conversation[1:-1]
                continue
            target = max(conversation, key=lambda item: len(item['content']))
            content = model.tokenize(target['content'].encode('utf-8'), add_bos=False)
            budget = len(content) - excess - 32
            if budget < 32:
                raise ValueError('입력 지시문이 너무 깁니다. 질문이나 메모를 줄여 주세요.')
            tail = min(512, budget // 2)
            target['content'] = (model.detokenize(content[:budget-tail]).decode('utf-8', errors='ignore')
                + '\n…\n' + model.detokenize(content[-tail:]).decode('utf-8', errors='ignore'))

    @property
    def active_device(self) -> str:
        return self._device

    def chat(self, system_prompt: str, user_prompt: str, *, on_token: Callable[[str], None] | None = None) -> str:
        return self.chat_messages(system_prompt, [{"role": "user", "content": user_prompt}], on_token=on_token)

    def chat_messages(self, system_prompt: str, messages: Sequence[Mapping[str, str]], *,
                      on_token: Callable[[str], None] | None = None) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        from .langchain_model import EXAONEChatModel

        conversation = [SystemMessage(content=text_input(system_prompt).strip())]
        for item in messages:
            text = text_input(item.get("content", "")).strip()
            role = item.get("role")
            if text and role in {"user", "assistant"}:
                conversation.append((HumanMessage if role == "user" else AIMessage)(content=text))
        if len(conversation) == 1 or not isinstance(conversation[-1], HumanMessage):
            raise ValueError("Chat messages must end with a non-empty user message")
        parts = []
        for chunk in EXAONEChatModel(client=self).stream(conversation):
            if chunk.content:
                parts.append(str(chunk.content))
                if on_token:
                    on_token(str(chunk.content))
        content = "".join(parts).strip()
        if not content:
            raise RuntimeError("EXAONE returned an empty response")
        return content

    def stream_messages(self, conversation: list[dict[str, str]]) -> Iterator[str]:
        conversation = text_messages(conversation)
        if self.isolated:
            yield from self._stream_process(conversation)
            return
        with self._lock:
            model = self._runtime()
            tokens = self._bounded_input(model, conversation)
            stream = model.create_completion(prompt=tokens, stream=True,
                max_tokens=model.n_ctx() - len(tokens), temperature=self.temperature,
                top_p=.95, stop=['[|endofturn|]', '[|endoftext|]'])
            try:
                for chunk in stream:
                    choice = chunk['choices'][0]
                    if choice.get('text'):
                        yield choice['text']
                    if choice.get('finish_reason') == 'length':
                        raise RuntimeError('모델의 문맥 용량에 도달해 생성이 중단되었습니다. 자료를 나누어 요약해 주세요.')
            finally:
                stream.close()
                model.reset()
                release_memory()

    def _stream_process(self, conversation):
        import json
        import os
        import subprocess
        import sys

        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self.unload()
                command = [sys.executable, "--llm-worker"] if getattr(sys, "frozen", False) else [sys.executable, "-m", "src.inference_worker"]
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
                environment["PYTHONIOENCODING"] = "utf-8"
                log_dir = Path(os.environ.get("LASS_DATA_DIR", "data")) / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                with (log_dir / "inference.log").open("w", encoding="utf-8") as log:
                    self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=log, env=environment,
                        **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
            request = {"config": {"model_dir": str(self.model_dir), "model_id": self.model_id,
                "device": self.requested_device, "temperature": self.temperature,
                "max_input_tokens": self.max_input_tokens}, "messages": conversation}
            completed = False
            try:
                self._process.stdin.write((json.dumps(request, ensure_ascii=True) + "\n").encode("ascii"))
                self._process.stdin.flush()
                for line in self._process.stdout:
                    message = json.loads(line.decode("ascii"))
                    if "error" in message:
                        raise RuntimeError(message["error"])
                    if "device" in message:
                        self._device = message["device"]
                    if message.get("done"):
                        self._device = message["device"]
                        completed = True
                        break
                    if "token" in message:
                        yield message["token"]
                if not completed:
                    raise RuntimeError("AI 프로세스가 중단되었습니다. 추론 로그를 확인해 주세요.")
            finally:
                if not completed:
                    self.unload()
