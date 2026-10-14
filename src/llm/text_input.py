"""Convert text-only documents and LangChain content blocks before tokenization."""
from collections.abc import Mapping


def text_input(value):
    if isinstance(value, str):
        # PDF/OCR text can contain lone surrogates rejected by Rust tokenizers.
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(part for item in value if (part := text_input(item)))
    if hasattr(value, "page_content"):
        return text_input(value.page_content)
    if isinstance(value, Mapping):
        if value.get("type", "text") not in {"text", "input_text"}:
            raise ValueError("이 모델은 텍스트 입력만 지원합니다. 이미지·음성 블록은 텍스트로 변환해 주세요.")
        for key in ("text", "page_content", "content"):
            if key in value:
                return text_input(value[key])
    raise ValueError(f"모델 입력은 문자열 또는 텍스트 문서여야 합니다. 받은 타입: {type(value).__name__}")


def text_messages(messages):
    conversation = []
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("모델 대화의 역할과 텍스트 형식이 올바르지 않습니다.")
        conversation.append({"role": message["role"], "content": text_input(message.get("content"))})
    if not conversation or not conversation[-1]["content"].strip():
        raise ValueError("모델에 전달할 질문을 입력해 주세요.")
    return conversation
