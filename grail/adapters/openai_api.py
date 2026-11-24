import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

DEFAULT_CHAT_MODEL = "gpt-4o"
DEFAULT_REASONING_MODEL = "gpt-4o"
DEFAULT_IMAGE_MODEL = "gpt-image-2.6-sunburst"

_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")


def _is_reasoning_model(model: str) -> bool:
    """Reasoning-tier models (o1/o3/o4 family) need different request shape."""
    model_base = model.split("/")[-1]
    return model_base.startswith(_REASONING_MODEL_PREFIXES)


def _encode_image_to_base64(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _image_path_to_data_uri(image_path: str) -> str:
    """Convert a local image file to a data URI suitable for vision inputs."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    b64 = _encode_image_to_base64(image_path)
    return f"data:{mime_type};base64,{b64}"


def _ensure_openai_key_present() -> None:
    """Raise a clear error if OPENAI_API_KEY is not configured."""
    if not os.getenv("OPENAI_API_KEY"):
        raise OSError("OPENAI_API_KEY is not set. Source .env or export the key first.")


def _make_client() -> OpenAI:
    """Standard OpenAI API client."""
    return OpenAI()


def chat_text(
    prompt_text: str,
    *,
    model: str = DEFAULT_CHAT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Plain text chat completion via the OpenAI API."""
    _ensure_openai_key_present()
    client = _make_client()

    reasoning = _is_reasoning_model(model)
    messages: list[dict[str, Any]] = []
    if system_prompt:
        role = "developer" if reasoning else "system"
        messages.append({"role": role, "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})

    create_kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if reasoning:
        create_kwargs["max_completion_tokens"] = max_tokens
    else:
        create_kwargs["max_tokens"] = max_tokens
        create_kwargs["temperature"] = temperature
    if response_format is not None:
        create_kwargs["response_format"] = response_format

    response = client.chat.completions.create(**create_kwargs)
    return (response.choices[0].message.content or "").strip()


def chat_with_image(
    prompt_text: str,
    image_path: str,
    *,
    model: str = DEFAULT_CHAT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: str | None = "You are a helpful vision assistant.",
    response_format: dict[str, Any] | None = None,
) -> str:
    """Vision chat completion: send a text prompt and a single image."""
    _ensure_openai_key_present()
    client = _make_client()

    data_uri = _image_path_to_data_uri(image_path)
    reasoning = _is_reasoning_model(model)

    messages: list[dict[str, Any]] = []
    if system_prompt:
        role = "developer" if reasoning else "system"
        messages.append({"role": role, "content": system_prompt})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
            ],
        }
    )

    create_kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if reasoning:
        create_kwargs["max_completion_tokens"] = max_tokens
    else:
        create_kwargs["max_tokens"] = max_tokens
        create_kwargs["temperature"] = temperature
    if response_format is not None:
        create_kwargs["response_format"] = response_format

    response = client.chat.completions.create(**create_kwargs)
    return (response.choices[0].message.content or "").strip()


def generate_image(
    prompt: str,
    output_path: Path | str,
    *,
    model: str = DEFAULT_IMAGE_MODEL,
    size: str = "1024x1024",
    system_prompt: str | None = None,
) -> Path:
    """
    Generate an image from a text prompt and save it as PNG to *output_path*.

    Uses the standard OpenAI Image API. ``system_prompt`` is prepended to the
    prompt because image generation requests accept a single prompt string.
    """
    _ensure_openai_key_present()
    client = _make_client()

    if system_prompt:
        prompt = f"{system_prompt.strip()}\n\n{prompt}"

    response = client.images.generate(model=model, prompt=prompt, size=size)
    b64 = response.data[0].b64_json
    if not b64:
        raise RuntimeError(f"No image data returned by model {model!r}.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(b64))
    return output_path
