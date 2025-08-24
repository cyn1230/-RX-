# call_llm.py —— 本地 Ollama（千问）专用
# 仅依赖 openai 官方 SDK（>=1.40），通过 /v1 兼容端点调用
import os
import time
import traceback
from typing import List, Dict, Optional, Union

from openai import OpenAI

# —— 环境变量，可按需覆盖 ————————————————————————————————————————————————
# Ollama /v1 兼容端点
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
# Ollama 不校验 API key，这里给一个非空值即可
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
# 默认千问模型（与你提供的一致）
DEFAULT_MODEL = os.environ.get("QWEN_MODEL_NAME", "qwen2.5:7b-instruct")
# 上下文长度（结合你的日志里 OLLAMA_CONTEXT_LENGTH:4096）
DEFAULT_NUM_CTX = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "4096"))

# 兼容你项目里可能传入的“别名”
MODEL_MAP = {
    "Qwen2.5-7B-Instruct": "qwen2.5:7b-instruct",
    "Qwen2.5-14B-Instruct": "qwen2.5:14b-instruct",
    "qwen2.5:7b-instruct": "qwen2.5:7b-instruct",
    "qwen2.5:14b-instruct": "qwen2.5:14b-instruct",
}

# 初始化 OpenAI 客户端指向 Ollama
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
    timeout=300,
    max_retries=0,  # 我们在函数内做自定义重试
)

def _resolve_model(name: Optional[str]) -> str:
    if not name:
        return DEFAULT_MODEL
    return MODEL_MAP.get(name, name)

def call_llm(
    messages: List[Dict[str, Union[str, dict]]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: Optional[float] = None,
    stop: Optional[Union[str, List[str]]] = None,
    seed: Optional[int] = None,
    stream: bool = False,
    max_retries: int = 6,
    llm_port_idx: Optional[int] = None,  # 为保持签名，不使用
) -> Union[str, Dict[str, Union[str, List[str]]]]:
    """
    与原项目签名保持最大兼容。
    - 返回：默认返回完整字符串；若 stream=True，返回增量拼接后的字符串。
    - 支持：stop、seed、top_p、temperature、max_tokens
    - 低显存建议：适当降低 max_tokens；必要时把 DEFAULT_NUM_CTX 设为 4096（已默认）
    """
    model_name = _resolve_model(model)
    last_err = None

    # 传递 Ollama 专属 options（通过 openai-python 的 extra_body）
    # 可按需添加：num_batch, num_gpu, repeat_penalty 等
    ollama_options = {
        "num_ctx": DEFAULT_NUM_CTX,
    }

    if temperature is not None:
        ollama_options["temperature"] = float(temperature)
    if top_p is not None:
        ollama_options["top_p"] = float(top_p)
    if seed is not None:
        # Ollama 支持 seed（从 0 开始）
        ollama_options["seed"] = int(seed)

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "extra_body": {"options": ollama_options},
    }
    if stop is not None:
        payload["stop"] = stop

    for attempt in range(1, max_retries + 1):
        try:
            if stream:
                chunks = []
                with client.chat.completions.stream(**payload) as s:
                    for event in s:
                        if event.type == "content.delta":
                            # 增量内容
                            delta = event.delta
                            if delta:
                                chunks.append(delta)
                        elif event.type == "error":
                            raise RuntimeError(f"Ollama stream error: {event.error}")
                    # 结束时 s.get_final_response() 可获取最终对象，这里直接拼接
                return "".join(chunks)
            else:
                resp = client.chat.completions.create(**payload)
                return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            # 简单指数退避
            wait = min(1.5 ** (attempt - 1), 10)
            print(f"[call_llm] attempt={attempt}/{max_retries} error: {e}")
            print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            time.sleep(wait)

    # 全部尝试失败
    raise RuntimeError(f"Ollama call failed after {max_retries} retries: {last_err}")
