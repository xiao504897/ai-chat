import logging
import time
from collections.abc import Iterator
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)
from config import OLLAMA_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)

# # 在线调用 DeepSeek 的代码：
# client = OpenAI(
#     api_key=API_KEY,
#     base_url="https://api.deepseek.com",
# )

# timeout 设长一点：本地 9B 模型首次加载 + 生成可能比较慢
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY,
    timeout=180.0,
)

MAX_RETRIES = 2  # 失败后最多再重试 2 次

def call_llm(messages: list[dict]) -> str:
    """调用本地模型，失败时重试；最终失败抛出 RuntimeError。"""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 2):
        start = time.perf_counter()
        try:
            logger.info(
                "调用模型 attempt=%d model=%s messages=%d",
                attempt, OLLAMA_MODEL, len(messages),
            )
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
            )
            content = response.choices[0].message.content
            elapsed = time.perf_counter() - start
            logger.info("调用成功 elapsed=%.2fs", elapsed)
            return content or ""

        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            logger.warning("连接/超时错误 attempt=%d error=%s", attempt, e)
            if attempt <= MAX_RETRIES:
                time.sleep(1)

        except APIStatusError as e:
            last_error = e
            logger.error("API 状态错误 status=%s error=%s", e.status_code, e)
            break  # 状态码错误（如模型名不对）重试也没用，直接退出

        except Exception as e:
            last_error = e
            logger.exception("未知错误")
            break

    raise RuntimeError(f"调用模型失败：{last_error}")

def call_llm_stream(messages: list[dict]) -> Iterator[str]:
    """流式调用本地模型，逐块产出文本。调用方负责累加完整回答。"""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 2):
        start = time.perf_counter()
        try:
            logger.info(
                "流式调用 attempt=%d model=%s messages=%d",
                attempt, OLLAMA_MODEL, len(messages),
            )
            stream = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                # print(delta)  # thinking过程
                # content 可能为 None 或空字符串，直接跳过
                if delta.content:
                    yield delta.content

            elapsed = time.perf_counter() - start
            logger.info("流式调用完成 elapsed=%.2fs", elapsed)
            return  # 正常结束

        except (APIConnectionError, APITimeoutError) as e:
            last_error = e
            logger.warning("流式连接/超时 attempt=%d error=%s", attempt, e)
            if attempt <= MAX_RETRIES:
                time.sleep(1)

        except APIStatusError as e:
            last_error = e
            logger.error("流式 API 状态错误 status=%s error=%s", e.status_code, e)
            raise RuntimeError(f"流式调用失败：{e}") from e

        except Exception as e:
            logger.exception("流式调用未知错误")
            raise RuntimeError(f"流式调用失败：{e}") from e

    raise RuntimeError(f"流式调用失败：{last_error}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    answer = call_llm([{"role": "user", "content": "用一句话解释什么是函数"}])
    print(answer)