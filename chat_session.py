from llm_client import call_llm, call_llm_stream
from collections.abc import Iterable


class ChatSession:
    """保存一次聊天会话中的所有消息。"""

    def __init__(self, system_prompt: str = "你是一个简洁、准确的中文助手。"):
        self.messages = [
            {"role": "system", "content": system_prompt},
        ]

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def get_messages(self) -> list[dict]:
        return self.messages

    def send(self, user_input: str) -> str:
        """记录用户输入 -> 调模型 -> 记录回答。失败时回滚用户消息。"""
        self.add_user_message(user_input)
        try:
            answer = call_llm(self.messages)
        except Exception:
            self.messages.pop()  # 失败时把刚加的用户消息撤掉，避免历史里留下“没有回答”的一轮
            raise
        self.add_assistant_message(answer)
        return answer

    def send_stream(self, user_input: str) -> Iterable[str]:
        """流式发送：注会产出文本，同时累加完整回答到历史"""
        self.add_user_message(user_input)
        full_answer = ''

        try:
            for token in call_llm_stream(self.messages):
                full_answer += token
                yield token
        except Exception:
            self.messages.pop()  # 失败时回滚用户消息
            raise
        # 流式正常结束后，把完整回答加入历史
        self.add_assistant_message(full_answer)