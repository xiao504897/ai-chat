import logging
from chat_session import ChatSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(leveltime)s] %(name)s: %(message)s",
)

# def main() -> None:
#     # user_input = input("你：")
#     # messages = build_messages(user_input)
#     # answer = call_llm(messages)
#     # print(f"助手：{answer}")
#     session = ChatSession()

#     # 第一轮
#     user_input_1 = input("你：")
#     answer_1 = session.send(user_input_1)
#     print(f"助手：{answer_1}")

#     #第二轮
#     user_input_2 = input("你：")
#     answer_2 = session.send(user_input_2)
#     print(f"助手：{answer_2}")

#     # 打印历史，确认消息顺序正确
#     print('\n, ---历史消息---')
#     for message in session.get_messages():
#         print(f'[{message["role"]}]: {message["content"]}')

def main() -> None:
    session = ChatSession()

    user_input_1 = input("你：")
    print("助手：", end="", flush=True)
    for token in session.send_stream(user_input_1):
        print(token, end="", flush=True)
    print()  # 换行

    user_input_2 = input("你：")
    print("助手：", end="", flush=True)
    for token in session.send_stream(user_input_2):
        print(token, end="", flush=True)
    print()

    print("\n--- 历史消息 ---")
    for msg in session.get_messages():
        print(f"[{msg['role']}] {msg['content']}")


if __name__ == "__main__":
    main()