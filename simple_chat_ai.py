import os
import tiktoken
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI

# .zshrc で設定された GROK_API_KEY を取得
grok_api_key = os.getenv("XAI_API_KEY")
if grok_api_key is None:
    st.error("XAI_API_KEYが環境変数に設定されていません。")
    st.stop()

###### dotenv を利用しない場合は消してください ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################

# 各モデルのトークン単価（仮の値）
MODEL_PRICES = {
    "input": {
        "o1": 0.5 / 1_000_000,
        "gemini-2.0-flash": 3.5 / 1_000_000,
        "grok-2-latest": 0.7 / 1_000_000,
        "claude-3-7-sonnet-latest": 3 / 1_000_000
    },
    "output": {
        "o1": 1.5 / 1_000_000,
        "gemini-2.0-flash": 10.5 / 1_000_000,
        "grok-2-latest": 2.0 / 1_000_000,
        "claude-3-7-sonnet-latest": 15 / 1_000_000
    }
}


def init_page():
    st.set_page_config(
        page_title="My Great Chat-LLMs",
        page_icon="🤗"
    )
    st.header("My Great Chat-LLMs 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    # clear_button が押された場合や message_history がまだ存在しない場合に初期化
    if clear_button or "message_history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "You are a helpful assistant.")
        ]


def select_model():
    # スライダーで temperature を選択 (0.0～2.0)
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

    # 新しい4つのモデルをラジオボタンで選択
    models = ("chatGPT o1", "Gemini 2.0 Flash", "grok-2", "Claude 3.7 Sonnet")
    model = st.sidebar.radio("Choose a model:", models)
    
    if model == "chatGPT o1":
        st.session_state.model_name = "o1"
        return ChatOpenAI(
            model_name=st.session_state.model_name
        )
    elif model == "Gemini 2.0 Flash":
        st.session_state.model_name = "gemini-2.0-flash"
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model=st.session_state.model_name
        )
    elif model == "grok-2":
        st.session_state.model_name = "grok-2-latest"
        return ChatXAI(
            temperature=temperature,
            model_name=st.session_state.model_name
        )
    elif model == "Claude 3.7 Sonnet":
        st.session_state.model_name = "claude-3-7-sonnet-latest"
        return ChatAnthropic(
            temperature=temperature,
            model_name=st.session_state.model_name
        )


def init_chain():
    st.session_state.llm = select_model()
    prompt = ChatPromptTemplate.from_messages([
        *st.session_state.message_history,
        ("user", "{user_input}")  # ユーザーの入力がここに入る
    ])
    output_parser = StrOutputParser()
    return prompt | st.session_state.llm | output_parser


def get_message_counts(text):
    # モデル名を小文字にして判定
    model_lower = st.session_state.model_name.lower()
    if "gemini" in model_lower:
        return st.session_state.llm.get_num_tokens(text)
    else:
        # Claude 3.7 Sonnet はトークナイザー非公開なので、tiktoken を利用
        if "gpt" in st.session_state.model_name or "grok" in st.session_state.model_name:
            encoding = tiktoken.encoding_for_model(st.session_state.model_name)
        else:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # 仮のエンコーディング
        return len(encoding.encode(text))

def get_message_counts(message):
    # ... (省略) ...
    # モデル名を取得 (st.session_state.model_name に grok-2-latest が設定されていることを前提)
    # model_name = st.session_state.model_name  # ← この行は不要になる可能性があります

    # エンコーディングを明示的に指定 (cl100k_base を例として使用)
    encoding = tiktoken.get_encoding("cl100k_base")  # または適切なエンコーディング名

    token_count = 0
    if isinstance(message, str):
        token_count = len(encoding.encode(message))
    elif isinstance(message, list):
        for item in message:
            if isinstance(item, dict) and "content" in item:
                token_count += len(encoding.encode(item["content"]))
    return token_count



def calc_and_display_costs():
    output_count = 0
    input_count = 0
    for role, message in st.session_state.message_history:
        token_count = get_message_counts(message)
        if role == "ai":
            output_count += token_count
        else:
            input_count += token_count

    # 初期状態（System Message のみ）の場合は API コールなし
    if len(st.session_state.message_history) == 1:
        return

    input_cost = MODEL_PRICES['input'][st.session_state.model_name] * input_count
    output_cost = MODEL_PRICES['output'][st.session_state.model_name] * output_count
    if "gemini" in st.session_state.model_name.lower() and (input_count + output_count) > 128000:
        input_cost *= 2
        output_cost *= 2

    cost = output_cost + input_cost

    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${cost:.5f}**")
    st.sidebar.markdown(f"- Input cost: ${input_cost:.5f}")
    st.sidebar.markdown(f"- Output cost: ${output_cost:.5f}")


def main():
    init_page()
    init_messages()
    chain = init_chain()

    # チャット履歴の表示
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)

    # ユーザー入力の取得
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.chat_message('user').markdown(user_input)

        # LLM の返答を Streaming 表示
        with st.chat_message('ai'):
            response = st.write_stream(chain.stream({"user_input": user_input}))

        # チャット履歴に追加
        st.session_state.message_history.append(("user", user_input))
        st.session_state.message_history.append(("ai", response))

    # コストの計算と表示
    calc_and_display_costs()


if __name__ == '__main__':
    main()