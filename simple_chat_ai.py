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

# dotenv が利用可能なら環境変数を読み込む
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 各モデルのトークン単価
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

# Streamlit Secrets 管理
def get_api_keys():
    """環境変数または Streamlit secrets から API キーを取得する"""
    api_keys = {}
    
    # まずは Streamlit secrets から取得（Streamlit Cloud 用）
    try:
        api_keys["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except (KeyError, TypeError):
        api_keys["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    try:
        api_keys["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except (KeyError, TypeError):
        api_keys["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
    try:
        api_keys["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, TypeError):
        api_keys["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
    
    try:
        api_keys["XAI_API_KEY"] = st.secrets["XAI_API_KEY"]
    except (KeyError, TypeError):
        api_keys["XAI_API_KEY"] = os.getenv("XAI_API_KEY")
    
    return api_keys

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
    api_keys = get_api_keys()
    
    # スライダーで temperature を選択 (0.0～2.0)
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

    # 新しい4つのモデルをラジオボタンで選択
    models = ("chatGPT o1", "Gemini 2.0 Flash", "grok-2", "Claude 3.7 Sonnet")
    model = st.sidebar.radio("Choose a model:", models)
    
    if model == "chatGPT o1":
        st.session_state.model_name = "o1"
        if not api_keys.get("OPENAI_API_KEY"):
            st.sidebar.error("OpenAI API key is missing. Please add it to your environment variables or Streamlit secrets.")
            return None
        return ChatOpenAI(
            model_name=st.session_state.model_name,
            temperature=temperature
        )
    elif model == "Gemini 2.0 Flash":
        st.session_state.model_name = "gemini-2.0-flash"
        if not api_keys.get("GOOGLE_API_KEY"):
            st.sidebar.error("Google API key is missing. Please add it to your environment variables or Streamlit secrets.")
            return None
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model=st.session_state.model_name
        )
    elif model == "grok-2":
        st.session_state.model_name = "grok-2-latest"
        if not api_keys.get("XAI_API_KEY"):
            st.sidebar.error("XAI API key is missing. Please add it to your environment variables or Streamlit secrets.")
            return None
        return ChatXAI(
            temperature=temperature,
            model_name=st.session_state.model_name
        )
    elif model == "Claude 3.7 Sonnet":
        st.session_state.model_name = "claude-3-7-sonnet-latest"
        if not api_keys.get("ANTHROPIC_API_KEY"):
            st.sidebar.error("Anthropic API key is missing. Please add it to your environment variables or Streamlit secrets.")
            return None
        return ChatAnthropic(
            temperature=temperature,
            model_name=st.session_state.model_name
        )


def init_chain():
    llm = select_model()
    if not llm:
        st.warning("Please configure the API key for the selected model.")
        st.stop()
        
    st.session_state.llm = llm
    prompt = ChatPromptTemplate.from_messages([
        *st.session_state.message_history,
        ("user", "{user_input}")  # ユーザーの入力がここに入る
    ])
    output_parser = StrOutputParser()
    return prompt | st.session_state.llm | output_parser


def get_message_counts(message):
    """
    現在のモデルに基づいて適切な方法でテキスト中のトークン数をカウントします。
    トークン数を整数で返します。
    """
    try:
        model_name = st.session_state.model_name.lower()
        
        # Geminiモデルの場合、内蔵のトークンカウンターを使用
        if "gemini" in model_name and hasattr(st.session_state.llm, "get_num_tokens"):
            return st.session_state.llm.get_num_tokens(message)
        
        # その他のモデルの場合、tiktoken を適切なエンコーディングで使用
        encoding_name = "cl100k_base"  # 新しいモデル用のデフォルト
        
        try:
            if "gpt" in model_name or "o1" in model_name:
                # OpenAIモデルの場合
                try:
                    encoding = tiktoken.encoding_for_model(model_name)
                except KeyError:
                    encoding = tiktoken.get_encoding(encoding_name)
            else:
                # Grok, Claude 等はデフォルトで cl100k_base を使用
                encoding = tiktoken.get_encoding(encoding_name)
                
            # テキストの型に応じたトークンカウント
            token_count = 0
            if isinstance(message, str):
                token_count = len(encoding.encode(message))
            elif isinstance(message, list):
                for item in message:
                    if isinstance(item, dict) and "content" in item:
                        token_count += len(encoding.encode(item["content"]))
                    elif isinstance(item, str):
                        token_count += len(encoding.encode(item))
            return token_count
        except Exception:
            # tiktoken エラー時は単語数 * 1.3 で概算（応急措置）
            if isinstance(message, str):
                return int(len(message.split()) * 1.3)
            return 0
    except Exception as e:
        st.sidebar.warning(f"Token counting error: {str(e)}")
        return 0  # エラー発生時は0を返し、アプリケーションのクラッシュを防止


def calc_and_display_costs():
    if not hasattr(st.session_state, "model_name"):
        return
        
    output_count = 0
    input_count = 0
    for role, message in st.session_state.message_history:
        token_count = get_message_counts(message)
        if role == "ai":
            output_count += token_count
        else:
            input_count += token_count

    # 初期状態（System Message のみ）の場合は API コールなし
    if len(st.session_state.message_history) <= 1:
        return

    # モデル名が存在するか確認
    if st.session_state.model_name not in MODEL_PRICES['input']:
        st.sidebar.warning(f"Price info for {st.session_state.model_name} not available")
        return
        
    input_cost = MODEL_PRICES['input'][st.session_state.model_name] * input_count
    output_cost = MODEL_PRICES['output'][st.session_state.model_name] * output_count
    
    # Gemini 128K+ token pricing adjustment
    if "gemini" in st.session_state.model_name.lower() and (input_count + output_count) > 128000:
        input_cost *= 2
        output_cost *= 2

    cost = output_cost + input_cost

    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${cost:.5f}**")
    st.sidebar.markdown(f"- Input cost: ${input_cost:.5f}")
    st.sidebar.markdown(f"- Output cost: ${output_cost:.5f}")
    st.sidebar.markdown(f"- Input tokens: {input_count}")
    st.sidebar.markdown(f"- Output tokens: {output_count}")


def main():
    init_page()
    init_messages()
    
    # サイドバーに API キーの状態を表示
    st.sidebar.markdown("## API Status")
    api_keys = get_api_keys()
    for key_name, key_value in api_keys.items():
        if key_value:
            st.sidebar.success(f"✅ {key_name} configured")
        else:
            st.sidebar.error(f"❌ {key_name} missing")
    
    # 必要に応じて、Streamlit Cloud での API キー設定方法の説明を表示
    if not all(api_keys.values()):
        with st.expander("How to configure API keys in Streamlit Cloud"):
            st.markdown("""
            To add your API keys to Streamlit Cloud:
            1. Go to your app dashboard in Streamlit Cloud
            2. Navigate to the app settings
            3. Under 'Secrets', add your API keys in this format:
            ```
            [secrets]
            OPENAI_API_KEY = "your-key-here"
            ANTHROPIC_API_KEY = "your-key-here"
            GOOGLE_API_KEY = "your-key-here"
            XAI_API_KEY = "your-key-here"
            ```
            4. Save and redeploy your app
            """)
    
    # チャットチェーンの初期化（進められる場合のみ）
    try:
        chain = init_chain()
    except Exception:
        st.warning("Please select a model and ensure its API key is configured.")
        return

    # チャット履歴の表示
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)

    # ユーザー入力の取得
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.chat_message('user').markdown(user_input)

        try:
            # LLM の返答を Streaming 表示
            with st.chat_message('ai'):
                response = st.write_stream(chain.stream({"user_input": user_input}))

            # チャット履歴に追加
            st.session_state.message_history.append(("user", user_input))
            st.session_state.message_history.append(("ai", response))
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")

    # コストの計算と表示
    calc_and_display_costs()


if __name__ == '__main__':
    main()
