import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_xai import ChatXAI
import traceback

openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
xai_api_key = st.secrets.get("XAI_API_KEY", os.getenv("XAI_API_KEY"))


def select_model():
    temperature = st.sidebar.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        help="値が大きいとランダムに、小さいと真面目になります。"
    )

    available_models = {
        "ChatGPT 4.1": "gpt-4.1-2025-04-14",
        "Gemini 2.5 Pro": "gemini-2.5-pro-exp-03-25",
        "Claude 3.7 Sonnet": "claude-3-7-sonnet-latest",
        "Grok-3 Mini": "grok-3-mini-fast-beta"
    }
    model_display_name = st.sidebar.radio(
        "Choose a model:",
        list(available_models.keys()),
        index=0,
        help="OpenAI、Google、XAI、Anthropicの最新モデルから選べます。"
    )

    model_name = available_models[model_display_name]
    st.session_state.model_name = model_name

    model = None
    error_message = None

    try:
        if model_display_name == "ChatGPT 4.1":
            if not openai_api_key:
                error_message = "OpenAI APIキーが設定されていません。"
            else:
                model = ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    api_key=openai_api_key
                )
        elif model_display_name == "Gemini 2.5 Pro":
            if not google_api_key:
                error_message = "Google APIキーが設定されていません。"
            else:
                model = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    google_api_key=google_api_key,
                    convert_system_message_to_human=True
                )
        elif model_display_name == "Grok-3 Mini":
             if not xai_api_key:
                 error_message = "XAI APIキーが設定されていません。"
             else:
                 model = ChatXAI(
                     model_name=model_name,
                     temperature=temperature,
                     api_key=xai_api_key
                 )
        elif model_display_name == "Claude 3.7 Sonnet":
            if not anthropic_api_key:
                error_message = "Anthropic APIキーが設定されていません。"
            else:
                model = ChatAnthropic(
                    model_name=model_name,
                    temperature=temperature,
                    anthropic_api_key=anthropic_api_key
                )

    except Exception as e:
        error_message = f"モデルの初期化中にエラーが発生しました: {e}"
        st.sidebar.error(error_message)
        st.sidebar.error(traceback.format_exc())
        model = None

    if error_message:
        st.sidebar.error(error_message)

    return model, error_message

def main():
    st.set_page_config(page_title="My Great LLM's", page_icon="🤗")
    st.header("My Great LLM's 🤗")

    model, error_message = select_model()

    if error_message:
        st.error(f"モデルの準備ができませんでした: {error_message}")
        st.warning("サイドバーで別のモデルを選択するか、APIキーの設定を確認してください。")
        return
    if model is None:
        st.error("モデルオブジェクトが正常に作成されませんでした。原因不明のエラーです。")
        return

    system_prompt = "You are a helpful assistant."

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": system_prompt}]

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_input = st.chat_input("聞きたいことを入力してね！")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        langchain_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "system":
                langchain_messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        with st.spinner(f"{st.session_state.model_name} is thinking..."):
            try:
                response = model.invoke(langchain_messages)

                if isinstance(response, AIMessage):
                    response_text = response.content
                elif isinstance(response, str):
                     response_text = response
                else:
                    st.error(f"予期しない応答形式を受け取りました: {type(response)}")
                    response_text = f"エラー: 応答の解析に失敗しました。形式: {type(response)}"
                    st.error(f"応答内容: {response}")

                st.session_state.messages.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)

            except Exception as e:
                st.error(f"応答の生成中にエラーが発生しました: {e}")
                st.error("詳細情報:")
                st.error(traceback.format_exc())

if __name__ == "__main__":
    main()
