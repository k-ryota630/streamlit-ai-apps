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
    st.sidebar.title("モデル設定")

    temperature = st.sidebar.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        help="値が高いほど創造的でランダムな応答に、低いほど決定的で一貫性のある応答になります。"
    )

    models = ("chatGPT 4.1", "Gemini 2.5 Pro", "grok-3 mini", "Claude 3.7 Sonnet")
    model_choice = st.sidebar.radio(
        "Choose a model:",
        models,
        index=0
    )

    model = None
    error_message = None

    try:
        if model_choice == "chatGPT 4.1":
            if not openai_api_key:
                error_message = "OpenAIのAPIキーが設定されていません。"
            else:
                st.session_state.model_name = "gpt-4.1-2025-04-14"
                model = ChatOpenAI(
                    model_name=st.session_state.model_name,
                    temperature=temperature,
                    api_key=openai_api_key
                )
        elif model_choice == "Gemini 2.5 Pro":
            if not google_api_key:
                error_message = "GoogleのAPIキーが設定されていません。"
            else:
                st.session_state.model_name = "gemini-2.5-pro-exp-03-25"
                model = ChatGoogleGenerativeAI(
                    model=st.session_state.model_name,
                    temperature=temperature,
                    google_api_key=google_api_key
                )
        elif model_choice == "grok-3 mini":
            if not xai_api_key:
                error_message = "XAIのAPIキーが設定されていません。"
            else:
                st.session_state.model_name = "grok-3-mini-fast-beta"
                model = ChatXAI(
                    model_name=st.session_state.model_name,
                    temperature=temperature,
                    api_key=xai_api_key
                )
        elif model_choice == "Claude 3.7 Sonnet":
            if not anthropic_api_key:
                error_message = "AnthropicのAPIキーが設定されていません。"
            else:
                st.session_state.model_name = "claude-3-7-sonnet-latest"
                model = ChatAnthropic(
                    model_name=st.session_state.model_name,
                    temperature=temperature,
                    anthropic_api_key=anthropic_api_key
                )

        if error_message:
            st.sidebar.error(error_message)
            return None
        else:
            st.session_state.selected_model_display_name = model_choice
            st.sidebar.success(f"モデル '{st.session_state.model_name}' を選択しました。")
            return model

    except Exception as e:
        st.sidebar.error(f"モデルの初期化中にエラーが発生しました: {e}")
        st.sidebar.error(traceback.format_exc())
        return None

def main():
    st.set_page_config(page_title="My Great LLM's", page_icon="🤗")
    st.header("My Great LLM's 🤗")

    model = select_model()

    if model is None:
        st.warning("サイドバーでモデルを選択し、必要なAPIキーが設定されているか確認してください。")
        st.stop()

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

        with st.spinner("LLM is thinking..."):
            try:
                response = model.invoke(langchain_messages)

                if isinstance(response, AIMessage):
                    response_text = response.content
                elif isinstance(response, str):
                     response_text = response
                else:
                    st.error(f"予期しない応答形式を受け取りました: {type(response)}")
                    response_text = "エラー: 応答の解析に失敗しました。予期しない形式です。"
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
