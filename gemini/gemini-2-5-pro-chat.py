import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import traceback

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    st.error("`langchain-google-genai` パッケージが見つかりません。")
    st.stop()

api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

if not api_key:
    st.error("GOOGLE_API_KEYが設定されていません。Streamlit Cloudを確認してください。")
    st.stop()

@st.cache_resource
def initialize_model(temperature, api_key):

    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro-exp-03-25",
            temperature=temperature,
            google_api_key=api_key 
        )

        print(f"モデルを temperature={temperature} で初期化しました。")
        return model
    except Exception as e:
        st.error(f"モデルの初期化中にエラーが発生しました: {e}")
        st.error("考えられる原因: 無効なAPIキー、指定されたモデル名へのアクセス権がない、Google Cloudの設定不備など。")
        st.stop()

def main():
    st.set_page_config(page_title="My Great Gemini 2.5 Pro", page_icon="🤗")
    st.header("My Great Gemini 2.5 Pro 🤗")

    temperature = st.sidebar.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.01,
        help="値が高いほどランダムに、低いほど一貫性のある内容になります。"
    )

    try:
        model = initialize_model(temperature, api_key)
    except Exception:
        return

    system_prompt = "あなたは親切で役立つアシスタントです。日本語で応答してください。"

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": system_prompt}]

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_input = st.chat_input("聞きたいことを入力してくださいね！")

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

        with st.spinner("Gemini 2.5 Pro is thinking..."):
            try:
                response = model.invoke(langchain_messages)

                if isinstance(response, AIMessage):
                    response_text = response.content
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
