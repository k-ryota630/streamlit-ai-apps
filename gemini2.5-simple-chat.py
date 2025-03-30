import os
import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel

# Vertex AI初期化
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Gemini 2.5モデルの読み込み
MODEL_ID = "gemini-2.5-pro-exp-03-25"
model = GenerativeModel(MODEL_ID)

# Streamlitページ設定
st.set_page_config(page_title="Gemini 2.5 Chatbot", page_icon="🤗")
st.header("Gemini 2.5 Chatbot 🤗")

# チャット履歴初期化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "あなたは親切で優秀なAIアシスタントです。"}
    ]

# Geminiのチャットセッションを初期化
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[message["content"] for message in st.session_state.chat_history])

# チャット履歴を画面に表示
for message in st.session_state.chat_history:
    if message["role"] != "system":
        st.chat_message(message["role"]).markdown(message["content"])

# ユーザー入力の受付
user_input = st.chat_input("質問を入力してください")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Geminiが考え中..."):
        response = st.session_state.chat.send_message(user_input)

    # レスポンスからテキストを取得
    response_text = response.text if not isinstance(response, list) else response[0].text

    st.chat_message("assistant").markdown(response_text)
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
