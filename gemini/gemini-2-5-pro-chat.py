import os
import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Google API Keyの設定
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEYが設定されていません。Streamlit CloudにAPIキーが登録されているか確認してください。")
    st.stop()

genai.configure(api_key=api_key)

# Gemini モデルの指定
MODEL_ID = "gemini-2.5-pro-exp-03-25"
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 2048,
}

safety_settings=None

def main():
    # Streamlitページ設定
    st.set_page_config(page_title="My Great Gemini 2.5 Pro", page_icon="🤗")
    st.header("My Great Gemini 2.5 Pro 🤗")

    # システムプロンプトを関数の先頭で定義
    system_prompt = "You are a helpful assistant."

    # チャット履歴初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # チャット履歴を画面に表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザー入力の受付
    user_input = st.chat_input("質問を入力してください")
    if user_input:
        # ユーザーメッセージをチャット履歴と画面に追加
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Geminiモデルの準備
        model = genai.GenerativeModel(
            model_name="gemini-2.5-pro-exp-03-25",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        # チャット履歴を適切な形式に変換
        chat_history = []
        for msg in st.session_state.messages:
            chat_history.append({"role": msg["role"], "parts": [msg["content"]]})

        # チャットセッションを開始
        chat = model.start_chat(history=chat_history)

        # システムプロンプトをコンテキストとして追加
        context_with_prompt = f"{system_prompt}\n\n{user_input}"

        # AIの応答を取得
        with st.spinner("Gemini is typing ..."):
            try:
                response = chat.send_message(context_with_prompt)
                response_text = response.text

                # AIの応答をチャット履歴と画面に追加
                st.session_state.messages.append({"role": "model", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")

# メイン関数を実行
if __name__ == "__main__":
    main()
