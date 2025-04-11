import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    st.error("`langchain_openai` が見つかりません。インストールしてください。")
    st.stop()

# --- APIキーが登録されているか確認する ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    if not api_key:
        st.error("GOOGLE_API_KEYが設定されていません。Streamlit Cloudの設定を確認してください。")
        st.stop()

@st.cache_resource 
def initialize_model(temperature, api_key):
    try:
        model = ChatGoogleGenerativeAI(
            model_name="gemini-2.5-pro-exp-03-25",
            temperature=temperature,
            api_key=api_key
        )
        print(f"モデルを temperature={temperature} で初期化しました。")
        return model
    except Exception as e:
        st.error(f"モデルの初期化中にエラーが発生しました: {e}")
        st.stop()

def main():
    st.set_page_config(page_title="My Great Gemini 2.5 Pro", page_icon="🤗")
    st.header("My Great Gemini 2.5 Pro 🤗")

    # --- サイドバーの設定項目 ---
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=0.7, step=0.01
    )

    model = initialize_model(temperature, api_key)

    system_prompt = "You are a helpful assistant."

    # --- チャット履歴の管理 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- 履歴の表示 ---
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    user_input = st.chat_input("聞きたいことを入力してね！")
    if user_input:
        # メッセージを履歴と画面に追加
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Gemini 2.5 Pro is thinking..."):
            try:
                langchain_messages = []
                # システムプロンプトをメッセージリストの先頭に追加
                langchain_messages.append(SystemMessage(content=system_prompt))

                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))

                # --- Gemini 2.5 Pro LLMの呼び出し
                response = model.invoke(langchain_messages)

                if isinstance(response, AIMessage):
                    response_text = response.content
                elif isinstance(response, str):
                    response_text = response
                elif hasattr(response, 'text'):
                    response_text = response.text
                else:
                    st.error(f"予期しない応答形式です: {type(response)}")
                    response_text = "エラー: 応答の解析に失敗しました。"

                # 応答を履歴と画面に追加 (roleは'assistant')
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)

            except Exception as e:
                st.error(f"応答の生成中にエラーが発生しました: {e}")
                import traceback
                st.error(traceback.format_exc()) 

if __name__ == "__main__":
    main()
