import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

###### dotenv を利用しない場合は消してください ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################

def main():
    st.set_page_config(
        page_title="My Great ChatGPT o3-mini",
        page_icon="🤗"
    )
    st.header("My Great ChatGPT o3-mini 🤗")

    # チャット履歴の初期化
    if "message_history" not in st.session_state:
        st.session_state.message_history = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

    # LLMの設定
    try:
        llm = ChatOpenAI(
            model_name="o3-mini-2025-01-31",
            temperature=1.0,
            max_tokens=1000
        )

        # プロンプトテンプレートの作成
        prompt = ChatPromptTemplate.from_messages([
            *st.session_state.message_history,
            {"role": "user", "content": "{user_input}"}
        ])

        # 出力パーサーの作成
        output_parser = StrOutputParser()

        # チェーンの作成
        chain = prompt | llm | output_parser

        # ユーザーの入力を監視
        if user_input := st.chat_input("聞きたいことを入力してね！"):
            with st.spinner("ChatGPT is typing ..."):
                response = chain.invoke({"user_input": user_input})

            # 履歴に追加
            st.session_state.message_history.append({"role": "user", "content": user_input})
            st.session_state.message_history.append({"role": "assistant", "content": response})

        # チャット履歴の表示
        for message in st.session_state.message_history:
            st.chat_message(message["role"]).markdown(message["content"])

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")

if __name__ == '__main__':
    main()
