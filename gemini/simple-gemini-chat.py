# main.py
import streamlit as st
import os # 環境変数を読み込むために必要
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import LangChainException
# google.api_core.exceptions もインポートしておくと、より具体的なエラーハンドリングが可能
# from google.api_core.exceptions import PermissionDenied, Unauthenticated

###### dotenv を利用するのはローカル開発時のみ ######
# Streamlit Cloud では Secrets を使用するため、dotenv は必須ではない
# ローカルで .env ファイルを使いたい場合は、この部分を残す
try:
    from dotenv import load_dotenv
    load_dotenv()
    # print("Loaded .env file (running locally)") # ローカル実行確認用
except ImportError:
    # dotenv がインストールされていない、または不要な環境 (Streamlit Cloudなど)
    # print("dotenv not found or not needed, skipping.")
    pass
################################################


def main():
    st.set_page_config(
        page_title="Gemini Chat on Streamlit Cloud", # ページタイトル変更
        page_icon="☁️" # ページアイコン変更
    )
    st.header("Gemini Chat on Streamlit Cloud ☁️") # ヘッダー変更

    # --- Streamlit Cloud Secrets または 環境変数からAPIキーを取得 ---
    # Streamlit Cloud の Secrets で設定した GOOGLE_API_KEY が環境変数として読み込まれる
    google_api_key = os.getenv("GOOGLE_API_KEY")

    # APIキーが設定されていない場合のエラー表示と処理停止
    if not google_api_key:
        st.error("⚠️ **エラー:** Google APIキーが設定されていません。")
        st.warning("Streamlit Cloudのアプリ設定 > Secrets に `GOOGLE_API_KEY = 'YOUR_API_KEY'` を追加してください。")
        st.info("APIキーは Google AI Studio (https://aistudio.google.com/app/apikey) で取得できます。")
        st.stop() # APIキーがない場合はアプリを停止

    # チャット履歴の初期化
    if "message_history" not in st.session_state:
        st.session_state.message_history = [
            ("system", "You are a helpful assistant.")
        ]

    try:
        # LLMモデルをGeminiに設定
        # APIキーは環境変数 GOOGLE_API_KEY から自動的に読み込まれる
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, convert_system_message_to_human=True)

        # プロンプトテンプレートの作成
        prompt = ChatPromptTemplate.from_messages([
            *st.session_state.message_history,
            ("user", "{user_input}")
        ])

        # 出力パーサー
        output_parser = StrOutputParser()

        # チェーンの作成
        chain = prompt | llm | output_parser

        # ユーザー入力
        if user_input := st.chat_input("聞きたいことを入力してね！"):
            with st.spinner("Gemini is thinking ..."):
                try:
                    # LLMを実行
                    response = chain.invoke({"user_input": user_input})

                    # 履歴に追加
                    st.session_state.message_history.append(("user", user_input))
                    st.session_state.message_history.append(("ai", response))

                # --- APIキー関連やその他のエラーハンドリング ---
                except (LangChainException) as e: # 必要に応じて他の例外 (PermissionDeniedなど) も捕捉
                    error_message = str(e)
                    # APIキーが無効、権限不足、認証失敗などのエラーメッセージをチェック
                    if "api key not valid" in error_message.lower() or \
                       "permission denied" in error_message.lower() or \
                       "unauthenticated" in error_message.lower() or \
                       "api_key" in error_message.lower(): # APIキー関連のエラーと推測される場合
                        st.error(f"🚫 **APIキーエラー:** {error_message}")
                        st.warning("Google APIキーが正しいか、Streamlit CloudのSecretsの設定を確認してください。")
                    else:
                        st.error(f"🤖 **LLMエラー:** 予期せぬエラーが発生しました。")
                        st.exception(e) # 詳細なエラー情報を表示
                except Exception as e: # その他の予期せぬエラー
                    st.error(f"💥 **システムエラー:** 予期せぬエラーが発生しました。")
                    st.exception(e) # 詳細なエラー情報を表示

        # チャット履歴の表示
        for role, message in st.session_state.get("message_history", []):
            if role != "system": # Systemメッセージは表示しない
                with st.chat_message(role):
                    st.markdown(message)

    except Exception as e:
        st.error(f"🚨 アプリケーションの初期化中にエラーが発生しました。")
        st.exception(e)


if __name__ == '__main__':
    main()
