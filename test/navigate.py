import streamlit as st
from streamlit_navigation_bar import st_navbar
import os # ロゴファイルのパス指定に使う場合

# ナビゲーションバーに表示するページのリスト
pages = ["Home", "Documentation", "Examples", "Community", "About"]

# ナビゲーションバーのスタイル（任意）
# 詳細はドキュメントを参照: https://github.com/gabrieltempass/streamlit-navigation-bar
styles = {
    "nav": {
        "background-color": "rgb(70, 70, 70)", # 背景色を濃い灰色に
        "justify-content": "left", # 左寄せにする場合
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(230, 230, 230)", # テキストの色を明るい灰色に
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        # アクティブな（選択中の）ページの背景色を半透明の白に
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        # マウスホバー時の背景色を少し濃い半透明の白に
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}

# # --- ロゴを使う場合（任意） ---
# # スクリプトファイルからの相対パスでロゴファイルへのパスを取得
# # 'your_logo.png' がスクリプトと同じディレクトリにあると仮定
# script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
# logo_path = os.path.join(script_dir, 'your_logo.png') # ロゴファイルが存在するか確認

# ナビゲーションバーのオプション（任意）
options = {
    "show_menu": False, # Streamlitのデフォルトメニューを隠すか
    "show_sidebar": False, # Streamlitのデフォルトサイドバーを隠すか
    # "logo_path": logo_path, # ロゴファイルを使う場合はコメント解除
    # "logo_width": 25, # ロゴの幅
    # "use_padding": False, # カスタムスタイルでパディングを調整する場合はFalseに
}

# --- Streamlit Cloud での注意点 ---
# Streamlit Cloud 上でページの自動検出がうまくいかない場合、
# URLのクエリパラメータなどを利用して手動で `selected` を設定する必要があるかもしれません。
# query_params = st.experimental_get_query_params() # 現在は st.query_params
# default_page = "Home"
# current_page_from_query = st.query_params.get("page", default_page) # pageパラメータを取得、なければHome
# selected_page = current_page_from_query if current_page_from_query in pages else default_page

# ナビゲーションバーを表示
# selected=selected_page のように手動設定が必要な場合もある
page = st_navbar(
    pages,
    styles=styles,
    options=options
)

# 選択されたページ名を表示
st.write(f"選択されたページ: {page}")

# 選択されたページに応じてコンテンツを表示
if page == "Home":
    st.header("ホームページ")
    st.write("ようこそ！")
elif page == "Documentation":
    st.header("ドキュメント")
    st.write("ここでドキュメントを参照できます。")
elif page == "Examples":
    st.header("使用例")
    st.write("これらの使用例をご覧ください。")
elif page == "Community":
    st.header("コミュニティ")
    st.write("コミュニティフォーラムに参加しましょう。")
elif page == "About":
    st.header("概要")
    st.write("このプロジェクトについて学びます。")

st.info("ナビゲーションバーが表示されない、または期待通りに動作しない場合は、Streamlit Cloud のログを確認してください。")
st.warning("Streamlit Cloud でデプロイするには、`requirements.txt` ファイルに `streamlit-navigation-bar` が含まれていることを確認してください。")

# requirements.txt の内容例:
# streamlit
# streamlit-navigation-bar
