import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def main():
    # Ensure OpenAI API key is set
    if not st.secrets.get("OPENAI_API_KEY"):
        st.error("OpenAI API key is not set. Please configure it in Streamlit secrets.")
        return

    # Initialize ChatOpenAI with error handling
    try:
        llm = ChatOpenAI(temperature=0)
    except Exception as e:
        st.error(f"Failed to initialize ChatOpenAI: {e}")
        return

    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤗"
    )
    st.header("My Great ChatGPT 🤗")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    # User input handling
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        try:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("ChatGPT is typing ..."):
                response = llm(st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=response.content))
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Display chat history
    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")

if __name__ == '__main__':
    main()
