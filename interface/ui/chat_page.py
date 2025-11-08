"""
Interface for chat page with streamlit.
"""
import streamlit as st
from services.rag_chat import RAGChatService

WELCOME = """Bonjour, je suis l'assistant virtuel de la mairie.
    Comment puis-je vous aider aujourd'hui?"""


class ChatPage:
    def __init__(self, service: RAGChatService, title: str):
        self.service = service
        self.title = title

    def _init_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant",
                                          "content": WELCOME}]

    def render(self):
        st.set_page_config(page_title=self.title, page_icon="🏛️")
        st.title(self.title)
        self._init_state()

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        if prompt := st.chat_input("Comment puis-je vous aider ?"):
            st.session_state.messages.append({"role": "user",
                                              "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                ph = st.empty()
                ph.markdown("_En cours..._")
                try:
                    reply = self.service.answer(
                        history=st.session_state.messages, question=prompt)
                except Exception as e:
                    reply = f"Erreur lors de la génération de la réponse : {e}"
                ph.write(reply)

            st.session_state.messages.append({"role": "assistant",
                                              "content": reply})
