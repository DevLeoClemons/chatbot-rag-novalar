import streamlit as st

from rag import carregar_indice, responder

st.set_page_config(page_title="Assistente NovaLar", page_icon="🛋️")
st.title("🛋️ Assistente NovaLar Móveis")
st.caption("Tire dúvidas sobre trocas, entrega, garantia e pagamento.")


@st.cache_resource
def get_indice():
    return carregar_indice()


def mostrar_fontes(fontes):
    with st.expander("Ver fontes"):
        for f in fontes:
            st.markdown(f"**{f['fonte']}**")
            st.text(f["texto"])


vetores, chunks = get_indice()

if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Mostra o histórico da conversa
for m in st.session_state.mensagens:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("fontes"):
            mostrar_fontes(m["fontes"])

# Caixa de entrada
pergunta = st.chat_input("Digite sua pergunta...")

if pergunta:
    st.session_state.mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando nos documentos..."):
            resposta, trechos = responder(pergunta, vetores, chunks)
        st.markdown(resposta)

        # Se o bot não encontrou a resposta, não mostra fontes
        fontes = [] if resposta.startswith("Não encontrei") else trechos
        if fontes:
            mostrar_fontes(fontes)

    st.session_state.mensagens.append(
        {"role": "assistant", "content": resposta, "fontes": fontes}
    )