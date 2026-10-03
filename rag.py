import json
from pathlib import Path

import numpy as np
import ollama

EMBED_MODEL = "bge-m3"
CHAT_MODEL = "llama3.2:3b"
INDEX_DIR = Path("index")
TOP_K = 3  # quantos pedaços usar como contexto

INSTRUCAO = (
    "Você é o assistente virtual da NovaLar Móveis. "
    "Responda sempre em português, de forma clara e objetiva, "
    "usando APENAS as informações do contexto fornecido. "
    "Se a resposta não estiver no contexto, diga exatamente: "
    "'Não encontrei essa informação nos documentos da NovaLar. "
    "Entre em contato com o SAC.' "
    "Não invente valores, prazos ou regras."
)


def carregar_indice():
    vetores = np.load(INDEX_DIR / "embeddings.npy")
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return vetores, chunks


def buscar(pergunta, vetores, chunks, k=TOP_K):
    """Devolve os k pedaços mais parecidos com a pergunta."""
    r = ollama.embed(model=EMBED_MODEL, input=pergunta)
    q = np.array(r["embeddings"][0], dtype=np.float32)
    q = q / np.linalg.norm(q)

    similaridades = vetores @ q  # vetores já estão normalizados
    melhores = np.argsort(similaridades)[::-1][:k]
    return [{**chunks[i], "score": float(similaridades[i])} for i in melhores]


def responder(pergunta, vetores, chunks):
    """Busca o contexto e pede a resposta ao LLM."""
    trechos = buscar(pergunta, vetores, chunks)
    contexto = "\n\n---\n\n".join(t["texto"] for t in trechos)

    mensagens = [
        {"role": "system", "content": INSTRUCAO},
        {
            "role": "user",
            "content": f"Contexto:\n{contexto}\n\nPergunta: {pergunta}",
        },
    ]
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=mensagens,
        options={"temperature": 0},
    )
    return resp["message"]["content"], trechos


if __name__ == "__main__":
    vetores, chunks = carregar_indice()
    print("Chatbot NovaLar (digite 'sair' para encerrar)\n")
    while True:
        pergunta = input("Você: ").strip()
        if pergunta.lower() == "sair":
            break
        if not pergunta:
            continue
        resposta, trechos = responder(pergunta, vetores, chunks)
        print(f"\nBot: {resposta}")
        fontes = sorted({t["fonte"] for t in trechos})
        print(f"[Fontes: {', '.join(fontes)}]\n")