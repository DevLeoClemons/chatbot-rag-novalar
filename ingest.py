import json
from pathlib import Path

import numpy as np
import ollama

EMBED_MODEL = "bge-m3"
DATA_DIR = Path("data")
INDEX_DIR = Path("index")
MAX_CHARS = 800   # tamanho máximo de um pedaço
OVERLAP = 100     # sobreposição entre pedaços longos


def dividir_texto_longo(texto):
    """Divide um texto grande em pedaços com sobreposição."""
    pedacos = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + MAX_CHARS
        pedacos.append(texto[inicio:fim])
        if fim >= len(texto):
            break
        inicio = fim - OVERLAP
    return pedacos


def criar_chunks(caminho):
    """Lê um documento e devolve uma lista de pedaços (um por seção)."""
    texto = caminho.read_text(encoding="utf-8")
    titulo = texto.splitlines()[0].lstrip("# ").strip()
    secoes = texto.split("\n## ")[1:]  # o primeiro item é só o título

    chunks = []
    for secao in secoes:
        conteudo = f"{titulo}\n{secao.strip()}"
        for pedaco in dividir_texto_longo(conteudo):
            chunks.append({"fonte": caminho.name, "texto": pedaco})
    return chunks


def main():
    todos_chunks = []
    for caminho in sorted(DATA_DIR.glob("*.md")):
        chunks = criar_chunks(caminho)
        print(f"{caminho.name}: {len(chunks)} pedaços")
        todos_chunks.extend(chunks)

    print(f"\nTotal: {len(todos_chunks)} pedaços. Gerando embeddings...")
    textos = [c["texto"] for c in todos_chunks]
    
    embeddings = []
    for i, texto in enumerate(textos, start=1):
        r = ollama.embed(model=EMBED_MODEL, input=texto)
        embeddings.append(r["embeddings"][0])
        print(f"  embedding {i}/{len(textos)}")

    vetores = np.array(embeddings, dtype=np.float32)

    # Normaliza os vetores: assim, produto escalar = similaridade de cosseno
    vetores = vetores / np.linalg.norm(vetores, axis=1, keepdims=True)

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", vetores)
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(todos_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Pronto! Índice salvo em '{INDEX_DIR}/' com vetores de tamanho {vetores.shape[1]}.")


if __name__ == "__main__":
    main()