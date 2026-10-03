from rag import carregar_indice, responder

# Cada teste: (pergunta, grupos de palavras esperadas na resposta)
# Passa se, para CADA grupo, pelo menos UMA das alternativas aparecer.
TESTES = [
    ("Em quantos dias posso desistir de uma compra online?", [["7 dias"]]),
    ("Quanto tempo tenho para trocar um produto com defeito?", [["30 dias"]]),
    ("Em quanto tempo recebo o estorno no cartão de crédito?", [["10 dias úteis"]]),
    ("Em quanto tempo o PIX é reembolsado?", [["3 dias úteis"]]),
    ("A partir de qual valor o frete é grátis no Distrito Federal?", [["1.500", "1500"]]),
    ("A montagem é gratuita para produtos acima de qual valor?", [["800"]]),
    ("Quantas tentativas de entrega a transportadora faz?", [["duas", "2 tentativas"]]),
    ("Qual a garantia de um colchão?", [["5 anos"]]),
    ("Qual a garantia da estrutura de um sofá?", [["3 anos"]]),
    ("Em quantas vezes posso parcelar uma compra de R$ 600?", [["10 vezes", "10x", "10 parcelas", "dez"]]),
    ("Qual o desconto para pagamento no boleto?", [["3%"]]),
    ("Que horas as lojas abrem aos domingos?", [["10h", "10:00", "10 horas"]]),
    ("Quantos dias tenho para retirar o pedido na loja?", [["3 dias"]]),
    ("Em quanto tempo o técnico visita após a confirmação do defeito?", [["7 dias úteis"]]),
    # Perguntas fora do escopo: o bot deve recusar
    ("Vocês vendem geladeiras?", [["não encontrei"]]),
    ("Qual o CNPJ da empresa?", [["não encontrei"]]),
    ("Qual é a capital da França?", [["não encontrei"]]),
]


def passou(resposta, grupos):
    r = resposta.lower()
    return all(any(alt.lower() in r for alt in grupo) for grupo in grupos)


def main():
    vetores, chunks = carregar_indice()
    acertos = 0
    for i, (pergunta, grupos) in enumerate(TESTES, start=1):
        resposta, _ = responder(pergunta, vetores, chunks)
        ok = passou(resposta, grupos)
        acertos += ok
        print(f"[{'OK' if ok else 'FALHOU'}] {i}. {pergunta}")
        if not ok:
            print(f"      Resposta do bot: {resposta}")
    print(f"\nResultado: {acertos}/{len(TESTES)} testes passaram.")


if __name__ == "__main__":
    main()