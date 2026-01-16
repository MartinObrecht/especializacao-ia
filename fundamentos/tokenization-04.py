import nltk
import numpy as np
from rank_bm25 import BM25Okapi


# Lista de documentos sobre machine learning em português
documents = [
    "Machine learning é um campo da inteligência artificial que permite que computadores aprendam padrões a partir de dados.",
    "O aprendizado de máquina dá aos sistemas a capacidade de melhorar seu desempenho sem serem explicitamente programados.",
    "Em vez de seguir apenas regras fixas, o machine learning descobre relações escondidas nos dados.",
    "Esse campo combina estatística, algoritmos e poder computacional para extrair conhecimento.",
    "O objetivo é criar modelos capazes de generalizar além dos exemplos vistos no treinamento.",
    "Aplicações de machine learning vão desde recomendações de filmes até diagnósticos médicos.",
    "Os algoritmos de aprendizado de máquina transformam dados brutos em previsões úteis.",
    "Diferente de um software tradicional, o ML adapta-se conforme novos dados chegam.",
    "O aprendizado pode ser supervisionado, não supervisionado ou por reforço, dependendo do tipo de problema.",
    "Na prática, machine learning é o motor que impulsiona muitos avanços em visão computacional e processamento de linguagem natural.",
    "Mais do que encontrar padrões, o machine learning ajuda a tomar decisões baseadas em evidências.",
]


# Função para pré-processar os textos
def preprocess(text):
    # Converte todo o texto para minúsculas
    text_lower = text.lower()
    # Tokeniza o texto em palavras usando o tokenizador do NLTK para português
    tokens = nltk.word_tokenize(text_lower, language="portuguese")
    # Filtra apenas tokens alfanuméricos (remove pontuação e símbolos)
    tokens = [token for token in tokens if token.isalnum()]
    return tokens


tokenized_docs = [preprocess(doc) for doc in documents]  # Tokeniza todos os documentos
tokenized_docs

bm25 = BM25Okapi(tokenized_docs)  # Cria o modelo BM25 com os documentos tokenizados

query = "machine learning"  # Define a consulta de busca


def search_bm25(query, bm25):
    # Pré-processa a consulta
    tokenized_query = preprocess(query)
    # Calcula os scores BM25 para todos os documentos em relação à consulta
    scores = bm25.get_scores(tokenized_query)
    return scores


results = search_bm25(query, bm25)  # Executa a busca BM25
results

for i in np.argsort(results)[::-1]:
    print(f"Documento {i}: {documents[i]} - Score: {results[i]}")
