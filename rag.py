import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

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

model = SentenceTransformer("all-MiniLM-L6-v2")

client = genai.Client()

documents_embeddings = model.encode(documents)
documents_embeddings


def cosine_similarity(vector_a, vector_b):
    # Calcula o produto escalar (dot product) entre os dois vetores
    escalar_product = np.dot(vector_a, vector_b)
    # Calcula a norma (magnitude/comprimento) do vetor A
    norm_a = np.linalg.norm(vector_a)
    # Calcula a norma (magnitude/comprimento) do vetor B
    norm_b = np.linalg.norm(vector_b)
    # Retorna a similaridade de cosseno: produto escalar dividido pelo produto das normas
    # Resultado varia de -1 (opostos) a 1 (idênticos), com 0 indicando ortogonalidade
    return escalar_product / (norm_a * norm_b)


def retrieve_similar_documents(query, top_k=3):
    # Codifica a query de entrada em um vetor de embedding (convertendo texto em representação numérica)
    query_embedding = model.encode([query])[0]

    # Inicializa uma lista vazia para armazenar os índices dos documentos e suas similaridades
    similarities = []

    # Itera sobre todos os documentos embeddings junto com seus índices
    for index, doc_embedding in enumerate(documents_embeddings):
        # Calcula a similaridade de cosseno entre o embedding da query e o embedding do documento atual
        similarity = cosine_similarity(query_embedding, doc_embedding)
        # Adiciona uma tupla contendo o índice do documento e sua similaridade à lista
        similarities.append((index, similarity))

    # Ordena a lista de similaridades em ordem decrescente (maior similaridade primeiro)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Retorna uma lista com os top_k documentos mais similares
    # Cada elemento é uma tupla contendo o texto do documento e sua pontuação de similaridade
    return [
        (documents[index], similarity) for index, similarity in similarities[:top_k]
    ]


def generate_answer(query, retrieved_docs):
    # Combina os documentos recuperados em um único contexto
    context = "\n".join([doc for doc, _ in retrieved_docs])

    # Gera a resposta usando o modelo Gemini com o contexto fornecido
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction="Atue como um especialista em machine learning. Use o contexto fornecido para responder à pergunta de forma precisa e concisa."
        ),
        contents=f"Contexto: {context}\n\nPergunta: {query}\nResposta:",
    )
    return response.text


def rag(query, top_k=3):
    retrieved_docs = retrieve_similar_documents(query, top_k)
    answer = generate_answer(query, retrieved_docs)
    return answer, retrieved_docs


# Exemplo de uso
query = "O que é machine learning?"
answer, docs = rag(query, top_k=3)
print("Resposta gerada:")
print(answer)

print("\nDocumentos recuperados:")
for doc, similarity in docs:
    print(f"(similaridade: {similarity:.4f}) - {doc}")
