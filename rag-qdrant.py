import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

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

qdrant = QdrantClient(":memory:")

vector_size = model.get_sentence_embedding_dimension()

qdrant.recreate_collection(
    collection_name="ml_documents",
    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
)

points = []
for idx, doc in enumerate(documents):
    embedding = model.encode(doc).tolist()
    point = PointStruct(id=idx, vector=embedding, payload={"text": doc})
    points.append(point)

qdrant.upsert(collection_name="ml_documents", points=points, wait=True)


def retrieve(query, top_k=3):
    query_embedding = model.encode(query).tolist()
    search_result = qdrant.query_points(
        collection_name="ml_documents",
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    return [(hit.payload["text"], hit.score) for hit in search_result.points]


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
    retrieved_docs = retrieve(query, top_k)
    answer = generate_answer(query, retrieved_docs)
    return answer, retrieved_docs


# Exemplo de uso
query = "O que é machine learning?"
answer, docs = rag(query)
print(answer)
print(docs)

print("\nDocumentos recuperados:")
for doc, similarity in docs:
    print(f"(similaridade: {similarity:.4f}) - {doc}")
