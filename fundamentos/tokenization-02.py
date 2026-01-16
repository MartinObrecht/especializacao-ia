# Importação das bibliotecas necessárias
import nltk  # Para tokenização de texto
from sklearn.feature_extraction.text import TfidfVectorizer  # Para vetorização TF-IDF
from sklearn.metrics.pairwise import (
    cosine_similarity,
)  # Para calcular similaridade entre vetores

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


# Pré-processa todos os documentos e junta os tokens em strings
processed_docs = [" ".join(preprocess(doc)) for doc in documents]
processed_docs

# Cria o vetorizador TF-IDF
vectorizer = TfidfVectorizer()
# Transforma os documentos processados em uma matriz TF-IDF
# Cada linha representa um documento e cada coluna representa um termo
tfidf_matrix = vectorizer.fit_transform(processed_docs)
tfidf_matrix

# Define a consulta de busca
query = "machine learning"


# Função para buscar documentos mais similares à consulta usando TF-IDF
def search_tfidf(query, vectorizer, tfidf_matrix):
    # Transforma a consulta em um vetor TF-IDF usando o mesmo vocabulário dos documentos
    query_vector = vectorizer.transform([query])
    # Calcula a similaridade do cosseno entre todos os documentos e a consulta
    similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    # Enumera as similaridades para manter o índice do documento
    sorted_similarities = list(enumerate(similarities))
    # Ordena os resultados por score de similaridade (maior para menor)
    results = sorted(sorted_similarities, key=lambda x: x[1], reverse=True)

    return results


# Executa a busca e obtém os scores de similaridade para todos os documentos
search_similarities = search_tfidf(query, vectorizer, tfidf_matrix)
search_similarities

# Exibe os 3 documentos mais relevantes para a consulta
print(f"top 3 documentos por score de similaridade {query}:")
for idx, score in search_similarities[:3]:
    print(f"Documento {idx} - Score: {score:.4f}")
    print(f"Conteúdo: {documents[idx]}\n")
