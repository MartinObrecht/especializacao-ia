import nltk  # Para tokenização de texto
import shutil  # Para operações de arquivo
from whoosh.index import create_in  # Para criação de índices Whoosh
from whoosh.fields import *  # Para definição de esquemas de índice
from whoosh.qparser import QueryParser  # Para análise de consultas Whoosh

import warnings

warnings.filterwarnings(
    "ignore", category=SyntaxWarning
)  # Ignorar avisos de sintaxe do Whoosh

# Baixa os recursos necessários do NLTK
nltk.download("stopwords")

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

    # Remove stopwords em português
    stopwords = set(nltk.corpus.stopwords.words("portuguese")) - {"e", "não", "ou"}
    # Filtra tokens que não são stopwords
    tokens = [token for token in tokens if token not in stopwords]
    return tokens


text = "Machine learning é um campo da inteligência artificial. que permite que computadores aprendam padrões a partir de dados."
preprocess(text)


if os.path.exists("index_dir"):
    shutil.rmtree("index_dir")  # Remove o diretório do índice se já existir
os.mkdir("index_dir")  # Cria um novo diretório para o índice

schema = Schema(title=ID(stored=True), content=TEXT(stored=True))

index = create_in("index_dir", schema)  # Cria o índice no diretório especificado
writer = index.writer()  # Cria um escritor para adicionar documentos ao índice

# Adiciona documentos ao índice após pré-processamento
for i, doc in enumerate(documents):
    processed_content = " ".join(preprocess(doc))
    writer.add_document(title=f"doc_{i + 1}", content=processed_content)

writer.commit()  # Salva as alterações no índice

query = "machine E learning"  # Define a consulta de busca


def boolean_search(query, index):
    parser = QueryParser(
        "content", schema=index.schema
    )  # Cria um analisador de consultas
    parsed_query = parser.parse(query)  # Analisa a consulta

    with index.searcher() as searcher:  # Abre um buscador para o índice
        results = searcher.search(parsed_query)  # Executa a busca
        return [
            (hit["title"], hit["content"]) for hit in results
        ]  # Retorna os títulos e conteúdos dos documentos encontrados


boolean_results = boolean_search(query, index)
boolean_results
