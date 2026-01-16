import nltk

nltk.download("punkt_tab")

text = "Machine learning é um campo da inteligência artificial que se concentra no desenvolvimento de algoritmos que permitem aos computadores aprenderem a partir de dados."

word_tokens = nltk.word_tokenize(text, language="portuguese")
print(word_tokens)


def preprocess(text):
    tokens = nltk.word_tokenize(text.lower(), language="portuguese")
    tokens = [token for token in tokens if token.isalnum()]
    return tokens


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

preprocessed_docs = [" ".join(preprocess(doc)) for doc in documents]
print(preprocessed_docs)
