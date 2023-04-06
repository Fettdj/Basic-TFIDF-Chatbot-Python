import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

# Conjunto de respuestas predefinidas
respuestas = [
    "Hola, ¿en qué puedo ayudarte?",
    "Me llamo Chatbot, estoy aquí para ayudarte.",
    "Estoy bien, gracias por preguntar.",
    "No lo sé, lo siento.",
    "Adiós, que tengas un buen día."
]

def normalizar_texto(texto):
    return nltk.word_tokenize(texto.lower().translate(str.maketrans('', '', string.punctuation)))

def chatbot_respuesta(texto_usuario):
    texto_usuario = normalizar_texto(texto_usuario)
    respuestas_con_usuario = respuestas + [' '.join(texto_usuario)]
    TfidfVec = TfidfVectorizer(tokenizer=normalizar_texto, stop_words='english')
    tfidf = TfidfVec.fit_transform(respuestas_con_usuario)
    similitud = cosine_similarity(tfidf[-1], tfidf)
    idx = similitud.argsort()[0][-2]
    similitud_ordenada = similitud.flatten()
    similitud_ordenada.sort()
    respuesta = respuestas[idx] if similitud_ordenada[-2] > 0 else "Lo siento, no entiendo tu pregunta."
    return respuesta

print("Chatbot: Hola, soy un chatbot básico. Escribe 'salir' para terminar la conversación.")
while True:
    texto_usuario = input("Tú: ")
    if texto_usuario.lower() == 'salir':
        print("Chatbot: ¡Hasta luego!")
        break
    else:
        respuesta = chatbot_respuesta(texto_usuario)
        print("Chatbot:", respuesta)
