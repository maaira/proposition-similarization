import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()


def stemmer_text(words : list):
    stemmer = SnowballStemmer("portuguese")
    words_ =[stemmer.stem(word) for word in words if word not in stopwords.words('portuguese')]    
    return words_


def lemmatize_text(proposition_words : list = []):
    if proposition_words:
        proposition_words = ' '.join([word for word in proposition_words if word.lower() not in stopwords.words('portuguese')])
        tokens = nltk.word_tokenize(proposition_words, language='portuguese')
        #   print(tokens)
        lemmatized_text = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_text
    else:
        return proposition_words

def transform_text(propositions: list = []):
    sentences = []
    for proposition in propositions:
        stemmer = " ".join(stemmer_text(proposition.split()))   
        sentences.append(stemmer)
    
    model = SentenceTransformer('rufimelo/Legal-BERTimbau-large-TSDAE-v4-GPL-sts')
    embeddings = model.encode(sentences)
    x = cosine_similarity(
        [embeddings[0]], [embeddings[1]]
    )
    return x

    