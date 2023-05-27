from typing import List
from fastapi import FastAPI, HTTPException
from model import SimilarityBody, SentenceBody, LemmatizeBody
from transform import transform_text, lemmatize_text, stemmer_text

app = FastAPI()

@app.post("/lemmatize")
async def lemmatize(body: LemmatizeBody) -> list[str]:
    if body.words:
        words_ = lemmatize_text(body.words) 
        print(words_)
        return words_
    else:
        return HTTPException(status_code=500, detail="Could not process the input.")
    
@app.post("/stemmer")
async def stemmer(body: LemmatizeBody) -> list[str]:
    if body.words:
        return stemmer_text(body.words)
    else:
        return HTTPException(status_code=500, detail="Could not process the input.")
    
@app.post("/lemmatize_and_stemmer")
async def lemmatize_and_stemmer(body: LemmatizeBody) -> list[str]:
    if body.words:
        words_ = stemmer_text(body.words)
        words_ = [lemmatize_text(word) for word in words_]
        return words_
    else:
        return HTTPException(status_code=500, detail="Could not process the input.")

@app.post("/similarity")
async def similarity(propositions : SentenceBody) -> SimilarityBody:
    try:
        sentences = [propositions.proposition1, propositions.proposition2]
        x = transform_text(sentences)
        return {
            "similarity": x
        }
    except Exception as e:
        return HTTPException(status_code=500, detail="Could not process the input.")
    
   
@app.get('/')
def index():
    return 'Welcome to Inference API!'
