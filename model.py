from pydantic import BaseModel

class SentenceBody(BaseModel):
    proposition1: str
    proposition2: str

class SimilarityBody(BaseModel):
    similarity: float

class LemmatizeBody(BaseModel):
    words: list[str]