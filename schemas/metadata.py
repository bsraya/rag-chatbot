from typing import List
from pydantic import BaseModel


class ImageMetadata(BaseModel):
    title: str
    summary: str
    key_objects: List[str]
    text_in_image: List[str]
    contextual_description: str
    tags: List[str]


class TextMetadata(BaseModel):
    summary: str
    keywords: List[str]
    entities: List[str]
    key_objects: List[str]
    tags: List[str]
    contextual_text: str
    hypothetical_questions: List[str]
