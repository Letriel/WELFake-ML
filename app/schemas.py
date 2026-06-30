"""Pydantic request/response models for the API."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    title: str = ""
    text: str = ""
    translate: bool = True
    verify: bool = True

    @model_validator(mode="after")
    def _at_least_one(self):
        if not (self.title.strip() or self.text.strip()):
            raise ValueError("Provide at least one of 'title' or 'text'.")
        return self


class VerifyRequest(BaseModel):
    title: str = ""
    text: str = ""
    translate: bool = True

    @model_validator(mode="after")
    def _at_least_one(self):
        if not (self.title.strip() or self.text.strip()):
            raise ValueError("Provide at least one of 'title' or 'text'.")
        return self


class NewsItem(BaseModel):
    title: str = ""
    text: str = ""


class Source(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""


class Verification(BaseModel):
    checked: bool = False
    method: str = "none"  # "none" | "search" | "search+llm"
    summary: Optional[str] = None
    sources: List[Source] = Field(default_factory=list)


class PredictResponse(BaseModel):
    label: Literal["REAL", "FAKE"]
    probability: float
    confidence: float
    input_was_translated: bool = False
    detected_language: Optional[str] = None
    translated_text: Optional[str] = None
    verification: Optional[Verification] = None


class BatchResponse(BaseModel):
    results: List[PredictResponse]
