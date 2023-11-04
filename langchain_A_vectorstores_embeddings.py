#!/usr/bin/env python
# coding: utf-8

# # Vectorstores and Embeddings

import os
import time
import json
import sys
from typing import Iterable, List
from langchain.docstore.document import Document

import openai

from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")
sys.path.append(os.getenv("PYTHONPATH"))



from utils import (
  load_pdf_vectordb,
  load_vectordb_from_file
  )


# ## Embeddings
#
# Let's take our splits and embed them.

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.embeddings import Embeddings
import numpy as np

from wcwidth import wcswidth

def align_text(text, width, align='left', fillchar=' '):
    text_width = wcswidth(text)
    padding = width - text_width
    if padding <= 0:
       return text

    if align == 'right':
        return fillchar * padding + text
    elif align == 'center':
        left_padding = padding // 2
        right_padding = padding - left_padding
        return fillchar * left_padding + text + fillchar * right_padding
    else:  # left by default
        return text + fillchar * padding


def test_embed(embedding : Embeddings) -> None:
  sentences = [
    "i like dogs",
    "i like canines",
    "나는 개를 좋아한다",
    "J'aime les chiens",
    "私は犬が好きです",
    "Я люблю собак",
    "ฉันชอบหมา",
    "the weather is ugly outside"
    ]
  embeddings = []

  for sentence in sentences:
    embeddings.append(embedding.embed_query(sentence))

  for idx, em1 in enumerate(embeddings) :
    for idx2, em2 in enumerate(embeddings) :
      if idx >= idx2:
        continue
      v = np.dot(em1, em2)
      l = f"'{sentences[idx]}',"
      r = f"'{sentences[idx2]}'"
      print(f"np.dot({align_text(l, 22)} {align_text(r, 30)} => {v}")
  return None


def test_embed_openai() -> None:
  print("test_embed_openai()")
  test_embed(OpenAIEmbeddings())
  return None


def test_embed_hugging_face() -> None:
  print("test_embed_hugging_face()")
  test_embed(HuggingFaceEmbeddings())
  return None


# ## Vectorstores

from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.schema.vectorstore import VectorStore


def test_pdf_vectordb() -> None:
  vectordb : VectorStore = load_pdf_vectordb("./data/프리랜서 가이드라인 (출판본).pdf")

  question = "정규직의 장점은?"
  docs = vectordb.similarity_search(question,k=3)

  print(f"len(docs)=>{len(docs)}")
  print(f"docs[0].page_content=>{docs[0].page_content}")
  return None


def test_csv_vectordb() -> None:
  vectordb : VectorStore = load_vectordb_from_file("data/OutdoorClothingCatalog_1000.csv")

  question = "A shirt that is wrinkle-resistant and breathable"
  docs = vectordb.similarity_search(question,k=3)

  print(f"len(docs)=>{len(docs)}")
  print(f"docs[0].page_content=>{docs[0].page_content}")

  question = "주름이 잘 잡히지 않으면서 통기성이 좋은 셔츠"
  docs = vectordb.similarity_search(question,k=3)

  print(f"len(docs)=>{len(docs)}")
  print(f"docs[0].page_content=>{docs[0].page_content}")
  return None


if __name__ == '__main__':
  test_embed_openai()
  test_embed_hugging_face()

  test_pdf_vectordb()
  test_csv_vectordb()

