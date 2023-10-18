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



from utils import ( load_pdfs, load_pdf )


# ## Embeddings
#
# Let's take our splits and embed them.

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.embeddings import Embeddings
import numpy as np


def test_embed(embedding : Embeddings) -> None:
  sentence1 = "i like dogs"
  sentence2 = "i like canines"
  sentence3 = "the weather is ugly outside"

  embedding1 = embedding.embed_query(sentence1)
  embedding2 = embedding.embed_query(sentence2)
  embedding3 = embedding.embed_query(sentence3)

  print(f"np.dot(embedding1, embedding2) => {np.dot(embedding1, embedding2)}")
  print(f"np.dot(embedding1, embedding3) => {np.dot(embedding1, embedding3)}")
  print(f"np.dot(embedding2, embedding3) => {np.dot(embedding2, embedding3)}")
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

persist_directory = 'vectordb/faiss-nj/'

def save_vectordb( embedding : Embeddings, documents: List[Document] ) -> VectorStore:
  # vectordb = Chroma.from_documents(
  vectordb = FAISS.from_documents(
    documents=documents,
    embedding=embedding
  )
  print(f"len(vectordb))=>{vectordb.index.ntotal / vectordb.index.d}")
  print(f"save local => {persist_directory}")
  vectordb.save_local(persist_directory)
  return vectordb

def load_vectordb(embedding : Embeddings) -> VectorStore:
  vectordb = FAISS.load_local(embeddings=embedding, folder_path=persist_directory)
  print(f"load local => {persist_directory}")
  print(f"len(vectordb))=>{vectordb.index.ntotal / vectordb.index.d}")
  return vectordb



def test_vectordb() -> None:
  vectordb : VectorStore = None
  documents : List[Document] = None
  if not os.path.exists(persist_directory):
    documents = load_pdf("./data/프리랜서 가이드라인 (출판본).pdf", True)
    vectordb = save_vectordb(OpenAIEmbeddings(), documents )
  else:
    vectordb = load_vectordb(OpenAIEmbeddings())

  question = "OKKY는 무엇인가?"
  docs = vectordb.similarity_search(question,k=3)

  print(f"len(docs)=>{len(docs)}")
  print(f"docs[0].page_content=>{docs[0].page_content}")
  return None


if __name__ == '__main__':
  test_embed_openai()
  test_embed_hugging_face()

  test_vectordb()

