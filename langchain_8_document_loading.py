#!/usr/bin/env python
# coding: utf-8

# # Document Loading


#! pip install langchain

import os
import time
import json

import openai

from dotenv import load_dotenv

load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")


# ## PDFs
#! pip install pypdf

from langchain.document_loaders import PyPDFLoader

def test_pdf() -> None:
  loader = PyPDFLoader("./data/프리랜서 가이드라인 (출판본).pdf")
  pages = loader.load()
  print(f"len(pages) => {len(pages)}")
  page = pages[0]
  print(page.page_content[0:500])
  print( f"page.metadata => {page.metadata}" )
  return None


# ## YouTube
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

# ! pip install yt_dlp
# ! pip install pydub

def test_youtube() -> None:
  url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
  save_dir="docs/youtube/"
  loader = GenericLoader(
    YoutubeAudioLoader([url],save_dir),
    OpenAIWhisperParser()
  )
  docs = loader.load()
  print( docs[0].page_content[0:500] )
  return None


# ## URLs

from langchain.document_loaders import WebBaseLoader

def test_url() -> None:
  loader = WebBaseLoader("https://ko.wikipedia.org/wiki/NewJeans")
  docs = loader.load()
  print(docs[0].page_content[:500])
  return None



if __name__ == '__main__':
  test_pdf()
  test_youtube()
  test_url()






