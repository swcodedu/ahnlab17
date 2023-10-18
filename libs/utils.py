import time
from typing import Iterable, List
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ExecutionTimeChecker:
  def __init__(self, with_start: bool = False):
    """
    ExecutionTimeChecker 클래스의 생성자입니다.

    :param with_start: 시간 측정을 시작할지 여부를 결정하는 부울 값
    """
    self.start_time = 0
    self.end_time = 0
    if with_start:
      self.start_time = time.time()

  def start(self) -> 'ExecutionTimeChecker':
    """
    시간 측정을 시작하고, self를 반환합니다.

    :return: self (ExecutionTimeChecker 객체)
    """
    self.start_time = time.time()
    self.end_time = 0
    return self

  def stop(self) -> float:
    """
    시간 측정을 종료하고 경과 시간을 반환합니다.

    :return: 경과 시간 (초 단위)
    """
    self.end_time = time.time()
    return self.get_elapsed_time()

  def get_elapsed_time(self) -> float:
    """
    시간 측정 결과인 경과 시간을 반환합니다.

    :return: 경과 시간 (초 단위)
    """
    if self.start_time:
      current_time = time.time()
      if self.end_time:
        return self.end_time - self.start_time
      else:
        return current_time - self.start_time
    else:
      return None


def split_docs(docs: Iterable[Document]) -> List[Document]:
  """
  문서를 분할하는 함수입니다.

  :param docs: Document 객체의 Iterable
  :return: 분할된 Document 객체의 List
  """
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
  )

  splits = text_splitter.split_documents(docs)
  # print(f"len(splits)=>{len(splits)}")
  return splits


def load_pdf(file: str, with_split: bool = False) -> List[Document]:
  """
  PDF 파일을 로드하는 함수입니다.

  :param file: PDF 파일의 경로
  :param with_split: 문서 분할 여부를 결정하는 부울 값
  :return: Document 객체의 List
  """
  loader = PyPDFLoader(file)
  pages = loader.load()
  if not with_split:
    return pages

  return split_docs(pages)



def load_pdfs( files: Iterable[str], with_split: bool = False) -> List[Document]:
  """
  PDF 파일들을 로드하는 함수입니다.

  :param files: PDF 파일의 목록
  :param with_split: 문서 분할 여부를 결정하는 부울 값
  :return: Document 객체의 List
  """

  docs = []
  for file in files:
    docs.extend(load_pdf(file, with_split))

  return docs

