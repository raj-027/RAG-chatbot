# from pathlib import Path
# from typing import List, Any
# from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, TextLoader, CSVLoader
# from langchain_community.document_loaders import Dox2txtLoader
# from langchain_community.document_loaders.excel import UnstructuresExcelLoader
# from langchain_community.document_loaders import JSONLoader


from langchain_community.document_loaders import PyPDFLoader

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        return documents