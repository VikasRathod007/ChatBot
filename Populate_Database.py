from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from typing import List
import concurrent.futures
from transformers import BertModel, BertTokenizer
import os
import torch
from pinecone import Pinecone, ServerlessSpec

DATA_PATH = "harry"


def main():
    def load_documents() -> List[Document]:
        try:
            document_loader = DirectoryLoader(DATA_PATH, use_multithreading=True)
            documents = document_loader.load()
            if not documents:
                print(f"No documents found in directory {DATA_PATH}")
            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []

    def split_document(doc: Document) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=500,
            length_function=len,
            is_separator_regex=False,
        )
        if hasattr(doc, "page_content"):
            print(f"Splitting document: {doc.metadata.get('source', 'Unknown source')}")
            chunks = text_splitter.split_text(doc.page_content)
            print(f"Number of chunks created: {len(chunks)}")
            return chunks
        else:
            print(f"Document does not have 'page_content' attribute: {doc}")
            return []

    def split_documents(documents: List[Document]) -> List[List[str]]:
        all_chunks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(split_document, documents))
            all_chunks.extend(results)
        return all_chunks

    def get_vectors(text_chunks):
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        model = BertModel.from_pretrained("bert-base-cased")

        max_length = 512
        embeddings = []

        for chunk in text_chunks:
            if isinstance(chunk, list):
                chunk = " ".join(chunk)

            encoding = tokenizer.encode_plus(
                chunk,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.last_hidden_state
                mean_embedding = last_hidden_state.mean(dim=1).squeeze().tolist()
                embeddings.append((mean_embedding, chunk))

        return embeddings

    documents = load_documents()
    if not documents:
        return

    print(f"Number of documents loaded: {len(documents)}")

    if documents:
        first_doc = documents[0]
        if hasattr(first_doc, "page_content"):
            print(f"Content of the first document: {first_doc.page_content[:1000]}...")

    chunks = split_documents(documents)
    flat_chunks = [chunk for sublist in chunks for chunk in sublist]
    pinecone = Pinecone(api_key="83945631-f182-436d-a4a2-f07e80aaa3c3")

    index_name = "example-namespace-example-index"

    if index_name not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pinecone.Index(index_name)

    embeddings_with_chunks = get_vectors(flat_chunks)
    vectors_to_upsert = [
        (str(i), embedding, {"text": chunk})
        for i, (embedding, chunk) in enumerate(embeddings_with_chunks)
    ]

    index.upsert(vectors=vectors_to_upsert)

    print("Vectors have been upserted to Pinecone.")


if __name__ == "__main__":
    main()
