import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

def load_and_split_pdf(pdf_path: str, chunk_size: int = 200, chunk_overlap: int = 30) -> List[Document]:
    """
    Load a PDF document and split it into smaller chunks for efficient processing.

    Args:
        pdf_path (str): The path to the PDF file.
        chunk_size (int, optional): The maximum size of each chunk in characters. Defaults to 200.
        chunk_overlap (int, optional): The number of overlapping characters between chunks. Defaults to 30.

    Returns:
        List[Document]: A list of Document objects representing the chunks of the PDF.
    """
    # Load the PDF document
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split the document into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    return chunks

def create_embeddings(api_key: str, model_name: str) -> HuggingFaceInferenceAPIEmbeddings:
    """
    Create embeddings using the Hugging Face Inference API.

    Args:
        api_key (str): The Hugging Face Inference API key.
        model_name (str): The name of the model to use for embeddings.

    Returns:
        HuggingFaceInferenceAPIEmbeddings: The embeddings object.
    """
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name=model_name)
    return embeddings

def create_vector_store(chunks: List[Document], embeddings: HuggingFaceInferenceAPIEmbeddings) -> Chroma:
    """
    Create a vector store from the given chunks and embeddings.

    Args:
        chunks (List[Document]): A list of Document objects representing the chunks of the PDF.
        embeddings (HuggingFaceInferenceAPIEmbeddings): The embeddings object.

    Returns:
        Chroma: The vector store object.
    """
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

def create_retrievers(chunks: List[Document], vectorstore: Chroma) -> EnsembleRetriever:
    """
    Create retrievers for the vector store and keyword search.

    Args:
        chunks (List[Document]): A list of Document objects representing the chunks of the PDF.
        vectorstore (Chroma): The vector store object.

    Returns:
        EnsembleRetriever: The ensemble retriever object combining the vector store and keyword retrievers.
    """
    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 3
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, keyword_retriever], weights=[0.3, 0.7]
    )
    return ensemble_retriever

def load_quantized_model(model_name: str) -> AutoModelForCausalLM:
    """
    Load a 4-bit quantized model from the specified model name or path.

    Args:
        model_name (str): Name or path of the model to be loaded.

    Returns:
        AutoModelForCausalLM: The loaded quantized model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    return model

def initialize_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Initialize the tokenizer for the specified model name or path.

    Args:
        model_name (str): Name or path of the model for tokenizer initialization.

    Returns:
        AutoTokenizer: The initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer

def create_pipeline(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> HuggingFacePipeline:
    """
    Create a text generation pipeline for the given model and tokenizer.

    Args:
        model (AutoModelForCausalLM): The loaded quantized model.
        tokenizer (AutoTokenizer): The initialized tokenizer.

    Returns:
        HuggingFacePipeline: The Hugging Face pipeline object.
    """
    pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2048,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    llm = HuggingFacePipeline(pipeline=pipeline)
    return llm

def create_qa_chains(llm: HuggingFacePipeline, vectorstore_retriever: Chroma.as_retriever, ensemble_retriever: EnsembleRetriever) -> Tuple[RetrievalQA, RetrievalQA]:
    """
    Create normal and hybrid retrieval QA chains.

    Args:
        llm (HuggingFacePipeline): The Hugging Face pipeline object.
        vectorstore_retriever (Chroma.as_retriever): The vector store retriever.
        ensemble_retriever (EnsembleRetriever): The ensemble retriever.

    Returns:
        Tuple[RetrievalQA, RetrievalQA]: The normal and hybrid retrieval QA chains.
    """
    normal_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore_retriever
    )
    hybrid_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=ensemble_retriever
    )
    return normal_chain, hybrid_chain

if __name__ == "__main__":
    # Get the PDF file path and Hugging Face Inference API key from environment variables
    pdf_path = os.environ.get("PDF_PATH")
    hf_token = os.environ.get("HF_TOKEN")

    # Load and split the PDF
    chunks = load_and_split_pdf(pdf_path)

    # Create embeddings
    embeddings = create_embeddings(hf_token, "BAAI/bge-base-en-v1.5")

    # Create vector store
    vectorstore = create_vector_store(chunks, embeddings)

    # Create retrievers
    ensemble_retriever = create_retrievers(chunks, vectorstore)

    # Load the quantized model and initialize the tokenizer
    model_name = "HuggingFaceH4/zephyr-7b-beta"
    model = load_quantized_model(model_name)
    tokenizer = initialize_tokenizer(model_name)

    # Create the text generation pipeline
    llm = create_pipeline(model, tokenizer)

    # Create the QA chains
    normal_chain, hybrid_chain = create_qa_chains(llm, vectorstore.as_retriever(), ensemble_retriever)

    # Example usage
    normal_response = normal_chain.invoke("Ask the question based out of PDF data")
    print(normal_response)
    
    # Example usage
    hybrid_response = hybrid_chain.invoke("Ask the question based out of PDF data")
    print(hybrid_response)
