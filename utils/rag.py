from db.database import graph
from keybert import KeyBERT
from PyPDF2 import PdfReader
from docx import Document
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import UploadFile

import numpy as np
import requests
import json
import os


# Load and Split Documents
def load_and_split_documents(file: UploadFile):
    print('Reached load and split')
    file_extension = file.filename.split(".")[-1].lower()

    # Load text from file
    if file_extension == "pdf":
        reader = PdfReader(file.file)
        raw_text = ''.join(page.extract_text() for page in reader.pages)
    elif file_extension == "txt":
        raw_text = file.file.read().decode("utf-8")
    elif file_extension == "docx":
        doc = Document(file.file)
        raw_text = '\n'.join(para.text for para in doc.paragraphs)
    else:
        raise ValueError("Unsupported file format")
    
    text = raw_text.replace(". ", ".\n")  # Add newline after full stop and space
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  
        chunk_overlap=24,  
        separators=["\n"]  # Split by newline
    )
    # text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)

    print('Reached split text')
    return text_splitter.split_text(text)

def create_embeddings(text: str):
    print('Reached create embeddings')
    # Create embeddings
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {os.getenv('JINA_API')}"
    }
    data = {
        "model": "jina-clip-v2",
        "dimensions": 512,
        "normalized": True,
        "embedding_type": "float",
        "input": [ {"text": text} ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()['data'][0]['embedding']

def extract_keywords(text: str, top_n: int = 5):
    # Use KeyBERT to extract keywords
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, top_n=top_n)
    return [kw[0] for kw in keywords]  # Return only the keywords

def cluster_embeddings(embeddings: list, n_clusters: int = 5):
    # Cluster embeddings using K-Means
    embeddings_array = np.array([e["embedding"] for e in embeddings])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_array)
    return clusters

def create_dynamic_concepts(embeddings: list, clusters: list):
    # Create concepts based on clusters and keywords
    for cluster_id in set(clusters):
        # Get all chunks in the cluster
        cluster_chunks = [embeddings[i]["text"] for i in range(len(embeddings)) if clusters[i] == cluster_id]
        
        # Extract keywords from the cluster
        cluster_text = " ".join(cluster_chunks)
        keywords = extract_keywords(cluster_text)
        
        # Create a concept node
        query = """
            CREATE (c:Concept {name: $name, keywords: $keywords})
            WITH c
            UNWIND $chunks AS chunk
            MATCH (n:RAG_Node {text_chunk: chunk})
            CREATE (n)-[r:PART_OF]->(c)
            RETURN r
            """
        graph.query(query, {"name": f"Concept_{cluster_id}", "keywords": keywords, "chunks": cluster_chunks})

def create_rag_node(text_chunk: str, embedding: list):
    query = """
        CREATE (n:RAG_Node {text_chunk: $text_chunk, embedding: $embedding})
        RETURN n
        """
    result = graph.query(query, {"text_chunk": text_chunk, "embedding": embedding})
    return result[0]["n"]

def create_semantic_relationships(embeddings: list, threshold: float = 0.8):
    # Compare embeddings and create relationships for similar chunks
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity([embeddings[i]["embedding"]], [embeddings[j]["embedding"]]).item()
            if sim > threshold:
                query = """
                    MATCH (a:RAG_Node {text_chunk: $text1}), (b:RAG_Node {text_chunk: $text2})
                    CREATE (a)-[r:RELATED_TO {similarity: $sim}]->(b)
                    RETURN r
                    """
                graph.query(query, {"text1": embeddings[i]["text"], "text2": embeddings[j]["text"], "sim": sim})

def create_hierarchical_structure(embeddings: list, concepts: dict):
    # Connect chunks to higher-level concepts
    for concept, keywords in concepts.items():
        query = """
            CREATE (c:Concept {name: $name})
            WITH c
            UNWIND $keywords AS keyword
            MATCH (n:RAG_Node)
            WHERE n.text_chunk CONTAINS keyword
            CREATE (n)-[r:PART_OF]->(c)
            RETURN r
            """
        graph.query(query, {"name": concept, "keywords": keywords})



def fetch_relevant_nodes(question: str, top_k: int = 3):
    print('Reached Fetch Relevant Nodes')
    # Generate embedding for the question
    question_embedding = create_embeddings(question)
    
    # Fetch all RAG_Nodes
    rag_query = """
    MATCH (n:RAG_Node)
    RETURN n.text_chunk AS text, n.embedding AS embedding
    """
    rag_results = graph.query(rag_query)
    
    # Calculate cosine similarity in Python
    rag_embeddings = [result["embedding"] for result in rag_results]
    similarities = cosine_similarity([question_embedding], rag_embeddings)[0]
    
    # Sort by similarity and get top_k results
    sorted_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_rag_results = [rag_results[i] for i in sorted_indices]
    
    # Fetch all Concepts
    concept_query = """
    MATCH (c:Concept)
    RETURN c.name AS name, c.keywords AS keywords
    """
    concept_results = graph.query(concept_query)
    
    return relevant_rag_results, concept_results