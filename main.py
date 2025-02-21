from fastapi import FastAPI, HTTPException
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
# from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from PyPDF2 import PdfReader
from docx import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

import numpy as np
import requests
import json
import os

app = FastAPI()

# Initialize Neo4j Graph
graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))

# Initialize Mistral tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Small-24B-Base-2501")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-Small-24B-Base-2501")
# Notes Cached the mode with local_files_only : If already downloaded the model once, it will be cached locally. Cache in ~/.cache/huggingface/transformers
# Initialize tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)

# Create a text-generation pipeline
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load and Split Documents
def load_and_split_documents(file_path: str):
    print('Reached load and split')
    # raw_documents = WikipediaLoader(query=query).load()
    # Load text from file
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        raw_text = ''.join(page.extract_text() for page in reader.pages)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            raw_text = file.read()
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        raw_text = '\n'.join(para.text for para in doc.paragraphs)
    else:
        raise ValueError("Unsupported file format")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  
        chunk_overlap=24,  
        separators=["\n\n", "\n", " ", ""]  # Split by paragraphs, sentences, and words
    )
    # text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    # return text_splitter.split_documents(raw_documents[:1])
    print('Reached split text')
    return text_splitter.split_text(raw_text)

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
    
    # Calculate cosine similarity for concepts (if needed)
    # (You can use the same approach as above if concepts have embeddings)
    
    return relevant_rag_results, concept_results

API_URL = "https://api-inference.huggingface.co/models/NousResearch/DeepHermes-3-Llama-3-8B-Preview"
headers = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE_ACESS_TOKEN')}"}

# def generate_answer(question: str, context: str):
#     # Combine question and context
#     input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
#     # Send request to Hugging Face Inference API
#     # payload = {"inputs": input_text}
#     response = requests.post(API_URL, headers=headers, json={"inputs": input_text})
    
#     # Check for errors
#     if response.status_code != 200:
#         raise Exception(f"API request failed: {response.text}")
    
#     # Extract and return the answer
#     answer = response.json()[0]["generated_text"]
#     return answer

pipe = pipeline("text-generation", model="gpt2", device_map="cpu")

def generate_answer(question: str, context: str):
    print('Reached generate answer')
    # Combine question and context
    input_text = f"Context: {context}\n\nQuestion ---- : {question}\n\nAnswer -----:"
    
    # Generate answer using the pipeline
    output = pipe(input_text, max_new_tokens=200, num_return_sequences=1)
    print('Done with output')
    answer = output[0]["generated_text"]
    print('Done with answer ......... ', answer)
    return answer

@app.get("/")
def read_root():
    tokenized_data = load_and_split_documents('photosynthesis.txt')
    embeddings = []
    for data in tokenized_data:
        print('Going through token data')
        embedding = create_embeddings(data)
        embeddings.append({"text": data, "embedding": embedding})
        create_rag_node(data, embedding)  # Create RAG_Node for each chunk

    # Cluster embeddings to create dynamic concepts
    clusters = cluster_embeddings(embeddings, n_clusters=5)
    create_dynamic_concepts(embeddings, clusters)

    # Create semantic relationships
    create_semantic_relationships(embeddings, threshold=0.8)

    return {"message": "Dynamic RAG Graph created!"}

@app.get("/ask")
def ask_question():
    question = 'What is oxidation in photosynthesis?'
    # Fetch relevant nodes
    rag_results, concept_results = fetch_relevant_nodes(question, top_k=1)
    
    # Combine context from RAG_Nodes
    rag_context = " ".join([result["text"] for result in rag_results])
    print('Rag context done...')
    # Combine context from Concepts
    concept_context = " ".join([f"{result['name']}: {', '.join(result['keywords'])}" for result in concept_results])
    print('Rag concept context...')
    # Full context
    context = f"{rag_context}\n{concept_context}"
    print('Context...')
    # Generate answer using Mistral
    answer = generate_answer(question, context)
    print('Question ====> ', question, '\nAnswer ====> ', answer)
    # Return the answer
    return {"question": question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)