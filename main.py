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
graph = Neo4jGraph(url="neo4j+s://c218b7a1.databases.neo4j.io", username="neo4j", password="2SPEIpuOjK707g1XKNwS5gdzIFlE3hWPJ7e9LnEZJds")

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
    text = raw_text.replace(". ", ".\n")  # Add newline after full stop and space
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  
        chunk_overlap=24,  
        separators=["\n"]  # Split by newline
    )
    # text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    # return text_splitter.split_documents(raw_documents[:1])
    print('Reached split text')
    return text_splitter.split_text(text)

def create_embeddings(text: str):
    print('Reached create embeddings')
    # Create embeddings
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
    'Content-Type': 'application/json',
    'Authorization': "Bearer jina_8cd60dc22e1b48d286009a83e4ec4dabCR5DpWxTbrsjgQ7u8_4ikwLNq3dZ"
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

def create_notes_node(subject: str, topic: str):
    query = """
    CREATE (n:Notes {subject: $subject, topic: $topic})
    RETURN n
    """
    result = graph.query(query, {"subject": subject, "topic": topic})
    return result[0]["n"]

def connect_concepts_to_notes(notes_node, concepts: list):
    query = """
        MATCH (n:Notes {subject: $subject, topic: $topic})
        WITH n
        UNWIND $concepts AS concept
        MATCH (c:Concept {name: concept})
        CREATE (c)-[r:PART_OF]->(n)
        RETURN r
        """
    graph.query(query, {"subject": notes_node["subject"], "topic": notes_node["topic"], "concepts": concepts})

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

def get_all_notes():
    query = """
    MATCH (n:Notes)
    RETURN elementId(n) AS id, n.subject AS subject, n.topic AS topic
    """
    result = graph.query(query)
    return [{"id": record["id"], "subject": record["subject"], "topic": record["topic"]} for record in result]

def get_rag_text_chunks_for_note(note_id: str):
    query = """
    MATCH (n:Notes)-[:PART_OF]-(c:Concept)-[:PART_OF]-(r:RAG_Node)
    WHERE elementId(n) = $note_id
    RETURN r.text_chunk AS text_chunk
    ORDER BY r.text_chunk ASC
    """
    result = graph.query(query, {"note_id": note_id})
    return [record["text_chunk"] for record in result]

def get_text_chunks_grouped_by_concept(note_id: str):
    query = """
    MATCH (n:Notes)-[:PART_OF]-(c:Concept)-[:PART_OF]-(r:RAG_Node)
    WHERE elementId(n) = $note_id
    RETURN c.name AS concept, collect(r.text_chunk) AS text_chunks
    """
    result = graph.query(query, {"note_id": note_id})
    return {record["concept"]: record["text_chunks"] for record in result}

pipe = pipeline("text2text-generation", model="google/flan-t5-base")     

def generate_answer(question: str, context: str):
    print('Reached generate answer')
    # Combine question and context
    input_text = f"Take the context given below and Answer the question asked.\n\nContext: {context}\n\nQuestion: {question}"
    
    # Generate answer using the pipeline
    output = pipe(input_text, max_new_tokens=200, num_return_sequences=1)
    generated_text = output[0]["generated_text"]
    # Remove the input text from the output to get only the answer
    answer = generated_text.replace(input_text, "").strip()
    return answer

larger_pipe = pipeline("text2text-generation", model="google/flan-t5-large")

def generate_questions_for_concept(text_chunks: list):
    context = " ".join(text_chunks)
    
    # Prompt to generate questions
    input_text = f"Generate a distinct short questions for students based on the following notes. Separate questions with a newline:\n\nNotes: {context}"
    
    output = pipe(input_text, max_new_tokens=200, num_return_sequences=1)
    generated_text = output[0]["generated_text"].strip()
    
    # Split the generated text into individual questions
    questions = [q.strip() for q in generated_text.split("\n") if q.strip()]
    
    return questions

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
    clusters = cluster_embeddings(embeddings, n_clusters=5) #5 clusters
    create_dynamic_concepts(embeddings, clusters)

    # Create semantic relationships
    create_semantic_relationships(embeddings, threshold=0.8)

    # Create Notes node and connect concepts
    notes_node = create_notes_node(subject="Biology", topic="Photosynthesis")
    concepts = [f"Concept_{i}" for i in range(5)]  # 5 concepts
    connect_concepts_to_notes(notes_node, concepts)

    return {"message": "Dynamic RAG Graph created!"}

@app.get("/ask")
def ask_question():
    question = 'What are photosynthetic reaction centers?' #What is photorespiration?
    # Fetch relevant nodes
    rag_results, concept_results = fetch_relevant_nodes(question, top_k=3)
    # print('Fetched relevant nodes', rag_results, concept_results)
    
    # Combine context from RAG_Nodes
    rag_context = " ".join(result["text"].strip().replace("\n", " ") for result in rag_results)
    # print('RAG Context ====> ', rag_context)
    
    # Generate answer using llm
    answer = generate_answer(question, rag_context)

    # Return the answer
    return {"question": question, "answer": answer}
    # return {"message": "Question asked!"}

@app.get("/generate_assignment/{note_id}")
def generate_assignment(note_id: str):
    # Get text chunks grouped by concept
    concept_to_text_chunks = get_text_chunks_grouped_by_concept(note_id)
    
    # Generate 2 questions per concept
    assignment_questions = {}
    for concept, text_chunks in concept_to_text_chunks.items():
        questions = generate_questions_for_concept(text_chunks)
        assignment_questions[concept] = questions
    
    return {"note_id": note_id, "assignment_questions": assignment_questions}

@app.get("/notes")
def get_notes():
    # Fetch all Notes nodes
    notes = get_all_notes()
    return {"notes": notes}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)