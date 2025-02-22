from fastapi import APIRouter, Form, File, UploadFile, Body
from utils.rag import fetch_relevant_nodes, load_and_split_documents, create_embeddings
from utils.rag import create_rag_node, cluster_embeddings, create_dynamic_concepts
from utils.rag import create_semantic_relationships
from utils.notes import get_all_notes, create_notes_node, connect_concepts_to_notes
from utils.utils import generate_answer

router = APIRouter()

@router.get("/notes")
def get_notes():
    # Fetch all Notes nodes
    notes = get_all_notes()
    return {"notes": notes}

@router.post("/upload-notes")
async def upload_file(
    subject: str = Form(...), 
    topic: str = Form(...), 
    teacher_id: str = Form(...), 
    file: UploadFile = File(...)
):
    # Process file content
    tokenized_data = load_and_split_documents(file)
    embeddings = []
    for data in tokenized_data:
        print('Going through token data')
        embedding = create_embeddings(data)
        embeddings.append({"text": data, "embedding": embedding})
        create_rag_node(data, embedding)  # Create RAG_Node for each chunk

    # Cluster embeddings to create dynamic concepts
    clusters = cluster_embeddings(embeddings, n_clusters=min(5, len(embeddings))) #minimum 5 clusters
    create_dynamic_concepts(embeddings, clusters)

    # Create semantic relationships
    create_semantic_relationships(embeddings, threshold=0.8)

    # Create Notes node and connect concepts
    notes_node = create_notes_node(subject=subject, topic=topic, teacher_id=teacher_id)
    concepts = [f"Concept_{i}" for i in range(5)]  # 5 concepts
    connect_concepts_to_notes(notes_node, concepts)

    return {"message": "Dynamic RAG Graph created!"}

@router.get("/ask-doubt")
def ask_question(question: str = Body(..., embed=True)):
    # question = 'What are photosynthetic reaction centers?' #What is photorespiration?
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