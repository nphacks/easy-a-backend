from db.database import graph

def get_all_notes(teacher_id: str):
    query = """
    MATCH (t:Teacher)-[:UPLOADED]->(n:Notes)
    WHERE elementId(t) = $teacher_id
    RETURN elementId(n) AS id, n.subject AS subject, n.topic AS topic
    """
    result = graph.query(query, {"teacher_id": teacher_id})
    return [{"id": record["id"], "subject": record["subject"], "topic": record["topic"]} for record in result]

def create_notes_node(subject: str, topic: str, teacher_id: str):
    query = """
        MATCH (t:Teacher) WHERE elementId(t) = $teacher_id
        CREATE (n:Notes {subject: $subject, topic: $topic})
        CREATE (t)-[:UPLOADED]->(n)
        RETURN n
        """
    result = graph.query(query, {"subject": subject, "topic": topic, "teacher_id": teacher_id})
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
    RETURN c.name AS concept, collect(r.text_chunk) AS text_chunks, n.topic AS topic
    """
    result = graph.query(query, {"note_id": note_id})
    if not result:
        return {}, ""
    
    topic = result[0]["topic"]
    concept_to_text_chunks = {record["concept"]: record["text_chunks"] for record in result}
    
    return concept_to_text_chunks, topic