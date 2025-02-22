from db.database import graph

def change_assignment_status(student_id: str, assignment_id: str, status: dict):
    query = """
    MATCH (s:Student), (a:Assignment)
    WHERE elementId(s) = $student_id AND elementId(a) = $assignment_id
    MERGE (s)-[r:STATUS]->(a)
    SET r.in_progress = $in_progress, r.completed = $completed
    """
    graph.query(query, {
        "student_id": student_id,
        "assignment_id": assignment_id,
        "in_progress": status.get("in_progress", False),
        "completed": status.get("completed", False)
    })

def get_assignment_status(student_id: str, assignment_id: str):
    query = """
    MATCH (s:Student)-[r:STATUS]->(a:Assignment)
    WHERE elementId(s) = $student_id AND elementId(a) = $assignment_id
    RETURN r.in_progress AS in_progress, r.completed AS completed
    """
    result = graph.query(query, {"student_id": student_id, "assignment_id": assignment_id})
    return result[0] if result else {"in_progress": False, "completed": False}

def change_assignment_report(student_id: str, assignment_id: str, question_id: str, query_type: str, query_q: str):
    query = """
    MATCH (s:Student), (a:Assignment), (q:Question)
    WHERE elementId(s) = $student_id AND elementId(a) = $assignment_id AND elementId(q) = $question_id
    MERGE (ar:Assignment_Report {assignment_id: $assignment_id})
    MERGE (qr:Question_Report {query_type: $query_type, query: $query_q})
    MERGE (s)-[:REPORTED]->(ar)
    MERGE (ar)-[:CONTAINS]->(qr)
    MERGE (a)-[:HAS_REPORT]->(ar)
    MERGE (q)-[:HAS_REPORT]->(qr)
    """
    graph.query(query, {
        "student_id": student_id,
        "assignment_id": assignment_id,
        "question_id": question_id,
        "query_type": query_type,
        "query_q": query_q
    })


def get_assignment_report(student_id: str, assignment_id: str):
    query = """
    MATCH (s:Student)-[:REPORTED]->(ar:Assignment_Report)-[:CONTAINS]->(qr:Question_Report)
    WHERE elementId(s) = $student_id AND ar.assignment_id = $assignment_id
    OPTIONAL MATCH (q:Question)-[:HAS_REPORT]->(qr)
    RETURN qr.query_type AS query_type, qr.query AS query, elementId(qr) AS question_report_id,
           elementId(q) AS question_id, q.question AS question, q.score AS score
    """
    result = graph.query(query, {"student_id": student_id, "assignment_id": assignment_id})
    return result
