from models.models import AssignmentQuestion
from db.database import graph

def create_assignment_in_db(teacher_id: str, title: str, questions: list[AssignmentQuestion]):
    query = """
    MATCH (t:Teacher) WHERE elementId(t) = $teacher_id
    CREATE (a:Assignment {title: $title})
    CREATE (t)-[:CREATED]->(a)
    WITH a
    UNWIND $questions AS question
    CREATE (q:Question {question: question.question, score: question.score, topic: question.topic})
    CREATE (a)-[:CONTAINS]->(q)
    """
    graph.query(query, {
        "teacher_id": teacher_id,
        "title": title,
        "questions": [q.dict() for q in questions]
    })

def update_assignment_question_in_db(assignment_id: str, question_id: str, updated_question: AssignmentQuestion):
    query = """
    MATCH (a:Assignment)-[:CONTAINS]->(q:Question)
    WHERE elementId(a) = $assignment_id AND elementId(q) = $question_id
    SET q.question = $question, q.score = $score
    """
    graph.query(query, {
        "assignment_id": assignment_id,
        "question_id": question_id,
        "question": updated_question.question,
        "score": updated_question.score
    })

def get_assignment_with_questions(assignment_id: str):
    query = """
    MATCH (a:Assignment)-[:CONTAINS]->(q:Question)
    WHERE elementId(a) = $assignment_id
    RETURN elementId(a) AS assignment_id, a.title AS title,
           elementId(q) AS question_id, q.question AS question, q.score AS score, q.topic AS topic
    """
    result = graph.query(query, {"assignment_id": assignment_id})
    return {
        "assignment_id": result[0]["assignment_id"],
        "title": result[0]["title"],
        "questions": [
            {
                "question_id": record["question_id"],
                "question": record["question"],
                "score": record["score"],
                "topic": record["topic"]
            }
            for record in result
        ]
    }

def get_teacher_assignments(teacher_id: str):
    query = """
    MATCH (t:Teacher)-[:CREATED]->(a:Assignment)
    WHERE elementId(t) = $teacher_id
    RETURN elementId(a) AS assignment_id, a.title AS title
    """
    result = graph.query(query, {"teacher_id": teacher_id})
    return [
        {"assignment_id": record["assignment_id"], "title": record["title"]}
        for record in result
    ]

def get_student_assignments(student_id: str):
    query = """
    MATCH (s:Student)<-[:ASSIGNED]-(a:Assignment)
    WHERE elementId(s) = $student_id
    RETURN elementId(a) AS assignment_id, a.title AS title
    """
    result = graph.query(query, {"student_id": student_id})
    return [
        {"assignment_id": record["assignment_id"], "title": record["title"]}
        for record in result
    ]