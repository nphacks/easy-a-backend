from models.models import AssignmentQuestion
from db.database import graph

def create_assignment_in_db(teacher_id: str, title: str, questions: list[AssignmentQuestion]):
    query = """
    MATCH (t:Teacher) WHERE elementId(t) = $teacher_id
    CREATE (a:Assignment {title: $title})
    CREATE (t)-[:CREATED]->(a)
    WITH a
    UNWIND $questions AS question
    CREATE (q:Question {question: question.question, score: question.score})
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