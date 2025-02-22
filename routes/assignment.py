from fastapi import APIRouter, Body, Query
from utils.notes import get_text_chunks_grouped_by_concept
from utils.utils import generate_questions_for_concept
from utils.assignment import create_assignment_in_db, update_assignment_question_in_db, get_assignment_with_questions
from models.models import AssignmentQuestion

router = APIRouter()

@router.get("/generate-assignment")
def generate_assignment(note_ids: list[str] = Body(...)):

    assignment_questions = []
    for note_id in note_ids:
        # Get text chunks grouped by concept
        concept_to_text_chunks, topic = get_text_chunks_grouped_by_concept(note_id)
        
        for text_chunks in concept_to_text_chunks.values():
            questions = generate_questions_for_concept(text_chunks)
            assignment_questions.extend([{"question": q, "topic": topic} for q in questions])

    return {"note_id": note_ids, "assignment_questions": assignment_questions}

@router.post("/create-assignment")
def create_assignment_questions(
        assignment_questions: list[AssignmentQuestion], 
        teacher_id: str = Query(...), 
        title: str = Query(...)):
    create_assignment_in_db(teacher_id, title, assignment_questions)
    return {"message": "Assignment is successfully generated"}

@router.get("/assignment/{assignment_id}")
def get_assignment(assignment_id: str):
    assignment = get_assignment_with_questions(assignment_id)
    return {"assignment": assignment}

@router.put("/edit-assignment/{assignment_id}/{question_id}")
def update_assignment_question(
        assignment_id: str, 
        question_id: str, 
        updated_question: AssignmentQuestion = Body(...)):
    update_assignment_question_in_db(assignment_id, question_id, updated_question)
    return {"message": "Updated changes to assignment"}