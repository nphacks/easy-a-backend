from fastapi import APIRouter, Query, Body
from db.database import graph
from utils.student import change_assignment_status, get_assignment_status, change_assignment_report, get_assignment_report
from utils.assignment import get_student_assignments

router = APIRouter()

@router.post("/assignment-status/{assignment_id}/{student_id}")
async def update_assignment_status(
        student_id: str, 
        assignment_id: str, 
        status: dict = Body(...)):
    change_assignment_status(student_id, assignment_id, status)
    return {"message": "Assignment status updated"}

@router.get("/assignment-status/{assignment_id}/{student_id}")
async def fetch_assignment_status(
        student_id: str, 
        assignment_id: str):
    status = get_assignment_status(student_id, assignment_id)
    return {"status": status}

@router.post("/assignment-report/{assignment_id}/{student_id}/{question_id}")
async def update_assignment_report(
        student_id: str, 
        assignment_id: str, 
        question_id: str, 
        query_type: str = Body(...), 
        query_q: str = Body(...)):
    change_assignment_report(student_id, assignment_id, question_id, query_type, query_q)
    return {"message": "Assignment report updated"}

@router.get("/assignment-report/{assignment_id}/{student_id}")
async def fetch_assignment_report(
        student_id: str, 
        assignment_id: str):
    report = get_assignment_report(student_id, assignment_id)
    return {"report": report}

@router.get("/student-assignments/{student_id}")
def get_student_assignments_route(student_id: str):
    assignments = get_student_assignments(student_id)
    return {"assignments": assignments}