from fastapi import APIRouter, Query, Body
from db.database import graph
from utils.student import change_assignment_status, get_assignment_status, change_assignment_report, get_assignment_report

router = APIRouter()

@router.post("/assignment-status")
async def update_assignment_status(
        student_id: str = Query(...), 
        assignment_id: str = Query(...), 
        status: dict = Body(...)):
    change_assignment_status(student_id, assignment_id, status)
    return {"message": "Assignment status updated"}

@router.get("/assignment-status")
async def fetch_assignment_status(
        student_id: str = Query(...), 
        assignment_id: str = Query(...)):
    status = get_assignment_status(student_id, assignment_id)
    return {"status": status}

@router.post("/assignment-report")
async def update_assignment_report(
        student_id: str = Query(...), 
        assignment_id: str = Query(...), 
        question_id: str = Query(...), 
        query_type: str = Query(...), 
        query: str = Query(...)):
    change_assignment_report(student_id, assignment_id, question_id, query_type, query)
    return {"message": "Assignment report updated"}

@router.get("/assignment-report")
async def fetch_assignment_report(
        student_id: str = Query(...), 
        assignment_id: str = Query(...)):
    report = get_assignment_report(student_id, assignment_id)
    return {"report": report}