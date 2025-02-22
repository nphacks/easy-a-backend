from fastapi import APIRouter, Query, Body
from utils.teacher import assign_assignment_to_class

router = APIRouter()

@router.post("/assignment-assignment")
async def assign_assignment(
        assignment_id: str = Query(...), 
        class_id: str = Query(...)
    ):
    assign_assignment_to_class(assignment_id, class_id)
    return {"message": "Assignment assigned to class"}

