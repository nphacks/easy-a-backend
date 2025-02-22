from fastapi import APIRouter, Query, Body
from utils.teacher import assign_assignment_to_class
from utils.assignment import get_teacher_assignments
from utils.classes import get_teacher_classes

router = APIRouter()

@router.post("/assign-assignment/{assignment_id}/{class_id}")
async def assign_assignment(
        assignment_id: str, 
        class_id: str
    ):
    assign_assignment_to_class(assignment_id, class_id)
    return {"message": "Assignment assigned to class"}

@router.get("/teacher-assignments/{teacher_id}")
def get_teacher_assignments_route(teacher_id: str):
    assignments = get_teacher_assignments(teacher_id)
    return {"assignments": assignments}

@router.get("/teacher-classes/{teacher_id}")
def get_teacher_classes_route(teacher_id: str):
    classes = get_teacher_classes(teacher_id)
    return {"classes": classes}