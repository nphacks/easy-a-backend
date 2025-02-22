from fastapi import APIRouter
from models.models import TeacherCreate, StudentCreate, UserLogin
from db.database import graph
from passlib.context import CryptContext
from fastapi import FastAPI, HTTPException, UploadFile, File

router = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/signup/teacher")
def create_teacher(teacher: TeacherCreate):
    # Check if email already exists
    query = """
    MATCH (t:Teacher {email: $email})
    RETURN t
    """
    result = graph.query(query, {"email": teacher.email})
    if result:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    # Hash password
    hashed_password = pwd_context.hash(teacher.password)
    
    # Create teacher node
    query = """
    CREATE (t:Teacher {
        email: $email,
        password: $password,
        name: $name,
        school: $school
    })
    RETURN t
    """
    graph.query(query, {
        "email": teacher.email,
        "password": hashed_password,
        "name": teacher.name,
        "school": teacher.school,
    })
    return {"message": "Teacher created successfully"}

@router.post("/signup/student") 
def create_student(student: StudentCreate):
    # Check if email already exists
    query = """
    MATCH (s:Student {email: $email})
    RETURN s
    """
    result = graph.query(query, {"email": student.email})
    if result:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Hash password
    hashed_password = pwd_context.hash(student.password)
    
    # Create student node
    query = """
    CREATE (s:Student {
        email: $email,
        password: $password,
        name: $name,
        grade: $grade
    })
    RETURN s
    """
    graph.query(query, {
        "email": student.email,
        "password": hashed_password,
        "name": student.name,
        "grade": student.grade
    })
    return {"message": "Student created successfully"}

@router.post("/login/teacher")
def login_teacher(user: UserLogin):
    query = """
        MATCH (t:Teacher {email: $email})
        RETURN elementId(t) AS id, t.name AS name, t.school AS school, t.password AS password
        """
    result = graph.query(query, {"email": user.email})
    if not result:
        raise HTTPException(status_code=400, detail="Teacher not found")
        
    teacher = result[0]
    if not pwd_context.verify(user.password, teacher["password"]):
        raise HTTPException(status_code=400, detail="Incorrect password")
        
    return {
        "message": "Login successful",
        "id": teacher["id"],
        "name": teacher["name"],
        "school": teacher["school"]
    }

@router.post("/login/student")
def login_student(user: UserLogin):
    query = """
    MATCH (s:Student {email: $email})
    RETURN s
    """
    result = graph.query(query, {"email": user.email})
    if not result:
        raise HTTPException(status_code=400, detail="Student not found")
        
    student = result[0]["s"]
    if not pwd_context.verify(user.password, student["password"]):
        raise HTTPException(status_code=400, detail="Incorrect password")
        
    return {"message": "Login successful"}