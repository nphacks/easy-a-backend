from pydantic import BaseModel

# Pydantic models for request validation
class UserBase(BaseModel):
    email: str
    password: str
    name: str

class TeacherCreate(UserBase):
    subject: str
    school: str
    
class StudentCreate(UserBase):
    grade: str

class UserLogin(BaseModel):
    email: str 
    password: str

class AssignmentQuestion(BaseModel):
    question: str
    score: str