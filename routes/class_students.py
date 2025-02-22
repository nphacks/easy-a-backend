from fastapi import APIRouter, UploadFile, File
from passlib.context import CryptContext
from db.database import graph
from io import StringIO
from utils.classes import create_class_and_link_to_teacher
import csv

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/upload/class-list")
async def upload_class_list(file: UploadFile = File(...)):
    # Read CSV file
    content = await file.read()
    csv_text = content.decode()
    csv_reader = csv.DictReader(StringIO(csv_text))

    # Process each row
    for row in csv_reader:
        student_email = row['Email']
        class_name = row['Class']
        
        # Extract and format name from email
        name = student_email.split('@')[0]  # Get part before @
        name = name.replace('.', ' ')  # Replace dots with spaces
        name = name.title()  # Capitalize each word
        
        # Create student if not exists
        student_query = """
        MERGE (s:Student {email: $student_email})
        ON CREATE SET s.password = $default_password,
                      s.name = $student_name,
                      s.grade = 'Unassigned'
        RETURN s
        """
        result = graph.query(student_query, {
            "student_email": student_email,
            "default_password": pwd_context.hash("changeme"),
            "student_name": name
        })
        
        # Create or find class node
        class_query = """
            MERGE (c:Class {class_name: $class_name})
            RETURN c
        """
        result2 = graph.query(class_query, {"class_name": class_name})
        
        # Link student to class
        link_query = """
            MATCH (s:Student {email: $student_email})
            MATCH (c:Class {class_name: $class_name})
            MERGE (s)-[:ENROLLED_IN]->(c)
            RETURN s, c
        """
        last = graph.query(link_query, {
            "student_email": student_email,
            "class_name": class_name
        })
        print(last)
    return {"message": "Class list uploaded successfully"}

@router.post("/create-class/{teacher_id}/{class_name}/{grade}/{section}")
def create_class(
    teacher_id: str, 
    class_name: str,
    grade: str, 
    section: str
):
    class_node = create_class_and_link_to_teacher(teacher_id, class_name, grade, section)
    return {"message": "Class created successfully", "class_id": class_node["id"]}

"""
Example CSV format (students.csv):
Email,Class
jake.brian@school.com,Mathematics 101
jane.smith@school.com,History 101
bob.wilson@school.com,History 101

Example curl command:
curl -X POST "http://localhost:8000/upload/class-list" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@students.csv"
"""