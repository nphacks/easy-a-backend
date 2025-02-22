from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from routes.auth import router as auth_router
from routes.notes import router as notes_router
from routes.class_students import router as classes_router
from routes.student import router as student_router
from routes.teacher import router as teacher_router
from routes.assignment import router as assignment_router

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(notes_router)
app.include_router(classes_router)
app.include_router(student_router)
app.include_router(teacher_router)
app.include_router(assignment_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)