from db.database import graph

def create_class_and_link_to_teacher(teacher_id: str, class_name: str, grade: str, section: str):
    query = """
    MATCH (t:Teacher) WHERE elementId(t) = $teacher_id
    MERGE (c:Class {class_name: $class_name, grade: $grade, section: $section})
    CREATE (t)-[:TEACH]->(c)
    RETURN c
    """
    result = graph.query(query, {
        "teacher_id": teacher_id,
        "class_name": class_name,
        "grade": grade,
        "section": section
    })
    return result[0]["c"]

def get_teacher_classes(teacher_id: str):
    query = """
    MATCH (t:Teacher)-[:TEACH]->(c:Class)
    WHERE elementId(t) = $teacher_id
    RETURN elementId(c) AS class_id, c.class_name AS class_name, c.grade AS grade, c.section AS section
    """
    result = graph.query(query, {"teacher_id": teacher_id})
    return [dict(record) for record in result]