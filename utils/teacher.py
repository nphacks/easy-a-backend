from db.database import graph

def assign_assignment_to_class(assignment_id: str, class_id: str):
    query = """
    MATCH (a:Assignment), (c:Class)
    WHERE elementId(a) = $assignment_id AND elementId(c) = $class_id
    MATCH (c)<-[:ADMITTED]-(s:Student)
    CREATE (a)-[:ASSIGNED]->(s)
    """
    graph.query(query, {
        "assignment_id": assignment_id,
        "class_id": class_id
    })