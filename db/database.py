import os
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

load_dotenv()

# Initialize Neo4j Graph
print("NEO4J_URI:", os.getenv("NEO4J_URI"))
print("NEO4J_USERNAME:", os.getenv("NEO4J_USERNAME")) 
print("NEO4J_PASSWORD:", os.getenv("NEO4J_PASSWORD"))
graph = Neo4jGraph(url=os.getenv("NEO4J_URI"), username=os.getenv("NEO4J_USERNAME"), password=os.getenv("NEO4J_PASSWORD"))