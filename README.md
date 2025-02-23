# EasyA Backend

This project is designed to interact with Neo4j, Hugging Face, and Jina AI to perform various tasks. Below are the steps to set up and run the project.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.8 or higher installed.
- Neo4j database access.
- Hugging Face account with write access.
- Jina AI API key.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/nphacks/easy-a-backend.git
   cd your-repo-name
   ```

2. **Set up a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Connect to Neo4j:**

   Set the following environment variables to connect to your Neo4j instance:

   ```bash
   export NEO4J_URI="bolt://your-neo4j-uri:7687"
   export NEO4J_USERNAME="your-username"
   export NEO4J_PASSWORD="your-password"
   ```

   Replace `your-neo4j-uri`, `your-username`, and `your-password` with your actual Neo4j credentials.

2. **Import Neo4j Graph Model:**

   Import the provided Neo4j graph model `neo4j_importer_model_2025-02-21.json` into your Neo4j instance for reference.

   ```bash
   neo4j-admin import --database=neo4j --nodes=neo4j_importer_model_2025-02-21.json
   ```

3. **Hugging Face Write Access Token:**

   Create a Hugging Face write access token from your Hugging Face account settings. Then, log in using the Hugging Face CLI:

   ```bash
   huggingface-cli login
   ```

   Enter your access token when prompted.

4. **Jina API Key:**

   Obtain your Jina API key from [Jina AI](https://jina.ai/embeddings/). Set it as an environment variable:

   ```bash
   export JINA_API_KEY="your-jina-api-key"
   ```

## Running the Project

To run the project, use the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

This will start the server on `http://0.0.0.0:8000`. The `--reload` flag enables auto-reloading on code changes.

## Usage

Once the server is running, you can interact with the API endpoints as defined in `main.py`.

## Acknowledgments

- Neo4j for the graph database.
- Hugging Face for the NLP models.
- Jina AI for the embeddings API.
