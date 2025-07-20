import pinecone
import openai

# Initialize the Pinecone client
def initialize_pinecone(api_key: str, environment: str):
    pinecone.init(api_key=api_key, environment=environment)

# Create an index in Pinecone
def create_index(index_name: str, dimension: int):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension)
    else:
        print(f"Index '{index_name}' already exists.")

# Insert vectors into the index
def insert_vectors(index_name: str, vectors: list):
    with pinecone.Client(index_name) as index:
        index.upsert(vectors)

# Query the index
def query_index(index_name: str, query_vector: list, top_k: int = 5):
    with pinecone.Client(index_name) as index:
        return index.query(query_vector, top_k=top_k)

# Example usage
if __name__ == "__main__":
    API_KEY = "your_api_key"
    ENVIRONMENT = "your_environment"
    INDEX_NAME = "example-index"
    DIMENSION = 128  # Example dimension size

    initialize_pinecone(API_KEY, ENVIRONMENT)
    create_index(INDEX_NAME, DIMENSION)

    # Example vectors to insert
    example_vectors = [(f"id_{i}", [0.1 * i] * DIMENSION) for i in range(10)]
    insert_vectors(INDEX_NAME, example_vectors)

    # Example query
    query_vector = [0.1] * DIMENSION
    results = query_index(INDEX_NAME, query_vector)
    print("Query results:", results)

    openai_client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY")

    documents = [
        "Document 1 text...",
        "Document 2 text...",
        "Document 3 text..."
    ]

    embeddings = []
    for i, doc in enumerate(documents):
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=[doc]
        )
        embedding = response.data[0].embedding
        embeddings.append((f"doc_{i}", embedding))

    insert_vectors(INDEX_NAME, embeddings)