from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

# Load environment variables
load_dotenv()

# OpenAI client
openai_client = OpenAI()

# Embedding model (MUST match ingestion time)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

# Connect to existing Qdrant collection
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="my_collection",
    embedding=embeddings
)

# Take user input
user_query = input("Ask Something: ")

# Similarity search
results = vector_db.similarity_search(
    query=user_query,
    k=4
)

# Build context
context = "\n\n".join([
    f"Page Content: {result.page_content}\n"
    f"Page Number: {result.metadata.get('page_label')}\n"
    f"File Location: {result.metadata.get('source')}"
    for result in results
])

SYSTEM_PROMPT = f"""
You are a helpful AI assistant.
Answer the user's question ONLY using the provided context from the PDF.
Mention page numbers so the user can verify the source.

CONTEXT:
{context}
"""

# Chat completion
response = openai_client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
)

# Output
print(f"\nHappyBot: {response.choices[0].message.content}")
