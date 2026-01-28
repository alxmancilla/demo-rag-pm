"""
MongoDB integration for the failure agent notebook.
Provides functions for vector search and data retrieval using Voyage AI embeddings.
"""

import json
from typing import Any, Dict, List, Optional
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

import requests
import voyageai


load_dotenv()

# Global MongoDB client
_mongo_client: Optional[MongoClient] = None


def get_mongo_client() -> MongoClient:
    """Get or create the MongoDB client."""
    global _mongo_client
    if _mongo_client is None:
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        _mongo_client = MongoClient(mongodb_uri)
    return _mongo_client


def get_database():
    """Get the MongoDB database instance."""
    client = get_mongo_client()
    db_name = os.getenv("DATABASE_NAME", "predictive_maintenance")
    return client[db_name]

def ingest_data_to_collection(collection_name: str, documents: List[Dict]) -> bool:
    """
    Ingest documents into a MongoDB collection
    
    Args:
        db: MongoDB database object
        collection_name: Name of the collection
        documents: List of documents to insert
        
    Returns:
        True if successful, False otherwise
    """
    db = get_database()

    if not documents:
        print(f"✗ No documents to ingest into {collection_name}")
        return False
    
    try:
        collection = db[collection_name]
        # Drop existing collection to start fresh
        collection.drop()
        
        # Insert documents
        result = collection.insert_many(documents)
        print(f"✓ Ingested {len(result.inserted_ids)} documents into '{collection_name}'")
        return True
    except Exception as e:
        print(f"✗ Error ingesting data into {collection_name}: {e}")
        return False


def generate_embedding(text: str) -> List[float]:
    """Generate embedding using Voyage AI's voyage-3-large model."""
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY environment variable is required")

    try:
        response = requests.post(
            "https://api.voyageai.com/v1/embeddings",
            json={
                "input": text,
                "model": os.getenv("EMBEDDING_MODEL", "voyage-3-large"),
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]
        return embedding
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to generate embedding: {e}")


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts using Voyage AI
    
    Args:
        texts: List of texts to embed
        model: Voyage AI model to use
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    try:
        # Create embeddings using Voyage AI
        model = os.getenv("EMBEDDING_MODEL", "voyage-3-large")
        voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"), base_url=os.getenv("VOYAGE_API_ENDPOINT"))
        response = voyage_client.embed(
            texts=texts,
            model=os.getenv("EMBEDDING_MODEL", "voyage-3-large"),
            input_type="document"
        )
        embeddings = [e for e in response.embeddings]
        print(f"✓ Generated {len(embeddings)} embeddings using {model}")
        return embeddings
    except Exception as e:
        print(f"✗ Error generating embeddings: {e}")
        return []


def add_embeddings_to_collection(collection_name: str, batch_size: int = 10) -> bool:
    """
    Generate embeddings for all documents in a collection and update them
    
    Args:
        db: MongoDB database object
        collection_name: Name of the collection to process
        batch_size: Number of documents to process per batch
        
    Returns:
        True if successful, False otherwise
    """
    try:
        db = get_database()
        collection = db[collection_name]
        documents = list(collection.find({}))
        
        if not documents:
            print(f"✗ No documents found in collection '{collection_name}'")
            return False
        
        print(f"\nProcessing {len(documents)} documents in '{collection_name}'...")
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Extract text from documents
            texts = [extract_text_for_embedding(doc) for doc in batch]
            
            # Generate embeddings for the batch
            embeddings = generate_embeddings_batch(texts)
            
            if not embeddings or len(embeddings) != len(batch):
                print(f"✗ Embedding generation failed for batch {i//batch_size + 1}")
                continue
            
            # Update documents with embeddings
            for doc, embedding in zip(batch, embeddings):
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embeddings": embedding}}
                )
            
            print(f"  ✓ Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        # Verify embeddings were added
        docs_with_embeddings = collection.count_documents({"embeddings": {"$exists": True}})
        print(f"✓ Updated {docs_with_embeddings} documents with embeddings in '{collection_name}'")
        return True
        
    except Exception as e:
        print(f"✗ Error adding embeddings to {collection_name}: {e}")
        return False

def create_vector_search_index(collection_name: str, embedding_dim: int = 1024) -> bool:
    """
    Create a vector search index on the embeddings field
    
    Note: This requires MongoDB Atlas with Atlas Vector Search enabled
    
    Args:
        db: MongoDB database object
        collection_name: Name of the collection
        embedding_dim: Dimension of the embeddings
        
    Returns:
        True if successful, False otherwise
    """
    try:
        db = get_database()
        collection = db[collection_name]
        
        # Vector search index definition for Atlas Vector Search


        search_index_model = SearchIndexModel(
                                    definition={
                                        "fields": [
                                        {
                                            "type": "vector",
                                            "path": "embeddings",
                                            "numDimensions": embedding_dim,
                                            "similarity": "cosine"
                                        }
                                        ]
                                    },
                                    name="vector_index",
                                    type="vectorSearch"
                                )


        
        # Create the index via the collection's create_search_indexes method
        # Note: This method requires MongoDB Python driver >= 4.6
        try:
            # Try using the newer search indexes API
            search_indexes = collection.list_search_indexes()
            existing_indexes = [idx.get('name') for idx in search_indexes]
            for index in existing_indexes:
                print(index)
            
            if 'vector_index' not in existing_indexes:
                collection.create_search_index(model=search_index_model)
                print(f"✓ Created vector search index for '{collection_name}'")
            else:
                print(f"✓ Vector search index already exists for '{collection_name}'")
                
        except AttributeError:
            # Fallback for older driver versions
            print(f"⚠ Vector search index creation requires MongoDB Atlas with Vector Search enabled")
            print(f"  Manually create the index in MongoDB Atlas UI with this definition:")
            print(f"  {json.dumps(index_definition, indent=2)}")
        
        return True
        
    except Exception as e:
        print(f"⚠ Note: {e}")
        print(f"  Vector indexes should be created in MongoDB Atlas UI")
        return False


def vector_search(
    query: str,
    collection_name: str,
    text_fields: Optional[List[str]] = None,
    n: int = 3,
    filter_query: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Perform vector search on a MongoDB collection using Voyage AI embeddings.

    Args:
        query: The search query string
        collection_name: Name of the MongoDB collection to search
        text_fields: List of field names to return in results
        n: Number of results to return
        filter_query: Optional MongoDB filter query

    Returns:
        JSON string containing search results
    """
    db = get_database()
    collection = db[collection_name]

    # Generate embedding for the query
    query_embedding = generate_embeddings_batch([query])

    # Build the projection
    project_fields = {"score": {"$meta": "vectorSearchScore"}}
    if text_fields:
        for field in text_fields:
            project_fields[field] = 1

    # Build aggregation pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_search_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": n * 20,
                "limit": n,
                **({"filter": filter_query} if filter_query else {}),
            },
        },
        {
            "$project": project_fields,
        },
    ]

    try:
        results = list(collection.aggregate(pipeline))
        return json.dumps(results)
    except Exception as e:
        raise RuntimeError(f"Vector search failed: {e}")


def vector_search_mongodb(collection_name: str, query_vector: List[float], num_results: int = 5) -> List[Dict]:
    """
    Perform vector similarity search on MongoDB collection
    
    Args:
        db: MongoDB database object
        collection_name: Name of the collection to search
        query_vector: Query embedding vector
        num_results: Number of results to return
        
    Returns:
        List of matching documents with similarity scores
    """
    try:
        db = get_database()
        collection = db[collection_name]
        
        # Use aggregation pipeline with vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    'queryVector': query_vector,
                    'numCandidates': 10, 
                    'limit': num_results
                }
            },
            {
                "$project": {
                    "'score": {"$meta": "vectorSearchScore"},
                    "document": "$$ROOT"
                }
            },
            {
                "$limit": num_results
            }
        ]
        
        # Try standard vector search first
        try:
            results = list(collection.aggregate(pipeline))
            return results
        except:
            # Fallback to simpler approach if aggregation fails
            # This works with documents that have embeddings field
            results = []
            documents = list(collection.find({"embeddings": {"$exists": True}}))
            
            if not documents:
                return []
            
            # Calculate similarity scores using cosine similarity
            import numpy as np
            query_vec = np.array(query_vector)
            
            for doc in documents:
                if 'embeddings' in doc:
                    doc_vec = np.array(doc['embeddings'])
                    # Cosine similarity
                    similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
                    results.append({
                        'similarityScore': float(similarity),
                        'document': doc
                    })
            
            # Sort by similarity score and return top results
            results.sort(key=lambda x: x['similarityScore'], reverse=True)
            return results[:num_results]
            
    except Exception as e:
        print(f"✗ Error performing vector search: {e}")
        return []



def retrieve_manuals(query: str, n: int = 3) -> str:
    """
    Retrieve technical manuals related to the query using vector search.

    Args:
        query: Search query
        n: Number of results to return

    Returns:
        JSON string with manual excerpts
    """
    return vector_search_mongodb(
        query,
        collection_name="manuals",
        n=n,
        text_fields=["section", "text", "pages"],
    )


def retrieve_work_orders(query: str, n: int = 3) -> str:
    """
    Retrieve work orders related to the query using vector search.

    Args:
        query: Search query
        n: Number of results to return

    Returns:
        JSON string with work order information
    """
    return vector_search_mongodb(
        query,
        collection_name="work_orders",
        n=n,
        text_fields=["title", "instructions", "observations", "root cause", "status"],
    )


def retrieve_interviews(query: str, n: int = 3) -> str:
    """
    Retrieve maintenance technician interviews related to the query using vector search.

    Args:
        query: Search query
        n: Number of results to return

    Returns:
        JSON string with interview excerpts
    """
    return vector_search_mongodb(
        query,
        collection_name="interviews",
        n=n,
        text_fields=["role", "text", "type"],
    )


def insert_incident_report(
    error_code: str,
    error_name: str,
    root_cause: str,
    repair_instructions: List[Dict[str, Any]],
    machine_id: str,
    timestamp: Optional[str] = None,
) -> str:
    """
    Insert an incident report into MongoDB.

    Args:
        error_code: The error code
        error_name: Human-readable error name
        root_cause: Root cause analysis
        repair_instructions: List of repair steps
        machine_id: Machine ID
        timestamp: Optional timestamp

    Returns:
        JSON string with incident report confirmation
    """
    from datetime import datetime

    db = get_database()
    collection = db["incident_reports"]

    
    report = {
        "name": f"Generate Incident Report - {error_code}",
        "error_code": error_code,
        "error_name": error_name,
        "root_cause": root_cause,
        "repair_instructions": repair_instructions,
        "machine_id": machine_id,
        "timestamp": timestamp or datetime.now(datetime.timezone.utc).isoformat(),
        "status": "created",
    }

    
    try:
        result = collection.insert_one(report)
        return json.dumps({
            "success": True,
            "incident_id": str(result.inserted_id),
            "message": "Incident report created successfully",
        })
    except Exception as e:
        raise RuntimeError(f"Failed to insert incident report: {e}")


def close_mongo_client():
    """Close the MongoDB client connection."""
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None


# Export all functions
__all__ = [
    "get_mongo_client",
    "get_database",
    "ingest_data_to_collection",
    "extract_text_for_embedding",
    "add_embeddings_to_collection",
    "create_vector_search_index",
    "vector_search_mongodb",
    "retrieve_manuals",
    "retrieve_work_orders",
    "retrieve_interviews",
    "insert_incident_report",
    "close_mongo_client",
]
