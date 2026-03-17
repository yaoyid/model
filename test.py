import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

class FinancialVectorIndex:
    """
    In-memory vector index for financial documents using .npy storage
    """
    
    def __init__(self, index_dir="vector_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        self.vectors = None  # Numpy array of vectors
        self.chunks = []     # List of text chunks
        self.metadata = []   # List of metadata dicts
        self.index_map = {}  # Index -> chunk_id mapping
        
    def build_index(self, chunks: List[str], metadata_list: List[Dict], 
                   embeddings_model, batch_size=100):
        """
        Build index from chunks and metadata
        """
        print(f"Building index with {len(chunks)} chunks...")
        start_time = time.time()
        
        # Generate embeddings in batches to manage memory
        all_vectors = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_vectors = embeddings_model.embed_documents(batch_chunks)
            all_vectors.extend(batch_vectors)
            print(f"  Processed batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        # Convert to numpy array
        self.vectors = np.array(all_vectors)
        self.chunks = chunks
        self.metadata = metadata_list
        
        # Build index map
        self.index_map = {i: metadata_list[i].get('chunk_id', f'chunk_{i}') 
                         for i in range(len(chunks))}
        
        # Save to disk
        self._save_index()
        
        elapsed = time.time() - start_time
        print(f"✅ Index built in {elapsed:.2f}s")
        print(f"   Vectors shape: {self.vectors.shape}")
        
    def _save_index(self):
        """Save index components to disk"""
        # Save vectors
        np.save(self.index_dir / "vectors.npy", self.vectors)
        
        # Save chunks and metadata as JSON
        with open(self.index_dir / "chunks.json", "w") as f:
            json.dump(self.chunks, f)
        
        with open(self.index_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f)
        
        # Save index map
        with open(self.index_dir / "index_map.json", "w") as f:
            json.dump(self.index_map, f)
        
        # Save config
        config = {
            "num_chunks": len(self.chunks),
            "vector_dim": self.vectors.shape[1],
            "dtype": str(self.vectors.dtype)
        }
        with open(self.index_dir / "config.json", "w") as f:
            json.dump(config, f)
            
    def load_index(self):
        """Load index from disk"""
        # Load vectors
        self.vectors = np.load(self.index_dir / "vectors.npy")
        
        # Load chunks and metadata
        with open(self.index_dir / "chunks.json", "r") as f:
            self.chunks = json.load(f)
        
        with open(self.index_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)
        
        with open(self.index_dir / "index_map.json", "r") as f:
            self.index_map = json.load(f)
        
        print(f"✅ Loaded {len(self.chunks)} chunks, vectors shape: {self.vectors.shape}")
        
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
               metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar vectors with optional metadata filtering
        
        Returns: List of results with chunk text, metadata, and similarity score
        """
        # If metadata filter is provided, pre-filter indices
        if metadata_filter:
            indices = self._filter_by_metadata(metadata_filter)
            if len(indices) == 0:
                return []
            
            # Get subset of vectors
            candidate_vectors = self.vectors[indices]
            
            # Compute similarity on filtered set
            similarities = np.dot(candidate_vectors, query_vector)
            
            # Get top k indices within filtered set
            top_indices_in_filtered = np.argsort(similarities)[-top_k:][::-1]
            
            # Map back to original indices
            top_original_indices = [indices[i] for i in top_indices_in_filtered]
            top_scores = [similarities[i] for i in top_indices_in_filtered]
        else:
            # Search all vectors
            similarities = np.dot(self.vectors, query_vector)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_original_indices = top_indices
            top_scores = [similarities[i] for i in top_indices]
        
        # Prepare results
        results = []
        for idx, score in zip(top_original_indices, top_scores):
            results.append({
                "chunk_id": self.index_map.get(str(idx), f"chunk_{idx}"),
                "text": self.chunks[idx][:500] + "..." if len(self.chunks[idx]) > 500 else self.chunks[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score),
                "index": int(idx)
            })
        
        return results
    
    def _filter_by_metadata(self, filter_dict: Dict) -> List[int]:
        """Return indices of chunks matching metadata filter"""
        indices = []
        for i, meta in enumerate(self.metadata):
            match = True
            for key, value in filter_dict.items():
                if key not in meta or meta[key] != value:
                    match = False
                    break
            if match:
                indices.append(i)
        return indices

# Usage Example
if __name__ == "__main__":
    from langchain_openai import OpenAIEmbeddings
    
    # Initialize
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index = FinancialVectorIndex("earnings_index")
    
    # Your chunks and metadata from PDF processing
    chunks = [
        "Q3 2024 revenue increased 15% to $1.2 billion...",
        "Gross margin expanded to 42.5% from 40.1%...",
        "Operating income grew 22% to $345 million...",
        # ... more chunks
    ]
    
    metadata_list = [
        {
            "title": "Revenue Performance",
            "page": 5,
            "time_period": "Q3 2024",
            "chart_type": "text",
            "content_category": "income_statement"
        },
        {
            "title": "Margin Analysis", 
            "page": 6,
            "time_period": "Q3 2024",
            "chart_type": "text",
            "content_category": "income_statement"
        },
        # ... matching metadata for each chunk
    ]
    
    # Build and save index
    index.build_index(chunks, metadata_list, embeddings)
    
    # Later: load index
    index.load_index()
    
    # Search example
    query = "What was the revenue growth in Q3 2024?"
    query_vector = np.array(embeddings.embed_query(query))
    
    results = index.search(
        query_vector, 
        top_k=5,
        metadata_filter={"time_period": "Q3 2024"}  # Optional filter
    )
    
    for r in results:
        print(f"Score: {r['similarity']:.3f} | Page: {r['metadata'].get('page')}")
        print(f"Text: {r['text'][:100]}...\n")
