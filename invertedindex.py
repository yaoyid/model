import json
import math
from collections import Counter, defaultdict
from typing import List, Dict, Set, Optional, Any
import heapq

class BM25InvertedIndex:
    def __init__(self, index_file: str, k1: float = 1.5, b: float = 0.75):
        """
        BM25 parameters:
        - k1: Term frequency saturation (1.2-2.0 typical)
        - b: Length normalization (0.75 typical)
        """
        with open(index_file, 'r') as f:
            self.raw_index = json.load(f)
        
        # Transform the index for efficient lookups
        self.index = self._transform_index(self.raw_index)
        
        # Initialize doc_metadata as empty dict (can be set later)
        self.doc_metadata = {}
        
        self.k1 = k1
        self.b = b
        
        # Precompute statistics
        self.total_docs = self._count_total_docs()
        self.doc_lengths = self._compute_doc_lengths()
        self.avg_doc_length = sum(self.doc_lengths.values()) / max(len(self.doc_lengths), 1)
        self.idf_cache = {}
        
        # Create lookup for term frequencies
        self.term_freq_lookup = self._create_term_freq_lookup()
        
    def _transform_index(self, raw_index):
        """
        Transform the raw index format to optimize for lookups
        Creates both term->docs mapping and term->tf mapping
        """
        transformed = {}
        for term, postings in raw_index.items():
            # Store list of doc_ids for quick membership testing
            transformed[term] = {
                'doc_ids': [p['doc_id'] for p in postings],
                'postings': postings  # Keep original for tf/positions
            }
        return transformed
    
    def _create_term_freq_lookup(self):
        """
        Create a nested dict for O(1) term frequency lookup:
        term_freq_lookup[term][doc_id] = tf
        """
        lookup = defaultdict(dict)
        for term, data in self.index.items():
            for posting in data['postings']:
                lookup[term][posting['doc_id']] = posting['tf']
        return lookup
    
    def _count_total_docs(self) -> int:
        """Count unique documents across all terms"""
        all_docs = set()
        for term_data in self.index.values():
            all_docs.update(term_data['doc_ids'])
        return len(all_docs)
    
    def _compute_doc_lengths(self) -> Dict[str, int]:
        """
        Compute document lengths using term frequencies
        This is more accurate than just counting occurrences
        """
        doc_lengths = Counter()
        
        for term, data in self.index.items():
            for posting in data['postings']:
                doc_id = posting['doc_id']
                tf = posting['tf']
                doc_lengths[doc_id] += tf
        
        return dict(doc_lengths)
    
    def compute_idf(self, term: str) -> float:
        """BM25 IDF formula"""
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        if term not in self.index:
            return 0
        
        # Number of documents containing this term
        doc_freq = len(self.index[term]['doc_ids'])
        
        # BM25 IDF: log((N - n + 0.5) / (n + 0.5))
        if doc_freq > 0:
            # Adding 1 to avoid negative values with small N
            idf = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        else:
            idf = 0
        
        self.idf_cache[term] = idf
        return idf
    
    def compute_bm25_score(self, term: str, doc_id: str) -> float:
        """Compute BM25 score for term-document pair"""
        if term not in self.term_freq_lookup:
            return 0
        
        # Get term frequency for this document
        tf = self.term_freq_lookup[term].get(doc_id, 0)
        if tf == 0:
            return 0
        
        # Document length
        doc_length = self.doc_lengths.get(doc_id, self.avg_doc_length)
        
        # BM25 formula
        idf = self.compute_idf(term)
        
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
        
        return idf * (numerator / denominator)
    
    def get_term_positions(self, term: str, doc_id: str) -> List[int]:
        """Get positions of term in document (for phrase/proximity queries)"""
        if term not in self.index:
            return []
        
        for posting in self.index[term]['postings']:
            if posting['doc_id'] == doc_id:
                return posting.get('positions', [])
        
        return []
    
    def tokenize_query(self, query: str) -> List[str]:
        """Simple query tokenization"""
        return query.lower().split()
    
    def search_ranked(self, query: str, top_n: int = 10,
                     metadata_filter: Dict = None) -> List[Dict]:
        """
        Ranked search using BM25 with optimized heap-based top-n selection
        """
        # Tokenize query
        query_terms = self.tokenize_query(query)
        if not query_terms:
            return []
        
        # Find candidate documents and track which terms they contain
        candidates = set()
        term_doc_map = {}
        
        for term in query_terms:
            if term in self.index:
                term_docs = set(self.index[term]['doc_ids'])
                term_doc_map[term] = term_docs
                candidates.update(term_docs)
        
        if not candidates:
            return []
        
        # Apply metadata filter if provided
        if metadata_filter and self.doc_metadata:
            filtered = set()
            for doc_id in candidates:
                meta = self.doc_metadata.get(doc_id, {})
                match = True
                for key, value in metadata_filter.items():
                    if key not in meta or meta[key] != value:
                        match = False
                        break
                if match:
                    filtered.add(doc_id)
            candidates = filtered
            
            if not candidates:
                return []
        
        # Score documents using min-heap for efficient top-n selection
        score_heap = []  # Min-heap of (score, doc_id, matched_terms)
        
        for doc_id in candidates:
            score = 0
            matched_terms = []
            
            for term in query_terms:
                term_score = self.compute_bm25_score(term, doc_id)
                if term_score > 0:
                    score += term_score
                    matched_terms.append(term)
            
            # Use negative score for max-heap behavior with min-heap
            if len(score_heap) < top_n:
                heapq.heappush(score_heap, (score, doc_id, matched_terms))
            elif score > score_heap[0][0]:
                heapq.heapreplace(score_heap, (score, doc_id, matched_terms))
        
        # Extract results from heap (in reverse order for descending score)
        results = []
        while score_heap:
            score, doc_id, matched_terms = heapq.heappop(score_heap)
            results.insert(0, {
                'doc_id': doc_id,
                'score': round(score, 4),
                'matched_terms': matched_terms,
                'metadata': self.doc_metadata.get(doc_id, {}),
                'term_count': len(matched_terms),
                'query_coverage': len(matched_terms) / len(query_terms) if query_terms else 0
            })
        
        return results
    
    def phrase_search(self, phrase: str, window: int = 0) -> List[Dict]:
        """
        Search for exact phrase or terms within a window
        
        Args:
            phrase: Space-separated terms to search for
            window: If 0, exact phrase; if >0, terms within window words
        """
        terms = phrase.split()
        if len(terms) < 2:
            return self.search_ranked(phrase)
        
        # Get all terms in the phrase
        query_terms = [t.lower() for t in terms]
        
        # Find documents containing all terms
        candidate_docs = set()
        for term in query_terms:
            if term in self.index:
                term_docs = set(self.index[term]['doc_ids'])
                if not candidate_docs:
                    candidate_docs = term_docs
                else:
                    candidate_docs.intersection_update(term_docs)
        
        results = []
        for doc_id in candidate_docs:
            # Get positions for each term
            positions_list = []
            valid = True
            
            for term in query_terms:
                positions = self.get_term_positions(term, doc_id)
                if not positions:
                    valid = False
                    break
                positions_list.append(positions)
            
            if not valid:
                continue
            
            # Check proximity
            if window == 0:
                # Exact phrase: positions must be consecutive
                first_positions = positions_list[0]
                for pos in first_positions:
                    match = True
                    for i in range(1, len(query_terms)):
                        if pos + i not in positions_list[i]:
                            match = False
                            break
                    if match:
                        # Calculate BM25 score for ranking
                        score = 0
                        for term in query_terms:
                            score += self.compute_bm25_score(term, doc_id)
                        
                        results.append({
                            'doc_id': doc_id,
                            'score': score,
                            'match_type': 'exact_phrase',
                            'metadata': self.doc_metadata.get(doc_id, {})
                        })
                        break
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def search_with_explanation(self, query: str, doc_id: str) -> Dict:
        """
        Explain why a specific document scored the way it did
        Useful for debugging and transparency
        """
        query_terms = self.tokenize_query(query)
        explanation = {
            'doc_id': doc_id,
            'query': query,
            'terms': {},
            'total_score': 0,
            'document_length': self.doc_lengths.get(doc_id, 'unknown'),
            'avg_doc_length': self.avg_doc_length
        }
        
        for term in query_terms:
            tf = self.term_freq_lookup.get(term, {}).get(doc_id, 0)
            if tf > 0:
                idf = self.compute_idf(term)
                doc_length = self.doc_lengths.get(doc_id, self.avg_doc_length)
                
                # Show BM25 components
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                term_score = idf * (numerator / denominator)
                
                explanation['terms'][term] = {
                    'tf': tf,
                    'idf': round(idf, 4),
                    'score': round(term_score, 4),
                    'positions': self.get_term_positions(term, doc_id)
                }
                explanation['total_score'] += term_score
            else:
                explanation['terms'][term] = {'tf': 0, 'score': 0}
        
        explanation['total_score'] = round(explanation['total_score'], 4)
        return explanation
    
    def get_term_stats(self, term: str) -> Dict:
        """Get statistics about a term in the index"""
        if term not in self.index:
            return {'error': f'Term "{term}" not found in index'}
        
        postings = self.index[term]['postings']
        
        return {
            'term': term,
            'document_frequency': len(postings),
            'total_occurrences': sum(p['tf'] for p in postings),
            'idf': self.compute_idf(term),
            'documents': [
                {
                    'doc_id': p['doc_id'],
                    'tf': p['tf'],
                    'positions_count': len(p.get('positions', []))
                }
                for p in postings[:10]  # First 10
            ]
        }
    
    def set_metadata(self, metadata_file: str):
        """Load document metadata from a file"""
        try:
            with open(metadata_file, 'r') as f:
                self.doc_metadata = json.load(f)
            print(f"✅ Loaded metadata for {len(self.doc_metadata)} documents")
        except FileNotFoundError:
            print(f"⚠️ Metadata file {metadata_file} not found")
        except Exception as e:
            print(f"⚠️ Error loading metadata: {e}")
    
    def _get_preview(self, doc_id: str, length: int = 200) -> str:
        """Get document preview (you'll need to store document content)"""
        try:
            with open('doc_content.json', 'r') as f:
                content = json.load(f)
            return content.get(doc_id, '')[:length] + '...'
        except:
            return f"Document {doc_id}"


# Example usage
if __name__ == "__main__":
    # Your exact index format
    bm25_ready_index = {
        "revenue": [
            {"doc_id": "q1_report_chunk3", "tf": 3, "positions": [5, 12, 18]},
            {"doc_id": "q1_report_chunk7", "tf": 1, "positions": [42]},
            {"doc_id": "annual_report_chunk12", "tf": 2, "positions": [8, 15]}
        ],
        "growth": [
            {"doc_id": "q1_report_chunk3", "tf": 2, "positions": [7, 22]},
            {"doc_id": "presentation_chunk5", "tf": 1, "positions": [31]}
        ]
    }
    
    # Save to file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(bm25_ready_index, f)
        index_file = f.name
    
    # Initialize BM25 index (without metadata first)
    bm25 = BM25InvertedIndex(index_file)
    
    # Load metadata separately
    doc_metadata = {
        "q1_report_chunk3": {"title": "Q1 Revenue", "page": 5, "time_period": "Q1 2024"},
        "q1_report_chunk7": {"title": "Q1 Margins", "page": 7, "time_period": "Q1 2024"},
        "annual_report_chunk12": {"title": "Annual Revenue", "page": 12, "time_period": "FY 2023"},
        "presentation_chunk5": {"title": "Growth Strategy", "page": 5, "time_period": "Q1 2024"}
    }
    
    # Create metadata file
    with open('doc_metadata.json', 'w') as f:
        json.dump(doc_metadata, f)
    
    # Set metadata using the new method
    bm25.set_metadata('doc_metadata.json')
    
    # Search
    results = bm25.search_ranked("revenue growth", top_n=5)
    
    print("SEARCH RESULTS:")
    print("=" * 50)
    for i, r in enumerate(results, 1):
        print(f"{i}. Document: {r['doc_id']}")
        print(f"   Score: {r['score']}")
        print(f"   Matched terms: {', '.join(r['matched_terms'])}")
        print(f"   Query coverage: {r['query_coverage']:.0%}")
        print(f"   Metadata: {r['metadata']}")
        print()
    
    # Search with metadata filter
    print("\nFILTERED SEARCH (time_period = Q1 2024):")
    print("=" * 50)
    filtered_results = bm25.search_ranked(
        "revenue", 
        top_n=5,
        metadata_filter={"time_period": "Q1 2024"}
    )
    
    for i, r in enumerate(filtered_results, 1):
        print(f"{i}. Document: {r['doc_id']}")
        print(f"   Score: {r['score']}")
        print(f"   Metadata: {r['metadata']}")
        print()
    
    # Explain scoring for a specific document
    print("\nSCORING EXPLANATION for q1_report_chunk3:")
    print("=" * 50)
    explanation = bm25.search_with_explanation("revenue growth", "q1_report_chunk3")
    print(f"Total Score: {explanation['total_score']}")
    print(f"Document Length: {explanation['document_length']} words")
    print(f"Average Document Length: {explanation['avg_doc_length']:.1f} words")
    print("\nTerm Contributions:")
    for term, details in explanation['terms'].items():
        if details.get('tf', 0) > 0:
            print(f"  {term}: TF={details['tf']}, IDF={details['idf']}, Score={details['score']}")
            if 'positions' in details and details['positions']:
                print(f"     Positions: {details['positions']}")
    
    # Phrase search
    print("\nPHRASE SEARCH (exact):")
    print("=" * 50)
    phrase_results = bm25.phrase_search("revenue growth", window=0)
    for i, r in enumerate(phrase_results, 1):
        print(f"{i}. Document: {r['doc_id']}")
        print(f"   Score: {r['score']}")
    
    # Clean up
    os.unlink(index_file)
    os.unlink('doc_metadata.json')
