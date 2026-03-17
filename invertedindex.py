class BM25InvertedIndex:
    def __init__(self, index_file: str, k1: float = 1.5, b: float = 0.75):
        """
        BM25 parameters:
        - k1: Term frequency saturation (1.2-2.0 typical)
        - b: Length normalization (0.75 typical)
        """
        with open(index_file, 'r') as f:
            self.index = json.load(f)
        
        self.doc_metadata = self._load_metadata()
        self.k1 = k1
        self.b = b
        
        # Precompute statistics
        self.total_docs = self._count_total_docs()
        self.doc_lengths = self._compute_doc_lengths()
        self.avg_doc_length = sum(self.doc_lengths.values()) / max(len(self.doc_lengths), 1)
        self.idf_cache = {}
        
    def _count_total_docs(self) -> int:
        all_docs = set()
        for doc_list in self.index.values():
            all_docs.update(doc_list)
        return len(all_docs)
    
    def _compute_doc_lengths(self) -> Dict[str, int]:
        doc_lengths = Counter()
        for term, doc_list in self.index.items():
            for doc_id in doc_list:
                doc_lengths[doc_id] += 1
        return dict(doc_lengths)
    
    def compute_idf(self, term: str) -> float:
        """BM25 IDF formula"""
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        doc_freq = len(self.index.get(term, []))
        
        # BM25 IDF: log((N - n + 0.5) / (n + 0.5))
        if doc_freq > 0:
            idf = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        else:
            idf = 0
        
        self.idf_cache[term] = idf
        return idf
    
    def compute_bm25_score(self, term: str, doc_id: str) -> float:
        """Compute BM25 score for term-document pair"""
        if term not in self.index or doc_id not in self.index[term]:
            return 0
        
        # Term frequency in document
        tf = self.index[term].count(doc_id)
        
        # Document length
        doc_length = self.doc_lengths.get(doc_id, 1)
        
        # BM25 formula
        idf = self.compute_idf(term)
        
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
        
        return idf * (numerator / denominator)
    
    def tokenize_query(self, query: str) -> List[str]:
        return query.lower().split()
    
    def search_ranked(self, query: str, top_n: int = 10,
                     metadata_filter: Dict = None) -> List[Dict]:
        """
        Ranked search using BM25
        """
        query_terms = self.tokenize_query(query)
        if not query_terms:
            return []
        
        # Find candidates
        candidates = set()
        term_doc_map = {}
        
        for term in query_terms:
            if term in self.index:
                term_docs = set(self.index[term])
                term_doc_map[term] = term_docs
                candidates.update(term_docs)
        
        # Apply metadata filter
        if metadata_filter and candidates:
            filtered = set()
            for doc_id in candidates:
                meta = self.doc_metadata.get(doc_id, {})
                match = all(meta.get(k) == v for k, v in metadata_filter.items())
                if match:
                    filtered.add(doc_id)
            candidates = filtered
        
        if not candidates:
            return []
        
        # Score each document
        doc_scores = {}
        for doc_id in candidates:
            score = 0
            matched_terms = []
            
            for term in query_terms:
                term_score = self.compute_bm25_score(term, doc_id)
                score += term_score
                if term_score > 0:
                    matched_terms.append(term)
            
            doc_scores[doc_id] = {
                'score': score,
                'matched_terms': matched_terms
            }
        
        # Sort and take top N
        sorted_docs = sorted(
            doc_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )[:top_n]
        
        # Format results
        results = []
        for doc_id, score_info in sorted_docs:
            results.append({
                'doc_id': doc_id,
                'score': round(score_info['score'], 4),
                'matched_terms': score_info['matched_terms'],
                'metadata': self.doc_metadata.get(doc_id, {}),
                'content_preview': self._get_preview(doc_id)
            })
        
        return results
    
    def _get_preview(self, doc_id: str, length: int = 200) -> str:
        """Get document preview (you'll need to store document content)"""
        try:
            with open('doc_content.json', 'r') as f:
                content = json.load(f)
            return content.get(doc_id, '')[:length] + '...'
        except:
            return f"Document {doc_id}"
