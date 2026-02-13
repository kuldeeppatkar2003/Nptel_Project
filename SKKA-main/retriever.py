"""
ChromaDB Advanced Retriever - Production Ready
==============================================

A comprehensive, enterprise-grade retrieval system for ChromaDB with:
- Query preprocessing and expansion
- Semantic search with configurable parameters
- Result reranking and filtering
- Caching layer for performance
- Comprehensive error handling
- Metrics and monitoring
- Type hints throughout

Author: Senior Python Developer
Version: 2.0.0
Python: 3.8+
"""

import logging
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
import json

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.api.types import QueryResult
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class RetrieverError(Exception):
    """Base exception for retriever"""
    pass


class QueryError(RetrieverError):
    """Raised when query execution fails"""
    pass


class CollectionNotFoundError(RetrieverError):
    """Raised when collection doesn't exist"""
    pass


class InvalidParametersError(RetrieverError):
    """Raised when invalid parameters are provided"""
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RetrieverConfig:
    """Configuration for the retriever system"""
    
    # ChromaDB settings
    chroma_dir: Path = field(default_factory=lambda: Path(os.getenv('CHROMA_PERSIST_DIR', 'chroma_db')))
    collection_name: str = field(default_factory=lambda: os.getenv('COLLECTION_NAME', 'nptel_transcripts'))
    
    # Embedding settings
    model_name: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2'))
    device: str = field(default_factory=lambda: os.getenv('DEVICE', 'cpu'))
    
    # Retrieval parameters
    default_k: int = 10
    max_k: int = 100
    min_similarity_score: float = 0.0
    
    # Query preprocessing
    enable_query_expansion: bool = True
    enable_spell_check: bool = False
    min_query_length: int = 3
    max_query_length: int = 1000
    
    # Result processing
    enable_reranking: bool = True
    enable_deduplication: bool = True
    dedup_threshold: float = 0.95
    
    # Performance
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    cache_max_size: int = 1000
    
    # Logging
    log_dir: Path = field(default_factory=lambda: Path(os.getenv('LOG_DIR', 'logs')))
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    
    # Metrics
    enable_metrics: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.chroma_dir.exists():
            # Try resolving relative to this file location
            # This helps when running from a parent directory
            resolved_path = Path(__file__).parent / self.chroma_dir
            if resolved_path.exists():
                self.chroma_dir = resolved_path
                
        if not self.chroma_dir.exists():
            raise FileNotFoundError(f"ChromaDB directory not found: {self.chroma_dir} (CWD: {os.getcwd()})")
        
        if self.default_k > self.max_k:
            raise InvalidParametersError(f"default_k ({self.default_k}) cannot exceed max_k ({self.max_k})")
        
        if not 0 <= self.min_similarity_score <= 1:
            raise InvalidParametersError(f"min_similarity_score must be between 0 and 1")
        
        self.log_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(config: RetrieverConfig) -> logging.Logger:
    """Setup logger with file and console handlers"""
    logger = logging.getLogger("ChromaDBRetriever")
    logger.setLevel(getattr(logging, config.log_level))
    
    if logger.handlers:
        return logger
    
    # File handler
    fh = logging.FileHandler(config.log_dir / 'retriever.log', encoding='utf-8')
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)-8s | %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# QUERY CACHE
# ============================================================================

class QueryCache:
    """LRU cache for query results with TTL"""
    
    def __init__(self, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.access_order: List[str] = []
    
    def _generate_key(self, query: str, k: int, filters: Optional[Dict]) -> str:
        """Generate cache key from query parameters"""
        filter_str = json.dumps(filters, sort_keys=True) if filters else ""
        raw = f"{query}|{k}|{filter_str}"
        return hashlib.sha256(raw.encode()).hexdigest()
    
    def get(self, query: str, k: int, filters: Optional[Dict] = None) -> Optional[Any]:
        """Retrieve from cache if exists and not expired"""
        key = self._generate_key(query, k, filters)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            
            # Check TTL
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                # Update access order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return result
            else:
                # Expired, remove from cache
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
        
        return None
    
    def set(self, query: str, k: int, result: Any, filters: Optional[Dict] = None):
        """Store result in cache"""
        key = self._generate_key(query, k, filters)
        
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            if self.access_order:
                oldest_key = self.access_order.pop(0)
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
        
        self.cache[key] = (result, datetime.now())
        
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_order.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds
        }


# ============================================================================
# QUERY PREPROCESSOR
# ============================================================================

class QueryPreprocessor:
    """Preprocesses and expands queries for better retrieval"""
    
    def __init__(self, config: RetrieverConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def preprocess(self, query: str) -> str:
        """Clean and normalize query"""
        # Strip whitespace
        query = query.strip()
        
        # Remove excessive whitespace
        query = ' '.join(query.split())
        
        # Validate length
        if len(query) < self.config.min_query_length:
            raise InvalidParametersError(
                f"Query too short. Minimum {self.config.min_query_length} characters required."
            )
        
        if len(query) > self.config.max_query_length:
            self.logger.warning(f"Query exceeds max length ({self.config.max_query_length}), truncating")
            query = query[:self.config.max_query_length]
        
        return query
    
    def expand(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        if not self.config.enable_query_expansion:
            return [query]
        
        queries = [query]
        
        # Add common academic variants
        expansions = {
            'ml': ['machine learning', 'ML'],
            'ai': ['artificial intelligence', 'AI'],
            'dl': ['deep learning', 'DL'],
            'nn': ['neural network', 'NN'],
            'cnn': ['convolutional neural network', 'CNN'],
            'rnn': ['recurrent neural network', 'RNN'],
            'nlp': ['natural language processing', 'NLP'],
            'cv': ['computer vision', 'CV'],
        }
        
        query_lower = query.lower()
        for abbrev, full_forms in expansions.items():
            if abbrev in query_lower.split():
                for full_form in full_forms:
                    expanded = query_lower.replace(abbrev, full_form)
                    if expanded != query_lower:
                        queries.append(expanded)
        
        return queries


# ============================================================================
# RESULT PROCESSOR
# ============================================================================

class ResultProcessor:
    """Processes and enhances retrieval results"""
    
    def __init__(self, config: RetrieverConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def filter_by_score(
        self,
        results: Dict[str, Any],
        min_score: float
    ) -> Dict[str, Any]:
        """Filter results by minimum similarity score"""
        if not results or not results.get('distances'):
            return results
        
        # Convert distances to similarity scores (1 - normalized_distance)
        distances = results['distances'][0]
        
        # Find indices that meet threshold
        valid_indices = [
            i for i, dist in enumerate(distances)
            if (1 - dist) >= min_score
        ]
        
        if not valid_indices:
            return {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
        
        # Filter results
        filtered = {
            'ids': [[results['ids'][0][i] for i in valid_indices]],
            'documents': [[results['documents'][0][i] for i in valid_indices]],
            'metadatas': [[results['metadatas'][0][i] for i in valid_indices]],
            'distances': [[results['distances'][0][i] for i in valid_indices]]
        }
        
        return filtered
    
    def deduplicate(
        self,
        results: Dict[str, Any],
        threshold: float = 0.95
    ) -> Dict[str, Any]:
        """Remove duplicate results based on content similarity"""
        if not self.config.enable_deduplication:
            return results
        
        if not results or not results.get('documents'):
            return results
        
        documents = results['documents'][0]
        if len(documents) <= 1:
            return results
        
        # Track seen documents
        seen: Set[str] = set()
        keep_indices = []
        
        for i, doc in enumerate(documents):
            # Use first N characters as fingerprint
            fingerprint = doc[:500] if len(doc) > 500 else doc
            
            if fingerprint not in seen:
                seen.add(fingerprint)
                keep_indices.append(i)
        
        # Filter results
        deduped = {
            'ids': [[results['ids'][0][i] for i in keep_indices]],
            'documents': [[results['documents'][0][i] for i in keep_indices]],
            'metadatas': [[results['metadatas'][0][i] for i in keep_indices]],
            'distances': [[results['distances'][0][i] for i in keep_indices]]
        }
        
        if len(keep_indices) < len(documents):
            self.logger.debug(f"Removed {len(documents) - len(keep_indices)} duplicates")
        
        return deduped
    
    def rerank(
        self,
        results: Dict[str, Any],
        query: str,
        method: str = 'bm25'
    ) -> Dict[str, Any]:
        """Rerank results using specified method"""
        if not self.config.enable_reranking:
            return results
        
        if not results or not results.get('documents'):
            return results
        
        documents = results['documents'][0]
        if len(documents) <= 1:
            return results
        
        # Simple keyword-based reranking
        query_terms = set(query.lower().split())
        
        scores = []
        for doc in documents:
            doc_terms = set(doc.lower().split())
            overlap = len(query_terms & doc_terms)
            scores.append(overlap)
        
        # Get sorted indices
        sorted_indices = sorted(
            range(len(scores)),
            key=lambda i: (scores[i], -results['distances'][0][i]),
            reverse=True
        )
        
        # Reorder results
        reranked = {
            'ids': [[results['ids'][0][i] for i in sorted_indices]],
            'documents': [[results['documents'][0][i] for i in sorted_indices]],
            'metadatas': [[results['metadatas'][0][i] for i in sorted_indices]],
            'distances': [[results['distances'][0][i] for i in sorted_indices]]
        }
        
        return reranked
    
    def format_results(
        self,
        results: Dict[str, Any],
        include_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """Format results into a clean structure"""
        if not results or not results.get('ids') or not results['ids'][0]:
            return []
        
        formatted = []
        
        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for i in range(len(ids)):
            item = {
                'id': ids[i],
                'content': documents[i],
                'metadata': metadatas[i]
            }
            
            if include_scores:
                # Convert distance to similarity score
                item['similarity_score'] = 1 - distances[i]
                item['distance'] = distances[i]
            
            formatted.append(item)
        
        return formatted


# ============================================================================
# METRICS TRACKER
# ============================================================================

@dataclass
class RetrievalMetrics:
    """Track retrieval metrics"""
    
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_results_returned: float = 0.0
    avg_query_time_ms: float = 0.0
    errors: int = 0
    
    query_times: List[float] = field(default_factory=list)
    result_counts: List[int] = field(default_factory=list)
    
    def record_query(self, duration_ms: float, result_count: int, cache_hit: bool):
        """Record query metrics"""
        self.total_queries += 1
        self.query_times.append(duration_ms)
        self.result_counts.append(result_count)
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_error(self):
        """Record error"""
        self.errors += 1
    
    def calculate_averages(self):
        """Calculate average metrics"""
        if self.query_times:
            self.avg_query_time_ms = sum(self.query_times) / len(self.query_times)
        
        if self.result_counts:
            self.avg_results_returned = sum(self.result_counts) / len(self.result_counts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        self.calculate_averages()
        return {
            'total_queries': self.total_queries,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': (
                self.cache_hits / self.total_queries * 100
                if self.total_queries > 0 else 0
            ),
            'avg_results_returned': round(self.avg_results_returned, 2),
            'avg_query_time_ms': round(self.avg_query_time_ms, 2),
            'errors': self.errors,
            'error_rate': (
                self.errors / self.total_queries * 100
                if self.total_queries > 0 else 0
            )
        }
    
    def reset(self):
        """Reset all metrics"""
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.avg_results_returned = 0.0
        self.avg_query_time_ms = 0.0
        self.errors = 0
        self.query_times.clear()
        self.result_counts.clear()


# ============================================================================
# MAIN RETRIEVER CLASS
# ============================================================================

class ChromaDBRetriever:
    """
    Production-ready ChromaDB retriever with advanced features.
    
    Features:
    - Query preprocessing and expansion
    - Result filtering and reranking
    - Caching layer for performance
    - Comprehensive error handling
    - Metrics tracking
    - Type hints throughout
    
    Example:
        >>> config = RetrieverConfig()
        >>> retriever = ChromaDBRetriever(config)
        >>> results = retriever.search("machine learning basics", k=5)
        >>> for result in results:
        ...     print(result['content'][:100])
    """
    
    def __init__(self, config: Optional[RetrieverConfig] = None):
        """Initialize retriever"""
        self.config = config or RetrieverConfig()
        self.logger = setup_logger(self.config)
        
        self.client = None
        self.collection = None
        self.embedding_fn = None
        
        # Initialize components
        self.query_preprocessor = QueryPreprocessor(self.config, self.logger)
        self.result_processor = ResultProcessor(self.config, self.logger)
        
        # Initialize cache
        self.cache = None
        if self.config.enable_caching:
            self.cache = QueryCache(
                self.config.cache_max_size,
                self.config.cache_ttl_seconds
            )
        
        # Initialize metrics
        self.metrics = RetrievalMetrics() if self.config.enable_metrics else None
        
        # Initialize ChromaDB
        self._init_client()
        
        self.logger.info("ChromaDB Retriever initialized successfully")
    
    def _init_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.logger.info(f"Connecting to ChromaDB at: {self.config.chroma_dir}")
            
            # Create client
            self.client = chromadb.PersistentClient(
                path=str(self.config.chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Initialize embedding function
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.model_name,
                device=self.config.device,
                normalize_embeddings=True
            )
            
            # Get collection
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_fn
                )
                count = self.collection.count()
                self.logger.info(
                    f"✓ Connected to collection '{self.config.collection_name}' "
                    f"with {count:,} documents"
                )
            except Exception:
                raise CollectionNotFoundError(
                    f"Collection '{self.config.collection_name}' not found. "
                    f"Please run the ingestion pipeline first."
                )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include_scores: bool = True,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query string
            k: Number of results to return (default: config.default_k)
            where: Metadata filters (e.g., {"course_name": "Machine Learning"})
            where_document: Document content filters
            include_scores: Include similarity scores in results
            min_score: Minimum similarity score threshold
        
        Returns:
            List of matching documents with metadata
        
        Raises:
            QueryError: If search fails
            InvalidParametersError: If parameters are invalid
        
        Example:
            >>> results = retriever.search(
            ...     "neural networks",
            ...     k=10,
            ...     where={"discipline": "Computer Science"}
            ... )
        """
        start_time = time.time()
        cache_hit = False
        
        try:
            # Validate k
            k = k or self.config.default_k
            if k > self.config.max_k:
                raise InvalidParametersError(
                    f"k ({k}) exceeds maximum allowed ({self.config.max_k})"
                )
            
            # Preprocess query
            query = self.query_preprocessor.preprocess(query)
            
            # Check cache
            if self.cache:
                cached_result = self.cache.get(query, k, where)
                if cached_result is not None:
                    cache_hit = True
                    duration_ms = (time.time() - start_time) * 1000
                    
                    if self.metrics:
                        self.metrics.record_query(duration_ms, len(cached_result), True)
                    
                    self.logger.debug(f"Cache hit for query: {query[:50]}...")
                    return cached_result
            
            # Execute query
            self.logger.debug(f"Executing query: {query[:100]}...")
            
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where,
                where_document=where_document
            )
            
            # Process results
            if min_score is not None:
                results = self.result_processor.filter_by_score(results, min_score)
            
            results = self.result_processor.deduplicate(results)
            results = self.result_processor.rerank(results, query)
            
            # Format results
            formatted_results = self.result_processor.format_results(
                results,
                include_scores
            )
            
            # Cache results
            if self.cache and not cache_hit:
                self.cache.set(query, k, formatted_results, where)
            
            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_query(duration_ms, len(formatted_results), cache_hit)
            
            self.logger.debug(
                f"Query completed in {duration_ms:.2f}ms, "
                f"returned {len(formatted_results)} results"
            )
            
            return formatted_results
            
        except InvalidParametersError:
            raise
        except Exception as e:
            if self.metrics:
                self.metrics.record_error()
            self.logger.error(f"Search failed: {e}", exc_info=True)
            raise QueryError(f"Search failed: {str(e)}") from e
    
    def search_by_course(
        self,
        query: str,
        course_name: str,
        k: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific course.
        
        Args:
            query: Search query
            course_name: Name of the course to search within
            k: Number of results
            **kwargs: Additional search parameters
        
        Returns:
            List of matching documents
        """
        where = {"course_name": course_name}
        return self.search(query, k=k, where=where, **kwargs)
    
    def search_by_discipline(
        self,
        query: str,
        discipline: str,
        k: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific discipline.
        
        Args:
            query: Search query
            discipline: Discipline to filter by
            k: Number of results
            **kwargs: Additional search parameters
        
        Returns:
            List of matching documents
        """
        where = {"discipline": discipline}
        return self.search(query, k=k, where=where, **kwargs)
    
    def search_by_professor(
        self,
        query: str,
        professor: str,
        k: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search lectures by a specific professor.
        
        Args:
            query: Search query
            professor: Professor name to filter by
            k: Number of results
            **kwargs: Additional search parameters
        
        Returns:
            List of matching documents
        """
        where = {"professor": professor}
        return self.search(query, k=k, where=where, **kwargs)
    
    def search_by_institute(
        self,
        query: str,
        institute: str,
        k: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search lectures from a specific institute.
        
        Args:
            query: Search query
            institute: Institute name to filter by
            k: Number of results
            **kwargs: Additional search parameters
        
        Returns:
            List of matching documents
        """
        where = {"institute": institute}
        return self.search(query, k=k, where=where, **kwargs)
    
    def search_by_lecture_range(
        self,
        query: str,
        course_name: str,
        start_lecture: int,
        end_lecture: int,
        k: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search within a specific range of lectures in a course.
        
        Args:
            query: Search query
            course_name: Course name
            start_lecture: Starting lecture number (inclusive)
            end_lecture: Ending lecture number (inclusive)
            k: Number of results
            **kwargs: Additional search parameters
        
        Returns:
            List of matching documents
        """
        # Note: ChromaDB doesn't support range queries directly
        # So we'll filter after retrieval
        results = self.search_by_course(query, course_name, k=k*2, **kwargs)
        
        # Filter by lecture range
        filtered = [
            r for r in results
            if start_lecture <= r['metadata'].get('lecture_no', 0) <= end_lecture
        ]
        
        return filtered[:k or self.config.default_k]
    
    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document if found, None otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id])
            
            if not result['ids']:
                return None
            
            return {
                'id': result['ids'][0],
                'content': result['documents'][0],
                'metadata': result['metadatas'][0]
            }
        except Exception as e:
            self.logger.error(f"Failed to retrieve document {doc_id}: {e}")
            return None
    
    def get_all_courses(self) -> List[str]:
        """
        Get list of all available courses.
        
        Returns:
            List of course names
        """
        try:
            # Sample some documents to extract courses
            results = self.collection.get(limit=1000)
            
            courses = set()
            if results and results['metadatas']:
                for meta in results['metadatas']:
                    if 'course_name' in meta:
                        courses.add(meta['course_name'])
            
            return sorted(list(courses))
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve courses: {e}")
            return []
    
    def get_all_disciplines(self) -> List[str]:
        """
        Get list of all available disciplines.
        
        Returns:
            List of discipline names
        """
        try:
            results = self.collection.get(limit=1000)
            
            disciplines = set()
            if results and results['metadatas']:
                for meta in results['metadatas']:
                    if 'discipline' in meta:
                        disciplines.add(meta['discipline'])
            
            return sorted(list(disciplines))
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve disciplines: {e}")
            return []
    
    def get_all_professors(self) -> List[str]:
        """
        Get list of all professors.
        
        Returns:
            List of professor names
        """
        try:
            results = self.collection.get(limit=1000)
            
            professors = set()
            if results and results['metadatas']:
                for meta in results['metadatas']:
                    if 'professor' in meta and meta['professor'] != 'N/A':
                        professors.add(meta['professor'])
            
            return sorted(list(professors))
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve professors: {e}")
            return []
    
    def get_all_institutes(self) -> List[str]:
        """
        Get list of all institutes.
        
        Returns:
            List of institute names
        """
        try:
            results = self.collection.get(limit=1000)
            
            institutes = set()
            if results and results['metadatas']:
                for meta in results['metadatas']:
                    if 'institute' in meta and meta['institute'] != 'N/A':
                        institutes.add(meta['institute'])
            
            return sorted(list(institutes))
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve institutes: {e}")
            return []
    
    def get_course_info(self, course_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a specific course.
        
        Args:
            course_name: Name of the course
        
        Returns:
            Dictionary with course information including all metadata fields
        """
        try:
            results = self.collection.get(
                where={"course_name": course_name},
                limit=1000
            )
            
            if not results or not results['metadatas']:
                return {}
            
            # Extract course info from first result
            meta = results['metadatas'][0]
            
            info = {
                'course_name': course_name,
                'course_id': meta.get('course_id', 'N/A'),
                'discipline': meta.get('discipline', 'N/A'),
                'professor': meta.get('professor', 'N/A'),
                'institute': meta.get('institute', 'N/A'),
                'course_url': meta.get('course_url', 'N/A'),
                'language': meta.get('language', 'N/A'),
                'lecture_count': len(results['ids']),
                'total_lectures': meta.get('total_lectures', len(results['ids'])),
                'source': meta.get('source', 'N/A')
            }
            
            # Calculate additional stats
            if results['documents']:
                total_chars = sum(len(doc) for doc in results['documents'])
                info['total_characters'] = total_chars
                info['avg_chars_per_lecture'] = total_chars / len(results['documents'])
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get course info: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the query cache"""
        if self.cache:
            self.cache.clear()
            self.logger.info("Cache cleared")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get retrieval metrics.
        
        Returns:
            Dictionary with metrics
        """
        if not self.metrics:
            return {}
        
        metrics = self.metrics.to_dict()
        
        if self.cache:
            metrics['cache_stats'] = self.cache.stats()
        
        return metrics
    
    def reset_metrics(self):
        """Reset all metrics"""
        if self.metrics:
            self.metrics.reset()
            self.logger.info("Metrics reset")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive collection statistics including all metadata.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            count = self.collection.count()
            courses = self.get_all_courses()
            disciplines = self.get_all_disciplines()
            professors = self.get_all_professors()
            institutes = self.get_all_institutes()
            
            return {
                'collection_name': self.config.collection_name,
                'total_documents': count,
                'total_courses': len(courses),
                'total_disciplines': len(disciplines),
                'total_professors': len(professors),
                'total_institutes': len(institutes),
                'courses': courses,
                'disciplines': disciplines,
                'professors': professors,
                'institutes': institutes
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the retriever.
        
        Returns:
            Dictionary with health status
        """
        status = {
            'healthy': True,
            'timestamp': datetime.now().isoformat(),
            'errors': []
        }
        
        try:
            # Check collection access
            count = self.collection.count()
            status['collection_accessible'] = True
            status['document_count'] = count
            
            # Test query
            test_results = self.search("test", k=1)
            status['query_functional'] = True
            
            # Check cache
            if self.cache:
                cache_stats = self.cache.stats()
                status['cache_functional'] = True
                status['cache_size'] = cache_stats['size']
            
        except Exception as e:
            status['healthy'] = False
            status['errors'].append(str(e))
            self.logger.error(f"Health check failed: {e}")
        
        return status


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_retriever(
    chroma_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    **kwargs
) -> ChromaDBRetriever:
    """
    Create a retriever with custom configuration.
    
    Args:
        chroma_dir: Path to ChromaDB directory
        collection_name: Name of the collection
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured ChromaDBRetriever instance
    
    Example:
        >>> retriever = create_retriever(
        ...     chroma_dir="./my_db",
        ...     collection_name="my_collection",
        ...     default_k=20
        ... )
    """
    config_params = {}
    
    if chroma_dir:
        config_params['chroma_dir'] = Path(chroma_dir)
    if collection_name:
        config_params['collection_name'] = collection_name
    
    config_params.update(kwargs)
    
    config = RetrieverConfig(**config_params)
    return ChromaDBRetriever(config)


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """
    Command-line interface for testing the retriever.
    
    Usage:
        python retriever.py
    """
    print("\n" + "=" * 80)
    print("ChromaDB Retriever - Interactive Mode")
    print("=" * 80 + "\n")
    
    try:
        # Initialize retriever
        config = RetrieverConfig()
        retriever = ChromaDBRetriever(config)
        
        # Show collection stats
        stats = retriever.get_collection_stats()
        print(f"Collection: {stats['collection_name']}")
        print(f"Documents: {stats['total_documents']:,}")
        print(f"Courses: {stats['total_courses']}")
        print(f"Disciplines: {stats['total_disciplines']}")
        print("\n" + "-" * 80 + "\n")
        
        # Interactive loop
        while True:
            query = input("Enter search query (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            try:
                # Search
                results = retriever.search(query, k=5)
                
                print(f"\nFound {len(results)} results:\n")
                
                for i, result in enumerate(results, 1):
                    print(f"Result {i}:")
                    print(f"  Course: {result['metadata'].get('course_name', 'N/A')}")
                    print(f"  Lecture: {result['metadata'].get('lecture_no', 'N/A')}")
                    print(f"  Score: {result.get('similarity_score', 0):.4f}")
                    print(f"  Content: {result['content'][:200]}...")
                    print()
                
            except Exception as e:
                print(f"Error: {e}\n")
        
        # Show metrics
        metrics = retriever.get_metrics()
        if metrics:
            print("\n" + "=" * 80)
            print("Session Metrics")
            print("=" * 80)
            print(json.dumps(metrics, indent=2))
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
