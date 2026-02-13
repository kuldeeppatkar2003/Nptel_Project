"""
ChromaDB Ingestion Pipeline - Production Optimized
==================================================
Async/parallel processing for high-performance ingestion of NPTEL transcripts.

Author: Senior Data Scientist
Version: 2.0.0
Python: 3.8+
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
import hashlib

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Validate required environment variables
REQUIRED_VARS = ['INPUT_JSON_PATH', 'CHROMA_PERSIST_DIR', 'COLLECTION_NAME', 'EMBEDDING_MODEL_NAME']
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration loaded from .env file"""
    
    # Paths
    input_json: Path = field(default_factory=lambda: Path(os.getenv('INPUT_JSON_PATH')))
    chroma_dir: Path = field(default_factory=lambda: Path(os.getenv('CHROMA_PERSIST_DIR')))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv('LOG_DIR', 'logs')))
    
    # Collection
    collection_name: str = field(default_factory=lambda: os.getenv('COLLECTION_NAME'))
    
    # Embedding
    model_name: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL_NAME'))
    device: str = field(default_factory=lambda: os.getenv('DEVICE', 'cpu'))
    batch_size: int = field(default_factory=lambda: int(os.getenv('BATCH_SIZE', '64')))
    
    # Processing
    duplicate_strategy: str = field(default_factory=lambda: os.getenv('DUPLICATE_STRATEGY', 'skip'))
    max_workers: int = field(default_factory=lambda: int(os.getenv('MAX_WORKERS', '4')))
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    
    def __post_init__(self):
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logger(config: Config) -> logging.Logger:
    """Setup logger with file and console handlers"""
    logger = logging.getLogger("ChromaDBIngestion")
    logger.setLevel(getattr(logging, config.log_level))
    
    if logger.handlers:
        return logger
    
    # File handler
    fh = logging.FileHandler(config.log_dir / 'ingestion.log', encoding='utf-8')
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(levelname)-8s | %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# CHROMADB MANAGER
# ============================================================================

class ChromaManager:
    """Manages ChromaDB operations with async support"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = None
        self.collection = None
        self.embedding_fn = None
        self._init_client()
    
    def _init_client(self):
        """Initialize ChromaDB client and embedding function"""
        self.logger.info(f"Initializing ChromaDB at: {self.config.chroma_dir}")
        
        # Create client
        self.client = chromadb.PersistentClient(
            path=str(self.config.chroma_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        # Initialize embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config.model_name,
            device=self.config.device,
            normalize_embeddings=True
        )
        
        # Test embedding
        test_emb = self.embedding_fn(["test"])
        dim = len(test_emb[0])
        self.logger.info(f"✓ Embedding model loaded: {self.config.model_name} (dim={dim}, device={self.config.device})")
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_fn
            )
            count = self.collection.count()
            self.logger.info(f"✓ Found existing collection '{self.config.collection_name}' with {count} documents")
        except:
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_fn,
                metadata={
                    'description': 'NPTEL lecture transcripts',
                    'created_at': datetime.now().isoformat(),
                    'model': self.config.model_name,
                    'dimension': dim
                }
            )
            self.logger.info(f"✓ Created new collection '{self.config.collection_name}'")
    
    def doc_exists(self, doc_id: str) -> bool:
        """Check if document exists"""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except:
            return False
    
    def add_batch(self, ids: List[str], texts: List[str], metadatas: List[Dict]) -> int:
        """Add batch of documents"""
        try:
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
            return len(ids)
        except Exception as e:
            self.logger.error(f"Batch add failed: {e}")
            return 0
    
    def update_batch(self, ids: List[str], texts: List[str], metadatas: List[Dict]) -> int:
        """Update batch of documents"""
        try:
            self.collection.update(ids=ids, documents=texts, metadatas=metadatas)
            return len(ids)
        except Exception as e:
            self.logger.error(f"Batch update failed: {e}")
            return 0


# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Process documents with async support"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """Load and validate documents from JSON"""
        self.logger.info(f"Loading documents from: {self.config.input_json}")
        
        if not self.config.input_json.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_json}")
        
        with open(self.config.input_json, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
        # Validate
        valid_docs = [
            doc for doc in docs
            if 'page_content' in doc 
            and 'metadata' in doc
            and doc['page_content']
            and doc['metadata'].get('course_name')
            and doc['metadata'].get('lecture_no') is not None
        ]
        
        self.logger.info(f"✓ Loaded {len(valid_docs)}/{len(docs)} valid documents")
        return valid_docs
    
    @staticmethod
    def generate_id(doc: Dict[str, Any]) -> str:
        """Generate deterministic document ID"""
        meta = doc['metadata']
        course = ''.join(c if c.isalnum() else '_' for c in meta['course_name'].lower()).strip('_')
        lecture = meta['lecture_no']
        content_hash = hashlib.md5(doc['page_content'].encode()).hexdigest()[:8]
        return f"{course}_L{lecture:03d}_{content_hash}"
    
    @staticmethod
    def prepare_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata for ChromaDB (only str, int, float, bool)"""
        clean = {}
        for k, v in metadata.items():
            if v is None:
                clean[k] = "N/A"
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, (list, dict)):
                clean[k] = json.dumps(v)
            else:
                clean[k] = str(v)
        return clean


# ============================================================================
# ASYNC INGESTION PIPELINE
# ============================================================================

class IngestionPipeline:
    """Main async ingestion pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logger(config)
        self.processor = DocumentProcessor(config, self.logger)
        self.chroma = ChromaManager(config, self.logger)
        
        # Metrics
        self.metrics = {
            'loaded': 0,
            'processed': 0,
            'ingested': 0,
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'start_time': datetime.now()
        }
    
    async def process_batch_async(self, batch: List[Dict[str, Any]]) -> Dict[str, int]:
        """Process a batch of documents asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Prepare batch data
        add_ids, add_texts, add_metas = [], [], []
        update_ids, update_texts, update_metas = [], [], []
        skipped, failed = 0, 0
        
        for doc in batch:
            try:
                doc_id = await loop.run_in_executor(None, self.processor.generate_id, doc)
                text = doc['page_content']
                meta = self.processor.prepare_metadata(doc['metadata'])
                
                # Check duplicate
                exists = await loop.run_in_executor(None, self.chroma.doc_exists, doc_id)
                
                if exists:
                    if self.config.duplicate_strategy == 'skip':
                        skipped += 1
                        continue
                    elif self.config.duplicate_strategy == 'update':
                        update_ids.append(doc_id)
                        update_texts.append(text)
                        update_metas.append(meta)
                        continue
                
                add_ids.append(doc_id)
                add_texts.append(text)
                add_metas.append(meta)
                
            except Exception as e:
                self.logger.debug(f"Document processing failed: {e}")
                failed += 1
        
        # Execute batch operations
        ingested = await loop.run_in_executor(
            None, self.chroma.add_batch, add_ids, add_texts, add_metas
        ) if add_ids else 0
        
        updated = await loop.run_in_executor(
            None, self.chroma.update_batch, update_ids, update_texts, update_metas
        ) if update_ids else 0
        
        return {
            'ingested': ingested,
            'updated': updated,
            'skipped': skipped,
            'failed': failed
        }
    
    async def run_async(self):
        """Execute async ingestion pipeline"""
        self.logger.info("=" * 80)
        self.logger.info("CHROMADB INGESTION PIPELINE - START")
        self.logger.info("=" * 80)
        
        # Load documents
        docs = self.processor.load_documents()
        self.metrics['loaded'] = len(docs)
        
        if not docs:
            self.logger.error("No documents to process!")
            return
        
        # Process in batches with progress bar
        batch_size = self.config.batch_size
        batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]
        
        self.logger.info(f"Processing {len(docs)} documents in {len(batches)} batches (batch_size={batch_size})")
        
        tasks = []
        for batch in batches:
            task = self.process_batch_async(batch)
            tasks.append(task)
        
        # Execute batches with progress tracking
        results = []
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Ingesting batches"):
            result = await coro
            results.append(result)
            
            # Update metrics
            self.metrics['ingested'] += result['ingested']
            self.metrics['updated'] += result['updated']
            self.metrics['skipped'] += result['skipped']
            self.metrics['failed'] += result['failed']
            self.metrics['processed'] += sum(result.values())
        
        # Validate
        await self._validate_async()
        
        # Print summary
        self._print_summary()
    
    async def _validate_async(self):
        """Validate ingestion"""
        loop = asyncio.get_event_loop()
        
        count = await loop.run_in_executor(None, self.chroma.collection.count)
        expected = self.metrics['ingested'] + self.metrics['updated']
        
        self.logger.info(f"Validation: {count} documents in collection")
        
        if count >= expected:
            self.logger.info("✓ Validation passed")
        else:
            self.logger.warning(f"⚠️  Expected {expected}, found {count}")
        
        # Test query
        try:
            results = await loop.run_in_executor(
                None,
                lambda: self.chroma.collection.query(
                    query_texts=["machine learning"],
                    n_results=1
                )
            )
            if results['ids']:
                self.logger.info("✓ Test query successful")
        except Exception as e:
            self.logger.error(f"Test query failed: {e}")
    
    def _print_summary(self):
        """Print execution summary"""
        duration = (datetime.now() - self.metrics['start_time']).total_seconds()
        
        print("\n" + "=" * 80)
        print("INGESTION SUMMARY")
        print("=" * 80)
        print(f"Duration: {duration:.2f}s")
        print(f"\n📄 Documents:")
        print(f"  Loaded:    {self.metrics['loaded']}")
        print(f"  Processed: {self.metrics['processed']}")
        print(f"  Ingested:  {self.metrics['ingested']}")
        print(f"  Updated:   {self.metrics['updated']}")
        print(f"  Skipped:   {self.metrics['skipped']}")
        print(f"  Failed:    {self.metrics['failed']}")
        
        if self.metrics['processed'] > 0:
            success_rate = (self.metrics['ingested'] + self.metrics['updated']) / self.metrics['processed'] * 100
            print(f"  Success:   {success_rate:.1f}%")
        
        print(f"\n🗄️  Collection: {self.config.collection_name}")
        print(f"  Location:  {self.config.chroma_dir}")
        print(f"  Total:     {self.chroma.collection.count()} documents")
        
        print("=" * 80 + "\n")
    
    def run(self):
        """Run the pipeline (sync wrapper for async)"""
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            self.logger.warning("\n⚠️  Pipeline interrupted by user")
            self._print_summary()
        except Exception as e:
            self.logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
            raise


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("ChromaDB Ingestion Pipeline (Async)")
    print("=" * 80 + "\n")
    
    try:
        config = Config()
        pipeline = IngestionPipeline(config)
        pipeline.run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()