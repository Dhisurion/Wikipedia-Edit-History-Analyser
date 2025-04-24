import os
import requests
import bz2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from collections import Counter, defaultdict
import logging
import multiprocessing
from multiprocessing import Value, Lock
import sqlite3
import time
import xml.sax
import json
from datetime import datetime
import gc
from urllib.parse import urljoin
import sys
import psutil
import traceback
import numpy as np
import seaborn as sns
import networkx as nx
from bs4 import BeautifulSoup

# Import WikipediaTagHandler class
from tag_handler import WikipediaTagHandler

# Logging configuration (set in main program)
logger = logging.getLogger("WikipediaDumpProcessor")

# Global configuration variables
DB_COMMIT_THRESHOLD = 2000  # Number of entries before executing DB commit

# This function is defined outside the class to work with multiprocessing
def process_dump_worker(dump_url, params, global_page_counter=None):
    """
    Worker function for parallel processing of a dump.
    Uses a global page counter for correct max_pages limitation.
    """
    try:
        # Unpack parameters
        extract_sentences = params.get('extract_sentences', False)
        max_pages = params.get('max_pages', None)
        max_revisions_per_page = params.get('max_revisions_per_page', None)
        force_download = params.get('force_download', False)
        skip_if_processed = params.get('skip_if_processed', True)
        processor_config = params.get('processor_config', {})
        
        # Create a dedicated processor for this worker
        processor = WikipediaDumpProcessor(**processor_config)
        
        # Extract filename for logs
        filename = dump_url.split('/')[-1]
        short_name = filename.split('-')[-1]  # Only the last part for compact logs
        logger.info(f"[Worker-{short_name}] Processing started")
        
        # Check if the dump has already been processed
        if skip_if_processed and processor.is_dump_processed(dump_url):
            logger.info(f"[Worker-{short_name}] Dump already processed, skipping")
            return dump_url, {"status": "skipped"}, filename
        
        # Download without progress bar
        logger.info(f"[Worker-{short_name}] Download starting...")
        try:
            dump_file = processor.download_dump_file(dump_url, force_download=force_download, 
                                              disable_progress=True)
            logger.info(f"[Worker-{short_name}] Download completed")
        except Exception as e:
            logger.error(f"[Worker-{short_name}] Download failed: {e}")
            return dump_url, {"error": f"Download failed: {e}"}, filename
        
        # Process with progress info in log
        logger.info(f"[Worker-{short_name}] XML parsing starting...")
        start_time = time.time()
        last_log_time = start_time
        last_page_count = 0
        last_revision_count = 0
        last_bytes = 0
        
        # Progress callback function that outputs status to the log
        def log_progress(percent, stats):
            nonlocal last_log_time, last_page_count, last_revision_count, last_bytes
            
            current_time = time.time()
            # Only update every 10 seconds (to avoid log spam)
            if current_time - last_log_time > 10:
                elapsed = current_time - start_time
                pages = stats['pages']
                revs = stats['revisions']
                mb = stats['uncompressed_mb']
                
                # Calculate changes since last update
                page_delta = pages - last_page_count
                rev_delta = revs - last_revision_count
                mb_delta = mb - last_bytes
                
                # Calculate processing speed
                interval = current_time - last_log_time
                pages_per_sec = page_delta / interval if interval > 0 else 0
                revs_per_sec = rev_delta / interval if interval > 0 else 0
                mb_per_sec = mb_delta / interval if interval > 0 else 0
                
                # Output status as log
                status_msg = (
                    f"[Worker-{short_name}] Status: {elapsed:.1f}s | "
                    f"Progress: {percent}% | "
                    f"Pages: {pages} (+{page_delta}, {pages_per_sec:.2f}/s) | "
                    f"Revisions: {revs} (+{rev_delta}, {revs_per_sec:.1f}/s) | "
                    f"Data: {mb:.1f} MB ({mb_per_sec:.2f} MB/s) | "
                    f"RAM: {psutil.Process(os.getpid()).memory_percent():.1f}%"
                )
                
                # If global counter exists, show its value too
                if global_page_counter is not None:
                    with global_page_counter.get_lock():
                        global_count = global_page_counter.value
                    status_msg += f" | Global: {global_count}/{max_pages} pages"
                
                logger.info(status_msg)
                
                # Update reference values
                last_log_time = current_time
                last_page_count = pages
                last_revision_count = revs
                last_bytes = mb
        
        # Process the dump with callback for progress display
        try:
            # Pass the global page counter to the process_dump_file_stream method
            tag_counter, tagged_revisions, sentence_changes, stats = processor.process_dump_file_stream(
                dump_file, dump_url, extract_sentences, max_pages, max_revisions_per_page,
                disable_progress=True,  # Disable internal progress bar
                progress_callback=log_progress,  # Use log-based callback
                global_page_counter=global_page_counter  # Pass global counter
            )
            
            processing_time = time.time() - start_time
            
            # Rest of the function remains unchanged
            if "page_count" in stats:
                logger.info(
                    f"[Worker-{short_name}] Parsing completed in {processing_time:.2f}s: "
                    f"{stats['page_count']} pages, {stats['revision_count']} revisions, "
                    f"{stats.get('uncompressed_bytes', 0)/1024/1024:.1f} MB uncompressed data"
                )
            else:
                logger.warning(f"[Worker-{short_name}] Parsing completed without complete statistics")
            
            if "processing_time" not in stats:
                stats["processing_time"] = processing_time
            
            if processor.cleanup_after_processing and os.path.exists(dump_file):
                os.remove(dump_file)
                logger.info(f"[Worker-{short_name}] Dump file deleted")

            stats['tagged_revision_count'] = tagged_revisions
            stats['sentence_change_count'] = sentence_changes
            
            gc.collect()
            
            return dump_url, stats, filename
            
        except Exception as e:
            logger.error(f"[Worker-{short_name}] Error during parsing: {e}")
            traceback_str = traceback.format_exc()
            logger.error(f"[Worker-{short_name}] Stacktrace: {traceback_str}")
            return dump_url, {"error": str(e)}, filename
            
    except Exception as e:
        logger.error(f"[Worker] Error processing {dump_url}: {e}")
        filename = dump_url.split('/')[-1]
        return dump_url, {"error": str(e)}, filename

class WikipediaDumpProcessor:
    """
    Highly optimized class for processing Wikipedia dump files
    with focus on speed and memory efficiency.
    """
    
    def __init__(self, dump_directory="./dumps", results_directory="./results", 
                 db_path="./wikipedia_analysis.db", cleanup_after_processing=False, 
                 cleanup_extracted_files=False, use_multiprocessing=True, max_workers=None):
        """
        Initializes the dump processor.
        
        Parameters:
        dump_directory (str): Directory for storing/reading dump files
        results_directory (str): Directory for storing results
        db_path (str): Path to SQLite database for incremental storage
        cleanup_after_processing (bool): Whether dump files should be deleted after processing
        cleanup_extracted_files (bool): Whether extracted files should be deleted after processing
        use_multiprocessing (bool): Whether multiple processes should be used for processing
        max_workers (int, optional): Maximum number of parallel processes (default: CPU cores)
        """
        self.dump_directory = dump_directory
        self.results_directory = results_directory
        self.db_path = db_path
        self.cleanup_after_processing = cleanup_after_processing
        self.cleanup_extracted_files = cleanup_extracted_files
        self.use_multiprocessing = use_multiprocessing
        
        # Performance optimization: Automatic worker configuration
        if max_workers is None:
            # Determine available resources
            physical_cores = psutil.cpu_count(logical=False) or 1
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Determine optimal worker count based on RAM and cores
            # More RAM allows more parallel processes
            if memory_gb > 32:
                optimal_workers = min(physical_cores, 32)
            elif memory_gb > 16:
                optimal_workers = min(physical_cores, 16)
            elif memory_gb > 8:
                optimal_workers = min(physical_cores, 8)
            else:
                optimal_workers = max(1, physical_cores // 2)
            
            self.max_workers = optimal_workers
            logger.info(f"Automatically configured: {self.max_workers} workers based on {physical_cores} cores and {memory_gb:.1f} GB RAM")
        else:
            self.max_workers = max_workers
        
        # Performance monitoring
        self._last_optimization = time.time()
        self._proc_speed_history = []
        
        # Define tag categories
        self.tag_categories = {
            'Content Policy': [
                'npov', 'pov', 'blp', 'or', 'verify', 'citation needed', 
                'copyvio', 'copyright', 'npov-section', 'coi', 'biased',
                'peacock', 'weasel', 'disputed', 'contentious', 'neutral point of view',
                'unbalanced', 'factual accuracy', 'disputed-section', 'undue weight',
                'fairuse', 'nfcc', 'promotional', 'advert', 'advertisement'
            ],
            'Content Quality': [
                'cleanup', 'stub', 'merge', 'split', 'wikify', 'factual', 
                'grammar', 'clarify', 'style', 'disambiguation', 'redirect',
                'copyedit', 'refactor', 'rewrite', 'expand', 'update',
                'expert-subject', 'tone', 'technical', 'lead rewrite',
                'reorganize', 'structure', 'prose', 'jargon', 'accessibility',
                'notable', 'notability', 'unreferenced', 'primary', 'secondary',
                'tertiary', 'fringe', 'context', 'unencyclopedic', 'essay-like',
                'howto', 'puffery', 'globalize', 'overcategorization'
            ],
            'Edit Type': [
                'minor', 'revert', 'rollback', 'undo', 'rv', 'afd', 'rfd',
                'move', 'merge', 'split', 'redirect', 'disambig', 'dab',
                'undid', 'restored', 'cleanup', 'fixed', 'corrected', 'added',
                'removed', 'section', 'formatting', 'links', 'categories',
                'rename', 'retarget', 'typo', 'spelling', 'grammar', 'punctuation',
                'capitalisation', 'wikify', 'dewikify', 'unsourced', 'unreferenced'
            ],
            'Edit Tools': [
                'visualeditor', 'visualeditor-source', 'visualeditor-wikitext',
                'mobileedit', 'mobile edit', 'mobile-web-edit', 'mobile-app-edit', 
                'awb', 'twinkle', 'huggle', 'popups', 'refill', 'reflinks',
                'cosmetic cleanup', 'citewatch', 'citationbot', 'hovercraft',
                'rater', 'teahouse', 'reviewer', 'rfa', 'rfb', 'mediation',
                'iaa', 'afe', 'afc', 'wikiproject', 'talk page', 'discussion'
            ],
            'Vandalism & Maintenance': [
                'vandalism', 'spam', 'socialuse', 'test', 'experiment',
                'sockpuppet', 'sock', 'meat', 'meatpuppet', 'disrupt',
                'disruption', 'lta', 'attack', 'personal attack', 'harassment',
                'outing', 'doxing', 'legal threat', 'block', 'banned',
                'rangeblock', 'checkuser', 'oversight', 'suppression',
                'speedy', 'prod', 'afd', 'g4', 'g6', 'g7', 'g11', 'g12',
                'copypaste', 'close paraphrasing', 'unsourced', 'hoax'
            ],
            'Automated Edits': [
                'bot', 'robot', 'automated', 'script', 'batch', 'massupload',
                'massmove', 'template bot', 'category bot', 'citation bot',
                'interwiki bot', 'cleanup bot', 'fix bot', 'userspace',
                'gadget', 'tool', 'api', 'pywikibot', 'mwclient', 'abusefilter'
            ],
            'Article Quality': [
                'FA', 'FL', 'GA', 'A', 'B', 'C', 'Start', 'Stub', 'Featured',
                'Good', 'List', 'quality', 'class', 'importance', 'priority',
                'top', 'high', 'mid', 'low', 'assess', 'reassess', 'rating'
            ],
            'Multimedia Content': [
                'image', 'file', 'picture', 'photo', 'diagram', 'svg', 'png',
                'jpg', 'gif', 'mp3', 'ogg', 'audio', 'video', 'media',
                'commons', 'caption', 'thumbnail', 'thumb', 'gallery',
                'illustration', 'map', 'chart', 'graph', 'plot', 'animation',
                'screenshot', 'portrait', 'figurative', 'infobox', 'sidebar'
            ],
            'Topics & Subject Areas': [
                'science', 'history', 'biology', 'chemistry', 'physics',
                'mathematics', 'literature', 'music', 'film', 'geography',
                'politics', 'religion', 'sports', 'medicine', 'technology',
                'computing', 'internet', 'art', 'architecture', 'business',
                'economics', 'philosophy', 'psychology', 'sociology', 'language',
                'education', 'entertainment', 'television', 'radio', 'food',
                'drink', 'travel', 'transportation', 'military', 'law',
                'crime', 'environment', 'space', 'biography', 'fiction'
            ]
        }
        
        # Create directories if they don't exist
        for directory in [dump_directory, results_directory]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize the database
        self.init_database()
        
        # Database caches for better performance
        self.db_cache = {
            'tags': [],
            'revisions': [],
            'sentence_changes': []
        }
    
    def init_database(self):
        """
        Initialize the SQLite database for incremental storage.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Performance optimization: Pragmas for better SQLite performance
        cursor.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for better concurrency
        cursor.execute("PRAGMA synchronous = NORMAL")  # Less disk synchronization
        cursor.execute("PRAGMA cache_size = 10000")  # Larger cache (in pages)
        cursor.execute("PRAGMA temp_store = MEMORY")  # Temporary tables in memory
        cursor.execute("PRAGMA mmap_size = 30000000000")  # Memory-Mapped I/O (30GB limit)

        try:
            cursor.execute("ALTER TABLE sentence_changes ADD COLUMN similarity REAL DEFAULT 0.0")
            logger.info("Added similarity column to sentence_changes")
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass
        
        # Table for processed dumps
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_dumps (
            dump_url TEXT PRIMARY KEY,
            filename TEXT,
            processed_at TIMESTAMP,
            page_count INTEGER,
            revision_count INTEGER,
            tagged_revision_count INTEGER,
            sentence_change_count INTEGER
        )
        ''')
        
        # Table for tags
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            tag TEXT,
            category TEXT,
            count INTEGER,
            dump_url TEXT,
            PRIMARY KEY (tag, dump_url)
        )
        ''')
        
        # Table for tagged revisions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tagged_revisions (
            revision_id TEXT,
            page_id TEXT,
            page_title TEXT,
            timestamp TEXT,
            comment TEXT,
            tags TEXT,
            dump_url TEXT,
            PRIMARY KEY (revision_id, dump_url)
        )
        ''')
        
        # Table for sentence changes
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentence_changes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            revision_id TEXT,
            page_id TEXT,
            page_title TEXT,
            timestamp TEXT,
            change_type TEXT,
            old_sentence TEXT,
            new_sentence TEXT,
            similarity REAL,  -- New field for similarity value
            tags TEXT,
            dump_url TEXT
        )
        ''')
        
        # Indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tagged_revisions_page_id ON tagged_revisions(page_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tagged_revisions_page_title ON tagged_revisions(page_title)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentence_changes_page_id ON sentence_changes(page_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentence_changes_change_type ON sentence_changes(change_type)")
        
        conn.commit()
        conn.close()
    
    def is_dump_processed(self, dump_url):
        """
        Checks if a dump has already been processed.
        
        Parameters:
        dump_url (str): URL of the dump
        
        Returns:
        bool: True if the dump has already been processed, False otherwise
        """
        normalized_dumpurl = dump_url.rstrip('/')
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM processed_dumps WHERE dump_url = ?", (normalized_dumpurl,))
        result = cursor.fetchone() is not None
        
        conn.close()
        return result
    
    def mark_dump_as_processed(self, dump_url, filename, page_count, revision_count, 
                              tagged_revision_count, sentence_change_count):
        """
        Marks a dump as processed in the database.
        
        Parameters:
        dump_url (str): URL of the dump
        filename (str): Filename of the dump
        page_count (int): Number of processed pages
        revision_count (int): Number of processed revisions
        tagged_revision_count (int): Number of tagged revisions
        sentence_change_count (int): Number of sentence changes
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT OR REPLACE INTO processed_dumps 
        (dump_url, filename, processed_at, page_count, revision_count, 
         tagged_revision_count, sentence_change_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            dump_url, filename, datetime.now().isoformat(),
            page_count, revision_count, tagged_revision_count, sentence_change_count
        ))
        
        conn.commit()
        conn.close()
        
        # Empty the database caches after completing a dump
        self.db_cache = {
            'tags': [],
            'revisions': [],
            'sentence_changes': []
        }
    
    def save_tags_to_db(self, tag_counter, dump_url):
        """
        Saves tags to the database.
        
        Parameters:
        tag_counter (Counter): Counter object with tags and their frequencies
        dump_url (str): URL of the dump
        """
        # Batch processing for better performance
        for tag, count in tag_counter.items():
            category = "Other"
            tag_lower = tag.lower()
            
            # Search for the matching category
            for cat_name, cat_tags in self.tag_categories.items():
                if any(cat_tag in tag_lower for cat_tag in cat_tags):
                    category = cat_name
                    break
            
            # Add to cache
            self.db_cache['tags'].append((tag, category, count, dump_url))
        
        # Check if the cache is full enough to save a batch
        if len(self.db_cache['tags']) >= DB_COMMIT_THRESHOLD:
            self._save_tags_batch()
    
    def _save_tags_batch(self):
        """Saves a batch of tags to the database."""
        if not self.db_cache['tags']:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Use a transaction for faster insertion
        cursor.execute("BEGIN TRANSACTION")
        
        for tag, category, count, dump_url in self.db_cache['tags']:
            cursor.execute("""
            INSERT OR REPLACE INTO tags (tag, category, count, dump_url)
            VALUES (?, ?, ?, ?)
            """, (tag, category, count, dump_url))
        
        cursor.execute("COMMIT")
        conn.close()
        
        # Empty the cache
        self.db_cache['tags'] = []
    
    def save_revisions_to_db(self, tagged_revisions, dump_url):
        """
        Saves tagged revisions to the database.
        
        Parameters:
        tagged_revisions (list): List of tagged revisions
        dump_url (str): URL of the dump
        """
        if not tagged_revisions:
            return
        
        # Batch processing: Add to cache
        for revision in tagged_revisions:
            self.db_cache['revisions'].append((
                revision["revision_id"],
                revision["page_id"],
                revision["page_title"],
                revision["timestamp"],
                revision["comment"],
                json.dumps(revision["tags"]),
                dump_url
            ))
        
        # Check if the cache is full enough to save a batch
        if len(self.db_cache['revisions']) >= DB_COMMIT_THRESHOLD:
            self._save_revisions_batch()
    
    def _save_revisions_batch(self):
        """Saves a batch of revisions to the database."""
        if not self.db_cache['revisions']:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        unique_ids = set()
        duplicates = 0
        for data in self.db_cache['revisions']:
            rev_id = data[0]  # revision_id is at position 0
            if rev_id in unique_ids:
                duplicates += 1
            else:
                unique_ids.add(rev_id)
        
        if duplicates > 0:
            logger.info(f"Found: {duplicates} duplicate revision IDs in cache before saving")
        
        # Use a transaction for faster insertion
        cursor.execute("BEGIN TRANSACTION")
        
        inserted = 0
        ignored = 0
        for data in self.db_cache['revisions']:
            # Check if the revision already exists in the DB
            cursor.execute("SELECT 1 FROM tagged_revisions WHERE revision_id = ? AND dump_url = ?", 
                        (data[0], data[6]))
            if cursor.fetchone():
                ignored += 1
            else:
                cursor.execute("""
                INSERT INTO tagged_revisions 
                (revision_id, page_id, page_title, timestamp, comment, tags, dump_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, data)
                inserted += 1
        
        cursor.execute("COMMIT")
        logger.info(f"Revisions saved to DB: {inserted} new, {ignored} ignored (duplicates)")
        
        conn.close()
        self.db_cache['revisions'] = []
    
    def save_sentence_changes_to_db(self, sentence_changes, dump_url):
        """
        Saves sentence changes to the database.
        """
        if not sentence_changes:
            return
        
        # Debug counters for similarity values
        modification_count = 0
        non_zero_count = 0
        
        # Batch processing: Add to cache
        for change in sentence_changes:
            # Performance optimization: Limit very long sentences
            old_sentence = change.get("old_sentence")
            if old_sentence and len(old_sentence) > 2000:
                old_sentence = old_sentence[:2000] + "..."
            
            new_sentence = change.get("new_sentence")
            if new_sentence and len(new_sentence) > 2000:
                new_sentence = new_sentence[:2000] + "..."
            
            # Extract the similarity value, ensure it's a float
            try:
                if change.get("change_type") == "modification":
                    modification_count += 1
                    similarity = float(change.get("similarity", 0.0))
                    if similarity > 0.0:
                        non_zero_count += 1
                        logger.debug(f"Non-zero similarity: {similarity} for change from '{old_sentence[:30]}...' to '{new_sentence[:30]}...'")
                else:
                    similarity = 0.0
            except (ValueError, TypeError):
                similarity = 0.0
                logger.warning(f"Invalid similarity value in change: {change.get('similarity')}")
            
            self.db_cache['sentence_changes'].append((
                change["revision_id"],
                change["page_id"],
                change["page_title"],
                change["timestamp"],
                change["change_type"],
                old_sentence,
                new_sentence,
                similarity,  # Use the correctly processed value
                json.dumps(change["tags"]),
                dump_url
            ))
        
        # Log summary
        if modification_count > 0:
            logger.info(f"Saving {modification_count} modifications, {non_zero_count} with similarity > 0")
        
        # Check if the cache is full enough to save a batch
        if len(self.db_cache['sentence_changes']) >= DB_COMMIT_THRESHOLD:
            self._save_sentence_changes_batch()
    
    def _save_sentence_changes_batch(self):
        """Saves a batch of sentence changes to the database."""
        if not self.db_cache['sentence_changes']:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if the database column for similarity exists and has the correct type
        try:
            cursor.execute("PRAGMA table_info(sentence_changes)")
            columns = {col[1]: col[2] for col in cursor.fetchall()}
            
            if 'similarity' not in columns:
                logger.warning("Column 'similarity' missing in table. Adding it...")
                cursor.execute("ALTER TABLE sentence_changes ADD COLUMN similarity REAL DEFAULT 0.0")
            elif columns['similarity'] != 'REAL':
                logger.warning(f"Column 'similarity' has wrong type: {columns['similarity']}. Should be REAL.")
        except Exception as e:
            logger.error(f"Error checking database structure: {e}")
        
        # Use a transaction for faster insertion
        cursor.execute("BEGIN TRANSACTION")
        
        try:
            # Counter for changes with similarity
            similarity_count = 0
            
            for data in self.db_cache['sentence_changes']:
                # Explicitly check similarity value
                if data[7] > 0.0:  # similarity is at position 7 in the tuple
                    similarity_count += 1
                
                cursor.execute("""
                INSERT INTO sentence_changes 
                (revision_id, page_id, page_title, timestamp, change_type, 
                old_sentence, new_sentence, similarity, tags, dump_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, data)
            
            # Log info about saved similarity values
            if similarity_count > 0:
                logger.info(f"Written to database: {similarity_count} entries with similarity > 0")
            
            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            logger.error(f"Error saving sentence_changes: {e}")
        finally:
            conn.close()
        
        # Empty the cache
        self.db_cache['sentence_changes'] = []
    
    def flush_all_caches(self):
        """Saves all pending cache entries to the database."""
        self._save_tags_batch()
        self._save_revisions_batch()
        self._save_sentence_changes_batch()
    
    def optimize_runtime_performance(self):
        """
        Optimizes runtime performance for long processing tasks.
        
        Returns:
        function: Performance monitoring function
        """
        # Activate more aggressive garbage collection
        gc.set_threshold(700, 10, 5)  # Default values are 700, 10, 10
        
        # Initialize performance monitoring
        self._last_optimization = time.time()
        self._proc_speed_history = []
        
        def performance_monitor(processed_bytes, elapsed_time):
            """Monitors performance and performs optimizations when needed."""
            current_time = time.time()
            
            # Only check every 2 minutes
            if current_time - self._last_optimization < 120:
                return
                
            speed = processed_bytes / elapsed_time if elapsed_time > 0 else 0
            self._proc_speed_history.append(speed)
            
            # Detect performance drop
            if len(self._proc_speed_history) >= 3:
                recent_speeds = self._proc_speed_history[-3:]
                if recent_speeds[0] > recent_speeds[-1] * 1.5:  # 50% drop
                    logger.warning(f"Performance drop detected: {recent_speeds[0]:.2f} KB/s → {recent_speeds[-1]:.2f} KB/s")
                    
                    # Flush all DB caches
                    self.flush_all_caches()
                    
                    # Aggressive memory cleanup
                    gc.collect(generation=2)
                    
                    # Clear cache (if available)
                    if 'process' in sys.modules:
                        if hasattr(sys.modules['process'], '_processCache'):
                            sys.modules['process']._processCache.clear()
                    
                    # Clear global fuzzy cache
                    from tag_handler import _fuzzy_cache
                    _fuzzy_cache.clear()
                    
                    # Limit the history
                    if len(self._proc_speed_history) > 10:
                        self._proc_speed_history = self._proc_speed_history[-10:]
                        
                    logger.info("Performance optimization completed")
                    
            self._last_optimization = current_time
        
        return performance_monitor
    
    def fetch_dump_links(self, base_url="https://dumps.wikimedia.org/enwiki/20250201/", 
                        file_pattern="pages-meta-history", extension=".bz2", 
                        page_ranges=None, max_dumps=None, verbose=True):
        """
        Extracts links to dump files from the Wikipedia dumps page.
        
        Parameters:
        base_url (str): Base URL for the dumps
        file_pattern (str): Pattern for filenames (e.g., 'pages-meta-history')
        extension (str): File extension (.bz2 or .7z)
        page_ranges (list, optional): List of page ranges
        max_dumps (int, optional): Maximum number of links to extract
        verbose (bool): Whether detailed information should be output
        
        Returns:
        list: List of URLs to dump files
        """
        if verbose:
            logger.info(f"Extracting dump links from {base_url}")
        
        # Normalize the base URL (ensure it ends with a slash)
        if not base_url.endswith('/'):
            base_url = base_url + '/'
        
        # Get the HTML page with timeout and retry mechanism
        max_retries = 3
        for retry in range(max_retries):
            try:
                response = requests.get(base_url, timeout=30)
                if response.status_code == 200:
                    break
                elif retry == max_retries - 1:
                    logger.error(f"Error retrieving the dump page: {response.status_code}")
                    return []
                else:
                    logger.warning(f"Retry {retry+1}/{max_retries}: Status {response.status_code}")
                    time.sleep(2)  # Short pause before retrying
            except requests.RequestException as e:
                if retry == max_retries - 1:
                    logger.error(f"Network error retrieving the dump page: {e}")
                    return []
                else:
                    logger.warning(f"Retry {retry+1}/{max_retries}: {e}")
                    time.sleep(2)
        
        # Parse the HTML page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links to dump files
        dump_links = []
        for link in soup.find_all('a'):
            href = link.get('href', '')
            
            # Filter links based on pattern and extension
            if file_pattern in href and href.endswith(extension):
                
                # Filter by page ranges, if specified
                if page_ranges:
                    if not any(range_pattern in href for range_pattern in page_ranges):
                        continue
                
                # Create the full URL
                full_url = urljoin(base_url, href)
                
                # Check if the dump has already been processed
                if self.is_dump_processed(full_url):
                    if verbose:
                        logger.info(f"Dump already processed, skipping: {href}")
                    continue
                
                dump_links.append(full_url)
                
                if verbose:
                    size_text = link.find_next('td').text.strip() if link.find_next('td') else 'Size unknown'
                    logger.info(f"Found: {href} ({size_text})")
        
        # Sort links to ensure a consistent order
        dump_links.sort()
        
        # Limit the number of links, if specified
        if max_dumps and len(dump_links) > max_dumps:
            dump_links = dump_links[:max_dumps]
            if verbose:
                logger.info(f"Limited to {max_dumps} dump files")
        
        if verbose:
            logger.info(f"Total of {len(dump_links)} new dump links found")
        
        return dump_links
    
    def download_dump_file(self, dump_url, save_path=None, force_download=False, disable_progress=False):
        """
        Downloads a Wikipedia dump file.
        
        Parameters:
        dump_url (str): URL of the dump file
        save_path (str, optional): Path to save
        force_download (bool): Whether the file should be downloaded again
        disable_progress (bool): Whether the progress display should be disabled
        
        Returns:
        str: Path to the downloaded file
        """
        if save_path is None:
            # Extract the filename from the URL
            filename = dump_url.split('/')[-1]
            save_path = os.path.join(self.dump_directory, filename)
        
        # Check if the file already exists
        if os.path.exists(save_path) and not force_download:
            logger.info(f"File {save_path} already exists, skipping download.")
            return save_path
        
        filename = os.path.basename(save_path)
        logger.info(f"Downloading {dump_url}...")
        
        # Use streaming for more efficient downloading with timeout and retry
        max_retries = 3
        for retry in range(max_retries):
            try:
                with requests.get(dump_url, stream=True, timeout=60) as response:
                    response.raise_for_status()
                    
                    # Determine the total size of the file (if available)
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(save_path, 'wb') as f:
                        if disable_progress:
                            # Without progress display
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        else:
                            # With progress display, but leave=False for less clutter
                            with tqdm(
                                total=total_size, 
                                unit='B', 
                                unit_scale=True, 
                                desc=f"Download {filename}",
                                leave=False
                            ) as progress_bar:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                                        progress_bar.update(len(chunk))
                
                # Download successful, break the loop
                break
                
            except (requests.RequestException, IOError) as e:
                # On error: Delete partially downloaded file
                if os.path.exists(save_path):
                    os.remove(save_path)
                
                if retry < max_retries - 1:
                    wait_time = 5 * (retry + 1)  # Progressive backoff
                    logger.warning(f"Download error (attempt {retry+1}/{max_retries}): {e}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Download failed after {max_retries} attempts: {e}")
                    raise
        
        logger.info(f"Download completed: {save_path}")
        return save_path
    
    def process_dump_file_stream(self, dump_path, dump_url, extract_sentences=False,
                         max_pages=None, max_revisions_per_page=None, 
                         disable_progress=False, progress_callback=None,
                         global_page_counter=None):  # New parameter
        """
        Processes a dump file directly from the compressed stream without full extraction.
        """
        logger.info(f"Processing compressed dump: {dump_path}")
        filename = os.path.basename(dump_path)
        start_time = time.time()
        
        # Helper function to format time
        def format_time(seconds):
            """Formats seconds into a human-readable format (HH:MM:SS)"""
            hours, remainder = divmod(int(seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                return f"{hours}h {minutes}m"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        
        if dump_path.endswith(".bz2"):
            # For bz2 files
            try:
                # Estimate the uncompressed size for the progress bar
                # Use constant ratio of 15:1 for Wikipedia dumps
                compressed_size = os.path.getsize(dump_path)
                known_ratio = 15.0 if "wiki" in dump_path.lower() else None
                estimated_size, compression_ratio = self.estimate_uncompressed_size(dump_path, known_ratio)
                
                # Log detailed information
                logger.info(
                    f"File info: {filename}\n"
                    f"  Compressed size: {self.format_size(compressed_size)}\n"
                    f"  Estimated uncompressed size: {self.format_size(estimated_size)}\n"
                    f"  Estimated compression ratio: {compression_ratio:.1f}:1"
                )
                
                # Log processing limits if present
                if max_pages or max_revisions_per_page:
                    limit_info = []
                    if max_pages:
                        limit_info.append(f"max_pages={max_pages}")
                    if max_revisions_per_page:
                        limit_info.append(f"max_revisions_per_page={max_revisions_per_page}")
                        
                    logger.info(f"Processing with limits: {', '.join(limit_info)}")
                
                # SAX parser for efficient streaming parsing
                handler = WikipediaTagHandler(
                    self.tag_categories, max_pages, max_revisions_per_page, 
                    extract_sentences, global_counter=global_page_counter  # Pass the global counter
    )
                
                # Flag for successful completion due to max_pages
                max_pages_reached = False
                
                with bz2.BZ2File(dump_path, 'rb') as dump_file:
                    parser = xml.sax.make_parser()
                    parser.setContentHandler(handler)
                    
                    # Initialize performance monitoring
                    performance_monitor = self.optimize_runtime_performance()
                    process_start_time = time.time()
                    
                    # Stream-based parsing with improved progress display
                    if disable_progress:
                        # Process without progress display
                        buffer_size = 1024 * 1024  # 1MB
                        processed = 0
                        uncompressed_bytes = 0
                        
                        try:
                            buffer = dump_file.read(buffer_size)
                            while buffer:
                                parser.feed(buffer)
                                processed += len(buffer)
                                
                                # Count uncompressed bytes (via stream position)
                                uncompressed_bytes = dump_file.tell()
                                
                                # Call performance monitoring
                                current_time = time.time()
                                elapsed = current_time - process_start_time
                                performance_monitor(uncompressed_bytes, elapsed)
                                
                                # Progress callback for external progress bar
                                if progress_callback:
                                    # Consider both bytes and page limits
                                    limit_percent = handler.estimate_completion_percent()
                                    if limit_percent is not None:
                                        # If limits exist, use this estimate
                                        progress_percent = int(limit_percent * 100)
                                    elif estimated_size > 0:
                                        # Otherwise fallback to byte-based estimate
                                        progress_percent = min(100, int(uncompressed_bytes * 100 / estimated_size))
                                    else:
                                        progress_percent = 0
                                        
                                    stats = {
                                        'pages': handler.page_count,
                                        'revisions': handler.revision_count,
                                        'uncompressed_mb': uncompressed_bytes / 1024 / 1024,
                                        'elapsed': elapsed,
                                        'limit_based': limit_percent is not None
                                    }
                                    progress_callback(progress_percent, stats)
                                
                                buffer = dump_file.read(buffer_size)
                        except xml.sax.SAXException as e:
                            if "Maximum number of pages reached" in str(e) or "Maximum number of article pages reached" in str(e) or "Maximum number of article pages globally reached" in str(e):
                                logger.info("Maximum number of pages reached, ending parsing.")
                                max_pages_reached = True  # Mark as successful!
                            else:
                                logger.warning(f"SAX error during parsing: {e}")
                        finally:
                            try:
                                parser.close()
                            except xml.sax.SAXException as close_error:
                                # Ignore errors when closing the parser if max_pages was reached
                                if max_pages_reached or "Maximum number of pages reached" in str(close_error) or "Maximum number of article pages reached" in str(close_error) or "Maximum number of article pages globally reached" in str(close_error):
                                    logger.info("Parser successfully closed after reaching page limit.")
                                    pass
                                else:
                                    logger.warning(f"SAX error when closing parser: {close_error}")
                    else:
                        # With progress display
                        buffer_size = 1024 * 1024  # 1MB
                        uncompressed_bytes = 0
                        
                        # Create a progress bar that will end at 100%
                        with tqdm(
                            total=100,  # Use percent instead of bytes
                            unit='%',
                            desc=f"Parsing {filename}",
                            leave=False
                        ) as pbar:
                            last_stats_update = time.time()
                            last_bytes = 0
                            last_percent = 0
                            speed_window = []  # For moving average
                            
                            try:
                                buffer = dump_file.read(buffer_size)
                                while buffer:
                                    parser.feed(buffer)
                                    
                                    # Determine the actual position in the uncompressed stream
                                    new_position = dump_file.tell()
                                    bytes_processed = new_position - uncompressed_bytes
                                    uncompressed_bytes = new_position
                                    
                                    # Update the progress bar based on limits
                                    limit_percent = handler.estimate_completion_percent()
                                    if limit_percent is not None:
                                        # If limits exist, use this estimate
                                        current_percent = int(limit_percent * 100)
                                    else:
                                        # Otherwise fallback to byte-based estimate
                                        current_percent = min(100, int(uncompressed_bytes * 100 / estimated_size))
                                    
                                    # Update the progress bar only if the percentage changes
                                    if current_percent > last_percent:
                                        pbar.update(current_percent - last_percent)
                                        last_percent = current_percent
                                    
                                    # Call performance monitoring
                                    current_time = time.time()
                                    elapsed = current_time - process_start_time
                                    
                                    # Update statistics every 2 seconds
                                    if current_time - last_stats_update > 2:
                                        # Calculate current speed (bytes/s)
                                        interval = current_time - last_stats_update
                                        if interval > 0:
                                            bytes_per_sec = (uncompressed_bytes - last_bytes) / interval
                                            speed_window.append(bytes_per_sec)
                                            # Keep only the last 5 measurements for moving average
                                            if len(speed_window) > 5:
                                                speed_window.pop(0)
                                        
                                        # Average speed over the window
                                        avg_speed = sum(speed_window) / len(speed_window) if speed_window else 0
                                        
                                        # Calculate estimated remaining time
                                        eta_str = "unknown"
                                        
                                        if limit_percent is not None and avg_speed > 0:
                                            # For limit-based progress
                                            if limit_percent > 0:
                                                # Estimate based on progress so far
                                                remaining_time = elapsed * (1 - limit_percent) / limit_percent
                                                eta_str = f"{format_time(remaining_time)}"
                                        elif avg_speed > 0 and uncompressed_bytes < estimated_size:
                                            # For size-based progress 
                                            remaining_bytes = estimated_size - uncompressed_bytes
                                            remaining_seconds = remaining_bytes / avg_speed
                                            eta_str = f"{format_time(remaining_seconds)}"
                                        
                                        # Additional status text for progress type
                                        progress_type = "Pages/Revisions" if limit_percent is not None else "Bytes"
                                        
                                        # Update the progress bar with detailed info
                                        pbar.set_postfix({
                                            'Pages': handler.page_count,
                                            'Revs': handler.revision_count,
                                            'MB/s': f"{avg_speed/1024/1024:.2f}",
                                            'ETA': eta_str,
                                            'Based on': progress_type
                                        })
                                        
                                        # Save values for next iteration
                                        last_stats_update = current_time
                                        last_bytes = uncompressed_bytes
                                        
                                        # Performance monitor for memory optimization
                                        performance_monitor(uncompressed_bytes, elapsed)
                                    
                                    # Progress callback for external progress bar, if desired
                                    if progress_callback:
                                        stats = {
                                            'pages': handler.page_count,
                                            'revisions': handler.revision_count,
                                            'uncompressed_mb': uncompressed_bytes / 1024 / 1024,
                                            'elapsed': elapsed,
                                            'limit_based': limit_percent is not None
                                        }
                                        progress_callback(current_percent, stats)
                                    
                                    buffer = dump_file.read(buffer_size)
                                    
                                    # Periodically save intermediate results to the database
                                    if len(handler.tagged_revisions) > DB_COMMIT_THRESHOLD:
                                        self.save_revisions_to_db(handler.tagged_revisions, dump_url)
                                        handler.tagged_revisions = []
                                    
                                    if len(handler.sentence_changes) > DB_COMMIT_THRESHOLD:
                                        self.save_sentence_changes_to_db(handler.sentence_changes, dump_url)
                                        handler.sentence_changes = []
                                        
                            except xml.sax.SAXException as e:
                                if "Maximum number of pages reached" in str(e):
                                    logger.info("Maximum number of pages reached, ending parsing.")
                                    max_pages_reached = True
                                    
                                    # Set progress to 100% when limit is reached
                                    if last_percent < 100:
                                        pbar.update(100 - last_percent)
                                else:
                                    logger.warning(f"SAX error during parsing: {e}")
                            except Exception as e:
                                logger.error(f"Error during parsing: {e}")
                                raise
                            finally:
                                try:
                                    parser.close()
                                except xml.sax.SAXException:
                                    if max_pages_reached:
                                        pass
                                    else:
                                        raise
                
                # Ensure all remaining data is saved
                self.save_tags_to_db(handler.tag_counter, dump_url)
                self.save_revisions_to_db(handler.tagged_revisions, dump_url)
                self.save_sentence_changes_to_db(handler.sentence_changes, dump_url)
                self.flush_all_caches()
                
                # Calculate actual compression ratio (if known)
                if uncompressed_bytes > 0:
                    actual_ratio = uncompressed_bytes / compressed_size
                    logger.info(f"Actual compression ratio: {actual_ratio:.1f}:1")
                    
                    # If the actual ratio differs significantly from the estimate, note for future reference
                    if abs(actual_ratio - compression_ratio) > 2.0:
                        logger.info(f"Note: Actual compression ratio differs from estimate "
                                f"({actual_ratio:.1f}:1 vs. {compression_ratio:.1f}:1)")
                
                processing_time = time.time() - start_time
                logger.info(f"Processing completed in {processing_time:.2f} seconds")
                logger.info(f"Pages: {handler.page_count}, Revisions: {handler.revision_count}, "
                        f"Tagged revisions: {len(handler.tagged_revisions)}, "
                        f"Sentence changes: {len(handler.sentence_changes)}")
                
                # Log memory usage
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                logger.info(f"Current memory usage: {memory_info.rss / 1024**2:.1f} MB ({process.memory_percent():.1f}%)")
                
                stats = {
                    "page_count": handler.page_count,
                    "revision_count": handler.revision_count,
                    "tagged_revision_count": len(handler.tagged_revisions),
                    "sentence_change_count": len(handler.sentence_changes),
                    "processing_time": processing_time,
                    "processing_stats": handler.processing_stats,
                    "max_pages_reached": max_pages_reached,
                    "uncompressed_bytes": uncompressed_bytes,
                    "compressed_size": compressed_size,
                    "compression_ratio": uncompressed_bytes / compressed_size if compressed_size > 0 else 0
                }
                
                # Explicit garbage collection at the end
                gc.collect()
                self.flush_all_caches()
                logger.info(f"All caches fully written to the database")
                
                return (handler.tag_counter, handler.tagged_revisions, 
                    handler.sentence_changes, stats)
            
            except Exception as e:
                logger.error(f"Error processing {dump_path}: {e}")
                traceback_str = traceback.format_exc()
                logger.error(f"Stacktrace: {traceback_str}")
                return Counter(), [], [], {"error": str(e)}
        else:
            logger.error(f"Unsupported file format: {dump_path}")
            return Counter(), [], [], {"error": "Unsupported file format"}
    
    def estimate_uncompressed_size(self, bz2_file_path, known_ratio=None):
        """
        Improved estimation of the uncompressed size of a BZ2 file.
        
        Parameters:
        bz2_file_path (str): Path to the BZ2 file
        known_ratio (float, optional): A known compression ratio, if available
        
        Returns:
        tuple: (estimated uncompressed size, compression ratio)
        """
        try:
            # Determine the size of the compressed file
            compressed_size = os.path.getsize(bz2_file_path)
            
            # If a known ratio was specified, use it
            if known_ratio is not None and known_ratio > 0:
                return int(compressed_size * known_ratio), known_ratio
            
            # Wikipedia XML dump specific adjustment:
            # These files typically have very high compression ratios (10:1 to 20:1)
            if "wikipedia" in bz2_file_path.lower() or "wiki" in bz2_file_path.lower():
                if compressed_size > 1024 * 1024 * 1024:  # Larger than 1GB
                    # XML dumps are particularly efficiently compressed, between 15:1 and 20:1
                    return compressed_size * 15, 15.0
                else:
                    # Smaller files somewhat less compressed
                    return compressed_size * 12, 12.0
            
            # Size-based estimate as fallback
            if "wikipedia" in bz2_file_path.lower() or "xml" in bz2_file_path.lower():
                # Wikipedia XML typically has a very high compression ratio
                return compressed_size * 15, 15.0
            else:
                # More conservative estimate for other files
                return compressed_size * 8, 8.0
                
        except Exception as e:
            logger.warning(f"Error during size estimation: {e}")
            # Safe fallback for Wikipedia XML files
            if "wiki" in bz2_file_path.lower() or "xml" in bz2_file_path.lower():
                return compressed_size * 15, 15.0
            else:
                return compressed_size * 8, 8.0


    def format_size(self, bytes):
        """Formats byte sizes in a user-friendly way"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024 or unit == 'TB':
                return f"{bytes:.2f} {unit}"
            bytes /= 1024
    
    def process_dump_files_simple_parallel(self, dump_urls, extract_sentences=False,
                                     max_pages_per_dump=None, max_revisions_per_page=None,
                                     force_download=False, skip_if_processed=True):
        """
        Processes multiple dump files in parallel.
        """
        if not dump_urls:
            logger.warning("No dump URLs specified.")
            return 0, 0, 0, 0
        
        start_time = time.time()
        total_dumps = len(dump_urls)
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        logger.info(f"Starting parallel processing of {total_dumps} dumps with {self.max_workers} workers")
        
        # Create parameters for worker function
        params = {
            'extract_sentences': extract_sentences,
            'max_pages': max_pages_per_dump,
            'max_revisions_per_page': max_revisions_per_page,
            'force_download': force_download,
            'skip_if_processed': skip_if_processed,
            'processor_config': {
                'dump_directory': self.dump_directory,
                'results_directory': self.results_directory,
                'db_path': self.db_path,
                'cleanup_after_processing': self.cleanup_after_processing,
                'cleanup_extracted_files': self.cleanup_extracted_files,
                'use_multiprocessing': False  # Workers don't use further multiprocessing
            }
        }
        
        # Use multiprocessing.Pool for parallel processing
        with multiprocessing.Pool(processes=self.max_workers) as pool:
            # Start parallel processing with a clear main progress bar
            with tqdm(total=total_dumps, desc="Processing dumps", unit="dump") as progress_bar:
                # Create a list of worker tasks
                results = []
                
                # Start all workers asynchronously
                for dump_url in dump_urls:
                    # Start async worker
                    result = pool.apply_async(process_dump_worker, (dump_url, params))
                    results.append(result)
                
                # Monitor running workers and update progress bar
                active_workers = len(results)
                
                while active_workers > 0:
                    # Check which workers are done
                    completed = []
                    for i, result in enumerate(results):
                        if result.ready():
                            try:
                                dump_url, stats, filename = result.get()
                                short_name = filename.split('-')[-1]
                                
                                # Process status for this file
                                if "error" in stats:
                                    logger.error(f"Error with {filename}: {stats['error']}")
                                    error_count += 1
                                    status = f"{short_name}: ❌ Error"
                                elif stats.get("status") == "skipped":
                                    logger.info(f"Skipped: {filename}")
                                    skipped_count += 1
                                    status = f"{short_name}: ⏭️ Skipped"
                                else:
                                    logger.info(f"Successfully processed: {filename}")
                                    processed_count += 1
                                    pages = stats.get('page_count', 0)
                                    revs = stats.get('revision_count', 0)
                                    proc_time = stats.get('processing_time', 0)
                                    mb = stats.get('uncompressed_bytes', 0) / 1024 / 1024
                                    tagged = len(stats.get('tagged_revision_count', 0))
                                    sentence_changes = len(stats.get('sentence_change_count', 0))
                                    self.mark_dump_as_processed(dump_url, filename, pages, revs, tagged, sentence_changes)
                                    status = f"{short_name}: ✓ OK ({pages} pages, {revs} revs, {mb:.1f} MB, {proc_time:.1f}s)"
                                
                                # Update Progress Bar
                                progress_bar.update(1)
                                
                                # Update display with better formatting
                                progress_bar.set_postfix({
                                    'OK': processed_count,
                                    'Skip': skipped_count,
                                    'Err': error_count,
                                    'RAM': f"{int(psutil.Process(os.getpid()).memory_info().rss / 1024**2)} MB"
                                })
                                
                                # Always show the latest status
                                tqdm.write(f"Completed: {status}")
                                
                                # Mark as completed
                                completed.append(i)
                            except Exception as e:
                                logger.error(f"Error processing the result: {e}")
                                error_count += 1
                                completed.append(i)
                    
                    # Remove completed workers from the list
                    for idx in sorted(completed, reverse=True):
                        results.pop(idx)
                    
                    active_workers = len(results)
                    
                    # Short pause to reduce CPU load
                    if active_workers > 0:
                        time.sleep(0.5)
        
        processing_time = time.time() - start_time
        
        # Summary
        minutes, seconds = divmod(processing_time, 60)
        hours, minutes = divmod(minutes, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        logger.info(f"Processing completed in {time_str}")
        logger.info(f"Result: {processed_count} processed, {skipped_count} skipped, "
                f"{error_count} with errors, out of a total of {total_dumps} dumps")
        
        return processed_count, skipped_count, error_count, processing_time
    
    def generate_report(self, output_prefix=""):
        """
        Generates a report about the analysis results from the database.
        
        Parameters:
        output_prefix (str): Prefix for the output files
        
        Returns:
        tuple: (tags_df, processed_dumps_df)
        """
        # Create the results directory if it doesn't exist
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory)
        
        start_time = time.time()
        logger.info("Generating report...")
        
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        
        # Performance optimization for database queries
        cursor = conn.cursor()
        cursor.execute("PRAGMA cache_size = 10000")
        
        # 1. Get processed dumps
        processed_dumps_df = pd.read_sql_query(
            "SELECT * FROM processed_dumps ORDER BY processed_at DESC", conn
        )
        
        # 2. Get aggregated tags
        tags_df = pd.read_sql_query(
            "SELECT tag, category, SUM(count) as total_count FROM tags GROUP BY tag, category ORDER BY total_count DESC", 
            conn
        )
        
        # 3. Get tag categories
        categories_df = pd.read_sql_query(
            """
            SELECT category, SUM(count) as total_count 
            FROM tags 
            GROUP BY category 
            ORDER BY total_count DESC
            """, 
            conn
        )
        
        # 4. Get NPOV-related tags
        npov_tags_df = pd.read_sql_query(
            """
            SELECT tag, category, SUM(count) as total_count 
            FROM tags 
            WHERE tag LIKE '%npov%' OR tag LIKE '%pov%' OR tag LIKE '%neutral%' OR tag LIKE '%bias%'
            GROUP BY tag, category
            ORDER BY total_count DESC
            """, 
            conn
        )
        
        # Close the database connection
        conn.close()
        
        # Performance optimization for CSV export: Fewer entries for large files
        max_tags = 10000  # Limit the number of tags for CSV export
        if len(tags_df) > max_tags:
            tags_csv_df = tags_df.head(max_tags)
            logger.info(f"Too many tags for CSV export - limiting to {max_tags} entries")
        else:
            tags_csv_df = tags_df
        
        # Save the DataFrames as CSV files
        processed_dumps_df.to_csv(os.path.join(self.results_directory, f"{output_prefix}processed_dumps.csv"), index=False)
        tags_csv_df.to_csv(os.path.join(self.results_directory, f"{output_prefix}tags_summary.csv"), index=False)
        categories_df.to_csv(os.path.join(self.results_directory, f"{output_prefix}tag_categories.csv"), index=False)
        npov_tags_df.to_csv(os.path.join(self.results_directory, f"{output_prefix}npov_tags.csv"), index=False)
        
        # Create visualizations
        # 1. Top Tags
        plt.figure(figsize=(14, 8))
        top_tags = tags_df.head(20)
        plt.barh(top_tags['tag'], top_tags['total_count'])
        plt.title('Top 20 Edit Tags in Wikipedia Dumps', fontsize=16)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Tag', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_directory, f"{output_prefix}top_tags.png"))
        plt.close()
        
        # 2. Tag categories
        plt.figure(figsize=(12, 6))
        plt.pie(categories_df['total_count'], labels=categories_df['category'], autopct='%1.1f%%', 
               shadow=True, startangle=140)
        plt.axis('equal')
        plt.title('Distribution of Tag Categories', fontsize=16)
        plt.savefig(os.path.join(self.results_directory, f"{output_prefix}tag_categories_pie.png"))
        plt.close()
        
        # 3. NPOV Tags
        if not npov_tags_df.empty:
            plt.figure(figsize=(12, 6))
            plt.barh(npov_tags_df['tag'].head(15), npov_tags_df['total_count'].head(15))
            plt.title('NPOV-Related Tags in Wikipedia Dumps', fontsize=16)
            plt.xlabel('Count', fontsize=12)
            plt.ylabel('Tag', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_directory, f"{output_prefix}npov_tags.png"))
            plt.close()
        
        # Create a summary report
        total_dumps = len(processed_dumps_df)
        total_pages = processed_dumps_df['page_count'].sum()
        total_revisions = processed_dumps_df['revision_count'].sum()
        total_tagged_revisions = processed_dumps_df['tagged_revision_count'].sum()
        total_sentence_changes = processed_dumps_df['sentence_change_count'].sum()
        
        with open(os.path.join(self.results_directory, f"{output_prefix}analysis_summary.txt"), 'w', encoding='utf-8') as f:
            f.write("=== Wikipedia Dump Analysis Summary ===\n\n")
            f.write(f"Total dumps processed: {total_dumps}\n")
            f.write(f"Total pages processed: {total_pages}\n")
            f.write(f"Total revisions processed: {total_revisions}\n")
            f.write(f"Total tagged revisions: {total_tagged_revisions}\n")
            f.write(f"Total sentence changes: {total_sentence_changes}\n\n")
            
            f.write("Top 20 Tags:\n")
            for _, row in top_tags.iterrows():
                f.write(f"  - {row['tag']} ({row['category']}): {row['total_count']}\n")
            
            f.write("\nTag Categories:\n")
            for _, row in categories_df.iterrows():
                f.write(f"  - {row['category']}: {row['total_count']}\n")
            
            f.write("\nNPOV-Related Tags:\n")
            for _, row in npov_tags_df.iterrows():
                f.write(f"  - {row['tag']}: {row['total_count']}\n")
        
        logger.info(f"Report generated in {time.time() - start_time:.2f} seconds")
        
        return tags_df, processed_dumps_df