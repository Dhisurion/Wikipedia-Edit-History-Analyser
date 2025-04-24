import xml.sax
import re
import time
import os
import gc
import logging
import psutil
from collections import Counter
from thefuzz import fuzz, process
from functools import lru_cache
import nltk
from nltk.tokenize import sent_tokenize
import difflib

# Pre-compiled regex patterns for better performance
REF_PATTERN = re.compile(r'<ref.*?\/ref>')
TEMPLATE_PATTERN = re.compile(r'\{\{.*?\}\}')
LINK_PATTERN = re.compile(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]')
REF_NUM_PATTERN = re.compile(r'\[\d+\]')
WHITESPACE_PATTERN = re.compile(r'\s+')
HEADING_PATTERN = re.compile(r'={2,}.+?={2,}')
FORMATTING_PATTERN = re.compile(r"'{2,}")
TAG_PATTERN = re.compile(r'(?:Tag[s]?:|tag[s]?\(|\[\[(?:Wikipedia:)?)([^\]\)]+)(?:\)|\]\]|$)')
BRACKET_PATTERN = re.compile(r'\[(.*?)\]')
HASHTAG_PATTERN = re.compile(r'#(\w+)')
TEMPLATE_TAG_PATTERN = re.compile(r'\{\{([^{}|]+)(?:\||\}\})')
PREFIX_PATTERN = re.compile(r'^(wikipedia:|wp:|category:|template:|talk:|user:|file:)')
SUFFIX_PATTERN = re.compile(r'(talk|template|category|disambiguation)$')

# Configuration constants
FUZZY_MATCH_ENABLED = True
FUZZY_MATCH_MAX_CANDIDATES = 100
FUZZY_MATCH_MIN_LENGTH = 25
MAX_TEXT_SIZE = 150000  # 100KB for text analysis
MEMORY_CHECK_INTERVAL = 240  # seconds
FUZZY_MATCH_THRESHOLD = 65

# Logging configuration (set in main program)
logger = logging.getLogger("WikipediaDumpProcessor")

# Cache for fuzzy matching
_fuzzy_cache = {}

@lru_cache(maxsize=1000)
def cached_token_sort_ratio(a, b):
    """Cached version of fuzz.token_sort_ratio for better performance."""
    return fuzz.token_sort_ratio(a, b)

def optimized_extract_one(query, choices, scorer=fuzz.ratio, threshold=0):
    """Optimized version of process.extractOne for better performance."""
    if not choices:
        return None
    
    # Create cache key
    if len(query) < 100:  # Only for shorter strings
        cache_key = (query, tuple(choices[:10]) if len(choices) > 10 else tuple(choices))
        if cache_key in _fuzzy_cache:
            return _fuzzy_cache[cache_key]
    
    # For very short queries or very few choices, calculate directly
    if len(query) < 20 or len(choices) < 5:
        best_match = None
        best_score = -1
        
        for choice in choices:
            score = scorer(query, choice)
            if score > best_score:
                best_match = choice
                best_score = score
        
        result = (best_match, best_score) if best_score >= threshold else None
        
        # Save in cache for later use
        if len(query) < 100:
            _fuzzy_cache[cache_key] = result
        
        return result
    
    # Otherwise process normally
    return process.extractOne(query, choices, scorer=scorer, score_cutoff=threshold)

class WikipediaTagHandler(xml.sax.ContentHandler):
    """
    SAX handler for efficient parsing of Wikipedia XML dumps.
    """
    
    def __init__(self, tag_categories, max_pages=None, max_revisions_per_page=None, 
             extract_sentences=False, global_counter=None):
        super().__init__()
        self.tag_categories = tag_categories
        self.max_pages = max_pages
        self.max_revisions_per_page = max_revisions_per_page
        self.extract_sentences = extract_sentences
        self.global_counter = global_counter
        
        # Processing counters
        self.page_count = 0
        self.article_page_count = 0
        self.revision_count = 0
        self.current_page_revision_count = 0
        self.processed_revisions_count = 0
        
        # Current status and data
        self.in_page = False
        self.in_revision = False
        self.current_element = None
        self.current_page = {}
        self.current_revision = {}
        self.buffer = ""
        
        # Results
        self.tag_counter = Counter()
        self.tagged_revisions = []
        self.sentence_changes = []
        
        # Previous revision for sentence changes
        self.previous_text = None
        
        # Cache for better performance
        self.sentence_cache = {}
        self.tag_cache = {}
        
        # Add tracking for element hierarchy
        self.element_stack = []
        self.in_contributor = False

        # Performance monitoring
        self.last_memory_check = time.time()
        self.last_gc_collection = time.time()
        self.processing_stats = {
            'text_processing_time': 0,
            'sentence_splits': 0,
            'tag_extractions': 0
        }
        
        # New flag for page counting
        self.page_counted = False

    def startElement(self, name, attrs):
        self.current_element = name
        self.element_stack.append(name)

        if name == "contributor":
            self.in_contributor = True
        
        if name == "page":
            self.in_page = True
            self.current_page = {"title": None, "ns": None, "id": None}
            self.current_page_revision_count = 0
            self.processed_revisions_count = 0  # New counter for actually processed revisions
            self.previous_text = None
            self.page_counted = False
            
        elif name == "revision":
            # Check if revision limit already reached
            if self.max_revisions_per_page and self.processed_revisions_count >= self.max_revisions_per_page:
                # If yes, ignore this revision
                self.in_revision = False
                self.current_element = None
                return
            
            self.in_revision = True
            self.current_revision = {"id": None, "timestamp": None, "comment": None, "tags": [], "text": None}
        
        # Clear the text buffer for the new element
        self.buffer = ""
    
    def characters(self, content):
    # Collect text content in buffer, but only if currently within a relevant revision
        if self.current_element and (not self.max_revisions_per_page or self.current_page_revision_count < self.max_revisions_per_page or not self.in_revision):
            self.buffer += content
        
    def endElement(self, name):
        is_revision_processed = False
        
        # Reset status when contributor element ends
        if name == "contributor":
            self.in_contributor = False
        
        # Remove element from stack (this should happen for EVERY element)
        if self.element_stack and self.element_stack[-1] == name:
            self.element_stack.pop()
        
        # Process text based on current element
        if self.in_page:
            if name == "title":
                self.current_page["title"] = self.buffer
            elif name == "ns":
                self.current_page["ns"] = int(self.buffer) if self.buffer.isdigit() else None
            elif name == "id" and not self.in_revision and not self.in_contributor:
                self.current_page["id"] = self.buffer
        
        if self.in_revision:
            if name == "id":
                # Distinguish between revision ID and contributor ID
                if not self.in_contributor:
                    self.current_revision["id"] = self.buffer
                else:
                    # This is a contributor ID, store separately
                    self.current_revision["contributor_id"] = self.buffer
            elif name == "timestamp":
                self.current_revision["timestamp"] = self.buffer
            elif name == "comment":
                self.current_revision["comment"] = self.buffer
                # Extract tags from comment
                if self.buffer:
                    start_time = time.time()
                    tags = self.extract_tags_from_comment(self.buffer)
                    self.processing_stats['tag_extractions'] += 1
                    self.processing_stats['text_processing_time'] += time.time() - start_time
                    self.current_revision["tags"].extend(tags)
                    for tag in tags:
                        self.tag_counter[tag] += 1
            elif name == "text":
                # Performance optimization: Skip very large texts if needed
                if self.buffer and len(self.buffer) > MAX_TEXT_SIZE:
                    self.current_revision["text"] = self.buffer[:MAX_TEXT_SIZE]
                else:
                    self.current_revision["text"] = self.buffer
            elif name == "tag":
                if self.buffer:
                    tag_name = self.buffer.lower()
                    self.current_revision["tags"].append(tag_name)
                    self.tag_counter[tag_name] += 1
            
            # End of a revision
            if name == "revision":
                logger.debug(f"extract_sentences parameter is: {self.extract_sentences}")
                # Periodic garbage collection and memory check
                current_time = time.time()
                if current_time - self.last_memory_check > MEMORY_CHECK_INTERVAL:
                    self.check_memory_usage()
                    self.last_memory_check = current_time
                
                self.in_revision = False
                self.revision_count += 1
                self.current_page_revision_count += 1
                
                # Save revisions with tags
                if self.current_page.get("ns") == 0 and self.current_revision["tags"]:
                    # Only save if limit not yet reached
                    if self.max_revisions_per_page and self.processed_revisions_count < self.max_revisions_per_page:
                        self.tagged_revisions.append({
                            "page_title": self.current_page["title"],
                            "page_id": self.current_page["id"],
                            "revision_id": self.current_revision["id"],
                            "timestamp": self.current_revision["timestamp"],
                            "comment": self.current_revision["comment"],
                            "tags": self.current_revision["tags"]
                        })
                        self.processed_revisions_count += 1
                        is_revision_processed = True
                        
                        logger.debug(f"Revision {self.current_page_revision_count} saved. Processed revisions: {self.processed_revisions_count}/{self.max_revisions_per_page}")
                    elif not self.max_revisions_per_page:
                        # Without limit save everything
                        self.tagged_revisions.append({
                            "page_title": self.current_page["title"],
                            "page_id": self.current_page["id"],
                            "revision_id": self.current_revision["id"],
                            "timestamp": self.current_revision["timestamp"],
                            "comment": self.current_revision["comment"],
                            "tags": self.current_revision["tags"]
                        })
                        self.processed_revisions_count += 1
                        is_revision_processed = True

                    # Extract sentence changes if the revision was processed
                    if is_revision_processed and self.extract_sentences and self.current_revision["text"] is not None:
                        start_time = time.time()
                        
                        # For the first revision of a page: mark all sentences as "addition"
                        if self.current_page_revision_count == 1:
                            # Extract all sentences from the first text
                            sentences = self.split_into_sentences(self.current_revision["text"]) 
                            # Create "addition" changes for each sentence
                            changes = []
                            for sentence in sentences:
                                changes.append({
                                    "type": "addition",
                                    "old_sentence": None,
                                    "new_sentence": sentence
                                })
                        # For subsequent revisions: perform normal change detection
                        elif self.previous_text is not None:
                            changes = self.extract_sentence_changes(self.previous_text, self.current_revision["text"])
                        else:
                            changes = []
                            
                        self.processing_stats['text_processing_time'] += time.time() - start_time
                        
                        if changes:
                            for change in changes:
                                # Explicit type conversion for similarity
                                similarity_value = 0.0
                                
                                if change["type"] == "modification" and "similarity" in change:
                                    # Ensure the value is a float and in the range 0.0-1.0
                                    try:
                                        raw_similarity = change["similarity"]
                                        if isinstance(raw_similarity, (int, float)):
                                            # Normalize values > 1.0 (in case it's still a 0-100 value)
                                            if raw_similarity > 1.0:
                                                similarity_value = float(raw_similarity) / 100.0
                                            else:
                                                similarity_value = float(raw_similarity)
                                            
                                            # Safety check for value range
                                            similarity_value = max(0.0, min(1.0, similarity_value))
                                            
                                            # Debug output to check the value
                                            logger.debug(f"Modification with similarity value: {similarity_value}")
                                    except (ValueError, TypeError) as e:
                                        logger.error(f"Error processing similarity: {e}")
                                
                                self.sentence_changes.append({
                                    "page_title": self.current_page["title"],
                                    "page_id": self.current_page["id"],
                                    "revision_id": self.current_revision["id"],
                                    "timestamp": self.current_revision["timestamp"],
                                    "change_type": change["type"],
                                    "old_sentence": change.get("old_sentence"),
                                    "new_sentence": change.get("new_sentence"),
                                    "similarity": similarity_value,  # Explicitly assign the processed value
                                    "tags": self.current_revision["tags"]
                                })
                    
                    # Save the current text for the next revision
                    if is_revision_processed:
                        self.previous_text = self.current_revision["text"]
                    
                    # If revision limit reached and not yet counted 
                    if self.max_revisions_per_page and self.processed_revisions_count >= self.max_revisions_per_page and not self.page_counted:
                        logger.info(f"Revision limit of {self.max_revisions_per_page} for the current page reached.")
                        
                        # Count the page if it's an article
                        if self.current_page.get("ns") == 0:
                            self.article_page_count += 1
                            self.page_counted = True
                            logger.info(f"Article page {self.article_page_count} of {self.max_pages if self.max_pages else '?'} completed (after limit).")
                            
                            # Check the global counter if available
                            if self.global_counter is not None:
                                with self.global_counter.get_lock():
                                    self.global_counter.value += 1
                                    if self.max_pages and self.global_counter.value >= self.max_pages:
                                        logger.info(f"Global maximum of {self.max_pages} article pages reached. Ending parsing.")
                                        raise xml.sax.SAXException("Maximum number of article pages globally reached")
                            
                            # Local check (if no global counter)
                            elif self.max_pages and self.article_page_count >= self.max_pages:
                                logger.info(f"Maximum number of {self.max_pages} article pages reached. Ending parsing.")
                                raise xml.sax.SAXException("Maximum number of article pages reached")
        
        # End of a page
        if name == "page":
            self.in_page = False
            self.page_count += 1
            
            # If the page hasn't been counted yet (revision limit not yet reached)
            if not self.page_counted and self.current_page.get("ns") == 0:
                self.article_page_count += 1
                logger.info(f"Article page {self.article_page_count} of {self.max_pages if self.max_pages else '?'} completed (normal).")
                
                # Check the global counter if available
                if self.global_counter is not None:
                    with self.global_counter.get_lock():
                        self.global_counter.value += 1
                        if self.max_pages and self.global_counter.value >= self.max_pages:
                            logger.info(f"Global maximum of {self.max_pages} article pages reached. Ending parsing.")
                            raise xml.sax.SAXException("Maximum number of article pages globally reached")
                
                # Local check (if no global counter)
                elif self.max_pages and self.article_page_count >= self.max_pages:
                    logger.info(f"Maximum number of {self.max_pages} article pages reached. Ending parsing.")
                    raise xml.sax.SAXException("Maximum number of article pages reached")
            
            logger.info(f"Page {self.page_count} completed with {self.current_page_revision_count} revisions.")
            
            # Reset for the next page
            self.current_page_revision_count = 0
            self.page_counted = False
        
        self.current_element = None
        
    def check_memory_usage(self):
        """Checks memory usage and performs optimizations if needed."""
        # Get current memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Perform optimizations at high memory usage (>75%)
        if memory_percent > 75:
            logger.warning(f"High memory usage: {memory_percent:.1f}% ({memory_info.rss / 1024**2:.1f} MB)")
            
            # Clear caches
            self.sentence_cache.clear()
            self.tag_cache.clear()
            _fuzzy_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Log new memory usage after optimization
            new_memory_percent = psutil.Process(os.getpid()).memory_percent()
            logger.info(f"Memory usage after optimization: {new_memory_percent:.1f}%")

    def estimate_completion_percent(self):
        """
        Estimates the percentage of completion based on limits.
        
        When max_pages or max_revisions are set, this method provides
        a better estimate of overall progress.
        
        Returns:
            float: Percentage of completion (0-1)
        """
        # If no limits are set, cannot estimate
        if not self.max_pages and not self.max_revisions_per_page:
            return None
            
        completion = 0.0
        
        # If page limit is set, estimate based on processed article pages
        if self.max_pages:
            # Use article_page_count instead of page_count
            page_completion = min(1.0, self.article_page_count / self.max_pages)
            completion = max(completion, page_completion)
        
        # If revision limit is set, consider that too
        if self.max_revisions_per_page and self.page_count > 0:
            # For current overall progress need:
            # 1. Completely finished pages: (page_count - 1) / max_pages
            # 2. Progress on current page: current_page_revision_count / max_revisions_per_page
            if self.max_pages:
                # If a page limit also exists
                page_fraction = 1.0 / self.max_pages
                pages_complete = max(0, self.page_count - 1) * page_fraction
                
                # Progress on current page
                current_page_percent = min(1.0, self.current_page_revision_count / self.max_revisions_per_page)
                current_page_contribution = current_page_percent * page_fraction
                
                rev_completion = pages_complete + current_page_contribution
                completion = max(completion, rev_completion)
            else:
                # If only a revision limit is given
                revision_percent = min(1.0, self.revision_count / (self.max_revisions_per_page * self.page_count))
                completion = max(completion, revision_percent)
        
        return completion
    
    def extract_tags_from_comment(self, comment):
        """
        Extracts tags from an edit comment with improved detection.
        Performance-optimized version.
        """
        if not comment:
            return []
        
        # Cache check for frequently occurring comments
        if comment in self.tag_cache:
            return self.tag_cache[comment]
        
        # Normalize comment for more consistent extraction
        comment_lower = comment.lower().strip()
        
        # Regular expressions for various tag formats
        # Explicit tags
        explicit_tags = TAG_PATTERN.findall(comment)
        
        # Square brackets
        bracket_tags = BRACKET_PATTERN.findall(comment)
        
        # Hashtags
        hashtag_tags = HASHTAG_PATTERN.findall(comment)
        
        # Keywords in templates
        template_tags = TEMPLATE_TAG_PATTERN.findall(comment)
        
        # Combine all found tags
        all_raw_tags = explicit_tags + bracket_tags + hashtag_tags + template_tags
        
        # Clean and normalize tags
        normalized_tags = []
        for tag in all_raw_tags:
            # Remove surrounding whitespace and convert to lowercase
            cleaned_tag = tag.strip().lower()
            
            # Remove typical prefixes and suffixes
            cleaned_tag = PREFIX_PATTERN.sub('', cleaned_tag)
            cleaned_tag = SUFFIX_PATTERN.sub('', cleaned_tag)
            
            if cleaned_tag:
                normalized_tags.append(cleaned_tag)
        
        # Extract implicit tags from content of comment
        # Look for keywords (only in short to medium-length comments)
        if len(comment_lower) < 500:  # Performance optimization
            keyword_mappings = {
                'npov': ['neutral', 'bias', 'pov', 'unbalanced', 'one-sided'],
                'citation needed': ['citation', 'source', 'reference', 'verify', 'citation needed'],
                'cleanup': ['cleanup', 'clean up', 'tidy', 'organize', 'fix'],
                'grammar': ['grammar', 'spelling', 'typo', 'punctuation', 'capitalization'],
                'style': ['style', 'format', 'formatting', 'layout', 'structure'],
                'revert': ['revert', 'undid', 'undo', 'reverted', 'rollback'],
                'vandalism': ['vandal', 'vandalism', 'spam', 'test', 'inappropriate'],
                'minor': ['minor', 'typo', 'typos', 'spacing', 'formatting'],
                'bot': ['bot', 'automated', 'auto', 'script', 'AWB'],
                'merge': ['merge', 'combined', 'consolidate', 'unify'],
                'split': ['split', 'separate', 'divide', 'partition'],
                'redirect': ['redirect', 'redir', '#REDIRECT', 'redirection'],
                'disambiguation': ['disambig', 'dab', 'disambiguation', 'clarify names'],
                'copyvio': ['copyright', 'copyvio', 'violation', 'infringement', 'plagiarism']
            }
            
            # Check if the comment contains keywords
            for tag, keywords in keyword_mappings.items():
                if tag not in normalized_tags and any(keyword in comment_lower for keyword in keywords):
                    normalized_tags.append(tag)
        
        # Recognize known tags via existing categories
        # Performance optimization: Only for short comments or when few tags found
        if len(normalized_tags) < 3 and len(comment_lower) < 1000:
            for category, tags in self.tag_categories.items():
                # Take only the first 10 tags per category for checking
                for tag in tags[:10]:
                    tag_lower = tag.lower()
                    if tag_lower in comment_lower and tag_lower not in normalized_tags:
                        normalized_tags.append(tag_lower)
        
        # Remove duplicates
        result = list(set(normalized_tags))
        
        # Cache the result for faster processing
        if len(comment) < 200:  # Only cache short comments
            self.tag_cache[comment] = result
        
        return result
    
    def clean_text(self, text):
        """Cleans Wiki markup from the text."""
        # Uses pre-compiled regex patterns
        cleaned = LINK_PATTERN.sub(r'\1', text)  # Wiki links
        cleaned = FORMATTING_PATTERN.sub('', cleaned)  # Bold and italic formatting
        cleaned = HEADING_PATTERN.sub('', cleaned)  # Headings
        cleaned = REF_PATTERN.sub(' ', cleaned)  # <ref> tags
        cleaned = TEMPLATE_PATTERN.sub(' ', cleaned)  # Templates
        cleaned = REF_NUM_PATTERN.sub(' ', cleaned)  # Reference numbers
        cleaned = WHITESPACE_PATTERN.sub(' ', cleaned)  # Multiple spaces/breaks
        return cleaned.strip()

    def split_into_sentences(self, text):
        """Splits text into individual sentences."""
        if not text:
            return []
        
        # Cache check for common text fragments
        if text in self.sentence_cache:
            return self.sentence_cache[text]
        
        # For very long texts, use only a sample
        if len(text) > 50000:
            sample_text = text[:50000]
        else:
            sample_text = text
        
        cleaned_text = self.clean_text(sample_text)
        if not cleaned_text:
            return []
        
        # Try NLTK or fall back to regex
        try:
            nltk.data.find('tokenizers/punkt')
            sentences = sent_tokenize(cleaned_text, 'english')
        except:
            sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        
        # Filtering empty or too short sentences
        valid_sentences = []
        for s in sentences:
            s = s.strip()
            # Check if the sentence actually contains words
            if s and len(s) > 15 and re.search(r'[a-zA-Z]{3,}', s):
                valid_sentences.append(s)
        
        # Caching for performance optimization
        if len(text) < 10000:  # Only cache smaller texts
            self.sentence_cache[text] = valid_sentences
            
        self.processing_stats['sentence_splits'] += 1
        return valid_sentences
    
    def extract_sentence_changes(self, old_text, new_text):
        """
        Extracts sentence changes between two text versions.
        """
        if not old_text or not new_text:
            return []
        
        # Quick check if the text has changed at all
        if old_text == new_text:
            return []
            
        old_sentences = self.split_into_sentences(old_text)
        new_sentences = self.split_into_sentences(new_text)
        
        # Optimized change detection
        changes = []
        
        # Use fast set-based matching for exact matches
        old_set = set(old_sentences)
        new_set = set(new_sentences)
        
        # Optimized fuzzy matching for sentences that don't match exactly
        remaining_old = [s for s in old_sentences if s not in new_set]
        remaining_new = [s for s in new_sentences if s not in old_set]
        
        # Keep track of sentences that are part of modifications
        modified_old_sentences = set()
        modified_new_sentences = set()
        
        # FIRST identify modifications using fuzzy matching
        if FUZZY_MATCH_ENABLED and remaining_old and remaining_new:
            modifications_found = 0  # Counter for found modifications
            
            # Limit the number of comparisons for better performance
            max_candidates = min(FUZZY_MATCH_MAX_CANDIDATES, len(remaining_old), len(remaining_new))
            
            if max_candidates > 0:
                # Choose the longest sentences for fuzzy matching
                remaining_old = sorted(remaining_old, key=len, reverse=True)[:max_candidates]
                
                for old_sent in remaining_old:
                    if len(old_sent) < FUZZY_MATCH_MIN_LENGTH:
                        continue  # Skip sentences that are too short
                    
                    # Use optimized fuzzy matching
                    try:
                        best_match = optimized_extract_one(
                            old_sent, 
                            remaining_new, 
                            scorer=cached_token_sort_ratio, 
                            threshold=FUZZY_MATCH_THRESHOLD
                        )
                        
                        if best_match:
                            # IMPORTANT: Normalize the similarity value from 0-100 to 0.0-1.0
                            # and safely convert to float
                            raw_score = best_match[1]
                            normalized_score = float(raw_score) / 100.0
                            
                            # Debug logging to check values
                            logger.debug(f"SIMILARITY: Raw={raw_score}, Normalized={normalized_score}")
                            
                            modifications_found += 1
                            
                            # Add to modification tracking sets
                            modified_old_sentences.add(old_sent)
                            modified_new_sentences.add(best_match[0])
                            
                            changes.append({
                                "type": "modification",
                                "old_sentence": old_sent,
                                "new_sentence": best_match[0],
                                "similarity": normalized_score  # Normalized value
                            })
                    except Exception as e:
                        logger.error(f"Error during fuzzy matching: {e}")
            
            # Log the number of modifications found
            if modifications_found > 0:
                logger.debug(f"Found: {modifications_found} sentence modifications with similarity values")
        
        # THEN add deletions and additions, but only for sentences not part of a modification
        for sentence in old_set - new_set:
            if sentence not in modified_old_sentences:
                changes.append({
                    "type": "deletion",
                    "old_sentence": sentence,
                    "new_sentence": None
                })
        
        for sentence in new_set - old_set:
            if sentence not in modified_new_sentences:
                changes.append({
                    "type": "addition",
                    "old_sentence": None,
                    "new_sentence": sentence
                })
        
        return changes
    
    def align_texts(self, old_text, new_text):
        """
        Performs detailed alignment between two texts.
        """
        if not old_text or not new_text:
            return []
        
        if old_text == new_text:
            return []
        
        # Performance optimization: Truncate large texts
        if len(old_text) > MAX_TEXT_SIZE:
            old_text = old_text[:MAX_TEXT_SIZE]
        if len(new_text) > MAX_TEXT_SIZE:
            new_text = new_text[:MAX_TEXT_SIZE]
        
        # Use Sequence Matcher for detailed alignment
        matcher = difflib.SequenceMatcher(None, old_text, new_text)
        opcodes = matcher.get_opcodes()
        
        operations = []
        for tag, i1, i2, j1, j2 in opcodes:
            # Skip 'equal' operations
            if tag == 'equal':
                continue
            
            # Determine operation type
            if tag == 'replace':
                operation_type = 'modification'
            elif tag == 'delete':
                operation_type = 'deletion'
            elif tag == 'insert':
                operation_type = 'addition'
            else:
                operation_type = 'unknown'
            
            old_segment = old_text[i1:i2] if i1 < i2 else None
            new_segment = new_text[j1:j2] if j1 < j2 else None
            
            operations.append({
                'type': operation_type,
                'old_segment': old_segment,
                'new_segment': new_segment,
                'position_old': (i1, i2),
                'position_new': (j1, j2),
                'size': max((i2 - i1), (j2 - j1))
            })
        
        return operations