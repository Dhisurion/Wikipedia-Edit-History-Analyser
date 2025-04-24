# Wikipedia Edit History Analyzer
A tool for processing and analyzing Wikipedia dump files with focus on edit patterns, content policy enforcement, and sentence-level changes.

## Features

- Extract and analyze edit tags from Wikipedia revision histories
- Track sentence-level changes (additions, deletions, modifications) across revisions
- High-performance processing pipeline with multiprocessing support
- Interactive dashboard for data exploration and visualization
- Network analysis of tag relationships and communities

## Installation

###Requirements

- Python 3.8+
- 8GB+ RAM recommended
- SQLite3

```
# Clone the repository
git clone https://github.com/yourusername/wikipedia-edit-analyzer.git
cd wikipedia-edit-analyzer

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Usage

### Process Wikipedia Dumps

```
# Basic processing with default settings
python main.py

# Limit processing scope for testing
python main.py --max-dumps 1 --max-pages 100

# Extract sentence changes with custom parameters
python main.py --extract-sentences --workers 4 --base-url https://dumps.wikimedia.org/enwiki/20250201/
```

### Run the Dashboard
```
streamlit run wikipedia_dashboard.py
```

The dashboard will be available at http://localhost:8501

## Command-Line-Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dump-dir` | Directory for dump files | `./dumps` |
| `--results-dir` | Directory for results | `./results` |
| `--db-path` | Path to SQLite database | `./wikipedia_analysis.db` |
| `--base-url` | Base URL for Wikipedia dumps | `https://dumps.wikimedia.org/enwiki/20250201/` |
| `--max-dumps` | Maximum number of dumps to process | All found dumps |
| `--max-pages` | Maximum pages per dump | No limit |
| `--max-revisions` | Maximum revisions per page | No limit |
| `--extract-sentences` | Extract sentence changes | False |
| `--workers` | Number of parallel processes | Auto-detected based on system |
| `--cleanup` | Delete dump files after processing | False |
| `--report`  | Generate report  | False |
| `--report-only` | Only generate report, skip processing | False |
| `--memory-check` | Output memory usage information | False |
| `--debug` | Enable detailed debug logging | False |

## Dashboard

The interactive dashboard provides five main analysis views:

1. Tag Analysis: Visualize tag frequencies, networks, and trends
2. Sentence Changes: Analyze content modification patterns
3. NPOV Analysis: Focus on neutrality enforcement
4. Text Explorer: Search and browse specific changes
5. Advanced Analysis: Network analysis of tag relationships

## Project Structure

wikipedia-edit-analyzer/
├── main.py              # Main entry point and CLI
├── dump_processor.py    # Dump processing pipeline
├── tag_handler.py       # XML parsing and tag extraction
├── wikipedia_dashboard.py # Interactive dashboard
├── dumps/               # Downloaded Wikipedia dumps
├── results/             # Generated reports
└── wikipedia_analysis.db # SQLite database

## Performance Considerations

- For large dumps, use --max-pages and --max-revisions to limit memory usage
- Sentence extraction requires significantly more resources
- The number of workers can be adjusted based on available RAM
- Initial processing builds the database; subsequent analyses are faster



# Technical Documentation

# Wikipedia Edit History Analyzer - Technical Documentation

This technical documentation provides in-depth information about the system architecture, implementation details, and design decisions for developers working with the Wikipedia Edit History Analyzer.

## System Architecture

The system consists of several key components designed for efficient processing of Wikipedia XML dumps:

### 1. Main Orchestration (`main.py`)

The entry point that:
- Parses command-line arguments
- Initializes components
- Orchestrates the dump processing workflow
- Configures logging

### 2. Dump Processor (`dump_processor.py`)

This component handles:
- Downloading dump files from Wikimedia servers
- Managing the processing queue for parallel execution
- Streaming decompression of BZ2 files
- Database operations for storing results
- Performance monitoring and optimization

Key classes:
- `WikipediaDumpProcessor`: Main processing class
- Processing functions for parallel execution

### 3. Tag Handler (`tag_handler.py`)

A SAX-based XML parser that:
- Extracts revision metadata
- Identifies and categorizes edit tags
- Processes text changes at the sentence level
- Implements memory-efficient streaming parsing

Key classes:
- `WikipediaTagHandler`: SAX content handler
- Utility functions for text processing and fuzzy matching

### 4. Interactive Dashboard (`wikipedia_dashboard.py`)

A Streamlit-based UI that provides:
- Interactive data exploration
- Visualizations and filtering
- Network analysis
- Export capabilities

## Processing Pipeline

### Dump Discovery and Download

1. Queries the Wikimedia dumps server to find available dump files
2. Filters by file pattern (e.g., `pages-meta-history`)
3. Downloads files with retry logic and progress tracking
4. Skips already processed dumps for incremental updates

### XML Parsing Strategy

The system uses SAX (Simple API for XML) instead of DOM parsing for memory efficiency:

1. Files are read in streaming mode without full decompression
2. The parser processes elements as they are encountered
3. Event-based callbacks extract relevant data
4. The handler maintains minimal state for the current page/revision

### Tag Extraction

Tags are extracted using several methods:

1. Explicit tags from structured comment patterns
2. Keywords in edit summaries matched against known tags
3. Regular expressions to detect common tag patterns
4. Fuzzy matching for misspelled or variant forms

Tags are organized into categories:
- Content Policy
- Content Quality
- Edit Type
- Edit Tools
- Vandalism & Maintenance
- Automated Edits
- Article Quality
- Multimedia Content
- Topics & Subject Areas

### Sentence-Level Change Detection

When enabled, the system tracks how individual sentences change:

1. Text normalization to remove wiki markup
2. Sentence tokenization using NLTK
3. Set-based comparison for additions/deletions
4. Fuzzy matching (using `thefuzz`) for modifications
5. Similarity scoring between old and new versions

## Database Schema

The SQLite database uses the following schema:

### `processed_dumps` Table
```sql
CREATE TABLE processed_dumps (
    dump_url TEXT PRIMARY KEY,
    filename TEXT,
    processed_at TIMESTAMP,
    page_count INTEGER,
    revision_count INTEGER,
    tagged_revision_count INTEGER,
    sentence_change_count INTEGER
)

### `tags` Table

```
CREATE TABLE tags (
    tag TEXT,
    category TEXT,
    count INTEGER,
    dump_url TEXT,
    PRIMARY KEY (tag, dump_url)
)
```
### `tagged_revisions` Table

```
CREATE TABLE tagged_revisions (
    revision_id TEXT,
    page_id TEXT,
    page_title TEXT,
    timestamp TEXT,
    comment TEXT,
    tags TEXT,  -- JSON array of tags
    dump_url TEXT,
    PRIMARY KEY (revision_id, dump_url)
)
```


### `sentence_changes` Table

```
CREATE TABLE sentence_changes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    revision_id TEXT,
    page_id TEXT,
    page_title TEXT,
    timestamp TEXT,
    change_type TEXT,  -- 'addition', 'deletion', or 'modification'
    old_sentence TEXT,
    new_sentence TEXT,
    similarity REAL,   -- Similarity score for modifications
    tags TEXT,         -- JSON array of tags
    dump_url TEXT
)
```

## Performance Optimizations

### Memory Management
- Streaming decompression avoids loading entire dumps into memory
- Batch database operations with configurable commit thresholds
- Cache clearing based on memory usage monitoring
- Automatic garbage collection triggering

### Multiprocessing
- Configurable worker pool based on available CPU cores and memory
- Shared progress tracking with atomic counters
- Global limit enforcement across workers
- Optimized work distribution

### Database Optimizations
- SQLite WAL (Write-Ahead Logging) mode for concurrent access
- Strategic indexing for query performance
- Batched operations to reduce commit overhead
- Memory-mapped I/O for large databases

### Processing Heuristics
- Early termination for non-article namespaces
- Text size limits for very large revisions
- Sample-based processing for extremely large dumps
- Adaptive compression ratio estimation

## Visualization Components

The dashboard uses several visualization libraries:
- **Plotly**: Interactive time series and bar charts
- **NetworkX**: Graph representation and analysis
- **Matplotlib/Seaborn**: Static visualizations
- **WordCloud**: Tag frequency visualization

## Network Analysis Features

The system implements several network analysis algorithms:

### Community Detection
- **Louvain Method**: Modularity-based community detection
- **Girvan-Newman**: Edge betweenness-based approach
- **Label Propagation**: Fast community detection through label propagation

### Centrality Measures
- **Degree Centrality**: Based on direct connections
- **Betweenness Centrality**: Based on shortest paths
- **Closeness Centrality**: Based on average distance
- **Eigenvector Centrality**: Based on influence of neighbors

## Error Handling and Resilience

The system implements multiple error handling strategies:
- Retry logic for network operations
- Transaction rollback for database errors
- Skip-ahead for corrupted XML sections
- Gradual degradation for memory pressure
- Detailed logging for troubleshooting

## Extending the System

### Adding New Tag Categories
1. Extend the `tag_categories` dictionary in `WikipediaDumpProcessor.__init__`
2. Add corresponding detection patterns in `extract_tags_from_comment`

### Processing Additional Metadata
1. Modify the `WikipediaTagHandler` to capture additional XML elements
2. Update the database schema to store new fields
3. Add corresponding visualizations to the dashboard

### Supporting Other Dump Formats
1. Implement new file handlers in `process_dump_file_stream`
2. Add decompression support for additional formats
3. Update the URL discovery regex patterns

## Configuration Constants

Important configuration values that may need adjustment:
- `DB_COMMIT_THRESHOLD`: Controls database batch size (default: 2000)
- `MAX_TEXT_SIZE`: Limits text processing size (default: 150000 bytes)
- `FUZZY_MATCH_THRESHOLD`: Similarity threshold (default: 65)
- `MEMORY_CHECK_INTERVAL`: Memory check frequency (default: 240 seconds)

## Dashboard Implementation Details

### State Management
The dashboard uses Streamlit's session state for:
- Tracking active tabs
- Storing export file references
- Maintaining filter selections

### Data Caching
Performance is improved through:
- `@st.cache_data` decorators for database queries
- In-memory data transformation caching
- Progressive loading of visualizations

### Interactive Components
The dashboard implements several interactive elements:
- Date range selectors
- Multi-select filters
- Pagination for text exploration
- Dynamic network parameter adjustment
- Export functionality with multiple formats