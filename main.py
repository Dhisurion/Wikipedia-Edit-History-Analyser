import os
import sys
import argparse
import logging
import traceback
import psutil
from collections import Counter

# Import the two modules
from tag_handler import WikipediaTagHandler
from dump_processor import WikipediaDumpProcessor

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("wikipedia_dump_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WikipediaDumpProcessor")

def main():
    """
    Main function for command line usage.
    """
    parser = argparse.ArgumentParser(description='Wikipedia Dump Processor')
    parser.add_argument('--dump-dir', type=str, default='./dumps', help='Directory for dump files')
    parser.add_argument('--results-dir', type=str, default='./results', help='Directory for results')
    parser.add_argument('--db-path', type=str, default='./wikipedia_analysis.db', help='Path to SQLite database')
    parser.add_argument('--cleanup', action='store_true', help='Delete dump files after processing')
    parser.add_argument('--base-url', type=str, default='https://dumps.wikimedia.org/enwiki/20250201/', 
                        help='Base URL for Wikipedia dumps')
    parser.add_argument('--extension', type=str, default='.bz2', choices=['.bz2', '.7z'], 
                        help='File extension of the dumps (.bz2 recommended for speed)')
    parser.add_argument('--max-dumps', type=int, default=None, help='Maximum number of dumps to process')
    parser.add_argument('--max-pages', type=int, default=None, help='Maximum number of pages to process per dump')
    parser.add_argument('--max-revisions', type=int, default=None, 
                        help='Maximum number of revisions to process per page')
    parser.add_argument('--extract-sentences', action='store_true', help='Extract sentence changes')
    parser.add_argument('--workers', type=int, default=None, 
                        help='Number of parallel processes (default: automatic based on system)')
    parser.add_argument('--report', action='store_true', 
    help='Generate report')
    parser.add_argument('--report-only', action='store_true', 
                        help='Only generate report, do not process dumps')
    
    # Memory usage output
    parser.add_argument('--memory-check', action='store_true', help='Output memory usage')
    # Debug
    parser.add_argument('--debug', action='store_true', help='Output detailed debug information')
    
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.info("Debug mode activated - more details in the logs.")
    
    # Create the processor
    processor = WikipediaDumpProcessor(
        dump_directory=args.dump_dir,
        results_directory=args.results_dir,
        db_path=args.db_path,
        cleanup_after_processing=args.cleanup,
        use_multiprocessing=True,
        max_workers=args.workers
    )
    
    # Output memory usage if desired
    if args.memory_check:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        logger.info(f"Memory usage: {mem_info.rss / 1024**2:.1f} MB ({process.memory_percent():.1f}%)")
        logger.info(f"Available system memory: {psutil.virtual_memory().available / 1024**2:.1f} MB")
    
    # Only generate report if desired
    if args.report_only:
        logger.info("Only generating report from existing data...")
        processor.generate_report()
        return
    
    # Process dumps
    dumps = processor.fetch_dump_links(
        base_url=args.base_url,
        file_pattern="pages-meta-history",
        extension=args.extension,
        max_dumps=args.max_dumps
    )
    
    if not dumps:
        logger.error("No dump URLs specified or found.")
        return
    
    # Parallel processing
    processor.process_dump_files_simple_parallel(
        dump_urls=dumps,
        extract_sentences=args.extract_sentences,
        max_pages_per_dump=args.max_pages,
        max_revisions_per_page=args.max_revisions,
        force_download=False,
        skip_if_processed=True
    )
    
    # Generate report
    if args.report:
        processor.generate_report()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main program: {e}")
        logger.error(traceback.format_exc())