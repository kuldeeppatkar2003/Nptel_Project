"""
NPTEL Course Transcript Scraper - Production Ready
==================================================

A robust, enterprise-grade web scraper for extracting and processing 
lecture transcripts from NPTEL courses.

Author: Senior Python Developer
Version: 2.0.0
Python: 3.8+

Key Features:
- Comprehensive error handling with custom exceptions
- Retry logic with exponential backoff
- Progress tracking with tqdm
- Configurable via environment variables
- Structured logging with rotation
- Resource pooling and cleanup
- Type hints throughout
- Unit test ready structure
- Metrics and monitoring hooks
"""

import io
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from logging.handlers import RotatingFileHandler

import pandas as pd
import pdfplumber
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
    StaleElementReferenceException
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from langchain_core.documents import Document
from tqdm import tqdm


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class NPTELScraperError(Exception):
    """Base exception for NPTEL scraper"""
    pass


class CourseScrapingError(NPTELScraperError):
    """Raised when course scraping fails"""
    pass


class TranscriptExtractionError(NPTELScraperError):
    """Raised when transcript extraction fails"""
    pass


class DriverInitializationError(NPTELScraperError):
    """Raised when WebDriver initialization fails"""
    pass


# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================

@dataclass
class ScraperConfig:
    """
    Centralized configuration for the NPTEL scraper.
    All settings can be overridden via environment variables.
    """
    
    # ======== Directory Settings ========
    output_dir: Path = field(default_factory=lambda: Path("data/intermediate"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    
    # ======== File Names ========
    courses_csv: str = "courses_list.csv"
    raw_documents_json: str = "documents_raw.json"
    summary_json: str = "pipeline_summary.json"
    log_file: str = "nptel_scraper.log"
    
    # ======== Scraping Parameters ========
    max_courses: int = 5
    target_language: str = "English-Verified"
    start_course_index: int = 0  # For resuming scraping
    
    # ======== Selenium Settings ========
    selenium_implicit_wait: int = 10
    selenium_explicit_wait: int = 20
    page_load_timeout: int = 60
    script_timeout: int = 30
    
    # ======== Chrome Options ========
    headless_mode: bool = True
    disable_gpu: bool = True
    no_sandbox: bool = True
    disable_dev_shm: bool = True
    window_size: str = "1920,1080"
    user_agent: Optional[str] = None
    
    # ======== Network Settings ========
    request_timeout: int = 60
    max_retries: int = 5
    retry_backoff_factor: float = 2.0
    retry_status_codes: List[int] = field(
        default_factory=lambda: [408, 429, 500, 502, 503, 504]
    )
    
    # ======== Rate Limiting ========
    delay_between_lectures: float = 1.0
    delay_between_courses: float = 2.0
    delay_after_download: float = 0.5
    
    # ======== Logging Settings ========
    log_level: str = "INFO"
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5
    console_log_level: str = "INFO"
    
    # ======== Performance Settings ========
    pdf_text_cache_size: int = 100
    enable_metrics: bool = True
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        """Initialize directories and load from environment"""
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load from environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Override configuration from environment variables"""
        env_mappings = {
            'NPTEL_MAX_COURSES': ('max_courses', int),
            'NPTEL_HEADLESS': ('headless_mode', lambda x: x.lower() == 'true'),
            'NPTEL_LOG_LEVEL': ('log_level', str),
            'NPTEL_OUTPUT_DIR': ('output_dir', Path),
            'NPTEL_REQUEST_TIMEOUT': ('request_timeout', int),
            'NPTEL_MAX_RETRIES': ('max_retries', int),
        }
        
        for env_var, (attr, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    setattr(self, attr, converter(value))
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {value}. Using default.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, handling Path objects"""
        config_dict = asdict(self)
        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        return config_dict


# ============================================================================
# LOGGING SETUP
# ============================================================================

class LoggerSetup:
    """Centralized logging configuration"""
    
    @staticmethod
    def setup(config: ScraperConfig) -> logging.Logger:
        """
        Configure logging with both file and console handlers.
        Includes rotation to prevent log files from growing too large.
        """
        logger = logging.getLogger("NPTELScraper")
        logger.setLevel(getattr(logging, config.log_level.upper()))
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler with rotation
        log_file_path = config.log_dir / config.log_file
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=config.log_max_bytes,
            backupCount=config.log_backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config.console_log_level.upper()))
        
        # Detailed formatter for file
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Simpler formatter for console
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger


# ============================================================================
# METRICS TRACKING
# ============================================================================

@dataclass
class ScraperMetrics:
    """Track scraper performance metrics"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    courses_attempted: int = 0
    courses_successful: int = 0
    courses_failed: int = 0
    
    lectures_attempted: int = 0
    lectures_successful: int = 0
    lectures_failed: int = 0
    
    total_characters_scraped: int = 0
    
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_error(self, error_type: str, message: str, context: Dict[str, Any]):
        """Record an error with context"""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': message,
            'context': context
        })
    
    def finalize(self):
        """Mark the scraping as complete"""
        self.end_time = datetime.now()
    
    def get_duration(self) -> float:
        """Get total duration in seconds"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.get_duration(),
            'courses': {
                'attempted': self.courses_attempted,
                'successful': self.courses_successful,
                'failed': self.courses_failed,
                'success_rate': (
                    self.courses_successful / self.courses_attempted * 100
                    if self.courses_attempted > 0 else 0
                )
            },
            'lectures': {
                'attempted': self.lectures_attempted,
                'successful': self.lectures_successful,
                'failed': self.lectures_failed,
                'success_rate': (
                    self.lectures_successful / self.lectures_attempted * 100
                    if self.lectures_attempted > 0 else 0
                )
            },
            'data': {
                'total_characters': self.total_characters_scraped,
                'avg_chars_per_lecture': (
                    self.total_characters_scraped / self.lectures_successful
                    if self.lectures_successful > 0 else 0
                )
            },
            'errors': self.errors
        }


# ============================================================================
# WEBDRIVER MANAGEMENT
# ============================================================================

class ChromeDriverManager:
    """Manages Chrome WebDriver lifecycle with robust error handling"""
    
    def __init__(self, config: ScraperConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._driver = None
    
    def _get_chrome_options(self) -> Options:
        """Configure Chrome options for stability and performance"""
        options = Options()
        
        # Basic options
        if self.config.headless_mode:
            options.add_argument('--headless=new')
        
        if self.config.disable_gpu:
            options.add_argument('--disable-gpu')
        
        if self.config.no_sandbox:
            options.add_argument('--no-sandbox')
        
        if self.config.disable_dev_shm:
            options.add_argument('--disable-dev-shm-usage')
        
        # Window size
        options.add_argument(f'--window-size={self.config.window_size}')
        
        # User agent
        if self.config.user_agent:
            options.add_argument(f'user-agent={self.config.user_agent}')
        
        # Additional stability options
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-popup-blocking')
        options.add_argument('--disable-notifications')
        options.add_argument('--disable-infobars')
        options.add_argument('--ignore-certificate-errors')
        
        # Performance options
        options.add_argument('--disable-logging')
        options.add_argument('--log-level=3')
        options.add_argument('--silent')
        
        # Exclude switches
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Preferences
        prefs = {
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": False,
            "profile.default_content_setting_values.notifications": 2
        }
        options.add_experimental_option("prefs", prefs)
        
        return options
    
    @contextmanager
    def get_driver(self):
        """Context manager for WebDriver with automatic cleanup"""
        driver = None
        try:
            options = self._get_chrome_options()
            driver = webdriver.Chrome(options=options)
            
            # Set timeouts
            driver.set_page_load_timeout(self.config.page_load_timeout)
            driver.set_script_timeout(self.config.script_timeout)
            driver.implicitly_wait(self.config.selenium_implicit_wait)
            
            self.logger.info("Chrome WebDriver initialized successfully")
            yield driver
            
        except WebDriverException as e:
            self.logger.error(f"Failed to initialize Chrome WebDriver: {e}")
            raise DriverInitializationError(f"WebDriver initialization failed: {e}")
        
        finally:
            if driver:
                try:
                    driver.quit()
                    self.logger.debug("Chrome WebDriver closed successfully")
                except Exception as e:
                    self.logger.warning(f"Error closing WebDriver: {e}")


# ============================================================================
# HTTP SESSION MANAGEMENT
# ============================================================================

class HTTPSessionManager:
    """Manages HTTP session with retry logic and connection pooling"""
    
    def __init__(self, config: ScraperConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff_factor,
            status_forcelist=self.config.retry_status_codes,
            allowed_methods=["GET", "POST", "HEAD"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        return session
    
    def download_file(self, url: str, timeout: Optional[int] = None) -> bytes:
        """Download file with retry logic"""
        timeout = timeout or self.config.request_timeout
        
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            self.logger.error(f"Failed to download from {url}: {e}")
            raise
    
    def close(self):
        """Close the session"""
        if self.session:
            self.session.close()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class TextUtils:
    """Text processing utilities"""
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for ID generation"""
        if not text:
            return "unknown"
        # Remove non-alphanumeric, replace with underscore
        normalized = re.sub(r'\W+', '_', text)
        # Remove leading/trailing underscores and convert to lowercase
        return normalized.strip('_').lower()
    
    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Clean excessive whitespace from text"""
        if not text:
            return ""
        # Replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def extract_drive_file_id(url: str) -> Optional[str]:
        """Extract Google Drive file ID from URL"""
        if not url:
            return None
        
        # Try different URL patterns
        patterns = [
            r'/d/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
            r'/file/d/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    @staticmethod
    def generate_document_id(course_name: str, lecture_no: int) -> str:
        """Generate stable document ID"""
        course = TextUtils.normalize(course_name)
        return f"{course}_L{lecture_no:03d}"  # Zero-padded lecture number


# ============================================================================
# COURSE LIST SCRAPER
# ============================================================================

class CourseListScraper:
    """Scrapes the main NPTEL courses listing page"""
    
    def __init__(
        self,
        config: ScraperConfig,
        logger: logging.Logger,
        driver_manager: ChromeDriverManager
    ):
        self.config = config
        self.logger = logger
        self.driver_manager = driver_manager
    
    def scrape(self) -> pd.DataFrame:
        """
        Scrape all courses from the NPTEL courses page.
        
        Returns:
            DataFrame with course information
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting to scrape NPTEL courses list")
        self.logger.info("=" * 80)
        
        with self.driver_manager.get_driver() as driver:
            try:
                # Navigate to courses page
                self.logger.info("Navigating to https://nptel.ac.in/courses")
                driver.get("https://nptel.ac.in/courses")
                
                # Wait for course cards to load
                wait = WebDriverWait(driver, self.config.selenium_explicit_wait)
                cards = wait.until(
                    EC.presence_of_all_elements_located(
                        (By.CSS_SELECTOR, "div.course-card")
                    )
                )
                
                self.logger.info(f"Found {len(cards)} course cards on the page")
                
                # Extract course information
                courses_data = []
                failed_count = 0
                
                for idx, card in enumerate(tqdm(cards, desc="Extracting courses"), 1):
                    try:
                        course_info = self._extract_course_info(card, idx)
                        if course_info:
                            courses_data.append(course_info)
                    except Exception as e:
                        failed_count += 1
                        self.logger.warning(f"Failed to extract course {idx}: {e}")
                        continue
                
                # Create DataFrame
                df = pd.DataFrame(courses_data)
                
                self.logger.info(f"Successfully extracted {len(df)} courses")
                self.logger.info(f"Failed to extract {failed_count} courses")
                
                # Save courses list
                output_path = self.config.output_dir / self.config.courses_csv
                df.to_csv(output_path, index=False, encoding='utf-8')
                self.logger.info(f"Saved courses list to: {output_path}")
                
                return df
                
            except TimeoutException:
                self.logger.error("Timeout waiting for course cards to load")
                raise CourseScrapingError("Course cards did not load in time")
            except Exception as e:
                self.logger.error(f"Unexpected error while scraping courses: {e}", exc_info=True)
                raise CourseScrapingError(f"Failed to scrape courses: {e}")
    
    def _extract_course_info(self, card, idx: int) -> Optional[Dict[str, str]]:
        """
        Extract information from a single course card.
        
        Args:
            card: Selenium WebElement representing a course card
            idx: Index of the course (for logging)
            
        Returns:
            Dictionary with course information or None if extraction fails
        """
        try:
            # Extract course name
            course_name = card.find_element(
                By.CSS_SELECTOR, "div.name"
            ).text.strip()
            
            # Extract discipline
            discipline = card.find_element(
                By.CSS_SELECTOR, "div.discipline"
            ).text.strip()
            
            # Extract metadata (professor and institute)
            meta_data = card.find_element(
                By.CSS_SELECTOR, "div.meta-data"
            ).text.strip()
            
            # Parse professor and institute
            lines = [line.strip() for line in meta_data.split("\n") if line.strip()]
            professor = lines[0] if len(lines) > 0 else "N/A"
            institute = lines[1] if len(lines) > 1 else "N/A"
            
            # Extract course URL
            course_url = card.find_element(By.TAG_NAME, "a").get_attribute("href")
            
            if not all([course_name, course_url]):
                self.logger.warning(f"Course {idx}: Missing required fields")
                return None
            
            return {
                "course_id": idx,
                "course_name": course_name,
                "discipline": discipline,
                "professor": professor,
                "institute": institute,
                "url": course_url
            }
            
        except NoSuchElementException as e:
            self.logger.debug(f"Course {idx}: Missing element - {e}")
            return None
        except StaleElementReferenceException:
            self.logger.debug(f"Course {idx}: Stale element reference")
            return None


# ============================================================================
# TRANSCRIPT SCRAPER
# ============================================================================

class TranscriptScraper:
    """Scrapes lecture transcripts for individual courses"""
    
    def __init__(
        self,
        config: ScraperConfig,
        logger: logging.Logger,
        driver_manager: ChromeDriverManager,
        http_manager: HTTPSessionManager,
        metrics: ScraperMetrics
    ):
        self.config = config
        self.logger = logger
        self.driver_manager = driver_manager
        self.http_manager = http_manager
        self.metrics = metrics
    
    def scrape_course(self, course_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Scrape all lecture transcripts for a single course.
        
        Args:
            course_info: Dictionary containing course information
            
        Returns:
            List of documents with page_content and metadata
        """
        course_name = course_info['course_name']
        course_url = course_info['url']
        
        self.logger.info("-" * 80)
        self.logger.info(f"Scraping course: {course_name}")
        self.logger.info(f"URL: {course_url}")
        self.logger.info("-" * 80)
        
        self.metrics.courses_attempted += 1
        documents = []
        
        with self.driver_manager.get_driver() as driver:
            try:
                # Navigate to course page
                driver.get(course_url)
                wait = WebDriverWait(driver, self.config.selenium_explicit_wait)
                
                # Navigate to transcripts section
                self._navigate_to_transcripts(driver, wait)
                
                # Get all lecture elements
                lecture_elements = self._get_lecture_elements(driver)
                total_lectures = len(lecture_elements)
                
                self.logger.info(f"Found {total_lectures} lectures for {course_name}")
                
                if total_lectures == 0:
                    self.logger.warning(f"No lectures found for {course_name}")
                    self.metrics.courses_failed += 1
                    return documents
                
                # Process each lecture
                for idx, lecture_elem in enumerate(lecture_elements, start=1):
                    self.metrics.lectures_attempted += 1
                    
                    try:
                        doc = self._process_lecture(
                            driver=driver,
                            lecture_element=lecture_elem,
                            course_info=course_info,
                            lecture_no=idx,
                            total_lectures=total_lectures
                        )
                        
                        if doc:
                            documents.append(doc)
                            self.metrics.lectures_successful += 1
                            self.metrics.total_characters_scraped += len(doc['page_content'])
                            
                            self.logger.info(
                                f"✓ [{idx}/{total_lectures}] {doc['metadata']['lecture_name']} "
                                f"({len(doc['page_content'])} chars)"
                            )
                        else:
                            self.metrics.lectures_failed += 1
                            self.logger.warning(f"✗ [{idx}/{total_lectures}] No content extracted")
                        
                        # Rate limiting
                        time.sleep(self.config.delay_between_lectures)
                        
                    except Exception as e:
                        self.metrics.lectures_failed += 1
                        self.logger.error(f"✗ [{idx}/{total_lectures}] Error: {e}")
                        self.metrics.record_error(
                            error_type="lecture_scraping",
                            message=str(e),
                            context={
                                'course': course_name,
                                'lecture_no': idx
                            }
                        )
                        continue
                
                # Update course statistics
                if documents:
                    self.metrics.courses_successful += 1
                    self.logger.info(
                        f"Successfully scraped {len(documents)}/{total_lectures} "
                        f"transcripts for {course_name}"
                    )
                else:
                    self.metrics.courses_failed += 1
                    self.logger.warning(f"No transcripts extracted for {course_name}")
                
            except Exception as e:
                self.metrics.courses_failed += 1
                self.logger.error(f"Failed to scrape course {course_name}: {e}", exc_info=True)
                self.metrics.record_error(
                    error_type="course_scraping",
                    message=str(e),
                    context={'course': course_name, 'url': course_url}
                )
        
        return documents
    
    def _navigate_to_transcripts(self, driver, wait):
        """Navigate to the transcripts section of the course page"""
        try:
            # Click Downloads tab
            self.logger.debug("Clicking Downloads tab...")
            downloads_tab = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//span[contains(@class,'tab') and contains(translate(., 'DOWNLOAD', 'download'), 'download')]")
                )
            )
            driver.execute_script("arguments[0].click();", downloads_tab)
            time.sleep(1.5)
            
            # Click Transcripts section
            self.logger.debug("Clicking Transcripts section...")
            transcripts_tab = wait.until(
                EC.element_to_be_clickable(
                    (By.XPATH, "//h3[contains(translate(., 'TRANSCRIPTS', 'transcripts'), 'transcripts')]")
                )
            )
            driver.execute_script("arguments[0].click();", transcripts_tab)
            time.sleep(1.5)
            
            self.logger.debug("Successfully navigated to transcripts section")
            
        except TimeoutException as e:
            self.logger.error("Failed to navigate to transcripts section")
            raise TranscriptExtractionError(f"Cannot find transcripts section: {e}")
    
    def _get_lecture_elements(self, driver) -> List:
        """Get all lecture elements from the page"""
        try:
            # Wait a moment for elements to be fully loaded
            time.sleep(2)
            
            lectures = driver.find_elements(
                By.XPATH,
                "//div[contains(@class,'d-data')][.//span[contains(@class,'c-name')]]"
            )
            
            self.logger.debug(f"Found {len(lectures)} lecture elements")
            return lectures
            
        except Exception as e:
            self.logger.error(f"Error getting lecture elements: {e}")
            return []
    
    def _process_lecture(
        self,
        driver,
        lecture_element,
        course_info: Dict[str, Any],
        lecture_no: int,
        total_lectures: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single lecture and extract its transcript.
        
        Args:
            driver: Selenium WebDriver
            lecture_element: WebElement for the lecture
            course_info: Course information dictionary
            lecture_no: Lecture number (1-indexed)
            total_lectures: Total number of lectures in course
            
        Returns:
            Document dictionary or None if extraction fails
        """
        try:
            # Extract lecture name
            lecture_name = lecture_element.find_element(
                By.XPATH, ".//span[contains(@class,'c-name')]"
            ).text.strip()
            
            if not lecture_name:
                self.logger.warning(f"Lecture {lecture_no}: No name found")
                return None
            
            # Select target language
            self._select_language(driver, lecture_element, lecture_no)
            
            # Get Google Drive link
            drive_link = self._get_drive_link(lecture_element, lecture_no)
            if not drive_link:
                return None
            
            # Extract file ID
            file_id = TextUtils.extract_drive_file_id(drive_link)
            if not file_id:
                self.logger.warning(f"Lecture {lecture_no}: Could not extract file ID from {drive_link}")
                return None
            
            # Download and extract transcript
            transcript_text = self._extract_transcript_from_drive(file_id, lecture_no)
            if not transcript_text:
                return None
            
            # Create document
            document = {
                "page_content": transcript_text,
                "metadata": {
                    "course_id": course_info.get('course_id', 0),
                    "course_name": course_info['course_name'],
                    "discipline": course_info.get('discipline', 'N/A'),
                    "professor": course_info.get('professor', 'N/A'),
                    "institute": course_info.get('institute', 'N/A'),
                    "course_url": course_info['url'],
                    "lecture_no": lecture_no,
                    "lecture_name": lecture_name,
                    "language": self.config.target_language,
                    "source": "google-drive",
                    "file_id": file_id,
                    "total_lectures": total_lectures,
                    "scrape_timestamp": datetime.now().isoformat()
                }
            }
            
            return document
            
        except Exception as e:
            self.logger.error(f"Lecture {lecture_no}: Processing failed - {e}")
            raise
    
    def _select_language(self, driver, lecture_element, lecture_no: int):
        """Select the target language for the transcript"""
        try:
            # Find and click language dropdown
            dropdown = lecture_element.find_element(
                By.XPATH, ".//div[contains(@class,'pseudo-input')]"
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", dropdown)
            time.sleep(0.3)
            driver.execute_script("arguments[0].click();", dropdown)
            time.sleep(0.5)
            
            # Select target language
            language_option = lecture_element.find_element(
                By.XPATH,
                f".//li[contains(translate(., 'ENGLISH', 'english'), 'english')]"
            )
            driver.execute_script("arguments[0].click();", language_option)
            time.sleep(0.8)
            
        except NoSuchElementException:
            self.logger.warning(f"Lecture {lecture_no}: Could not find language selector")
            raise
    
    def _get_drive_link(self, lecture_element, lecture_no: int) -> Optional[str]:
        """Extract Google Drive link from lecture element"""
        try:
            drive_link = lecture_element.find_element(
                By.XPATH, ".//a[contains(@href,'drive.google.com')]"
            ).get_attribute("href")
            
            return drive_link if drive_link else None
            
        except NoSuchElementException:
            self.logger.warning(f"Lecture {lecture_no}: No Google Drive link found")
            return None
    
    def _extract_transcript_from_drive(
        self,
        file_id: str,
        lecture_no: int
    ) -> Optional[str]:
        """
        Download PDF from Google Drive and extract text.
        
        Args:
            file_id: Google Drive file ID
            lecture_no: Lecture number (for logging)
            
        Returns:
            Extracted text or None if extraction fails
        """
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        try:
            # Download PDF
            self.logger.debug(f"Lecture {lecture_no}: Downloading PDF (file_id: {file_id})")
            pdf_content = self.http_manager.download_file(download_url)
            
            # Extract text from PDF
            text_content = []
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                total_pages = len(pdf.pages)
                self.logger.debug(f"Lecture {lecture_no}: Processing {total_pages} pages")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
            
            # Combine all pages
            full_text = "\n".join(text_content).strip()
            
            if not full_text:
                self.logger.warning(f"Lecture {lecture_no}: No text extracted from PDF")
                return None
            
            # Clean the text
            full_text = TextUtils.clean_whitespace(full_text)
            
            self.logger.debug(
                f"Lecture {lecture_no}: Extracted {len(full_text)} characters "
                f"from {total_pages} pages"
            )
            
            # Rate limiting after download
            time.sleep(self.config.delay_after_download)
            
            return full_text
            
        except requests.RequestException as e:
            self.logger.error(f"Lecture {lecture_no}: Failed to download PDF - {e}")
            return None
        except Exception as e:
            self.logger.error(f"Lecture {lecture_no}: Failed to extract text from PDF - {e}")
            return None


# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Handles document conversion and chunking operations"""
    
    def __init__(
        self,
        config: ScraperConfig,
        logger: logging.Logger,
        metrics: ScraperMetrics
    ):
        self.config = config
        self.logger = logger
        self.metrics = metrics
    
    def raw_to_langchain(self, raw_documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert raw document dictionaries to LangChain Document objects.
        
        Args:
            raw_documents: List of dictionaries with 'page_content' and 'metadata'
            
        Returns:
            List of LangChain Document objects
        """
        self.logger.info(f"Converting {len(raw_documents)} raw documents to LangChain format")
        
        documents = []
        for idx, doc_dict in enumerate(raw_documents, 1):
            try:
                documents.append(
                    Document(
                        page_content=doc_dict['page_content'],
                        metadata=doc_dict['metadata']
                    )
                )
            except KeyError as e:
                self.logger.warning(f"Document {idx}: Missing required key {e}")
                continue
        
        self.logger.info(f"Successfully converted {len(documents)} documents")
        return documents
    
    def langchain_to_json(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Convert LangChain Documents to JSON-serializable dictionaries.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of dictionaries
        """
        return [
            {
                'page_content': doc.page_content,
                'metadata': doc.metadata
            }
            for doc in documents
        ]


# ============================================================================
# FILE MANAGER
# ============================================================================

class FileManager:
    """Handles file I/O operations"""
    
    def __init__(self, config: ScraperConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def save_json(self, data: Any, filename: str, indent: int = 2):
        """Save data to JSON file"""
        filepath = self.config.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            
            self.logger.info(f"✓ Saved data to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save JSON to {filepath}: {e}")
            raise
    
    def load_json(self, filename: str) -> Any:
        """Load data from JSON file"""
        filepath = self.config.output_dir / filename
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"✓ Loaded data from {filepath}")
            return data
            
        except FileNotFoundError:
            self.logger.warning(f"File not found: {filepath}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load JSON from {filepath}: {e}")
            raise
    
    def save_dataframe(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to CSV file"""
        filepath = self.config.output_dir / filename
        
        try:
            df.to_csv(filepath, index=False, encoding='utf-8')
            self.logger.info(f"✓ Saved DataFrame to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save DataFrame to {filepath}: {e}")
            raise


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class NPTELScraperPipeline:
    """Main orchestration pipeline for NPTEL scraping"""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.logger = LoggerSetup.setup(config)
        self.metrics = ScraperMetrics()
        
        # Initialize managers
        self.driver_manager = ChromeDriverManager(config, self.logger)
        self.http_manager = HTTPSessionManager(config, self.logger)
        self.file_manager = FileManager(config, self.logger)
        
        # Initialize scrapers and processors
        self.course_scraper = CourseListScraper(
            config, self.logger, self.driver_manager
        )
        self.transcript_scraper = TranscriptScraper(
            config, self.logger, self.driver_manager,
            self.http_manager, self.metrics
        )
        self.document_processor = DocumentProcessor(
            config, self.logger, self.metrics
        )
    
    def run(self):
        """Execute the complete scraping and processing pipeline"""
        self.logger.info("=" * 80)
        self.logger.info("NPTEL SCRAPER PIPELINE - START")
        self.logger.info("=" * 80)
        self.logger.info(f"Configuration: {self.config.to_dict()}")
        self.logger.info("=" * 80)
        
        try:
            # Step 1: Scrape courses list
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 1: Scraping Courses List")
            self.logger.info("=" * 80)
            
            courses_df = self.course_scraper.scrape()
            
            if courses_df.empty:
                self.logger.error("No courses found! Exiting.")
                return
            
            # Step 2: Scrape transcripts
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 2: Scraping Course Transcripts")
            self.logger.info(f"Processing {self.config.max_courses} courses")
            self.logger.info("=" * 80)
            
            all_documents = []
            
            # Select courses to process
            courses_to_process = courses_df.iloc[
                self.config.start_course_index:
                self.config.start_course_index + self.config.max_courses
            ]
            
            for idx, (_, course) in enumerate(courses_to_process.iterrows(), 1):
                self.logger.info(
                    f"\n{'=' * 80}\n"
                    f"Processing Course {idx}/{len(courses_to_process)}\n"
                    f"{'=' * 80}"
                )
                
                course_docs = self.transcript_scraper.scrape_course(course.to_dict())
                all_documents.extend(course_docs)
                
                # Save intermediate results
                if self.config.save_intermediate_results and course_docs:
                    self._save_intermediate(all_documents, idx)
                
                # Rate limiting between courses
                if idx < len(courses_to_process):
                    time.sleep(self.config.delay_between_courses)
            
            if not all_documents:
                self.logger.warning("No documents were scraped!")
                return
            
            # Step 3: Save raw documents
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 3: Saving Raw Documents")
            self.logger.info("=" * 80)
            
            self.file_manager.save_json(
                all_documents,
                self.config.raw_documents_json
            )
            
            # Step 4: Convert to LangChain format
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 4: Converting to LangChain Format")
            self.logger.info("=" * 80)
            
            lc_documents = self.document_processor.raw_to_langchain(all_documents)
            
            # Step 5: Generate and save summary
            self.logger.info("\n" + "=" * 80)
            self.logger.info("STEP 7: Generating Summary")
            self.logger.info("=" * 80)
            
            self._generate_summary(courses_df, all_documents, lc_documents)
            
            # Finalize
            self.metrics.finalize()
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info("=" * 80)
            self._print_summary()
            
        except KeyboardInterrupt:
            self.logger.warning("\n\nPipeline interrupted by user (Ctrl+C)")
            self.metrics.finalize()
            self._print_summary()
            
        except Exception as e:
            self.logger.error(f"\n\nPipeline failed with error: {e}", exc_info=True)
            self.metrics.finalize()
            raise
            
        finally:
            # Cleanup
            self.http_manager.close()
            self.logger.info("Cleanup completed")
    
    def _save_intermediate(self, documents: List[Dict], batch_num: int):
        """Save intermediate results"""
        filename = f"documents_intermediate_batch_{batch_num}.json"
        self.file_manager.save_json(documents, filename)
    
    def _generate_summary(
        self,
        courses_df: pd.DataFrame,
        documents: List[Dict],
        lc_documents: List[Document]  # RENAME TO lc_documents
    ):
        """Generate and save comprehensive pipeline summary"""
        
        # Group documents by course
        course_stats = {}
        for doc in documents:
            course_name = doc['metadata']['course_name']
            if course_name not in course_stats:
                course_stats[course_name] = {
                    'course_name': course_name,
                    'discipline': doc['metadata']['discipline'],
                    'professor': doc['metadata']['professor'],
                    'institute': doc['metadata']['institute'],
                    'lecture_count': 0,
                    'total_characters': 0,
                    'avg_chars_per_lecture': 0
                }
            
            course_stats[course_name]['lecture_count'] += 1
            course_stats[course_name]['total_characters'] += len(doc['page_content'])
        
        # Calculate averages
        for course in course_stats.values():
            if course['lecture_count'] > 0:
                course['avg_chars_per_lecture'] = (
                    course['total_characters'] / course['lecture_count']
                )
        
        # Create summary
        summary = {
            'pipeline_info': {
                'version': '2.0.0',
                'execution_timestamp': datetime.now().isoformat(),
                'configuration': self.config.to_dict()
            },
            'statistics': {
                'total_courses_available': len(courses_df),
                'courses_processed': self.config.max_courses,
                'courses_successful': self.metrics.courses_successful,
                'courses_failed': self.metrics.courses_failed,
                'total_lectures_scraped': self.metrics.lectures_successful,
                'total_lectures_failed': self.metrics.lectures_failed,
                'total_documents': len(documents),
                'total_characters_scraped': self.metrics.total_characters_scraped
            },
            'course_details': list(course_stats.values()),
            'metrics': self.metrics.to_dict()
        }
        
        # Save summary
        self.file_manager.save_json(summary, self.config.summary_json)
        
        self.logger.info("Summary generated and saved successfully")
    
    def _print_summary(self):
        """Print execution summary to console"""
        metrics = self.metrics.to_dict()
        
        print("\n" + "=" * 80)
        print("EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Duration: {metrics['duration_seconds']:.2f} seconds")
        print(f"\nCourses:")
        print(f"  Attempted: {metrics['courses']['attempted']}")
        print(f"  Successful: {metrics['courses']['successful']}")
        print(f"  Failed: {metrics['courses']['failed']}")
        print(f"  Success Rate: {metrics['courses']['success_rate']:.1f}%")
        print(f"\nLectures:")
        print(f"  Attempted: {metrics['lectures']['attempted']}")
        print(f"  Successful: {metrics['lectures']['successful']}")
        print(f"  Failed: {metrics['lectures']['failed']}")
        print(f"  Success Rate: {metrics['lectures']['success_rate']:.1f}%")
        print(f"\nData:")
        print(f"  Total Characters: {metrics['data']['total_characters']:,}")
        print(f"  Avg Chars/Lecture: {metrics['data']['avg_chars_per_lecture']:.0f}")
        
        if metrics['errors']:
            print(f"\nErrors: {len(metrics['errors'])} errors occurred")
        
        print("=" * 80 + "\n")


# ============================================================================
# CLI / ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the NPTEL scraper.
    
    This function can be called directly or used as a command-line tool.
    Configuration can be customized by modifying the ScraperConfig parameters.
    """
    
    # Create configuration
    # You can customize these parameters or load from environment variables
    config = ScraperConfig(
        # Start with 1 course for testing, increase for production
        max_courses=1,
        
        # Set to True for production/server environments
        headless_mode=True,
        
        # Performance settings
        delay_between_lectures=1.0,
        delay_between_courses=2.0,
        
        # Enable metrics tracking
        enable_metrics=True,
        
        # Logging
        log_level="INFO"
    )
    
    # Initialize and run pipeline
    pipeline = NPTELScraperPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()