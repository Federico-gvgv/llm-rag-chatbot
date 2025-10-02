# Web scraping module for crawling and extracting text from web pages

import re, time, hashlib, json
from pathlib import Path
from urllib.parse import urlparse, urljoin
import requests
import trafilatura
from bs4 import BeautifulSoup
from tqdm import tqdm

USER_AGENT = "Mozilla/5.0 (RAG-edu-bot)"

# Normalize URL by removing fragments and trailing slashes
def normalize(url: str) -> str:
    u = url.split("#")[0] # Remove fragment
    if u.endswith('/'): # Remove trailing slash
        u = u[:-1]
    return u # Return normalized URL

# Check if URL is in the same domain as seed_domain
def in_domain(url: str, seed_domain: str) -> bool:
    parsed = urlparse(url)
    return seed_domain in parsed.netloc

# Extract and normalize links from HTML content
def extract_links(html: str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser") # Parse HTML
    links = []
    for a in soup.find_all("a", href=True): # Find all anchor tags with href
        href = urljoin(base, a['href']) # Resolve relative URLs
        if href.startswith("http"):
            links.append(normalize(href))
    return links

# Fetch URL content with timeout and return HTML and extracted text
def fetch(url: str, timeout=15) -> tuple[str, str] | None:
    headers = {"User-Agent": USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200 and 'text/html' in response.headers.get('Content-Type', ''):
            html = response.text
            text = trafilatura.extract(html, include_comments=False) or ""
            return html, text
    except requests.RequestException:
        return None
    return None

# Crawl starting from seed URLs, respecting domain and page limits
def crawl(seeds: list[str], seed_domain: str, out_dir: Path, max_pages=400, same_domain_only=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw" # Directory for raw HTML and text
    raw_dir.mkdir(exist_ok=True)
    seen, queue = set(), list(map(normalize, seeds)) # Initialize seen set and queue with normalized seeds
    docs = []

    with tqdm(total=max_pages, desc="Crawl") as pbar:
        while queue and len(docs) < max_pages:
            url = queue.pop(0)
            if url in seen: continue
            seen.add(url)

            resp = fetch(url)
            if not resp: continue
            html, text = resp
            if not text.strip(): # Skip if no text extracted
                continue

            # Save raw HTML and text
            h = hashlib.md5(url.encode()).hexdigest() # Create a hash of the URL for filename
            (raw_dir / f"{h}.html").write_text(html, encoding="utf-8")
            
            docs.append({"url": url, "text": text}) # Store URL and text
            pbar.update(1) # Update progress bar

            # Discover links
            for link in set(extract_links(html, url)):
                if same_domain_only and not in_domain(link, seed_domain):
                    continue
                if link not in seen:
                    queue.append(link)

    return docs
                