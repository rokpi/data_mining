"""
E-Commerce Data Scraper - FIXED VERSION

Properly scrapes from https://web-scraping.dev/
"""

import json
import time
import random
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Base URLs
BASE_URL = "https://web-scraping.dev"
PRODUCTS_URL = f"{BASE_URL}/products"
REVIEWS_URL = f"{BASE_URL}/reviews"
GRAPHQL_URL = f"{BASE_URL}/api/graphql"
TESTIMONIALS_URL = f"{BASE_URL}/testimonials"
TESTIMONIALS_API_URL = f"{BASE_URL}/api/testimonials"

# Request headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

REQUEST_DELAY = (0.5, 1.5)


def rate_limit():
    """Apply rate limiting between requests."""
    time.sleep(random.uniform(*REQUEST_DELAY))


def make_request(url: str, headers: dict = None, **kwargs) -> Optional[requests.Response]:
    """Make HTTP request with retry logic."""
    max_retries = 3
    request_headers = {**HEADERS, **(headers or {})}
    
    for attempt in range(max_retries):
        try:
            rate_limit()
            response = requests.get(url, headers=request_headers, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"All retries failed for {url}")
                return None
    return None


def scrape_products() -> list[dict[str, Any]]:
    """Scrape all products with pagination and proper categorization."""
    logger.info("Starting product scraping...")
    products = []
    product_id = 1
    
    # First, scrape the categories from the filters section
    logger.info("Discovering categories...")
    response = make_request(PRODUCTS_URL)
    if not response:
        logger.error("Failed to fetch products page to discover categories")
        return products
    
    soup = BeautifulSoup(response.text, 'html.parser')
    filters_div = soup.find('div', class_='filters')
    
    categories = {}
    if filters_div:
        category_links = filters_div.find_all('a')
        for link in category_links:
            category_text = link.get_text(strip=True)
            href = link.get('href', '')
            
            # Skip empty or invalid links
            if not category_text:
                continue
            
            # Extract category parameter from URL
            if 'category=' in href:
                # Extract the category value from URL like "?category=apparel"
                category_param = href.split('category=')[1].split('&')[0]
                # Capitalize first letter for display name
                category_name = category_text.capitalize()
                categories[category_param] = category_name
            elif category_text.lower() == 'all':
                # Skip 'all' to avoid duplicates
                continue
    
    logger.info(f"Found categories: {categories}")
    
    # If no categories found, use fallback
    if not categories:
        logger.warning("No categories found, using fallback")
        categories = {
            'apparel': 'Apparel',
            'consumables': 'Consumables',
            'household': 'Household'
        }
    
    for category_param, category_name in categories.items():
        page = 1
        
        while page <= 10:  # Safety limit per category
            # Build URL
            if category_param == 'all':
                url = f"{PRODUCTS_URL}?page={page}"
            else:
                url = f"{PRODUCTS_URL}?category={category_param}&page={page}"
            
            logger.info(f"Scraping {category_name} - page {page}: {url}")
            
            response = make_request(url)
            if not response:
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find product containers - the class is just 'product' in a 'row'
            product_divs = soup.find_all('div', class_='row product')
            
            if not product_divs:
                logger.info(f"No products found on {category_name} page {page}")
                break
                
            for div in product_divs:
                try:
                    # Extract product name from h3 > a
                    h3_elem = div.find('h3', class_='mb-0')
                    if not h3_elem:
                        continue
                    
                    name_link = h3_elem.find('a')
                    if not name_link:
                        continue
                    
                    name = name_link.get_text(strip=True)
                    
                    # Extract price from div.price
                    price_elem = div.find('div', class_='price')
                    if not price_elem:
                        continue
                    
                    try:
                        price = float(price_elem.get_text(strip=True))
                    except ValueError:
                        price = 0.0
                    
                    # Extract description
                    desc_elem = div.find('div', class_='short-description')
                    description = desc_elem.get_text(strip=True) if desc_elem else ""
                    
                    # Extract image
                    img_elem = div.find('img', class_='img-thumbnail')
                    image_url = img_elem.get('src', '') if img_elem else ''
                    if image_url and not image_url.startswith('http'):
                        image_url = f"https://web-scraping.dev{image_url}"
                    
                    # Extract product URL
                    product_url = name_link.get('href', '')
                    if product_url and not product_url.startswith('http'):
                        product_url = f"https://web-scraping.dev{product_url}"
                    
                    # Check if product already exists (avoid duplicates from 'all' category)
                    if any(p['name'] == name for p in products):
                        continue
                    
                    product = {
                        "id": product_id,
                        "name": name,
                        "category": category_name,
                        "price": price,
                        "image_url": image_url,
                        "product_url": product_url,
                    }
                    
                    products.append(product)
                    product_id += 1
                    
                except Exception as e:
                    logger.warning(f"Error parsing product: {e}")
                    continue
            
            # Check for next page
            pagination = soup.find('div', class_='paging')
            if not pagination:
                break
            
            # Check if there's a next page link
            next_link = None
            page_links = pagination.find_all('a')
            for link in page_links:
                if link.get_text(strip=True) == '>':
                    next_link = link
                    break
            
            if not next_link:
                break
            
            page += 1
    
    logger.info(f"Scraped {len(products)} products total")
    return products

def normalize_name(name: str) -> str:
    """Normalize product name for matching."""
    # Convert to lowercase
    name = name.lower()
    # Remove all apostrophes first
    name = name.replace("'", "")
    # Handle possessives: "womens" -> "women", "mens" -> "men"
    # But be careful not to break other words ending in 's'
    name = name.replace("womens", "women").replace("mens", "men")
    # Replace hyphens with spaces  
    name = name.replace("-", " ")
    # Remove extra spaces
    name = " ".join(name.split())
    return name

def scrape_reviews_graphql(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Scrape ALL reviews using GraphQL API - paginate until no more data."""
    logger.info("Starting review scraping via GraphQL...")
    reviews = []
    review_id = 1
    cursor = None
    page = 1
    
    # Create a mapping of product names (from rid) to product IDs
    product_name_to_id = {}
    product_name_variants = {}
    for product in products:
        # Normalize the product name for matching
        normalized_name = normalize_name(product['name'])
        product_name_to_id[normalized_name] = product['id']
        
        # Create word-based index for fuzzy matching
        words = normalized_name.split()
        for i in range(len(words)):
            # Store each substring combination
            for j in range(i+1, len(words)+1):
                key = " ".join(words[i:j])
                if key not in product_name_variants:
                    product_name_variants[key] = []
                product_name_variants[key].append((product['id'], product['name'], normalized_name))
    
    # CORRECTED GraphQL query - only fields that exist
    query = """
    query GetReviews($first: Int, $after: String) {
        reviews(first: $first, after: $after) {
            edges {
                node {
                    rid
                    text
                    rating
                    date
                }
                cursor
            }
            pageInfo {
                hasNextPage
                endCursor
            }
        }
    }
    """
    
    # Keep fetching until hasNextPage is False
    while True:
        logger.info(f"Fetching reviews page {page}...")
        
        variables = {
            "first": 20,
            "after": cursor
        }
        
        payload = {
            "query": query,
            "variables": variables
        }
        
        try:
            rate_limit()
            graphql_headers = {**HEADERS, 'Content-Type': 'application/json'}
            response = requests.post(
                GRAPHQL_URL,
                headers=graphql_headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            if 'errors' in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                break
            
            reviews_data = data.get('data', {}).get('reviews', {})
            edges = reviews_data.get('edges', [])
            page_info = reviews_data.get('pageInfo', {})
            
            if not edges:
                logger.info("No more reviews found")
                break
            
            for edge in edges:
                node = edge.get('node', {})
                
                # Parse date
                date_str = node.get('date', '')
                
                # Extract product name from rid
                # rid format: "teal-potion-4" -> product name: "Teal Potion"
                # rid format: "classic-leather-sneakers-10" -> "Classic Leather Sneakers"
                rid = node.get('rid', '')
                
                # Remove the trailing number (e.g., "-4", "-10")
                # Split by '-' and remove the last part if it's a number
                parts = rid.split('-')
                if parts and parts[-1].isdigit():
                    parts = parts[:-1]
                
                # Join back and title case
                product_name_from_rid = ' '.join(parts).title()
                normalized_rid = normalize_name(product_name_from_rid)
                
                # Try to match with actual product
                product_id = None
                product_name_matched = None
                
                # Try exact match first
                if normalized_rid in product_name_to_id:
                    product_id = product_name_to_id[normalized_rid]
                    for p in products:
                        if p['id'] == product_id:
                            product_name_matched = p['name']
                            break
                else:
                    # Try direct lookup in variants
                    if normalized_rid in product_name_variants:
                        matches = product_name_variants[normalized_rid]
                        if matches:
                            product_id, product_name_matched, _ = matches[0]
                    else:
                        # Fuzzy matching - find product where rid words are subset of product words
                        rid_words = set(normalized_rid.split())
                        best_match = None
                        best_score = 0
                        
                        for normalized_prod_name, prod_id in product_name_to_id.items():
                            prod_words = set(normalized_prod_name.split())
                            
                            # Check if all rid words are in product words
                            if rid_words.issubset(prod_words):
                                # Score based on how many words match
                                score = len(rid_words)
                                if score > best_score:
                                    best_score = score
                                    best_match = prod_id
                        
                        if best_match:
                            product_id = best_match
                            for p in products:
                                if p['id'] == product_id:
                                    product_name_matched = p['name']
                                    break

                if not product_id:
                    logger.warning(f"Could not match product for rid: {rid} (normalized: '{normalized_rid}')")
                    continue

                # ONLY REAL DATA - no synthetic fields
                review = {
                    "id": review_id,
                    "product_id": product_id,
                    "product_name": product_name_matched,
                    "rating": int(node.get('rating', 0)),
                    "text": node.get('text', ''),
                    "date": date_str,
                }
                reviews.append(review)
                review_id += 1
            
            # Check for next page - this is the "Load More" button logic
            has_next = page_info.get('hasNextPage', False)
            cursor = page_info.get('endCursor')
            
            # If hasNextPage is False, the button would have "d-none" class
            if not has_next or not cursor:
                logger.info("No more pages (Load More button would be hidden)")
                break
            
            page += 1
            
            # Safety limit
            if page > 200:
                logger.warning("Reached safety limit of 200 pages")
                break
                
        except Exception as e:
            logger.error(f"Error fetching reviews: {e}")
            break
    
    logger.info(f"Scraped {len(reviews)} reviews")
    return reviews


def scrape_testimonials() -> list[dict[str, Any]]:
    """Scrape ALL testimonials using infinite scroll API - NO synthetic data."""
    logger.info("Starting testimonial scraping...")
    testimonials = []
    testimonial_id = 1
    
    # Start with main page
    logger.info("Fetching main testimonials page...")
    response = make_request(TESTIMONIALS_URL)
    
    if not response:
        logger.error("Failed to fetch main testimonials page")
        return testimonials
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract x-secret-token from the page
    app_data_script = soup.find('script', id='appData', type='application/json')
    secret_token = 'secret123'  # Default
    if app_data_script:
        try:
            app_data = json.loads(app_data_script.string)
            secret_token = app_data.get('x-secret-token', 'secret123')
            logger.info(f"Found secret token: {secret_token}")
        except:
            logger.warning("Could not parse app data, using default token")
    
    # Parse initial testimonials
    testimonial_divs = soup.find_all('div', class_='testimonial')
    
    # Track the last testimonial div that has hx-get attribute (for next page)
    next_page_url = None
    
    for div in testimonial_divs:
        try:
            # Check if this div has the hx-get attribute (infinite scroll trigger)
            hx_get = div.get('hx-get')
            if hx_get:
                next_page_url = hx_get
                logger.info(f"Found next page URL: {next_page_url}")
            
            # Extract rating (count SVG stars)
            rating_span = div.find('span', class_='rating')
            rating = 0
            if rating_span:
                stars = rating_span.find_all('svg')
                rating = len(stars)
            
            # Extract text
            text_elem = div.find('p', class_='text')
            text = text_elem.get_text(strip=True) if text_elem else ""
            
            if not text:
                continue
            
            # ONLY REAL DATA from HTML
            testimonial = {
                "id": testimonial_id,
                "text": text,
                "rating": rating if rating > 0 else 0,
            }
            
            testimonials.append(testimonial)
            testimonial_id += 1
            
        except Exception as e:
            logger.warning(f"Error parsing testimonial: {e}")
            continue
    
    logger.info(f"Scraped {len(testimonials)} testimonials from main page")
    
    # Now follow the infinite scroll chain
    max_pages = 100  # Safety limit
    page_count = 1
    
    while next_page_url and page_count < max_pages:
        # Make the URL absolute if needed
        if not next_page_url.startswith('http'):
            next_page_url = f"{BASE_URL}{next_page_url}" if next_page_url.startswith('/') else f"{BASE_URL}/{next_page_url}"
        
        logger.info(f"Fetching infinite scroll page {page_count + 1}: {next_page_url}")
        
        # Use requests.Session to make the request with proper headers
        # The key is to NOT use make_request() which uses requests.get
        # Instead, make a direct request with all the right headers
        try:
            rate_limit()
            
            # Build headers that match HTMX request
            api_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.5',
                'x-secret-token': secret_token,
                'HX-Request': 'true',
                'HX-Trigger': 'revealed',
                'HX-Current-URL': TESTIMONIALS_URL,
                'Referer': TESTIMONIALS_URL,
            }
            
            response = requests.get(next_page_url, headers=api_headers, timeout=30)
            response.raise_for_status()
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 422:
                logger.error(f"422 Error - likely missing or invalid x-secret-token header")
                logger.error(f"Response: {e.response.text[:200]}")
            logger.warning(f"Failed to fetch page: {e}")
            break
        except Exception as e:
            logger.warning(f"Failed to fetch page: {e}")
            break
        
        # Check if response is empty or has no content
        if not response.text or len(response.text.strip()) < 50:
            logger.info(f"Empty response, no more testimonials")
            break
        
        # Parse HTML response from API
        soup = BeautifulSoup(response.text, 'html.parser')
        testimonial_divs = soup.find_all('div', class_='testimonial')
        
        if not testimonial_divs:
            logger.info(f"No more testimonials found, stopping")
            break
        
        page_testimonials_count = 0
        next_page_url = None  # Reset for this iteration
        
        for div in testimonial_divs:
            try:
                # Check if this div has the hx-get attribute (next infinite scroll trigger)
                hx_get = div.get('hx-get')
                if hx_get:
                    next_page_url = hx_get
                    logger.info(f"Found next page URL: {next_page_url}")
                
                # Extract rating (count SVG stars)
                rating_span = div.find('span', class_='rating')
                rating = 0
                if rating_span:
                    stars = rating_span.find_all('svg')
                    rating = len(stars)
                
                # Extract text
                text_elem = div.find('p', class_='text')
                text = text_elem.get_text(strip=True) if text_elem else ""
                
                if not text:
                    continue
                
                # ONLY REAL DATA from HTML
                testimonial = {
                    "id": testimonial_id,
                    "text": text,
                    "rating": rating if rating > 0 else 0,
                }
                
                testimonials.append(testimonial)
                testimonial_id += 1
                page_testimonials_count += 1
                
            except Exception as e:
                logger.warning(f"Error parsing testimonial: {e}")
                continue
        
        logger.info(f"Scraped {page_testimonials_count} testimonials from page {page_count + 1}")
        
        # If we got no testimonials on this page, stop
        if page_testimonials_count == 0:
            break
        
        # If no next page URL found, we've reached the end
        if not next_page_url:
            logger.info("No more pages to load (no hx-get attribute found)")
            break
        
        page_count += 1
    
    logger.info(f"Total scraped {len(testimonials)} testimonials")
    return testimonials

def save_to_json(data: list[dict[str, Any]], filename: str) -> None:
    """Save data to JSON file."""
    filepath = DATA_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(data)} records to {filepath}")


def main():
    """Main scraping function."""
    logger.info("=" * 60)
    logger.info("E-Commerce Data Scraper - REAL DATA ONLY")
    logger.info("Target: https://web-scraping.dev/")
    logger.info("=" * 60)
    
    logger.info("\n[1/3] Scraping Products...")
    products = scrape_products()
    
    logger.info("\n[2/3] Scraping Reviews...")
    reviews = scrape_reviews_graphql(products)  # Pass products here
    
    logger.info("\n[3/3] Scraping Testimonials...")
    testimonials = scrape_testimonials()
    
    logger.info("\nSaving data to JSON files...")
    save_to_json(products, "products.json")
    save_to_json(reviews, "reviews.json")
    save_to_json(testimonials, "testimonials.json")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Scraping Complete! Summary:")
    logger.info(f"  - Products: {len(products)}")
    logger.info(f"  - Reviews: {len(reviews)}")
    logger.info(f"  - Testimonials: {len(testimonials)}")
    logger.info("=" * 60)
    
    # Review distribution
    from collections import Counter
    months_counter = Counter()
    for review in reviews:
        try:
            date = datetime.strptime(review['date'], "%Y-%m-%d")
            months_counter[date.strftime("%B")] += 1
        except:
            pass
    
    logger.info("\nReviews by Month (2023):")
    for month in ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]:
        count = months_counter.get(month, 0)
        if count > 0:
            logger.info(f"  - {month}: {count}")


if __name__ == "__main__":
    main()