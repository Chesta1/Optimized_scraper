
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import openai
import html2text
import tiktoken
import time
import os
import pandas as pd
import json
from dotenv import load_dotenv, find_dotenv
from selenium.webdriver.chrome.service import Service
import re
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import timedelta
import redis
import hashlib
import pickle
import functools

def get_redis_client():
    return redis.Redis(host='localhost', port=6379, db=0)

def redis_cache(ttl=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            redis_client = get_redis_client()
            key = generate_cache_key(func.__name__, args, kwargs)
            cached_result = redis_client.get(key)
            if cached_result:
                return pickle.loads(cached_result)
            else:
                result = func(*args, **kwargs)
                redis_client.set(key, pickle.dumps(result), ex=ttl)
                return result
        return wrapper
    return decorator

def generate_cache_key(func_name, args, kwargs):
    args_str = json.dumps(args, sort_keys=True, default=str)
    kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
    key_data = f"{func_name}:{args_str}:{kwargs_str}"
    key_hash = hashlib.sha256(key_data.encode()).hexdigest()
    return f"cache:{func_name}:{key_hash}"


openai.api_key = st.secrets["OPENAI_API_KEY"]

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options

# from selenium.webdriver.chrome.service import Service
# import streamlit as st
# import os

def get_driver():
    """Create and return a configured WebDriver instance."""
    try:
        # First check installed versions
        import subprocess
        import re
        
        try:
            # Get Chromium version
            chrome_version_output = subprocess.check_output(['chromium', '--version']).decode()
            st.write(f"Installed Chromium: {chrome_version_output.strip()}")
            
            # Get ChromeDriver version
            chromedriver_version_output = subprocess.check_output(['chromedriver', '--version']).decode()
            st.write(f"Installed ChromeDriver: {chromedriver_version_output.strip()}")
            
        except Exception as e:
            st.warning(f"Version check failed: {str(e)}")
        
        # Initialize Chrome options
        chrome_options = Options()
        
        # Basic required options
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        
        # Set binary location
        chrome_options.binary_location = "/usr/bin/chromium"
        
        # Additional options
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--remote-debugging-port=9222")  # Enable debugging
        
        # Generic user agent
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/stable Safari/537.36")

        # Initialize service with logging
        service = Service(
            executable_path='/usr/bin/chromedriver',
            log_path='/tmp/chromedriver.log',
            service_args=['--verbose']
        )

        st.write("Attempting to initialize ChromeDriver...")
        
        # Try to create driver
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        # Verify browser capabilities
        st.write("Driver capabilities:")
        st.write(f"Browser version: {driver.capabilities.get('browserVersion', 'unknown')}")
        st.write(f"ChromeDriver version: {driver.capabilities.get('chrome', {}).get('chromedriverVersion', 'unknown')}")
        
        return driver
        
    except Exception as e:
        st.error(f"Failed to initialize ChromeDriver: {str(e)}")
        
        # Try to read ChromeDriver log
        try:
            with open('/tmp/chromedriver.log', 'r') as f:
                st.code(f.read(), language='text')
        except:
            st.warning("Could not read ChromeDriver log")
            
        import traceback
        st.code(traceback.format_exc())
        raise



def format_location_key(location):
    """Format location string to use as a cache key."""
    return location.lower().strip().replace(" ", "-")

@redis_cache(ttl=86400)
def get_total_pages_and_urls(search_location):
    """
    Calculate the total number of pages and collect the URLs of each page
    by following the 'Next' href attribute in <a> tags.
    """
    search_url = f"https://www.airbnb.co.in/s/{search_location}/homes"
    driver = get_driver()
    try:
        driver.get(search_url)
        time.sleep(10)

        total_pages = 1  # Start with at least one page
        page_urls = [driver.current_url]  # Store the URL of the first page

        while True:
            try:
                # Wait for the "Next" link to be present
                next_link = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//a[@aria-label='Next']"))
                )
            except Exception as e:
                print("No 'Next' link found. Assuming this is the last page.")
                break  # Exit the loop if the "Next" link is not found

            # Attempt to click the "Next" link
            try:
                next_link.click()
                time.sleep(3)  # Wait for the next page to load

                # Increment the page counter
                total_pages += 1

                # Add the current page's URL to the list
                page_urls.append(driver.current_url)
            except Exception as e:
                print(f"Failed to click the 'Next' link: {e}")
                print("Reached the last page.")
                break  # Exit the loop if the click fails

        return total_pages, page_urls

    except Exception as e:
        print(f"Error calculating total pages: {e}")
        return 1, []  # Return 1 and an empty list if there's an error
    

    
@redis_cache(ttl=86400)
def get_html_content( listing_url,max_retries=3):
    """Fetch HTML content from the current page with retries."""
    driver = get_driver()
    for attempt in range(max_retries):
        try:
            driver.get(listing_url)
            time.sleep(10)
            WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            
            # Scroll gradually to trigger lazy-loading
            total_height = driver.execute_script("return document.body.scrollHeight")
            for i in range(3):
                driver.execute_script(f"window.scrollTo(0, {total_height * (i+1) / 3});")
                time.sleep(2)
            
            return driver.page_source
        except TimeoutException:
            if attempt == max_retries - 1:
                st.error(f"Timeout while loading the page after {max_retries} attempts")
                return None
            time.sleep(5)  # Wait before retrying
        finally:
            driver.quit()
    return None


def parse_html(html_content):
    """Parse and clean HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    for element in soup(['script', 'style', 'footer']):
        element.decompose()
    return str(soup)


def clean_content(parsed_html):
    """Convert HTML to clean text."""
    h = html2text.HTML2Text()
    h.ignore_links = h.ignore_images = True
    h.body_width = 0
    return h.handle(parsed_html)

def split_text(text, max_tokens=2000):
    """Split text into chunks."""
    encoding = tiktoken.get_encoding("cl100k_base")## encoding to represents the token model can understand and process
    tokens = encoding.encode(text)
    return [encoding.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

@redis_cache(ttl=86400)
def extract_content_with_openai(cleaned_content, fields_to_extract=None):
    """
    Extract relevant information from Airbnb listings using OpenAI API.

    Args:
        cleaned_content (str): The text content of the listing.
        fields_to_extract (list): A list of fields to extract.

    Returns:
        str: Extracted information in structured format or an error message.
    """

    try:
        # Prepare the fields to extract
        if fields_to_extract is None:
            fields_to_extract = ['property name', 'property location', 'ratings', 'amenities']
        else:
            # Ensure fields are in a proper string format
            fields_to_extract = [str(field).strip() for field in fields_to_extract]
        
        fields_str = ', '.join(fields_to_extract)

        # Call the OpenAI API
        response = openai.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise data extractor for Airbnb listings. "
                        "Focus only on the current listing and extract exactly what is asked. "
                        "Format the output clearly with labels."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"From this single Airbnb listing, extract only: {fields_str}\n\n"
                        f"Listing Content:\n{cleaned_content}"
                    )
                }
            ],
            max_tokens=500,
            temperature=0.3,  # Consistent output
        )
        
        # Extract the assistant's reply
        assistant_reply = response.choices[0].message.content.strip()
        
        # Return the assistant's reply
        print(assistant_reply)
        return assistant_reply

    except Exception as e:
        # Log error and return
        error_message = f"Error in content extraction: {str(e)}"
        print(error_message)  # Add logging here if necessary
        return error_message

@redis_cache(ttl=86400)
def process_listing(listing_url):
    """Process an individual listing and extract relevant information."""
    if not listing_url.startswith('http'):
        listing_url = f"https://{listing_url if listing_url.startswith('/') else '/' + listing_url}"
    
    try:
        # Get HTML content
        html_content = get_html_content(listing_url)
        if not html_content:
            return f"Error: Unable to load content for {listing_url}"
        
        # Process the content
        parsed_html = parse_html(html_content)
        clean_content_data = clean_content(parsed_html)
        
        # Only split if content is too long
        if len(clean_content_data) > 4000:
            text_chunks = split_text(clean_content_data)
            extracted_info = []
            for chunk in text_chunks:
                info = extract_content_with_openai(chunk)
                extracted_info.append(info)
            result = " ".join(extracted_info)
        else:
            result = extract_content_with_openai(clean_content_data)
            
        return result
            
    except Exception as e:
        print(f"Error processing listing: {str(e)}")
        return f"Error processing listing: {str(e)}"

@redis_cache(ttl=86400)
def get_listing_links_for_page(page_url):
    """Get links to individual listings from a specific page URL."""
    driver = get_driver()
    try:
        driver.get(page_url)
        time.sleep(5)
        
        # Scroll to load all content
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        listings_info = []

        listing_elements = soup.find_all('div', {'itemprop': 'itemListElement'})
        for element in listing_elements:
            url_meta = element.find('meta', {'itemprop': 'url'})
            price_span = element.find('span', class_="_11jcbg2")
            if url_meta and url_meta.get('content') and price_span:
                url = url_meta['content'].split('?')[0]
                price = price_span.text.strip() if price_span else "No price available"
                listings_info.append({'url': url, 'price': price})
        
        return listings_info
        
    finally:
        driver.quit()

def process_listing_wrapper(listing):
    """Wrapper function for multiprocessing. Processes an individual listing."""
    link = listing['url']
    price = listing['price']

    try:
        result = process_listing(link)
        if result:
            return {'link': link, 'price': price, 'data': result}
        return None
    except Exception as e:
        print(f"Error processing listing {link}: {str(e)}")
        return None


def extract_property_name(text):
    match = re.search(r'Property Name[:*]*\s*(.*)', text, re.IGNORECASE)
    if match:
        return match.group(1).split('\n', 1)[0].strip('* ').strip()
    else:
        return None

def extract_property_location(text):
    match = re.search(r'Property Location[:*]*\s*(.*)', text, re.IGNORECASE)
    if match:
        return match.group(1).split('\n', 1)[0].strip('* ').strip()
    else:
        return None
    
def extract_ratings(text):
    ratings_section = re.search(r'Ratings[:*]*\s*(.*?)(\n\n|$)', text, re.IGNORECASE | re.DOTALL)
    if ratings_section:
        ratings_text = ratings_section.group(1).strip()
        if 'Not specified' in ratings_text or 'Not provided' in ratings_text:
            return None
        else:
            # Extract individual ratings
            ratings = {}
            lines = ratings_text.split('\n')
            for line in lines:
                rating_match = re.search(r'-\s*(.*?):\s*([\d.]+)\s*out of 5', line)
                if rating_match:
                    category = rating_match.group(1).strip()
                    score = float(rating_match.group(2))
                    ratings[category] = score
                else:
                    # Overall rating
                    overall_match = re.search(r'([\d.]+)\s*out of 5', line)
                    if overall_match:
                        ratings['Overall'] = float(overall_match.group(1))
            return ratings if ratings else None
    else:
        return None
def extract_amenities(text):
    amenities_section = re.search(r'Amenities[:*]*\s*(.*?)(\n\n|$)', text, re.IGNORECASE | re.DOTALL)
    if amenities_section:
        amenities_text = amenities_section.group(1).strip()
        if 'Not specified' in amenities_text or 'Not provided' in amenities_text:
            return None
        else:
            # Extract amenities as a list
            amenities = re.findall(r'-\s*(.*)', amenities_text)
            return amenities if amenities else None
    else:
        return None



def process_and_cache_data(location, results):
    """
    Process and cache the scraped data for a location.
    Args:
        location (str): The search location
        results (list): List of scraped listing results
    Returns:
        pd.DataFrame: Processed data as a DataFrame
    """
    processed_data = []
    for item in results:
        data_text = item['data']
        processed_data.append({
            'Link': item['link'],
            'Price': item['price'],
            'PropertyName': extract_property_name(data_text),
            'PropertyLocation': extract_property_location(data_text),
            'Ratings': extract_ratings(data_text),
            'Amenities': extract_amenities(data_text)
        })
    
    return pd.DataFrame(processed_data)   



# Modified main function with explicit data caching
def main():
    st.title("Airbnb Listings Scraper")
    
    location = st.text_input("Enter location to search:", "Riyadh, Saudi Arabia")
    formatted_location = format_location_key(location)
    
    cached_df = None
    if st.button("Start Scraping"):
        try:
            # First, try to get cached results
            try:
                redis_client = get_redis_client()
                cache_key = f"airbnb_data:{formatted_location}"
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    df = pd.read_json(cached_data)
                    st.success("Retrieved data from cache!")
                    st.subheader("Aggregated Data (Cached)")
                    st.dataframe(df)
                    cached_df = df
            except Exception as e:
                cached_df = None
            
            # If no cached data, proceed with scraping
            if cached_df is None:
                with st.spinner('Finding pages to scrape...'):
                    search_location = location.replace(' ', '--').replace(',', '')
                    total_pages, page_urls = get_total_pages_and_urls(search_location)
                    st.write(f"Found {total_pages} pages to scrape")
                
                all_listings = []
                with st.spinner('Collecting listings from all pages...'):
                    progress_bar = st.progress(0)
                    for idx, page_url in enumerate(page_urls):
                        st.write(f"Scraping page {idx + 1}/{len(page_urls)}")
                        listings = get_listing_links_for_page(page_url)
                        all_listings.extend(listings)
                        progress_bar.progress((idx + 1) / len(page_urls))
                
                st.write(f"Total listings found: {len(all_listings)}")
                
                results = []
                with st.spinner('Processing individual listings...'):
                    progress_bar = st.progress(0)
                    num_processes = min(cpu_count(), 4)
                    
                    with Pool(processes=num_processes) as pool:
                        for idx, result in enumerate(pool.imap_unordered(process_listing_wrapper, all_listings)):
                            if result:
                                results.append(result)
                            progress_bar.progress((idx + 1) / len(all_listings))
                
                if results:
                    # Process and cache the data
                    df = process_and_cache_data(formatted_location, results)
                   
                    st.subheader("Aggregated Data (Fresh)")
                    st.dataframe(df)
                    redis_client.set(cache_key,  df.to_json(), ex=86400)
                    st.success(f"Data cached for location: {location}")
                    
                    # Save with timestamp
                    # timestamp = time.strftime("%Y%m%d-%H%M%S")
                    # filename = f'airbnb_listings_{formatted_location}_{timestamp}.csv'
                    # df.to_csv(filename, index=False)
                    # st.success(f"Data saved to {filename}")
                    
                    # Show cache info
                    cache_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.info(f"Data cached at: {cache_time}")
                else:
                    st.error("No listings were successfully processed")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            raise

# Add cache management functions
def get_cache_info():
    """Get information about the current cache state."""
    try:
        cache_entries = process_and_cache_data.cache_info()
        return {
            "hits": cache_entries.hits,
            "misses": cache_entries.misses,
            "size": cache_entries.currsize,
            "max_size": cache_entries.maxsize
        }
    except Exception as e:
        return {"error": str(e)}

def clear_cache_for_location(location):
    redis_client = get_redis_client()
    cache_key = f"airbnb_data:{format_location_key(location)}"
    redis_client.delete(cache_key)

# def show_cache_management():
#     with st.sidebar:
#         st.header("Cache Management")
#         location_to_clear = st.text_input("Location to clear cache for:")
#         if st.button("Clear Cache for Location"):
#             clear_cache_for_location(location_to_clear)
    
#             st.success(f"Cache cleared for location: {location_to_clear}")
def show_cache_management():
    redis_client = get_redis_client()
    with st.sidebar:
        st.header("Cache Management")
        
        # Retrieve all keys related to cached data
        keys = redis_client.keys('airbnb_data:*')
        cached_locations = []
        for key in keys:
            location_key = key.decode('utf-8')
            location = location_key.split(':')[1]
            # Get remaining TTL
            ttl = redis_client.ttl(key)
            expiration_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + ttl))
            cached_locations.append((location, expiration_time))
        
        if cached_locations:
            st.subheader("Cached Locations")
            for loc, exp_time in cached_locations:
                display_loc = loc.replace('--', ' ').title()
                st.write(f"- **{display_loc}** (expires at {exp_time})")
        else:
            st.write("No data is currently cached.")
        
        location_to_clear = st.text_input("Location to clear cache for:")
        if st.button("Clear Cache for Location"):
            clear_cache_for_location(location_to_clear)
            st.success(f"Cache cleared for location: {location_to_clear}")

     
if __name__ == '__main__':
    show_cache_management()
    main()