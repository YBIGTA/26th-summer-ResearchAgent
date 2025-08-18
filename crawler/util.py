import base64
import requests
import logging

logger = logging.getLogger(__name__)

def get_raw_data(url):
    """Fetches a file from a URL and returns it as a Base64 encoded string."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        encoded_string = base64.b64encode(response.content).decode('utf-8')
        # Get content type to form the data URI
        content_type = response.headers.get('Content-Type', 'application/octet-stream')
        return f"data:{content_type};base64,{encoded_string}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching {url}: {e}")
        return None