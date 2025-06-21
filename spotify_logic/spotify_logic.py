import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re
def extract_album_id(url_or_uri):
    """
    Extracts the album ID from a full Spotify album URL or URI.
    """
    match = re.search(r"(album[:/])([a-zA-Z0-9]+)", url_or_uri)
    return match.group(2) if match else url_or_uri.strip()

# Keep 'sp_param' here, as it's the parameter you're expecting from the caller (app.py)
