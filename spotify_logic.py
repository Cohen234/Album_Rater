import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re

# Set up your Spotify credentials here (use environment variables or config in production)
SPOTIPY_CLIENT_ID = "5979ecb4cfa040f2a9c7ff06e819f240"
SPOTIPY_CLIENT_SECRET = "710ef028cbf141eb9cbb7d72dd31a373"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET
))

def extract_album_id(url_or_uri):
    """
    Extracts the album ID from a full Spotify album URL or URI.
    """
    match = re.search(r"(album[:/])([a-zA-Z0-9]+)", url_or_uri)
    return match.group(2) if match else url_or_uri.strip()

def load_album_data(spotify_url):
    album_id = extract_album_id(spotify_url)
    album = sp.album(album_id)
    tracks = sp.album_tracks(album_id)

    album_name = album['name']
    artist_name = album['artists'][0]['name']
    album_cover_url = album['images'][0]['url'] if album['images'] else ""

    songs = []
    for item in tracks['items']:
        song_name = item['name']
        song_id = item['id']
        songs.append({
            'song_name': song_name,
            'song_id': song_id
        })

    return {
        'album_name': album_name,
        'artist_name': artist_name,
        'album_cover_url': album_cover_url,
        'songs': songs
    }

def get_albums_by_artist(artist_name):
    results = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
    items = results.get('artists', {}).get('items', [])
    if not items:
        return []

    artist_id = items[0]['id']

    # THE FIX: Expanded list of keywords to catch more live albums
    live_keywords = ['(live', 'live at', 'live from', 'unplugged', 'sessions']

    all_api_albums = []
    offset = 0
    # Paginate through all results from Spotify
    while True:
        # THE FIX: Set album_type='album' to exclude singles
        response = sp.artist_albums(artist_id, album_type='album', limit=50, offset=offset)

        if not response['items']:
            break

        all_api_albums.extend(response['items'])
        offset += 50

    # Process and filter the collected albums
    album_list = []
    for album in all_api_albums:
        album_name_lower = album['name'].lower()

        # Check if any live keyword is in the album title
        is_live_album = any(keyword in album_name_lower for keyword in live_keywords)

        if not is_live_album:
            album_list.append({
                'name': album['name'],
                'id': album['id'],
                'url': album['external_urls']['spotify'],
                'image': album['images'][0]['url'] if album['images'] else ""
            })

    # Remove duplicates based on a cleaned name (e.g., "Donda" and "Donda (Deluxe)" are treated separately)
    # This keeps the earliest released version if names are identical after cleaning
    unique_albums = []
    seen_names = set()
    # Spotify returns newest first, so we reverse to process oldest first
    for album in reversed(album_list):
        # Clean the name by removing content in parentheses for better de-duplication
        cleaned_name = re.sub(r'\s*\([^)]*\)$', '', album['name']).strip()
        if cleaned_name.lower() not in seen_names:
            unique_albums.append(album)
            seen_names.add(cleaned_name.lower())

    # Return the unique list, reversed again to show newest first
    return unique_albums[::-1]

