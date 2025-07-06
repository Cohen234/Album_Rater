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
    results = sp.search(q=f'artist:{artist_name}', type='artist', limit=1)
    if not results['artists']['items']:
        return []

    artist_id = results['artists']['items'][0]['id']

    albums = []
    offset = 0
    limit = 50

    # Define keywords that typically indicate a live album
    live_keywords = ['live at', 'live from', 'unplugged', 'sessions', 'live in']

    while True:
        response = sp.artist_albums(artist_id, album_type='album,single', limit=limit, offset=offset)
        if not response['items']:
            break

        for album in response['items']:
            album_name_lower = album['name'].lower()

            # THE FIX: Check if any live keyword is in the album title
            is_live_album = any(keyword in album_name_lower for keyword in live_keywords)

            # Only add the album if it's NOT a live album
            if not is_live_album:
                albums.append({
                    'id': album['id'],
                    'name': album['name'],
                    'image': album['images'][0]['url'] if album['images'] else "",
                    'url': album['external_urls'].get('spotify', '')
                })

        offset += limit

    # Remove duplicate albums based on name
    unique_albums = []
    seen_names = set()
    for album in albums:
        if album['name'] not in seen_names:
            unique_albums.append(album)
            seen_names.add(album['name'])

    return unique_albums

