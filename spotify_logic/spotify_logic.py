import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Set up your Spotify credentials here (use environment variables or config in production)
SPOTIPY_CLIENT_ID = 'your_spotify_client_id'
SPOTIPY_CLIENT_SECRET = 'your_spotify_client_secret'

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

def load_album_data(sp_param, album_id): # Changed 'sp' to 'sp_param' for clarity, or just keep it 'sp'
    # Use the parameter passed to the function, not a different variable name
    album = sp_param.album(album_id) # <-- Use the parameter name
    tracks = sp_param.album_tracks(album_id) # <-- Use the parameter name

    album_name = album['name']
    artist_name = album['artists'][0]['name']
    album_cover_url = album['images'][0]['url'] if album['images'] else ""
    album_url = album['external_urls'].get('spotify', '')

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
        'url': album_url,
        'songs': songs
    }