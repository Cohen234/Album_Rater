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
def load_album_data(sp_param, album_id):
    # Use the parameter passed to the function
    album = sp_param.album(album_id)
    tracks = sp_param.album_tracks(album_id)

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