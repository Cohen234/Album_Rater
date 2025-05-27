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
    albums = sp.artist_albums(artist_id=artist_id, album_type='album', limit=50)
    sorted_albums = sorted(albums['items'], key=lambda x: x.get('release_date', '9999-99-99'))
    album_set = set()
    album_list = []

    for album in sorted_albums:
        name = album['name']
        if name not in album_set:
            album_set.add(name)
            album_list.append({
                'name': name,
                'id': album['id'],
                'url': album['external_urls']['spotify'],
                'image': album['images'][0]['url'] if album['images'] else ""
            })
    return album_list