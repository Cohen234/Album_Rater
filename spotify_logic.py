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


# In spotify_logic.py
import re  # Make sure 're' is imported at the top of your file


def get_albums_by_artist(artist_name):
    """
    Fetches all studio albums for a given artist, using a regex to filter out unwanted variants.
    Keeps Deluxe, Extended, Edition, Bonus, Super Deluxe, etc.
    Removes Remaster, Live, Instrumental, Acoustic, Clean, Explicit, Karaoke, Demo, etc.
    """
    results = sp.search(q=f"artist:{artist_name}", type="artist", limit=1)
    items = results.get('artists', {}).get('items', [])
    if not items:
        return []
    artist_id = items[0]['id']

    # Exclude only these types of albums (case insensitive, at end of name or in parenthesis/brackets)
    exclude_keywords = [
        r'remaster', r'remastered', r'live', r'instrumental', r'acoustic',
        r'karaoke', r'clean', r'explicit', r'commentary', r'demo',
        r'edit', r'edit version', r'b-side', r'remix', r'single',
        r'cover', r'performance', r'voice memo', r'voice notes'
    ]
    # Build a pattern that matches (Remastered), [Live], (Instrumental), etc at end
    pattern = re.compile(
        r'[\(\[\{][^)\]\}]*(' + '|'.join(exclude_keywords) + r')[^)\]\}]*[\)\]\}]$',
        re.IGNORECASE
    )
    # Also exclude if keyword appears at end (not in parenthesis)
    pattern2 = re.compile(
        r'(' + '|'.join(exclude_keywords) + r')\s*$', re.IGNORECASE
    )

    all_api_albums = []
    offset = 0
    while True:
        response = sp.artist_albums(artist_id, album_type='album', limit=50, offset=offset)
        if not response['items']:
            break
        all_api_albums.extend(response['items'])
        offset += 50

    album_list = []
    for album in all_api_albums:
        name = album['name']
        # Filter out only if the pattern matches
        if not pattern.search(name) and not pattern2.search(name):
            album_list.append({
                'name': name,
                'id': album['id'],
                'url': album['external_urls']['spotify'],
                'image': album['images'][0]['url'] if album['images'] else ""
            })

    # De-duplication logic: allow multiple editions (deluxe, extended, bonus, etc) to remain!
    # Only dedupe exact matches (case-insensitive)
    unique_albums = []
    seen_ids = set()
    for album in album_list:
        if album['id'] not in seen_ids:
            unique_albums.append(album)
            seen_ids.add(album['id'])

    return unique_albums