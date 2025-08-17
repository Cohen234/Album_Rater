import json
import os

BLOCKLIST_FILE = "album_blocklist.json"

def load_blocklist():
    if not os.path.exists(BLOCKLIST_FILE):
        return {}
    with open(BLOCKLIST_FILE, "r") as f:
        return json.load(f)

def save_blocklist(blocklist):
    with open(BLOCKLIST_FILE, "w") as f:
        json.dump(blocklist, f)

def load_blocklist_for_artist(artist_name):
    blocklist = load_blocklist()
    return set(blocklist.get(artist_name.lower(), []))

def add_to_blocklist(artist_name, album_id):
    blocklist = load_blocklist()
    artist_key = artist_name.lower()
    if artist_key not in blocklist:
        blocklist[artist_key] = []
    if album_id not in blocklist[artist_key]:
        blocklist[artist_key].append(album_id)
    save_blocklist(blocklist)

def get_visible_studio_albums_for_artist(artist_name, spotify_client, blocklist_loader, deduplicate_fn,
                                         is_live_album_fn):
    from spotify_logic import get_albums_by_artist
    # 1. Get all albums for artist from Spotify
    albums_from_spotify = get_albums_by_artist(spotify_client, artist_name)

    # 2. Fetch tracks and filter out live albums
    filtered_albums = []
    for album_data in albums_from_spotify:
        album_id_spotify = album_data.get("id")
        try:
            tracks = spotify_client.album_tracks(album_id_spotify)['items']
            if is_live_album_fn(tracks):
                continue  # Skip "live" albums
            album_data['tracks'] = tracks
            filtered_albums.append(album_data)
        except Exception as e:
            # Could not fetch tracks, skip this album
            continue

    # 3. Deduplicate and filter out compilations, deluxe, etc.
    studio_albums = deduplicate_fn(filtered_albums)

    # 4. Apply blocklist
    blocklist = blocklist_loader(artist_name)
    visible_albums = [a for a in studio_albums if a['id'] not in blocklist]

    return visible_albums