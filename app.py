from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.oauth2.service_account import Credentials
from datetime import datetime
import pandas as pd
from colorthief import ColorThief
import requests
from io import BytesIO
import os
import json # Only need to import json once
from collections import Counter
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import traceback # Already imported, keeping it explicit for clarity
import re # Added for extract_album_id if it's not exclusively in spotify_logic

from dotenv import load_dotenv # Load dotenv as early as possible
load_dotenv()
import logging

# Import functions from spotify_logic after all core imports
from spotify_logic import get_albums_by_artist, extract_album_id
import os
import psycopg2

def get_db_connection():
    return psycopg2.connect(
        os.environ["SUPABASE_DATABASE_URL"],
        sslmode="require"
    )


# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_temporary_dev_key')

# --- Spotify API Initialization ---
# Use consistent variable names for Spotify Client ID/Secret
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID') # Changed from SPOTIPY_CLIENT_ID_APP
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET') # Changed from SPOTIPY_CLIENT_SECRET_APP

sp = None # Initialize sp to None
if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    try:
        # THE FIX: Add a 'retries' parameter to automatically retry failed connections.
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ), retries=3, status_forcelist=[429, 500, 502, 503, 504])
        print("DEBUG: Spotify client (sp) initialized successfully in app.py.")
    except Exception as e:
        print(f"ERROR: Failed to initialize Spotify client (sp) in app.py: {e}")
        print("Ensure SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET are correctly set.")
        sp = None # Ensure sp is None if initialization fails
else:
    print("WARNING: SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET environment variables not found for app.py.")
    print("Spotify functionality may be limited.")

# --- Helper Functions (move these up here if they are used globally) ---
# Your group_ranked_songs, get_dominant_color, get_album_stats functions should
# come after the sp and client initialization if they use them, but before routes.
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
def calculate_score_value(position, total_songs, rank_group_val):
    """
    Calculates a precise score for a song where the rank group is the floor.
    """
    try:
        rank_group_val = float(rank_group_val)
    except (ValueError, TypeError):
        return 0.0

    if total_songs <= 1:
        return rank_group_val

    score_spread = 0.49
    highest_score = rank_group_val + score_spread
    # Avoid division by zero if there's only one song
    step = score_spread / (total_songs - 1) if total_songs > 1 else 0
    new_score = highest_score - (step * position)

    # Return with high precision to prevent score collisions
    return round(new_score, 6)

def get_album_averages_df():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM "Re-Ranking and Song History (Album Averages)";')
    albums_data = cursor.fetchall()
    albums_df = pd.DataFrame(albums_data, columns=[desc[0] for desc in cursor.description])

    # Perform DataFrame cleanup
    expected_cols = ['album_id', 'album_name', 'artist_name', 'average_score', 'weighted_average_score',
                     'original_weighted_score', 'previous_weighted_score', 'times_ranked',
                     'last_ranked_date', 'rerank_history', 'score_history', 'album_cover_url']

    for col in expected_cols:
        if col not in albums_df.columns:
            albums_df[col] = pd.NA if col not in ['rerank_history', 'score_history'] else '[]'

    albums_df['times_ranked'] = pd.to_numeric(albums_df['times_ranked'], errors='coerce').fillna(0).astype(int)
    albums_df = albums_df.fillna({
        'rerank_history': '[]',
        'score_history': '[]',
        'album_cover_url': ''
    })

    return albums_df
def group_ranked_songs(sheet_rows):
    group_bins = {round(x * 0.5, 1): [] for x in range(2, 21)}  # 1.0 to 10.0
    for row in sheet_rows:
        try:
            rank = float(row["Ranking"])
            group = round(rank * 2) / 2  # Ensure .5 steps
            group = min(max(group, 1.0), 10.0)  # Clamp between 1.0 and 10.0
            group_bins[group].append({
                "artist": row["Artist Name"],
                "title": row["Song Name"],
                "rank": rank,
                "date": row["Ranked Date"],
                "position": row.get("Position In Group", None)
            })
        except Exception as e: # Catch specific exceptions or general Exception
            print(f"WARNING: Skipping bad row in group_ranked_songs: {row} - {e}")
            continue
    return group_bins
def get_album_release_dates(sp_instance, album_ids):
    """Fetches release dates for a list of album IDs robustly."""
    release_dates = {}
    if not album_ids:
        return release_dates

    album_ids = [str(aid) for aid in album_ids]
    for i in range(0, len(album_ids), 20):
        batch = album_ids[i:i + 20]
        try:
            albums_info = sp_instance.albums(batch)
            for album in albums_info['albums']:
                if album:
                    release_dates[album['id']] = album.get('release_date')
        except Exception as e:
            logging.error(f"Could not fetch album release dates batch {batch}: {e}")
            # Optionally, retry once after a short delay
            import time
            time.sleep(2)
            try:
                albums_info = sp_instance.albums(batch)
                for album in albums_info['albums']:
                    if album:
                        release_dates[album['id']] = album.get('release_date')
            except Exception as e2:
                logging.error(f"Second attempt failed for batch {batch}: {e2}")
                # Optionally, set release date to None for these IDs
                for aid in batch:
                    release_dates[aid] = None

    return release_dates
from collections import defaultdict
from gspread.exceptions import APIError
import numpy as np
@app.route("/")
@app.route("/profile")
def profile_page():
    user_name = "Cohen Callaway"
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Load main song/album dataframes
        cursor.execute('SELECT * FROM "Current Positions";')
        songs_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description]).fillna("")

        albums_df = get_album_averages_df()

        cursor.execute('SELECT * FROM "Preliminary Ranks";')
        prelim_df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description]).fillna("")
    except psycopg2.Error as e:
        logging.error(f"Database error occurred: {e}")
        return print("Error")

    # --- Standardization ---
    def std_cols(df):
        df.columns = [c.replace(' ', '_').lower() for c in df.columns]
        return df

    songs_df = std_cols(songs_df)
    albums_df = std_cols(albums_df)
    prelim_df = std_cols(prelim_df)

    # Ensure numeric for scores/dates
    songs_df['ranking'] = pd.to_numeric(songs_df['ranking'], errors='coerce')
    songs_df['ranked_date'] = pd.to_datetime(songs_df['ranked_date'], errors='coerce')
    albums_df['weighted_average_score'] = pd.to_numeric(albums_df['weighted_average_score'], errors='coerce')
    albums_df['last_ranked_date'] = pd.to_datetime(albums_df['last_ranked_date'], errors='coerce')

    # --- Totals ---
    albums_ranked = albums_df[albums_df['times_ranked'] > 0]
    if 'ranking_status' in songs_df.columns:
        songs_ranked = songs_df[songs_df['ranking_status'].str.lower() == 'final'].copy()
    else:
        songs_ranked = songs_df.copy()

    # ... previous code ...
    song_scores = songs_ranked['ranking'].dropna().values
    mean_song = np.mean(song_scores) if len(song_scores) else 0
    std_song = np.std(song_scores) if len(song_scores) else 1

    bins = [round(x * 0.5, 1) for x in range(2, 22)]  # 1.0 ... 10.0 inclusive
    songs_ranked.loc[:, 'score_bin'] = pd.cut(
        songs_ranked['ranking'],
        bins=[0] + bins,
        labels=[str(b) for b in bins],
        include_lowest=True
    )

    songs_ranked.loc[:, 'standardized_score'] = (songs_ranked['ranking'] - mean_song) / std_song
    # ... rest of your code ...
    artists_ranked = songs_ranked['artist_name'].str.strip().str.lower().nunique()

    num_albums = len(albums_ranked)
    num_songs = len(songs_ranked)
    num_artists = artists_ranked

    # --- Last Ranked ---
    last_album = albums_ranked.sort_values('last_ranked_date', ascending=False).head(1)
    last_album_info = None
    if not last_album.empty:
        last_album_row = last_album.iloc[0]
        last_album_info = {
            "name": last_album_row['album_name'],
            "score": last_album_row['weighted_average_score'],
            "date": last_album_row['last_ranked_date'].strftime("%b %d, %Y") if pd.notnull(last_album_row['last_ranked_date']) else ""
        }
    last_song = songs_ranked.sort_values('ranked_date', ascending=False).head(1)
    last_song_info = None
    if not last_song.empty:
        last_song_row = last_song.iloc[0]
        last_song_info = {
            "name": last_song_row['song_name'],
            "score": last_song_row['ranking'],
            "date": last_song_row['ranked_date'].strftime("%b %d, %Y") if pd.notnull(last_song_row['ranked_date']) else ""
        }

    # --- Recently Ranked Artists ---
    recent_artists = songs_ranked.sort_values('ranked_date', ascending=False)['artist_name'].drop_duplicates().head(8).tolist()

    # --- Paused Albums (Preliminary) ---
    if not prelim_df.empty and 'album_name' in prelim_df.columns and 'ranking_status' in prelim_df.columns:
        paused_albums = (
            prelim_df[prelim_df['ranking_status'].str.lower() == 'preliminary']
            .sort_values('ranked_date', ascending=False)
            .drop_duplicates(['album_name'])
        )[['artist_name', 'album_name', 'ranked_date']].head(8).to_dict('records')
    else:
        paused_albums = []

    # --- Averages, Medians, Stdevs ---
    avg_album_score = albums_ranked['weighted_average_score'].mean() if not albums_ranked.empty else 0
    avg_song_score = songs_ranked['ranking'].mean() if not songs_ranked.empty else 0
    median_album_score = albums_ranked['weighted_average_score'].median() if not albums_ranked.empty else 0
    median_song_score = songs_ranked['ranking'].median() if not songs_ranked.empty else 0
    std_album_score = albums_ranked['weighted_average_score'].std() if not albums_ranked.empty else 0
    std_song_score = songs_ranked['ranking'].std() if not songs_ranked.empty else 0



    # --- Favorite Albums by Decade (Top 3 per decade) ---
    if 'release_date' not in albums_df.columns or albums_df['release_date'].isnull().all():
        if 'album_id' in albums_df.columns:
            album_ids = albums_df['album_id'].dropna().unique().tolist()
            release_dates_map = get_album_release_dates(sp, album_ids)
            albums_df['release_date'] = albums_df['album_id'].map(release_dates_map)
        else:
            albums_df['release_date'] = pd.NaT

    # Always do this after filling the column, to ensure it's really datetime
    albums_df['release_date'] = pd.to_datetime(albums_df['release_date'], errors='coerce')

    # Now you can safely use .dt
    albums_df['release_year'] = albums_df['release_date'].dt.year
    decade_bins = list(range(1960, datetime.now().year + 10, 10))
    albums_df['decade'] = pd.cut(albums_df['release_year'], bins=decade_bins, right=False, labels=[f"{y}s" for y in decade_bins[:-1]])
    favorite_by_decade = defaultdict(list)
    for decade, group in albums_df.groupby('decade', observed=False):
        favs = group.sort_values('weighted_average_score', ascending=False).head(3)
        for _, row in favs.iterrows():
            favorite_by_decade[decade].append({
                "album_name": row['album_name'],
                "artist_name": row['artist_name'],
                "score": row['weighted_average_score'],
                "release_year": row['release_year']
            })

    # --- Decade Stats Table ---
    decade_stats = []
    for decade, group in albums_df.groupby('decade', observed=False):
        if not group.empty:
            decade_stats.append({
                "decade": decade,
                "avg_album_score": group['weighted_average_score'].mean(),
                "median_album_score": group['weighted_average_score'].median(),
                "std_album_score": group['weighted_average_score'].std(),
                "album_count": len(group)
            })

    # --- First Album Ranked ---
    if not albums_ranked.empty and 'last_ranked_date' in albums_ranked.columns:
        first_album_row = albums_ranked.sort_values('last_ranked_date').iloc[0]
        first_album_ranked = {
            "name": first_album_row['album_name'],
            "date": first_album_row['last_ranked_date'].strftime("%b %d, %Y") if pd.notnull(first_album_row['last_ranked_date']) else ""
        }
    else:
        first_album_ranked = None

    # --- Ranking Distribution Polar Chart (Songs) ---
    bins = [round(x * 0.5, 1) for x in range(2, 22)]  # 1.0 ... 10.0 inclusive
    songs_ranked.loc[:, 'score_bin'] = pd.cut(
        songs_ranked['ranking'],
        bins=[0] + bins,
        labels=[str(b) for b in bins],
        include_lowest=True
    )

    songs_ranked.loc[:, 'standardized_score'] = (songs_ranked['ranking'] - mean_song) / std_song
    rank_dist = songs_ranked['score_bin'].value_counts().sort_index()
    polar_chart_data = {
        "labels": [str(label) for label in rank_dist.index],
        "data": [int(x) for x in rank_dist.values]
    }

    # --- Timeline Data (Albums Ranked Chronologically) ---
    timeline_albums = albums_ranked.sort_values('last_ranked_date')
    timeline_events = []
    for _, row in timeline_albums.iterrows():
        timeline_events.append({
            "album_name": row['album_name'],
            "artist_name": row['artist_name'],
            "score": row['weighted_average_score'],
            "date": row['last_ranked_date'].strftime("%b %d, %Y") if pd.notnull(row['last_ranked_date']) else "",
            "album_cover_url": row.get('album_cover_url', '')
        })

    if 'release_date' in albums_ranked.columns:
        era_albums = albums_ranked[albums_ranked['release_date'].notnull()]
    else:
        era_albums = pd.DataFrame(columns=albums_ranked.columns)

    era_chart_albums = []
    if not era_albums.empty and 'release_date' in era_albums.columns:
        era_chart_albums = [{
            "x": row['release_date'].strftime("%Y-%m-%d"),
            "y": row['weighted_average_score'],
            "label": row['album_name'],
            "artist": row['artist_name'],
            "cover": row.get('album_cover_url', '')
        } for _, row in era_albums.iterrows() if pd.notnull(row['release_date'])]

        era_albums['release_year'] = pd.to_datetime(era_albums['release_date'], errors='coerce').dt.year
        era_yearly = era_albums.groupby('release_year').agg(
            avg_score=('weighted_average_score', 'mean'),
            std_score=('weighted_average_score', 'std'),
            count=('weighted_average_score', 'count'),
        ).reset_index()
        era_chart_years = [{
            "x": f"{int(row['release_year'])}-07-01",
            "y": row['avg_score'],
            "sem": row['std_score'] if not np.isnan(row['std_score']) else 0,
            "n": int(row['count']),
            "year": int(row['release_year'])
        } for _, row in era_yearly.iterrows()]
    else:
        era_chart_albums = []
        era_chart_years = []

    # --- Search/Autocomplete Data ---
    ranked_artists = sorted(songs_ranked['artist_name'].dropna().unique())
    ranked_albums = sorted(albums_ranked['album_name'].dropna().unique())

    # --- Standardized Song Scores ---
    # Assuming albums_ranked is a DataFrame of albums with album_name and album_cover_url
    album_cover_map = dict(zip(albums_ranked['album_name'], albums_ranked['album_cover_url']))

    standardized_songs = (
        songs_ranked[['song_name', 'artist_name', 'ranking', 'standardized_score', 'album_name']]
        .sort_values('standardized_score', ascending=False)
        .head(10)
        .assign(album_cover_url=lambda df: df['album_name'].map(album_cover_map))
        .to_dict('records')
    )
    return render_template(
        "profile.html",
        user_name=user_name,
        num_albums=num_albums,
        num_songs=num_songs,
        num_artists=num_artists,
        last_album_info=last_album_info,
        last_song_info=last_song_info,
        recent_artists=recent_artists,
        paused_albums=paused_albums,
        avg_album_score=avg_album_score,
        avg_song_score=avg_song_score,
        median_album_score=median_album_score,
        median_song_score=median_song_score,
        std_album_score=std_album_score,
        std_song_score=std_song_score,
        favorite_by_decade=favorite_by_decade,
        decade_stats=decade_stats,
        first_album_ranked=first_album_ranked,
        polar_chart_data=polar_chart_data,
        timeline_events=timeline_events,
        era_chart_albums=era_chart_albums,
        era_chart_years=era_chart_years,
        ranked_artists=ranked_artists,
        ranked_albums=ranked_albums,
        standardized_songs=standardized_songs
    )

from PIL import Image, UnidentifiedImageError
def get_dominant_color(image_url):
    try:
        # Fetch the image with a timeout
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Raise HTTP errors if any

        # Load the image data into a buffer
        image_data = BytesIO(response.content)

        # Try to extract the dominant color using ColorThief
        color_thief = ColorThief(image_data)
        rgb = color_thief.get_color(quality=1)  # Might raise an error for unsupported images
        return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch album image from {image_url}. Error: {e}")
        return "#000000"  # Default to black if download fails

    except UnidentifiedImageError as e:
        logging.error(f"Invalid image data fetched from {image_url}. Error: {e}")
        return "#000000"  # Default to black if image is invalid

    except Exception as e:
        logging.error(f"Unexpected error in get_dominant_color for {image_url}: {e}")
        return "#000000"  # Default to black for all other errors
@app.route("/api/find_album")
def api_find_album():
    album_name = request.args.get("album_name", "").strip().lower()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM "Re-Ranking and Song History (Album Averages)";')
    album_data = cursor.fetchall()
    cursor.execute('SELECT * FROM "Re-Ranking and Song History (Album Averages)" WHERE LOWER(album_name) = %s;', (album_name,))
    row = cursor.fetchall()
    if row:
        r = row.iloc[0]
        url = url_for("album_page", artist_name=r['artist_name'], album_name=quote_plus(r['album_name']), album_id=r['album_id'])
        return jsonify({"found": True, "url": url})
    return jsonify({"found": False})
def get_ordinal_suffix(n):
    """Converts a number to its ordinal form (e.g., 1 -> 1st, 2 -> 2nd)."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{'st' if n % 10 == 1 else 'nd' if n % 10 == 2 else 'rd' if n % 10 == 3 else 'th'}"


def calculate_streak(score_history):
    """Calculates a streak based on the complete score history."""
    # Need at least 4 ranking events to have 3 consecutive changes
    if len(score_history) < 4:
        return 'none'

    # Get the last 3 score changes from the history
    last_three_changes = [score_history[i] - score_history[i - 1] for i in
                          range(len(score_history) - 3, len(score_history))]

    is_hot_streak = all(change >= 0 for change in last_three_changes) and any(
        change > 0 for change in last_three_changes)
    is_cold_streak = all(change <= 0 for change in last_three_changes) and any(
        change < 0 for change in last_three_changes)

    if is_hot_streak:
        return 'hot_streak'
    if is_cold_streak:
        return 'cold_streak'

    return 'none'
def merge_album_with_rankings(album_tracks, sheet_rows, artist_name):
    merged_tracks = []
    for track in album_tracks:
        # Handle both string and dict cases
        track_name = track if isinstance(track, str) else track.get("song_name", "")
        tn_lower = track_name.strip().lower()
        artist_lower = artist_name.strip().lower()

        matches = [
            row for row in sheet_rows
            if row['Artist Name'].strip().lower() == artist_lower and
               row['Song Name'].strip().lower() == tn_lower
        ]

        if matches:
            rank_count = len(matches)
            avg_rank = round(sum(float(r['Ranking']) for r in matches) / rank_count, 2)
            latest_rank_date = max(r['Ranked Date'] for r in matches)
            prelim_rank = matches[-1]['Ranking']
        else:
            rank_count = 0
            avg_rank = None
            latest_rank_date = None
            prelim_rank = ""

        merged_tracks.append({
            "song_name": track_name,
            "rank_count": rank_count,
            "avg_rank": avg_rank,
            "latest_rank_date": latest_rank_date,
            "prelim_rank": prelim_rank
        })

    return merged_tracks

def load_google_sheet_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    # This function uses `client`, so it must be defined after `client` is initialized
    cursor.execute('SELECT * FROM "Current Positions";')
    song_data = cursor.fetchall()
    return pd.DataFrame(song_data, columns=[desc[0] for desc in cursor.description])


def clean_title(title):
    """
    Clean a song or album title for display:
    - Remove years (e.g. 2009, 2015), 'Remastered', 'Mix', 'Edition', etc.
    - Remove anything in parentheses.
    - Remove dashes and descriptors at the end.
    """
    # Remove anything in parentheses, e.g. (Remastered 2009), (Deluxe Edition)
    title = re.sub(r'\s*\([^)]*\)', '', title)
    # Remove dashes and common descriptors at end of string (and years), e.g. " - 2009 Mix", " - Remastered 2015"
    title = re.sub(r'\s*-\s*(Remastered|Remaster(ed)?|[0-9]{4} Mix|Mix|Extended Edition|Bonus Track|Deluxe Edition|Mono Version|Stereo Version|Edit|Version|Live|Single Version|From [^,\.]*|[0-9]{4})\s*$', '', title, flags=re.IGNORECASE)
    # Remove "Remastered YYYY" or "YYYY Remaster" at end
    title = re.sub(r'\s*Remaster(ed)? ?[0-9]*$', '', title, flags=re.IGNORECASE)
    # Remove years at end or in middle of string
    title = re.sub(r'\s*\b(19|20)\d{2}\b', '', title)
    # Remove extra whitespace and stray dashes
    title = re.sub(r'\s*-\s*$', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title.strip()
# In app.py
from urllib.parse import quote_plus
@app.route("/artist/<string:artist_name>")
def artist_page_v2(artist_name):
    import numpy as np
    import pandas as pd
    import math
    import json
    from jinja2 import Undefined
    from album_blocklist import get_visible_studio_albums_for_artist, load_blocklist_for_artist
    visible_studio_albums = get_visible_studio_albums_for_artist(
        artist_name,
        spotify_client=sp,  # or whatever your Spotify client variable is called
        blocklist_loader=load_blocklist_for_artist,  # your existing blocklist loader function
        deduplicate_fn=deduplicate_by_track_overlap,  # your deduplication function
        is_live_album_fn=is_live_album  # your live album filter function
    )
    visible_album_ids = set(a['id'] for a in visible_studio_albums)


    def standardize_columns(df):
        df.columns = [c.replace(' ', '_') for c in df.columns]
        if 'Rank_Group' not in df.columns and 'Rank_Group' in df.columns:
            pass
        elif 'Rank_Group' not in df.columns and 'Rank_Group' not in df.columns and 'Rank_Group' in df.columns:
            df['Rank_Group'] = df['Rank_Group']
        elif 'Rank_Group' not in df.columns and 'Rank_Group' not in df.columns and 'Rank_Group' not in df.columns and 'Rank Group' in df.columns:
            df['Rank_Group'] = df['Rank Group']
        return df

    def filter_df_by_artist(df, artist_col, artist_name):
        artist_query = artist_name.lower().strip()

        def artist_matcher(x):
            try:
                return artist_query == str(x).strip().lower()
            except Exception:
                return False

        return df[df[artist_col].apply(artist_matcher)].copy()

    try:
        logging.info(f"--- Loading Artist Stats Page for: {artist_name} ---")
        conn = get_db_connection()
        cursor = conn.cursor()
        # 1. --- Load All Base Data ---
        cursor.execute('SELECT * FROM "Current Positions";')
        songs_data = cursor.fetchall()
        all_songs_df = pd.DataFrame(songs_data, columns=[desc[0] for desc in cursor.description]).fillna("")
        cursor.execute('SELECT * FROM "Re-Ranking and Song History (Album Averages)";')
        album_data = cursor.fetchall()
        all_albums_df = pd.DataFrame(album_data, columns=[desc[0] for desc in cursor.description])

        # --- Normalize columns: spaces to underscores everywhere ---
        all_songs_df = standardize_columns(all_songs_df)
        all_albums_df = standardize_columns(all_albums_df)
        for col in ['Song_Name', 'Artist_Name', 'Ranking_Status', 'Album_Name']:
            if col in all_songs_df.columns:
                all_songs_df[col] = all_songs_df[col].astype(str)
        for col in ['album_name', 'artist_name']:
            if col in all_albums_df.columns:
                all_albums_df[col] = all_albums_df[col].astype(str)
        logging.info(f"all_songs_df dtypes:\n{all_songs_df.dtypes}")
        logging.info(f"all_albums_df dtypes:\n{all_albums_df.dtypes}")


        # --- Type conversions ---
        all_songs_df['Ranking'] = pd.to_numeric(all_songs_df['Ranking'], errors='coerce')
        all_albums_df['weighted_average_score'] = pd.to_numeric(all_albums_df['weighted_average_score'],
                                                                errors='coerce')

        # --- Keep only FINAL rows ---
        all_songs_df = all_songs_df[all_songs_df['Ranking_Status'].astype(str).str.lower() == 'final']

        # --- Remove duplicates: keep only the latest by Ranked_Date ---
        all_songs_df = all_songs_df.sort_values('Ranked_Date').drop_duplicates(['Song_Name', 'Artist_Name'],
                                                                               keep='last')

        # --- Filter for valid song/artist/ranking ---
        all_songs_df = all_songs_df[
            (all_songs_df['Song_Name'].str.strip() != "") &
            (all_songs_df['Artist_Name'].str.strip() != "") &
            (all_songs_df['Ranking'].notnull())
            ]

        all_albums_df = all_albums_df[
            (all_albums_df['album_name'].str.strip() != "") &
            (all_albums_df['artist_name'].str.strip() != "") &
            (all_albums_df['weighted_average_score'].notnull())
            ]

        # --- Assign Universal Rank *after* deduplication ---
        all_songs_df = all_songs_df.sort_values(by='Ranking', ascending=False)
        all_songs_df['Universal_Rank'] = range(1, len(all_songs_df) + 1)
        all_albums_df = all_albums_df.sort_values(by='weighted_average_score', ascending=False)
        all_albums_df['Global_Rank'] = range(1, len(all_albums_df) + 1)

        # --- Duplicate check for debugging ---
        all_songs_df['album_name_clean'] = all_songs_df['Album_Name'].astype(str).str.strip().str.lower()
        all_albums_df['album_name_clean'] = all_albums_df['album_name'].astype(str).str.strip().str.lower()
        album_first_ranked = all_songs_df.groupby('album_name_clean')['Ranked_Date'].min()
        album_first_score = all_songs_df.groupby('album_name_clean')['Ranking'].first()
        all_albums_df['first_ranked_date'] = all_albums_df['album_name_clean'].map(album_first_ranked)
        all_albums_df['first_score'] = all_albums_df['album_name_clean'].map(album_first_score)
        all_albums_df['first_ranked_date'] = pd.to_datetime(all_albums_df['first_ranked_date'], errors='coerce')
        all_albums_df = all_albums_df[all_albums_df['first_ranked_date'].notnull()]

        artist_songs_df = filter_df_by_artist(all_songs_df, 'Artist_Name', artist_name)
        artist_albums_df = filter_df_by_artist(all_albums_df, 'artist_name', artist_name)

        if artist_songs_df.empty and artist_albums_df.empty:
            return redirect(url_for('load_albums_by_artist_route', artist_name=artist_name))

        # Standardize columns for artist_songs_df
        artist_songs_df = standardize_columns(artist_songs_df)



        if 'Rank_Group' not in artist_songs_df.columns and 'Rank Group' in artist_songs_df.columns:
            artist_songs_df['Rank_Group'] = artist_songs_df['Rank Group']

        # Filter out interludes
        actual_songs_df = artist_songs_df[artist_songs_df['Rank_Group'] != "I"].copy()

        # Recompute contiguous artist rank AFTER filtering (so no gaps)
        actual_songs_df = actual_songs_df.sort_values(by='Ranking', ascending=False).reset_index(drop=True)
        actual_songs_df['Artist_Rank'] = actual_songs_df.index + 1

        # 2. --- Calculate New Stats ----

        # ARTIST MASTERY
        # After all select-album-screen filtering and blocklist application

        artist_albums_df = artist_albums_df[artist_albums_df['album_id'].isin(visible_album_ids)]

        albums_ranked_at_least_once = artist_albums_df['times_ranked'].fillna(0).astype(int) >= 1
        mastery_points = artist_albums_df['times_ranked'].fillna(0).astype(int).clip(upper=3).sum()
        max_mastery_points = len(visible_album_ids) * 3
        mastery_percentage = (mastery_points / max_mastery_points) * 100 if max_mastery_points > 0 else 0


        # LEADERBOARD POINTS
        total_songs = len(all_songs_df)
        total_albums = len(all_albums_df)
        song_points = artist_songs_df['Universal_Rank'].apply(lambda x: total_songs - x + 1).sum() if not artist_songs_df.empty else 0
        album_points = ((total_albums - artist_albums_df['Global_Rank'] + 1) * 10).sum() if not artist_albums_df.empty else 0
        total_leaderboard_points = song_points + album_points
        ranked_albums_count = (artist_albums_df['times_ranked'].fillna(0).astype(int) >= 1).sum()

        # ARTIST SCORE
        album_percentile = ((total_albums - artist_albums_df['Global_Rank'].mean()) / total_albums) * 100 if total_albums > 0 and not artist_albums_df.empty else 0
        song_percentile = ((total_songs - artist_songs_df['Universal_Rank'].mean()) / total_songs) * 100 if total_songs > 0 and not artist_songs_df.empty else 0
        artist_score = (album_percentile * 0.6) + (song_percentile * 0.4) if ranked_albums_count > 0 else 0



        def get_album_placement_on_rank_date(album_id, rank_date, all_albums_df):
            # Only include albums ranked on or before this date
            eligible = all_albums_df[all_albums_df['first_ranked_date'] <= rank_date].copy()
            eligible = eligible.sort_values('weighted_average_score', ascending=False).reset_index(drop=True)
            try:
                placement = eligible[eligible['album_id'].astype(str) == str(album_id)].index[0] + 1
                return placement
            except Exception:
                return None

        # 2. In your timeline event loop, use all_albums_df for placement:
        timeline_events = []
        for _, row in artist_albums_df.iterrows():
            dt = pd.to_datetime(row['first_ranked_date'], errors='coerce')
            if pd.isnull(dt): continue
            placement = get_album_placement_on_rank_date(row['album_id'], dt, all_albums_df)
            timeline_events.append({
                'date_obj': dt,
                'ranking_date_str': dt.strftime('%b %d, %Y %I:%M:%S %p') if not pd.isnull(dt) else 'N/A',
                'score': row.get('weighted_average_score'),
                'placement': placement,  # This is now global!
                'album_name': row['album_name'],
                'album_cover_url': row.get('album_cover_url', '')
            })

        for event in timeline_events:
            event['album_name'] = clean_title(event['album_name'])
        valid_timeline_events = [event for event in timeline_events if not pd.isnull(event['date_obj'])]
        ranking_timeline_data = sorted(valid_timeline_events, key=lambda x: x['date_obj'])

        # RELEASE HISTORY HISTOGRAM
        ranked_album_ids = artist_albums_df['album_id'].tolist() if 'album_id' in artist_albums_df else []
        release_dates = get_album_release_dates(sp, ranked_album_ids) if ranked_album_ids else {}

        if 'album_id' in artist_albums_df:
            artist_albums_df['release_date'] = artist_albums_df['album_id'].map(release_dates)
            era_history_data = artist_albums_df.sort_values(by='release_date')
        else:
            era_history_data = artist_albums_df

        # Calculate SEM and mean (ONLY ACTUAL SONGS)
        sd_by_album = actual_songs_df.groupby('Album_Name')['Ranking'].std()
        mean_by_album = actual_songs_df.groupby('Album_Name')['Ranking'].mean()

        # Get album metadata
        album_info = artist_albums_df.set_index('album_name')
        era_chart_data = []
        for album_name, mean in mean_by_album.items():
            sem = sd_by_album.get(album_name, 0)
            if album_name in album_info.index:
                row = album_info.loc[album_name]
                era_chart_data.append({
                    'x': row['release_date'],
                    'y': mean,
                    'sem': sem,
                    'label': album_name,
                    'image': row.get('album_cover_url', '')
                })

        # --- Prepare Leaderboard and other stats ---
        artist_songs_df.sort_values(by='Ranking', ascending=False, inplace=True)
        artist_songs_df['Artist_Rank'] = range(1, len(artist_songs_df) + 1)
        artist_albums_df.sort_values(by='weighted_average_score', ascending=False, inplace=True)
        artist_albums_df['Artist_Rank'] = range(1, len(artist_albums_df) + 1)
        artist_average_score = artist_albums_df['weighted_average_score'].mean() if not artist_albums_df.empty else 0

        # For polar chart
        all_rank_groups = [f"{i / 2:.1f}" for i in range(2, 21)]
        all_songs_df['Rank_Group_Str'] = all_songs_df['Rank_Group'].astype(str)
        polar_data_series = pd.Series(index=all_rank_groups + ['I'], dtype=int).fillna(0)
        song_counts = all_songs_df['Rank_Group_Str'].value_counts()
        polar_data_series.update(song_counts)
        polar_chart_data = {
            'labels': polar_data_series.index.tolist(),
            'data': polar_data_series.values.tolist()
        }

        # Use only actual songs for ALL STATS and song_score_distribution!
        average_song_score = actual_songs_df['Ranking'].mean() if not actual_songs_df.empty else 0
        median_song_score = actual_songs_df['Ranking'].median() if not actual_songs_df.empty else 0
        std_song_score = actual_songs_df['Ranking'].std() if not actual_songs_df.empty else 0



        if not actual_songs_df.empty:
            top_song_row = actual_songs_df.loc[actual_songs_df['Ranking'].idxmax()]
            low_song_row = actual_songs_df.loc[actual_songs_df['Ranking'].idxmin()]
            top_song_name = top_song_row['Song_Name']
            top_song_score = top_song_row['Ranking']
            top_song_album = top_song_row['Album_Name']
            low_song_name = low_song_row['Song_Name']
            low_song_score = low_song_row['Ranking']
            low_song_album = low_song_row['Album_Name']

            # Find album art for highest song
            top_album_row = artist_albums_df[
                artist_albums_df['album_name'].str.strip().str.lower() == str(top_song_album).strip().lower()
                ]
            if not top_album_row.empty:
                top_song_cover = top_album_row['album_cover_url'].values[0]
            else:
                top_song_cover = 'https://placehold.co/60x60'

            # Find album art for lowest song
            low_album_row = artist_albums_df[
                artist_albums_df['album_name'].str.strip().str.lower() == str(low_song_album).strip().lower()
                ]
            if not low_album_row.empty:
                low_song_cover = low_album_row['album_cover_url'].values[0]
            else:
                low_song_cover = 'https://placehold.co/60x60'

            top_song_link = url_for(
                'album_page',
                artist_name=artist_name,
                album_name=quote_plus(top_song_row['Album_Name']),
                album_id=top_album_row['album_id'].values[0] if not top_album_row.empty else ''
            ) if not top_album_row.empty else "#"

            low_song_link = url_for(
                'album_page',
                artist_name=artist_name,
                album_name=quote_plus(low_song_row['Album_Name']),
                album_id=low_album_row['album_id'].values[0] if not low_album_row.empty else ''
            ) if not low_album_row.empty else "#"
        else:
            top_song_name = top_song_score = top_song_cover = top_song_link = ''
            low_song_name = low_song_score = low_song_cover = low_song_link = ''

        # Highest and lowest ranked albums
        if not artist_albums_df.empty:
            top_album_row = artist_albums_df.loc[artist_albums_df['weighted_average_score'].idxmax()]
            low_album_row = artist_albums_df.loc[artist_albums_df['weighted_average_score'].idxmin()]
            top_album_name = top_album_row['album_name']
            top_album_score = top_album_row['weighted_average_score']
            top_album_cover = top_album_row.get('album_cover_url', '')
            top_album_link = url_for(
                'album_page',
                artist_name=artist_name,
                album_name=quote_plus(top_album_row['album_name']),
                album_id=top_album_row['album_id']
            ) if top_album_row.get('album_id') else "#"
            low_album_name = low_album_row['album_name']
            low_album_score = low_album_row['weighted_average_score']
            low_album_cover = low_album_row.get('album_cover_url', '')
            low_album_link = url_for(
                'album_page',
                artist_name=artist_name,
                album_name=quote_plus(low_album_row['album_name']),
                album_id=low_album_row['album_id']
            ) if low_album_row.get('album_id') else "#"
        else:
            top_album_name = top_album_score = top_album_cover = top_album_link = ''
            low_album_name = low_album_score = low_album_cover = low_album_link = ''

        global_avg_song_score = all_songs_df['Ranking'].mean() if not all_songs_df.empty else 0
        most_improved_song_name = ""
        most_improved_song_delta = 0

        if not actual_songs_df.empty and 'Song_Name' in actual_songs_df.columns and 'Ranking' in actual_songs_df.columns:
            improvement_data = []
            for song_name, group in actual_songs_df.groupby('Song_Name'):
                group_sorted = group.sort_values('Ranked_Date')
                if len(group_sorted) > 1:
                    first_rank = group_sorted.iloc[0]['Ranking']
                    last_rank = group_sorted.iloc[-1]['Ranking']
                    delta = last_rank - first_rank
                    improvement_data.append((song_name, delta, last_rank))
            if improvement_data:
                most_improved = max(improvement_data, key=lambda x: x[1])
                most_improved_song_name = most_improved[0]
                most_improved_song_delta = most_improved[1]

        def safe_float(val):
            try:
                if isinstance(val, float) and math.isnan(val):
                    return 0.0
                return float(val)
            except Exception:
                return 0.0

        ranking_trajectory_data = {
            "labels": [],
            "datasets": [{
                "label": "Avg Song Score",
                "data": [],
                "borderColor": "rgba(29, 185, 84, 1)",
                "backgroundColor": "rgba(29, 185, 84, 0.2)",
            }]
        }

        # Only use actual_songs_df for song score distribution!
        song_scores = actual_songs_df['Ranking'].tolist()

        # Song leaderboard: Only use actual_songs_df!
        song_leaderboard_clean = []
        for row in actual_songs_df.to_dict('records'):
            # For display
            row['display_name'] = clean_title(row.get('Song_Name', ''))
            row['Universal Rank'] = row.get('Universal_Rank', '')
            row['Artist Rank'] = row.get('Artist_Rank', '')
            # For linking: pass the REAL sheet song name in the url (url-encoded)
            row['song_link'] = url_for(
                'song_page',
                artist_name=artist_name,
                song_name=quote_plus(row['Song_Name'])
            )
            song_leaderboard_clean.append(row)

        # For album leaderboard (not changed)
        album_leaderboard_clean = []
        for row in artist_albums_df.to_dict('records'):
            row['album_name'] = clean_title(row['album_name'])
            row['Global Rank'] = row.get('Global_Rank', '')
            row['Artist Rank'] = row.get('Artist_Rank', '')  # <-- make sure you include this!
            album_leaderboard_clean.append(row)

        # --- First/last album ranked info ---
        if not actual_songs_df.empty and 'Ranked_Date' in actual_songs_df.columns:
            actual_songs_df['Ranked_Date'] = pd.to_datetime(actual_songs_df['Ranked_Date'], errors='coerce')
            # Approach 2: Get the earliest and latest ranked date per album
            album_dates = (
                actual_songs_df.groupby('Album_Name')['Ranked_Date']
                .agg(['min', 'max'])
                .reset_index()
            )
            # Find the album with the earliest ranking date (first ranked)
            first_album_row = album_dates.loc[album_dates['min'].idxmin()]
            # Find the album with the latest ranking date (most recent ranked)
            last_album_row = album_dates.loc[album_dates['max'].idxmax()]

            first_album_ranked_name = first_album_row['Album_Name']
            first_album_ranked_date = first_album_row['min'].strftime('%b %d, %Y') if pd.notnull(
                first_album_row['min']) else ""
            last_album_ranked_name = last_album_row['Album_Name']
            last_album_ranked_date = last_album_row['max'].strftime('%b %d, %Y') if pd.notnull(
                last_album_row['max']) else ""
        else:
            first_album_ranked_name = ""
            first_album_ranked_date = ""
            last_album_ranked_name = ""
            last_album_ranked_date = ""

        # Clean up for template
        top_album_display_name = clean_title(top_album_name)
        low_album_display_name = clean_title(low_album_name)
        top_song_display_name = clean_title(top_song_name)
        low_song_display_name = clean_title(low_song_name)
        first_album_ranked_name_display = clean_title(first_album_ranked_name)
        last_album_ranked_name_display = clean_title(last_album_ranked_name)

        # --- Points and chart boundaries ---
        songs_sorted = actual_songs_df.sort_values(['Album_Name', 'Ranked_Date'])
        points = []

        from datetime import datetime

        def safe_json_val(val):
            if isinstance(val, Undefined):
                return None
            if val is None:
                return None
            if isinstance(val, float) and np.isnan(val):
                return None
            if pd.isna(val):
                return None
            return str(val) if isinstance(val, (pd.Timestamp, np.datetime64)) else val

        album_cover_map = dict(
            zip(artist_albums_df['album_name'].map(clean_title), artist_albums_df['album_cover_url']))

        album_boundaries = []
        album_labels = []
        album_arts = []
        last_album = None
        for idx, row in enumerate(songs_sorted.itertuples()):
            album = clean_title(getattr(row, 'Album_Name', 'Unknown Album'))
            album_cover = album_cover_map.get(album, 'https://placehold.co/36x36')
            song = clean_title(getattr(row, 'Song_Name', ''))
            ranked_date = getattr(row, 'Ranked_Date', None)
            score = getattr(row, 'Ranking', None)
            points.append({
                "x": idx,
                "y": safe_json_val(score),
                "album": safe_json_val(album),
                "song": safe_json_val(song),
                "date": safe_json_val(ranked_date.strftime('%b %d, %Y') if ranked_date and pd.notnull(ranked_date) else "")
            })
            if album != last_album:
                album_boundaries.append(idx)
                album_labels.append(album)
                album_arts.append(album_cover)
                last_album = album
        for point in points:
            for k, v in point.items():
                point[k] = safe_json_val(v)
        album_labels = [safe_json_val(x) for x in album_labels]
        album_boundaries = [int(x) for x in album_boundaries]

        days = 7
        now = pd.Timestamp.now()
        recent = actual_songs_df[actual_songs_df['Ranked_Date'] >= (now - pd.Timedelta(days=days))] if not actual_songs_df.empty else pd.DataFrame()
        old = actual_songs_df[actual_songs_df['Ranked_Date'] < (now - pd.Timedelta(days=days))] if not actual_songs_df.empty else pd.DataFrame()
        recent_avg = recent['Ranking'].mean() if not recent.empty else 0
        old_avg = old['Ranking'].mean() if not old.empty else 0
        arrow_delta = recent_avg - old_avg
        arrow_direction = "up" if arrow_delta > 0 else "down" if arrow_delta < 0 else "flat"

        def safe_date(val):
            try:
                return datetime.strptime(val, "%Y-%m-%d") if isinstance(val, str) else val
            except Exception:
                return datetime(1900, 1, 1)

        era_chart_data = sorted(era_chart_data, key=lambda d: safe_date(d['x']))

        return render_template(
            "artist_page_v2.html",
            artist_name=artist_name,
            artist_mastery=mastery_percentage,
            leaderboard_points=total_leaderboard_points,
            artist_average_score=artist_average_score,
            ranking_timeline_data=ranking_timeline_data,
            polar_chart_data=polar_chart_data,
            song_leaderboard=song_leaderboard_clean,
            album_leaderboard=album_leaderboard_clean,
            artist_score=artist_score,
            first_album_ranked_name=first_album_ranked_name_display,
            first_album_ranked_date=first_album_ranked_date,
            last_album_ranked_name=last_album_ranked_name_display,
            last_album_ranked_date=last_album_ranked_date,
            average_song_score=average_song_score,
            median_song_score=median_song_score,
            ranking_trajectory_data=ranking_trajectory_data,
            std_song_score=std_song_score,
            top_album_name=top_album_display_name,
            top_album_score=top_album_score,
            top_album_cover=top_album_cover,
            top_album_link=top_album_link,
            low_album_name=low_album_display_name,
            low_album_score=low_album_score,
            low_album_cover=low_album_cover,
            low_album_link=low_album_link,
            top_song_name=top_song_display_name,
            top_song_score=top_song_score,
            top_song_cover=top_song_cover,
            top_song_link=top_song_link,
            low_song_name=low_song_display_name,
            low_song_score=low_song_score,
            low_song_cover=low_song_cover,
            low_song_link=low_song_link,
            most_improved_song_name=most_improved_song_name,
            most_improved_song_delta=most_improved_song_delta,
            global_avg_song_score=global_avg_song_score,
            arrow_direction=arrow_direction,
            arrow_delta=arrow_delta,
            album_boundaries=album_boundaries,
            album_labels=album_labels,
            points=points,
            album_arts=album_arts,
            song_scores=song_scores,
            era_chart_data=era_chart_data
        )
    except Exception as e:
        logging.critical(f"🔥 CRITICAL ERROR loading artist page for {artist_name}: {e}")
        try:
            logging.critical("Sample all_songs_df row:\n%s", all_songs_df.head().to_dict())
            logging.critical("Sample all_albums_df row:\n%s", all_albums_df.head().to_dict())
        except Exception as inner_e:
            logging.critical(f"Could not print sample rows: {inner_e}")
        return f"An error occurred: {e}", 500

from flask import abort
from urllib.parse import unquote

@app.route("/artist/<artist_name>/album/<path:album_name>/<album_id>")
def album_page(artist_name, album_name, album_id):
    album_name = unquote(album_name)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Query album data from `album_averages` table
        cursor.execute('SELECT * FROM "Re-Ranking and Song History (Album Averages)" WHERE album_id = %s;', (album_id,))
        album_data_row = cursor.fetchone()

        if not album_data_row:
            logging.error("ALBUM DATA NOT FOUND!")
            abort(404)

        album_data = dict(zip([desc[0] for desc in cursor.description], album_data_row))

        # Additional queries for songs, if needed
        cursor.execute("SELECT * FROM Current Positions WHERE spotify_album_id = %s;", (album_id,))
        album_songs = cursor.fetchall()
        album_data['album_songs'] = pd.DataFrame(album_songs, columns=[desc[0] for desc in cursor.description]).to_dict(
            'records')

    except Exception as e:
        logging.error(f"Error fetching album data from database: {e}")
        abort(500)


    return render_template(
        "album_page.html",  # Use your album page template filename here
        artist_name=artist_name,
        album_name=album_data['album_name'],
        album_cover_url=album_data['album_cover_url'],
        release_date=album_data['release_date'],
        album_length=album_data['album_length'],
        album_length_sec=album_data['album_length_sec'],
        album_score=album_data['album_score'],
        avg_song_score=album_data['avg_song_score'],
        median_song_score=album_data['median_song_score'],
        std_song_score=album_data['std_song_score'],
        global_album_rank=album_data['global_album_rank'],
        top_3_songs=album_data['top_3_songs'],
        lowest_song=album_data['lowest_song'],
        most_improved_song=album_data['most_improved_song'],
        worst_improved_song=album_data.get('worst_improved_song'),
        artist_avg_song_score=album_data['artist_avg_song_score'],
        global_avg_song_score=album_data['global_avg_song_score'],
        album_ranking_timeline=album_data.get('album_ranking_timeline'),
        album_ranking_delta=album_data.get('album_ranking_delta'),
        album_songs=album_data['album_songs'],
        last_song_end_min=album_data['last_song_end_min']
    )
@app.route('/get_album_stats/<album_id>')
def get_album_stats(album_id):
    from album_blocklist import get_visible_studio_albums_for_artist, load_blocklist_for_artist
    try:
        logging.info(f"Received album_id: {album_id}")
        conn = get_db_connection()
        cursor = conn.cursor()
        # 1. Load data
        # Fetch song-level data from `song_data` table
        cursor.execute('SELECT * FROM "Current Positions";')
        main_data = cursor.fetchall()
        main_df = pd.DataFrame(main_data, columns=[desc[0] for desc in cursor.description]).fillna("")

        # Fetch album averages data from `album_averages`
        cursor.execute('SELECT * FROM "Re-Ranking and Song History (Album Averages)";')
        averages_data = cursor.fetchall()
        averages_df = pd.DataFrame(averages_data, columns=[desc[0] for desc in cursor.description]).fillna("")

        # 2. Find the specific album's data
        album_stats = averages_df[averages_df['album_id'].astype(str) == str(album_id)]
        logging.info(f"album_stats shape: {album_stats.shape}")
        if album_stats.empty:
            return jsonify({'error': 'Album not found in averages sheet.'}), 404

        album_stats = album_stats.iloc[0]
        album_artist = str(album_stats.get('artist_name', '')).strip()  # <-- get album artist

        current_score = pd.to_numeric(album_stats.get('weighted_average_score'), errors='coerce')
        previous_score = pd.to_numeric(album_stats.get('previous_weighted_score'), errors='coerce')
        original_score = pd.to_numeric(album_stats.get('original_weighted_score'), errors='coerce')

        # Score drift is now based on the most recent previous score
        score_drift = (current_score - previous_score) if pd.notna(current_score) and pd.notna(previous_score) else 0

        # Load and parse the rerank history
        history_str = album_stats.get('rerank_history', '[]')
        try:
            rerank_history = json.loads(history_str) if history_str and pd.notna(history_str) else []
        except json.JSONDecodeError:
            rerank_history = []

        # 3. Find Best/Worst Songs
        album_songs_df = main_df[
            (main_df['Spotify Album ID'] == album_id) & (main_df['Rank Group'].astype(str) != 'I')].copy()
        album_songs_df['Ranking'] = pd.to_numeric(album_songs_df['Ranking'], errors='coerce')
        best_song = album_songs_df.loc[album_songs_df['Ranking'].idxmax()] if not album_songs_df.empty and not \
        album_songs_df['Ranking'].isnull().all() else None
        worst_song = album_songs_df.loc[album_songs_df['Ranking'].idxmin()] if not album_songs_df.empty and not \
        album_songs_df['Ranking'].isnull().all() else None

        averages_df['weighted_average_score'] = pd.to_numeric(averages_df['weighted_average_score'], errors='coerce')
        averages_df.dropna(subset=['weighted_average_score'], inplace=True)
        averages_df.sort_values(by='weighted_average_score', ascending=False, inplace=True)
        averages_df.reset_index(drop=True, inplace=True)

        averages_df.sort_values(by='weighted_average_score', ascending=False, inplace=True)
        averages_df.reset_index(drop=True, inplace=True)
        placement_series = averages_df.index[averages_df['album_id'].astype(str) == str(album_id)]
        leaderboard_placement = int(placement_series[0] + 1) if not placement_series.empty else 'N/A'

        last_ranked_date = pd.to_datetime(album_stats.get('last_ranked_date'), errors='coerce')
        times_ranked_val = pd.to_numeric(album_stats.get('times_ranked'), errors='coerce')
        times_ranked = 0 if pd.isna(times_ranked_val) else int(times_ranked_val)
        next_rerank_date = 'N/A'
        if pd.notna(last_ranked_date):
            days_to_add = 45 if times_ranked > 1 else 15
            next_rerank_date = (last_ranked_date + pd.Timedelta(days=days_to_add)).strftime('%Y-%m-%d')


        # 1. Standardize columns (spaces to underscores)
        main_df.columns = [c.replace(' ', '_') for c in main_df.columns]

        # 2. Coerce types
        for col in ['Ranking', 'Artist_Name', 'Song_Name', 'Ranking_Status']:
            if col in main_df.columns:
                if col == 'Ranking':
                    main_df[col] = pd.to_numeric(main_df[col], errors='coerce')
                else:
                    main_df[col] = main_df[col].astype(str)

        # 3. Only FINAL rankings
        if 'Ranking_Status' in main_df.columns:
            main_df = main_df[main_df['Ranking_Status'].str.lower() == 'final']

        # 4. Remove duplicates: keep only the latest by Ranked_Date
        if all(col in main_df.columns for col in ['Song_Name', 'Artist_Name', 'Ranked_Date']):
            main_df = main_df.sort_values('Ranked_Date').drop_duplicates(['Song_Name', 'Artist_Name'], keep='last')

        # 5. Filter valid entries
        main_df = main_df[
            (main_df['Song_Name'].str.strip() != "") &
            (main_df['Artist_Name'].str.strip() != "") &
            (main_df['Ranking'].notnull())
            ]

        # 6. Handle 'Rank_Group' column (legacy compatibility)
        if 'Rank_Group' not in main_df.columns and 'Rank Group' in main_df.columns:
            main_df['Rank_Group'] = main_df['Rank Group']

        # 7. Remove interludes
        main_df = main_df[main_df['Rank_Group'] != "I"]
        artist_canonical = album_artist.strip().lower()
        visible_studio_albums = get_visible_studio_albums_for_artist(
            artist_canonical,
            spotify_client=sp,
            blocklist_loader=load_blocklist_for_artist,
            deduplicate_fn=deduplicate_by_track_overlap,
            is_live_album_fn=is_live_album
        )
        visible_album_ids = set(a['id'] for a in visible_studio_albums)


        # --- Restrict main_df to visible albums ---
        if 'Spotify_Album_ID' in main_df.columns:
            main_df = main_df[main_df['Spotify_Album_ID'].isin(visible_album_ids)]
        elif 'spotify_album_id' in main_df.columns:
            main_df = main_df[main_df['spotify_album_id'].isin(visible_album_ids)]

        # --- Artist matcher: exact match in comma-separated list ---
        def artist_matcher_field(artists_string):
            if not isinstance(artists_string, str):
                return False
            return any(artist_canonical == a.strip().lower() for a in artists_string.split(','))

        artist_songs_df = main_df[main_df['Artist_Name'].apply(artist_matcher_field)]



        artist_avg = artist_songs_df['Ranking'].mean() if not artist_songs_df.empty else None

        response_data = {
            'original_score': f"{original_score:.2f}" if pd.notna(original_score) else 'N/A',
            'best_song': {'name': str(best_song['Song Name']),
                          'score': f"{best_song['Ranking']:.2f}"} if best_song is not None else {'name': 'N/A',
                                                                                                 'score': ''},
            'worst_song': {'name': str(worst_song['Song Name']),
                           'score': f"{worst_song['Ranking']:.2f}"} if worst_song is not None else {'name': 'N/A',
                                                                                                    'score': ''},
            'leaderboard_placement': leaderboard_placement,
            'change_from_last_rank': f"{score_drift:+.2f}",  # This now uses the new drift calculation
            'next_rerank_date': next_rerank_date,
            'rerank_history': rerank_history,  # Pass the history to the frontend
            'artist_avg': f"{artist_avg:.2f}" if artist_avg is not None else 'N/A'  # <-- RETURN THIS TO FRONTEND!
        }
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error in get_album_stats for {album_id}: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred: {e}'}), 500

@app.route("/submit_rankings", methods=["POST"])
def submit_rankings():
    global sp, client
    import gspread
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'Invalid data received.'}), 400

        # --- 1. Get Core Data from Frontend ---
        album_id = data.get("album_id")
        artist_name = data.get("artist_name")
        album_name = data.get("album_name")
        album_cover_url = data.get("album_cover_url")
        all_ranked_songs_from_js = data.get("all_ranked_data", [])
        prelim_ranks_from_js = data.get("prelim_rank_data", [])
        submission_status = data.get("status", "final")
        is_rerank = data.get("is_rerank_mode", False)

        logging.info(f"\n--- SUBMIT RANKINGS START for Album ID: {album_id} (Status: {submission_status}) ---")
        if submission_status == 'draft':
            prelim_ranks_from_js = data.get("prelim_rank_data", [])
            if not prelim_ranks_from_js:
                return jsonify({'status': 'error', 'message': 'No preliminary ranks to save.'}), 400

            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM "Preliminary Ranks";')
                prelim_data = cursor.fetchall()
                prelim_df = pd.DataFrame(prelim_data, columns=[desc[0] for desc in cursor.description]).fillna("")
            except psycopg2.Error as e:
                logging.error(f"Database error occurred while fetching `preliminary_rankings`: {e}")
                # Handle the error gracefully (maybe return an empty DataFrame or raise the error)
                prelim_df = pd.DataFrame(
                    columns=['album_id', 'album_name', 'artist_name', 'album_cover_url', 'song_id', 'song_name',
                             'prelim_rank', 'timestamp'])

            # Filter out any old draft rows for this album before adding new ones
            if 'album_id' in prelim_df.columns:
                prelim_df = prelim_df[prelim_df['album_id'].astype(str) != str(album_id)]

            new_prelim_rows = [{
                'album_id': album_id, 'album_name': album_name, 'artist_name': artist_name,
                'album_cover_url': album_cover_url, 'song_id': p.get('song_id'),
                'song_name': p.get('song_name'), 'prelim_rank': p.get('prelim_rank'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            } for p in prelim_ranks_from_js]

            final_prelim_df = pd.concat([prelim_df, pd.DataFrame(new_prelim_rows)], ignore_index=True)
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO 'Preliminary Ranks' (album_id, album_name, artist_name, album_cover_url, song_id, song_name, prelim_rank, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (album_id, song_id) DO UPDATE SET
                    prelim_rank = EXCLUDED.prelim_rank, timestamp = EXCLUDED.timestamp;
                """,
                final_prelim_df.values.tolist()
            )
            conn.commit()

            dominant_color = get_dominant_color(album_cover_url)
            return jsonify({
                'status': 'success',
                'animation_data': {
                    'album_name': album_name, 'artist_name': artist_name,
                    'album_cover_url': album_cover_url, 'dominant_color': dominant_color
                }
            })

            # --- FINAL SUBMISSION LOGIC (existing logic, now in an 'else' block) ---
        else:


            old_score = 0
            old_placement = 0
            if is_rerank:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM "Re-Ranking and Song History (Album Averages)";')
                averages_data = cursor.fetchall()
                averages_df_before = pd.DataFrame(averages_data, columns=[desc[0] for desc in cursor.description])

                if not averages_df_before.empty:
                    old_album_data = averages_df_before[averages_df_before['album_id'].astype(str) == str(album_id)]
                    old_score = old_album_data.iloc[0]['weighted_average_score']
                    # Sort to find old placement
                    cursor.execute(
                        "SELECT album_id, weighted_average_score FROM 'Re-Ranking and Song History (Album Averages)' ORDER BY weighted_average_score DESC;"
                    )
                    averages_sorted_data = cursor.fetchall()
                    averages_sorted_df = pd.DataFrame(
                        averages_sorted_data,
                        columns=['album_id', 'weighted_average_score']
                    ).reset_index(drop=True)

                    if not averages_sorted_df.empty:
                        old_placement_series = averages_sorted_df.index[
                            averages_sorted_df['album_id'].astype(str) == str(album_id)
                            ]
                        old_placement = int(old_placement_series[0] + 1) if not old_placement_series.empty else 1
                    else:
                        old_placement = 1  # Default placement if no data is found

            # --- 4. Update Google Sheets with New Final Rankings ---
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM "Current Positions";')
            main_data = cursor.fetchall()
            main_df = pd.DataFrame(main_data, columns=[desc[0] for desc in cursor.description]).fillna("")

            all_song_ids = [s.get('song_id') for s in all_ranked_songs_from_js if s.get('song_id')]
            song_details_map = {}
            if sp and all_song_ids:
                try:
                    all_song_ids = [str(sid) for sid in all_song_ids]
                    for i in range(0, len(all_song_ids), 20):
                        batch = all_song_ids[i:i + 20]
                        tracks_info = sp.tracks(batch)
                        for track in tracks_info['tracks']:
                            if track: song_details_map[track['id']] = {'name': track['name'],
                                                                       'duration_ms': track['duration_ms']}
                except Exception as e:
                    logging.error(f"Failed to fetch batch track details from Spotify: {e}")

            submitted_song_ids = {str(s.get('song_id')) for s in all_ranked_songs_from_js}
            main_df_filtered = main_df[~main_df['Spotify Song ID'].astype(str).isin(
                submitted_song_ids)] if 'Spotify Song ID' in main_df.columns and submitted_song_ids else main_df

            new_final_rows_data = []
            for ranked_song_data in all_ranked_songs_from_js:
                song_id = str(ranked_song_data.get('song_id'))
                details = song_details_map.get(song_id, {})
                ranked_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Use current timestamp for new songs

                cursor.execute(
                    """
                    INSERT INTO 'Current Positions' (album_name, artist_name, spotify_album_id, song_name, ranking, duration_ms, ranking_status, ranked_date, position_in_group, rank_group, spotify_song_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (spotify_song_id) DO UPDATE SET
                        ranking = EXCLUDED.ranking,
                        ranking_status = EXCLUDED.ranking_status,
                        ranked_date = EXCLUDED.ranked_date;
                    """,
                    (
                        ranked_song_data.get('album_name'),
                        ranked_song_data.get('artist_name'),
                        ranked_song_data.get('album_id'),
                        details.get('name', ranked_song_data.get('song_name')),
                        ranked_song_data.get('calculated_score', 0.0),
                        details.get('duration_ms', 0),
                        'final',
                        ranked_date,  # Fixed timestamp
                        str(ranked_song_data.get('position_in_group', '')),
                        str(ranked_song_data.get('rank_group')),
                        song_id,
                    )
                )
            conn.commit()

            logging.info("Updating album averages...")
            try:
                # Fetch necessary data for recalculations
                cursor.execute(
                    "SELECT spotify_album_id, AVG(ranking) AS average_score FROM 'Current Positions' WHERE rank_group != 'I' GROUP BY spotify_album_id;")
                averages = cursor.fetchall()

                for album_id, avg_score in averages:
                    cursor.execute(
                        """
                        UPDATE album_averages SET average_score = %s, weighted_average_score = %s, last_ranked_date = %s, times_ranked = times_ranked + 1
                        WHERE album_id = %s;
                        """,
                        (
                            avg_score,  # Regular average score
                            avg_score,  # Weighted average needs custom calculation logic (placeholder here)
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Update last ranked date
                            album_id,  # Specific album ID
                        )
                    )
                conn.commit()
                logging.info("Album averages updated.")
            except psycopg2.Error as e:
                logging.error(f"Error updating album averages: {e}")

            # Recalculate and Update Album Averages Sheet
            # Query valid song data from the database
            cursor.execute("""
                SELECT spotify_album_id, ranking, duration_ms
                FROM 'Current Positions'
                WHERE rank_group != 'I' AND ranking IS NOT NULL AND duration_ms IS NOT NULL;
            """)
            valid_song_data = cursor.fetchall()

            # Convert the data into a Pandas DataFrame for convenience in calculating averages
            col_names = ['Spotify Album ID', 'Ranking', 'Duration (ms)']
            df_for_calc = pd.DataFrame(valid_song_data, columns=col_names)

            # Ensure numeric data types
            df_for_calc['Ranking'] = pd.to_numeric(df_for_calc['Ranking'], errors='coerce')
            df_for_calc['Duration (ms)'] = pd.to_numeric(df_for_calc['Duration (ms)'], errors='coerce')
            df_for_calc.dropna(inplace=True)  # Drop rows with missing data
            df_for_calc_no_interludes = df_for_calc[df_for_calc['Spotify Album ID'].notnull()]

            if not df_for_calc_no_interludes.empty:
                cursor.execute("""
                    SELECT spotify_album_id, AVG(ranking) AS simple_average
                    FROM 'Current Positions'
                    WHERE rank_group != 'I'
                    GROUP BY spotify_album_id;
                """)
                simple_averages = pd.DataFrame(cursor.fetchall(), columns=['Spotify Album ID', 'Simple Average'])
                cursor.execute("""
                    SELECT spotify_album_id, 
                           SUM(ranking * duration_ms) / SUM(duration_ms) AS weighted_average
                    FROM 'Current Positions'
                    WHERE rank_group != 'I' AND duration_ms > 0
                    GROUP BY spotify_album_id;
                """)
                weighted_averages = pd.DataFrame(cursor.fetchall(), columns=['Spotify Album ID', 'Weighted Average'])
                cursor.execute("""
                    SELECT spotify_album_id
                    FROM 'Re-Ranking and Song History (Album Averages)'
                    ORDER BY weighted_average_score DESC;
                """)
                sorted_album_ids = [row[0] for row in cursor.fetchall()]
                new_placement = sorted_album_ids.index(album_id) + 1 if album_id in sorted_album_ids else None
                total_albums = len(sorted_album_ids)

                cursor.execute("""
                        SELECT spotify_album_id, album_name, artist_name
                        FROM 'Current Positions'
                        GROUP BY spotify_album_id, album_name, artist_name;
                    """)
                album_info_map = {
                    row[0]: {'Album Name': row[1], 'Artist Name': row[2]}
                    for row in cursor.fetchall()
                }

                logging.info(f"Successfully calculated album metrics and placement for album {album_id}.")

                # THE FIX: This loop now updates the score history for ALL albums
                for album_id_to_update in weighted_averages.keys():  # Iterating through all albums to update or insert
                    try:
                        # Fetch current album values
                        cursor.execute("""
                            SELECT weighted_average_score, times_ranked, score_history, average_score
                            FROM 'Re-Ranking and Song History (Album Averages)' WHERE album_id = %s;
                        """, (album_id_to_update,))
                        album_data = cursor.fetchone()

                        if album_data:  # If the album exists
                            current_weighted_avg = album_data[0] or 0
                            current_times_ranked = album_data[1] or 0
                            current_score_history = json.loads(album_data[2] or '[]')
                            current_average_score = album_data[3]

                            new_weighted_avg = weighted_averages[album_id_to_update]
                            new_simple_avg = simple_averages.get(album_id_to_update, None)

                            # Update score history
                            current_score_history.append(new_weighted_avg)
                            updated_score_history = json.dumps(current_score_history)

                            # Update the database values for the album
                            cursor.execute("""
                                UPDATE 'Re-Ranking and Song History (Album Averages)'
                                SET score_history = %s, average_score = %s, weighted_average_score = %s, 
                                    times_ranked = %s, last_ranked_date = %s
                                WHERE album_id = %s;
                            """, (
                                updated_score_history,
                                new_simple_avg,
                                new_weighted_avg,
                                current_times_ranked + 1,
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                album_id_to_update
                            ))
                            conn.commit()

                            # Handle rerank history if reranking
                            if album_id_to_update == album_id and is_rerank:
                                cursor.execute("""
                                    SELECT rerank_history
                                    FROM 'Re-Ranking and Song History (Album Averages)'
                                    WHERE album_id = %s;
                                """, (album_id,))
                                result = cursor.fetchone()
                                history_str = result[0] or '[]' if result else '[]'
                                rerank_history = json.loads(history_str)

                                # Append rerank history
                                rerank_history.append({
                                    'date': datetime.now().strftime('%Y-%m-%d'),
                                    'score': new_weighted_avg,
                                    'placement': f"{new_placement}/{total_albums}"
                                })

                                updated_rerank_history = json.dumps(rerank_history)
                                cursor.execute("""
                                    UPDATE 'Re-Ranking and Song History (Album Averages)'
                                    SET rerank_history = %s
                                    WHERE album_id = %s;
                                """, (updated_rerank_history, album_id))
                                conn.commit()

                        else:  # If the album does not exist, insert it into the database
                            info = album_info_map.get(album_id_to_update)
                            if info:
                                try:
                                    initial_history = [{
                                        'date': datetime.now().strftime('%Y-%m-%d'),
                                        'score': weighted_averages[album_id_to_update]
                                    }]
                                    cursor.execute("""
                                        INSERT INTO 'Re-Ranking and Song History (Album Averages)' (
                                            album_id, album_name, artist_name, average_score, weighted_average_score,
                                            original_weighted_score, previous_weighted_score, times_ranked, 
                                            last_ranked_date, rerank_history, score_history, album_cover_url
                                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                                    """, (
                                        album_id_to_update,
                                        info['Album Name'],
                                        info['Artist Name'],
                                        simple_averages.get(album_id_to_update, None),
                                        weighted_averages[album_id_to_update],
                                        weighted_averages[album_id_to_update],
                                        weighted_averages[album_id_to_update],
                                        1,
                                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        json.dumps([]),  # No rerank history initially
                                        json.dumps([weighted_averages[album_id_to_update]]),  # Initial score history
                                        info.get('album_cover_url', '')
                                    ))
                                    conn.commit()
                                except psycopg2.Error as e:
                                    logging.error(f"Error inserting new album into album_averages: {e}")

                    except psycopg2.Error as e:
                        logging.error(f"Error processing album ID {album_id_to_update}: {e}")

                    # Commit new rows or updates in `album_averages`
                    logging.info("Successfully recalculated and saved all averages into the database.")

                    # Fetch latest song data
                    logging.info("Fetching song data for processing...")
                    cursor.execute("""
                            SELECT event_number, ranked_date, album_name, artist_name, spotify_album_id,
                                   song_name, spotify_song_id, ranking, placement, percentile
                            FROM 'Current Positions'
                        """)
                    song_data = cursor.fetchall()
                    song_data_df = pd.DataFrame(song_data, columns=[desc[0] for desc in cursor.description]).fillna("")

                    # Group and calculate `song_grouped` data
                    logging.info("Grouping song data for the event...")
                    song_grouped = (
                        song_data_df.groupby(
                            ["Spotify Album ID", "Album Name", "Artist Name", "Spotify Song ID", "Song Name"]
                        )
                        .agg({"Ranking": "mean"})
                        .reset_index()
                        .sort_values("Ranking", ascending=False)
                        .reset_index(drop=True)
                    )
                    song_grouped["Placement"] = song_grouped.index + 1
                    total_songs = len(song_grouped)
                    song_grouped["Percentile"] = song_grouped["Placement"] / total_songs * 100

                    # Compute the next event number
                    logging.info("Calculating the next event number...")
                    cursor.execute("SELECT MAX(event_number) FROM 'Current Positions';")
                    result = cursor.fetchone()
                    event_number = int(result[0] or 0) + 1

                    # Prepare song event data for insertion
                    logging.info("Preparing song events for insertion...")
                    song_event_data = [
                        (
                            event_number,
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            row["Album Name"],
                            row["Artist Name"],
                            row["Spotify Album ID"],
                            row["Song Name"],
                            row["Spotify Song ID"],
                            round(row["Ranking"], 4),
                            int(row["Placement"]),
                            round(row["Percentile"], 4)
                        )
                        for _, row in song_grouped.iterrows()
                    ]

                    # Insert new song events into the database
                    logging.info("Inserting song events into the database...")
                    cursor.executemany("""
                            INSERT INTO song_data (event_number, ranked_date, album_name, artist_name, spotify_album_id,
                                                   song_name, spotify_song_id, ranking, placement, percentile)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, song_event_data)
                    conn.commit()
                    logging.info(f"Inserted {len(song_event_data)} rows into `song_data` for event {event_number}.")



                # Prepare drift data
            try:
                # Compute the next event number
                logging.info("Calculating the next event number...")
                try:
                    cursor.execute("SELECT MAX(event_number) FROM song_data;")
                    result = cursor.fetchone()
                    # Default event_number to 1 if no previous events exist
                    event_number = int(result[0] or 0) + 1
                    logging.info(f"Next event number determined: {event_number}")
                except psycopg2.Error as e:
                    logging.error(f"Error fetching maximum event number: {e}")
                    event_number = 1  # Default to 1 in case of error or no prior events

                # Fetch the necessary baseline data from `song_data`
                logging.info("Fetching final song data for drift calculations...")
                cursor.execute("""
                    SELECT spotify_album_id AS "Spotify Album ID", artist_name AS "Artist Name", album_name AS "Album Name",
                           song_name AS "Song Name", spotify_song_id AS "Spotify Song ID", ranking AS "Ranking",
                           rank_group AS "Rank Group", ranking_status AS "Ranking Status"
                    FROM song_data
                    WHERE rank_group != 'I' AND ranking_status = 'final' AND ranking IS NOT NULL;
                """)
                song_data = cursor.fetchall()

                # Create a `song_data_df` DataFrame for drift calculations
                song_data_df = pd.DataFrame(song_data, columns=[desc[0] for desc in cursor.description]).fillna("")
                logging.info("Final song data fetched and loaded into Pandas DataFrame.")

                # Group by relevant columns to calculate drift
                logging.info("Grouping song data for drift calculations...")
                song_grouped = (
                    song_data_df.groupby(
                        ['Song Name', 'Artist Name', 'Spotify Song ID', 'Album Name', 'Spotify Album ID']
                    ).agg({
                        'Ranking': 'mean'  # Calculate mean ranking for each grouped song
                    }).reset_index()
                    .sort_values('Ranking', ascending=False)
                    .reset_index(drop=True)  # Sort grouped data in descending order of ranking
                )

                # Compute placements and percentiles within the grouped data
                total_songs = len(song_grouped)
                logging.info(f"Total number of songs grouped: {total_songs}")

                song_grouped['Placement'] = song_grouped.index + 1
                song_grouped['Percentile'] = (
                    (song_grouped['Placement'] - 1) / (total_songs - 1) * 100 if total_songs > 1 else 0
                )

                # Define the ranked date for this drift calculation event
                event_ranked_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Prepare batch data for insertion
                logging.info("Preparing batch drift data for insertion...")
                batch_data = [
                    (
                        int(event_number),
                        event_ranked_date,
                        row['Album Name'],
                        row['Artist Name'],
                        row['Spotify Album ID'],
                        row['Song Name'],
                        row['Spotify Song ID'],
                        round(row['Ranking'], 4),
                        int(row['Placement']),
                        round(row['Percentile'], 4)
                    )
                    for _, row in song_grouped.iterrows()
                ]

                # Insert drift data into `song_data`
                logging.info("Inserting batch drift data into the database...")
                try:
                    cursor.executemany("""
                        INSERT INTO 'Current Positions' (event_number, ranked_date, album_name, artist_name, spotify_album_id,
                                               song_name, spotify_song_id, ranking, placement, percentile)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """, batch_data)
                    conn.commit()
                    logging.info(f"Drift data for {len(batch_data)} songs successfully inserted.")
                except psycopg2.Error as e:
                    logging.error(f"Error inserting batch drift data: {e}")

            except Exception as e:
                logging.error(f"Error calculating and inserting drift data: {e}")



            # --- 5. Get "AFTER" data for the animation ---
            try:
                # --- Step 1: Fetch the "after" album averages directly from the database ---
                logging.info("Fetching 'after' album averages for animations...")
                cursor.execute("""
                    SELECT album_id, album_name, artist_name, weighted_average_score, times_ranked, album_cover_url
                    FROM 'Re-Ranking and Song History (Album Averages)'
                    WHERE weighted_average_score IS NOT NULL
                    ORDER BY weighted_average_score DESC;
                """)
                album_averages_after = cursor.fetchall()

                if not album_averages_after:
                    return jsonify({'status': 'error', 'message': 'No album averages found after ranking.'}), 500

                # Convert query results into a sorted list
                sorted_album_averages = [
                    {
                        'album_id': row[0],
                        'album_name': row[1],
                        'artist_name': row[2],
                        'weighted_average_score': float(row[3]),
                        'times_ranked': int(row[4]),
                        'album_cover_url': row[5]
                    }
                    for row in album_averages_after
                ]

                # --- Step 2: Find the current album's data ---
                album_data = next((album for album in sorted_album_averages if str(album['album_id']) == str(album_id)),
                                  None)
                if not album_data:
                    return jsonify(
                        {'status': 'error', 'message': f"Could not find album {album_id} after ranking."}), 500

                # Extract the new scores, placements, etc.
                new_score = album_data['weighted_average_score']
                times_ranked = album_data['times_ranked']
                new_placement = sorted_album_averages.index(album_data) + 1  # Placement is 1-based
                total_albums = len(sorted_album_averages)
                dominant_color = get_dominant_color(album_cover_url)

                # --- Step 3: Return JSON response to the frontend ---
                if is_rerank:
                    return jsonify({
                        'status': 'success',
                        'rerank_animation_data': {
                            'album_name': album_data['album_name'],
                            'artist_name': album_data['artist_name'],
                            'album_cover_url': album_data['album_cover_url'],
                            'old_score': old_score,
                            'new_score': new_score,
                            'old_placement': old_placement,
                            'new_placement': new_placement,
                            'total_albums': total_albums,
                            'times_ranked': times_ranked,
                            'dominant_color': dominant_color
                        }
                    })
                else:
                    return jsonify({
                        'status': 'success',
                        'animation_data': {
                            'album_name': album_data['album_name'],
                            'artist_name': album_data['artist_name'],
                            'album_cover_url': album_data['album_cover_url'],
                            'final_score': new_score,
                            'final_rank': new_placement,
                            'total_albums': total_albums,
                            'dominant_color': dominant_color,
                            'album_id': album_id
                        }
                    })

            except Exception as e:
                logging.critical(f"\n🔥 CRITICAL ERROR in /submit_rankings (Final Animation Section): {e}",
                                 exc_info=True)
    except Exception as e:
        logging.critical(f"Error")
    return jsonify({'status': 'error', 'message': f"An unexpected error occurred: {e}"}), 500

def get_album_data(artist_name, album_name, album_id):
    import json

    def format_seconds(seconds):
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h}:{m:02}:{s:02}"
        else:
            return f"{m}:{s:02}"

    try:
        # Ensure clean inputs for matching
        album_name_clean = album_name.strip().lower()
        artist_name_clean = artist_name.strip().lower()
        conn = get_db_connection()
        cursor = conn.cursor()

        # --- Step 1: Fetch album metadata and statistics ---
        cursor.execute("""
            SELECT album_id, album_name, artist_name, weighted_average_score, average_score, times_ranked,
                   release_date, album_cover_url, rerank_history
            FROM 'Re-Ranking and Song History (Album Averages)'
            WHERE LOWER(album_name) = %s AND LOWER(artist_name) = %s AND album_id = %s;
        """, (album_name_clean, artist_name_clean, album_id))
        album_row = cursor.fetchone()

        if not album_row:
            logging.error(f"Album not found: {artist_name_clean}, {album_name_clean}, ID: {album_id}")
            return None

        album_data_processed = {
            'album_name': album_row[1],
            'artist_name': album_row[2],
            'album_score': float(album_row[3]),
            'avg_song_score': float(album_row[4]),
            'global_album_rank': None,  # Will compute later
            'times_ranked': int(album_row[5]),
            'release_date': album_row[6] or "Unknown",
            'album_cover_url': album_row[7],
            'album_ranking_timeline': json.loads(album_row[8]) if album_row[8] else []
        }

        # --- Step 2: Sort album for global ranking ---
        cursor.execute("""
            SELECT album_id, weighted_average_score
            FROM 'Re-Ranking and Song History (Album Averages)'
            WHERE weighted_average_score IS NOT NULL
            ORDER BY weighted_average_score DESC;
        """)
        global_ranking = cursor.fetchall()
        for rank, album in enumerate(global_ranking, start=1):
            if str(album[0]) == str(album_id):
                album_data_processed['global_album_rank'] = rank
                break

        # --- Step 3: Fetch song-level statistics ---
        cursor.execute("""
            SELECT song_name, spotify_song_id, duration, ranking, rank_group, ranked_date
            FROM 'Current Positions'
            WHERE LOWER(album_name) = %s AND LOWER(artist_name) = %s AND spotify_album_id = %s;
        """, (album_name_clean, artist_name_clean, album_id))
        song_rows = cursor.fetchall()

        song_data = []
        song_scores = []
        top_3_songs = []
        lowest_song = None

        # Process song rows for statistics
        logging.info("Processing song-level data...")
        for song in song_rows:
            song_name = song[0]
            spotify_song_id = song[1]
            duration = int(song[2]) if song[2] else 0
            formatted_duration = format_seconds(duration // 1000)
            score = float(song[3]) if song[3] else None
            rank_group = song[4]
            ranked_date = song[5]

            # Exclude "interludes" (Rank Group == "I") from averages
            if rank_group != "I" and score is not None:
                song_scores.append(score)

            song_data.append({
                'song_name': song_name,
                'spotify_song_id': spotify_song_id,
                'formatted_duration': formatted_duration,
                'ranking': score,
                'rank_group': rank_group,
                'ranked_date': ranked_date,
            })

        # --- Statistics on Songs ---
        logging.info("Calculating song-level statistics...")
        avg_song_score = sum(song_scores) / len(song_scores) if song_scores else 0
        median_song_score = sorted(song_scores)[len(song_scores) // 2] if song_scores else 0
        std_song_score = (sum((x - avg_song_score) ** 2 for x in song_scores) / len(song_scores)) ** 0.5 if song_scores else 0

        # Top 3 songs
        top_3_songs = sorted(song_data, key=lambda x: x['ranking'] or 0, reverse=True)[:3]
        top_3_songs = [{'title': song['song_name'], 'score': song['ranking']} for song in top_3_songs]

        # Lowest ranked song
        if song_scores:
            lowest_song_score = min(song_scores)
            lowest_song = next(({'title': x['song_name'], 'score': x['ranking']} for x in song_data if x['ranking'] == lowest_song_score), None)

        # Most improved song logic
        most_improved_song = {
            'title': '',
            'delta': 0
        }
        logging.info("Processing song ranking deltas (improvement)...")
        for song_name, group in pd.DataFrame(song_data).groupby("song_name"):
            group = group.sort_values("ranked_date")
            if len(group) > 1:
                delta = group.iloc[-1]['ranking'] - group.iloc[0]['ranking']
                if delta > most_improved_song['delta']:
                    most_improved_song = {'title': song_name, 'delta': delta}

        # --- Step 4: Album Length ---
        album_length_sec = sum([row['duration'] // 1000 for row in song_data])
        album_length = format_seconds(album_length_sec)

        # --- Step 5: Return Combined Results ---
        album_data_processed.update({
            'album_length': album_length,
            'album_length_sec': album_length_sec,
            'avg_song_score': avg_song_score,
            'median_song_score': median_song_score,
            'std_song_score': std_song_score,
            'top_3_songs': top_3_songs,
            'lowest_song': lowest_song,
            'most_improved_song': most_improved_song,
            'album_songs': song_data
        })

        return album_data_processed

    except psycopg2.Error as e:
        logging.error(f"Database error: {e}")
        return {'status': 'error', 'message': str(e)}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {'status': 'error', 'message': str(e)}
@app.route("/search_albums", methods=["GET"])
def search_albums():
    """
    AJAX endpoint for album search.
    Accepts ?q=search_term and returns up to 10 albums with: album_id, album_name, artist_name, album_cover_url
    """
    query = request.args.get("q", "").strip().lower()

    if not query or len(query) < 2:
        # Return empty JSON array if the query is too short or empty
        return jsonify([])

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Perform case-insensitive substring search using SQL's ILIKE operator
        cursor.execute("""
            SELECT album_id, album_name, artist_name, album_cover_url
            FROM 'Re-Ranking and Song History (Album Averages)'
            WHERE LOWER(album_name) LIKE %s OR LOWER(artist_name) LIKE %s
            LIMIT 10;
        """, (f"%{query}%", f"%{query}%"))  # Search for the query in album_name or artist_name

        album_results = cursor.fetchall()

        # Map the query results into a JSON response
        albums = [
            {
                "album_id": str(album[0]),
                "album_name": album[1],
                "artist_name": album[2],
                "album_cover_url": album[3] or "",  # Use an empty string if album_cover_url is null
            }
            for album in album_results
        ]
        return jsonify(albums)

    except psycopg2.Error as e:
        logging.error(f"Database error in /search_albums: {e}")
        return jsonify([])  # Return an empty JSON array on error

@app.route('/compare')
def compare_page():
    # Render your compare page template
    return render_template('compare_albums.html')
@app.route('/prelim_success')
def prelim_success():
    dominant_color = request.args.get('dominant_color', '#121212')
    return render_template(
        'prelim_success.html',
        album_name=request.args.get('album_name'),
        artist_name=request.args.get('artist_name'),
        album_cover_url=request.args.get('album_cover_url'),
        dominant_color=dominant_color
    )
from flask import request, jsonify
def format_seconds(seconds):
    """
    Converts seconds into hh:mm:ss or mm:ss format.
    """
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02}:{s:02}"  # Format as hh:mm:ss
    else:
        return f"{m}:{s:02}"         # Format as mm:ss
@app.route("/compare_albums", methods=["GET"])
def compare_albums():
    """
    Compares up to 4 albums based on their statistics and song scores.
    """
    try:
        album_ids = request.args.getlist("album_ids")  # Fetch album ids from the query
        print("Compare called with:", album_ids)

        # Restrict to up to 4 albums
        album_ids = album_ids[:4]
        if not album_ids:
            return jsonify({"error": "No album IDs provided"}), 400

        # Colors for graphs/charts
        colors = ["#1DB954", "#e74c3c", "#3498db", "#ffd700"]

        # --- Step 1: Fetch Album Data ---
        logging.info("Fetching album data for comparison...")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.album_id, a.album_name, a.artist_name, a.weighted_average_score, a.average_score, a.times_ranked,
                   a.release_date, a.album_cover_url, a.std_dev,
                   s.song_name, s.ranking, s.duration AS song_duration, s.rank_group, s.spotify_song_id
            FROM 'Re-Ranking and Song History (Album Averages)' a
            LEFT JOIN 'Current Positions' s ON a.spotify_album_id = s.spotify_album_id
            WHERE a.album_id = ANY(%s)
        """, (album_ids,))
        query_results = cursor.fetchall()

        # Organize album data into a dictionary
        albums = {}
        for row in query_results:
            album_id = str(row[0])

            if album_id not in albums:
                albums[album_id] = {
                    "album_id": row[0],
                    "album_name": row[1],
                    "artist_name": row[2],
                    "album_score": float(row[3]),
                    "avg_song_score": float(row[4]),
                    "std_dev": row[8],
                    "times_ranked": int(row[5]),
                    "release_date": row[6] or "Unknown",
                    "album_cover_url": row[7] or "",
                    "songs": []
                }

            # Add each song's details to the "songs" list for the album
            song_data = {
                "song_name": row[9],
                "score": float(row[10]) if row[10] else None,
                "duration": int(row[11]) if row[11] else 0,
                "rank_group": row[12],
                "spotify_song_id": row[13]
            }
            albums[album_id]["songs"].append(song_data)

        # --- Step 2: Prepare Data for Each Album ---
        all_data = []
        for i, album_id in enumerate(album_ids):
            album = albums.get(album_id)
            if not album:
                logging.warning(f"Album {album_id} not found in the database.")
                continue

            # --- Calculate song stats ---
            songs_with_start = []
            cumulative_seconds = 0
            main_songs = []  # Only include non-interlude songs for graphs and stats
            for song in album["songs"]:
                if song["rank_group"] != "I":  # Exclude interludes
                    main_songs.append(song)

                # Calculate `start_min` for all songs
                start_min = cumulative_seconds / 60
                songs_with_start.append({
                    **song,
                    "start_min": start_min
                })
                cumulative_seconds += song["duration"]

            # Prepare runtime graph points
            points = [
                {
                    "x": song["start_min"],
                    "y": song["score"],
                    "song": song["song_name"],
                    "album": album["album_name"],
                }
                for song in songs_with_start if song["rank_group"] != "I" and song["score"] is not None
            ]

            # Prepare box plot data
            song_scores = [song["score"] for song in main_songs if song["score"] is not None]

            # Find best and worst songs
            best_song = max(main_songs, key=lambda s: s["score"], default=None)
            worst_song = min(main_songs, key=lambda s: s["score"], default=None)

            # Append the data for this album
            all_data.append({
                "id": album_id,
                "name": album["album_name"],
                "artist": album["artist_name"],
                "album_cover_url": album["album_cover_url"],
                "release_date": album["release_date"],
                "album_score": album["album_score"],
                "avg_song_score": album["avg_song_score"],
                "placement": None,  # Could add placement in global rank later
                "std_dev": album["std_dev"],
                "length": format_seconds(cumulative_seconds),
                "color": colors[i % len(colors)],
                "points": points,
                "album_length_sec": cumulative_seconds,
                "song_scores": song_scores,
                "best_song": {
                    "title": best_song["song_name"],
                    "score": best_song["score"]
                } if best_song else {
                    "title": "", "score": None
                },
                "worst_song": {
                    "title": worst_song["song_name"],
                    "score": worst_song["score"]
                } if worst_song else {
                    "title": "", "score": None
                },
            })

        # Return comparison data
        logging.info("Comparison data prepared successfully.")
        return jsonify({"albums": all_data})

    except psycopg2.Error as e:
        logging.error(f"Database error in /compare_albums: {e}")
        return jsonify({"error": "Database error occurred. Please try again later."}), 500
    except Exception as e:
        logging.error(f"Error in /compare_albums: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/rerank_success')
def rerank_success():
    """
    Render a rerank success page.
    """
    try:
        # Fetch and validate query parameters
        album_name = request.args.get('album_name', 'Unknown Album')
        artist_name = request.args.get('artist_name', 'Unknown Artist')
        album_cover_url = request.args.get('album_cover_url', '')
        old_score = float(request.args.get('old_score', 0))
        new_score = float(request.args.get('new_score', 0))
        old_placement = int(request.args.get('old_placement', 1))
        new_placement = int(request.args.get('new_placement', 1))
        total_albums = int(request.args.get('total_albums', 1))
        times_ranked = int(request.args.get('times_ranked', 2))
        dominant_color = request.args.get('dominant_color', '#121212')

        # Determine score colors based on thresholds
        def get_score_color(score):
            if score >= 7:
                return 'green'
            elif score >= 4:
                return 'yellow'
            else:
                return 'red'

        old_score_color = get_score_color(old_score)
        new_score_color = get_score_color(new_score)

        # Render the rerank success page
        return render_template(
            'rerank_success.html',
            album_name=album_name, artist_name=artist_name, album_cover_url=album_cover_url,
            old_score=old_score, new_score=new_score, old_placement=old_placement,
            new_placement=new_placement, total_albums=total_albums, times_ranked=times_ranked,
            dominant_color=dominant_color, old_score_color=old_score_color, new_score_color=new_score_color
        )
    except Exception as e:
        logging.error(f"Error in `rerank_success`: {e}")
        return redirect(url_for('error_page'))  # Fallback to an error page

@app.route('/ranking_success')
def ranking_success():
    album_name = request.args.get('album_name')
    artist_name = request.args.get('artist_name')
    album_cover_url = request.args.get('album_cover_url')
    final_score = float(request.args.get('final_score', 0))
    final_rank = int(request.args.get('final_rank', 1))
    total_albums = int(request.args.get('total_albums', 1))
    dominant_color = request.args.get('dominant_color', '#121212')
    album_id = request.args.get('album_id')   # <-- ADD THIS LINE

    # Determine the color for the score text
    if final_score >= 7:
        score_color = 'green'
    elif final_score >= 4:
        score_color = 'yellow'
    else:
        score_color = 'red'

    start_rank = total_albums

    return render_template(
        'ranking_success.html',
        album_name=album_name,
        artist_name=artist_name,
        album_cover_url=album_cover_url,
        final_score=final_score,
        final_rank=final_rank,
        total_albums=total_albums,
        dominant_color=dominant_color,
        score_color=score_color,
        start_rank=start_rank,
        album_id=album_id        # <-- AND THIS
    )
@app.route('/search', methods=['POST'])
def search_artist():
    """
    Handle form submission for artist search and redirect appropriately.
    """
    try:
        artist_name = request.form.get('artist_name', '').strip()

        if artist_name:
            # Redirect to artist page (assuming route is `artist_page_v2`)
            logging.info(f"Redirecting to artist page for {artist_name}.")
            return redirect(url_for('artist_page_v2', artist_name=artist_name))
        else:
            # Flash a warning message and redirect to the profile page
            flash("Please enter an artist name.")
            logging.warning("No artist name provided in search form.")
            return redirect(url_for('profile'))

    except Exception as e:
        logging.error(f"Error in `search_artist`: {e}")
        flash("Something went wrong. Please try again.")
        return redirect(url_for('profile'))  # Redirect fallback
def deduplicate_by_track_overlap(albums):
    import re
    from collections import defaultdict
    print("ALL albums returned from Spotify API:")
    for album in albums:
        print(f"- {album.get('name')} (ID: {album.get('id')})")

    def normalize_track_name(name):
        name = re.sub(r'\(.*?\)', '', name)
        name = re.sub(r'-.*$', '', name)
        return name.strip().lower()

    def normalize_album_title(title):
        return re.sub(r'\(.*?\)', '', title).strip().lower()

    EXCLUDE_KEYWORDS = [
        'anthology', 'alternate', 'bonus', 'remix', 'karaoke',
        'commentary', 'version', 'expanded', 'world', 'instrumental', 'voice memo',
        'demo', 'soundtrack', 'tour', 'surprise', 'original motion picture',
        'motion picture', 'score', 'session', 'introduction', 'bbc', 'compilation', 'mothership'
    ]

    def should_exclude_by_title(title):
        title = title.lower()
        if any(kw in title for kw in EXCLUDE_KEYWORDS):
            return True
        if re.search(r'\b(live|tour|soundtrack|session|karaoke|score|surprise|compilation|mothership)\b', title):
            return True
        return False

    filtered_albums = []
    for album in albums:
        title = album.get('name', '')
        if should_exclude_by_title(title):
            continue
        filtered_albums.append(album)
    albums = filtered_albums


    track_appearances = defaultdict(list)
    for a in albums:
        release_date = a.get('release_date', '1900-01-01')
        if len(release_date) == 4:
            release_date = release_date + '-01-01'
        for t in a['tracks']:
            norm = normalize_track_name(t['name'])
            track_appearances[norm].append((release_date, a['id']))

    track_to_earliest_album = {}
    for track, appearances in track_appearances.items():
        earliest_album = min(appearances, key=lambda x: x[0])[1]
        track_to_earliest_album[track] = earliest_album

    albums_to_exclude = set()
    for a in albums:
        album_id = a['id']
        tracks = [normalize_track_name(t['name']) for t in a['tracks']]
        if not tracks:
            continue
        num_first_appearance = sum(1 for t in tracks if track_to_earliest_album[t] == album_id)
        percent_first = num_first_appearance / len(tracks)
        if percent_first < 0.3:
            albums_to_exclude.add(album_id)

    albums_by_title = defaultdict(list)

    for a in albums:
        norm_title = normalize_album_title(a.get('name', ''))
        albums_by_title[norm_title].append(a)

    canonical_album_ids = set()
    for norm_title, title_albums in albums_by_title.items():
        # 1. True original (not deluxe, remaster, edition anywhere)
        originals = [
            album for album in title_albums
            if not re.search(r'(deluxe|remaster|edition)', album.get('name', '').lower())
        ]
        # 2. Remaster only (must have remaster or remastered, but not deluxe or edition)
        remasters = [
            album for album in title_albums
            if re.search(r'remaster(ed)?', album.get('name', '').lower())
            and not re.search(r'deluxe|edition', album.get('name', '').lower())
        ]
        # 3. Deluxe/Edition as last resort
        deluxe_editions = [
            album for album in title_albums
            if re.search(r'deluxe|edition', album.get('name', '').lower())
        ]

        def get_date(album):
            date = album.get('release_date', '9999-12-31')
            if len(date) == 4:
                date = date + '-01-01'
            return date

        if originals:
            canonical_album = min(originals, key=get_date)
        elif remasters:
            canonical_album = min(remasters, key=get_date)
        elif deluxe_editions:
            canonical_album = min(deluxe_editions, key=get_date)
        else:
            canonical_album = min(title_albums, key=get_date)
        canonical_album_ids.add(canonical_album['id'])

    return [a for a in albums if a['id'] in canonical_album_ids]
def percentile_from_rank(rank, total):
    return round(100 * (rank - 1) / total + 0.00001, 1)  # e.g. rank=1 of 1000 is 0.0%

@app.route('/song/<artist_name>/<path:song_name>')
def song_page(artist_name, song_name):
    from urllib.parse import unquote
    import numpy as np
    import re

    try:
        # Decode and normalize the artist and song names
        artist_name = unquote(artist_name).replace('+', ' ').strip().lower()
        song_name = unquote(song_name).replace('+', ' ').strip().lower()

        # --- Step 1: Fetch Song Data ---
        logging.info(f"Fetching song data for artist: {artist_name}, song: {song_name}")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.song_name, s.spotify_song_id, s.ranking, s.rank_group, s.ranked_date, s.placement, s.percentile,
                   a.album_name, a.album_id, a.release_date, a.album_cover_url
            FROM song_data s
            LEFT JOIN album_averages a ON s.spotify_album_id = a.spotify_album_id
            WHERE LOWER(s.song_name) = %s AND LOWER(s.artist_name) = %s;
        """, (song_name, artist_name))
        song_results = cursor.fetchall()

        if not song_results:
            logging.warning(f"No match for artist: {artist_name}, song: {song_name}")
            abort(404, f"Song not found for artist: {artist_name}, song: {song_name}")

        # Parse song and album data
        song_df = [
            {
                "song_name": row[0],
                "spotify_song_id": row[1],
                "ranking": float(row[2]) if row[2] else None,
                "rank_group": row[3],
                "ranked_date": row[4],
                "placement": row[5],
                "percentile": float(row[6]) if row[6] else None,
                "album_name": row[7],
                "album_id": row[8],
                "release_date": row[9],
                "album_cover_url": row[10]
            }
            for row in song_results
        ]

        # Use the first song entry as the main representative
        rep = song_df[0]
        album_name = rep["album_name"]
        album_id = rep["album_id"]
        album_cover_url = rep["album_cover_url"] or ""
        release_date = rep["release_date"] or "Unknown"

        # --- Step 2: Fetch Global and Artist Rankings ---
        logging.info(f"Calculating rankings for artist: {artist_name}, song: {song_name}")
        cursor.execute("""
            WITH artist_songs AS (
                SELECT song_name, artist_name, AVG(ranking) AS avg_ranking
                FROM song_data
                WHERE LOWER(artist_name) = %s AND ranking IS NOT NULL
                GROUP BY song_name, artist_name
            )
            SELECT s.song_name, s.artist_name, s.avg_ranking, RANK() OVER (ORDER BY s.avg_ranking DESC) AS universal_rank
            FROM artist_songs s;
        """, (artist_name,))
        ranking_results = cursor.fetchall()

        # Parse global and artist rankings
        global_rank_data = [
            {"song_name": row[0], "artist_name": row[1], "average_ranking": row[2], "global_rank": row[3]}
            for row in ranking_results
        ]
        song_global_rank = next(
            (rank["global_rank"] for rank in global_rank_data if rank["song_name"].lower() == song_name), "N/A"
        )
        total_songs = len(global_rank_data)

        # Calculate artist-specific rank
        artist_rank_data = [
            rank for rank in global_rank_data if rank["artist_name"].lower() == artist_name
        ]
        artist_rank = next(
            (rank["global_rank"] for rank in artist_rank_data if rank["song_name"].lower() == song_name), "N/A"
        )

        # --- Step 3: Timeline Data ---
        logging.info("Fetching timeline data for song.")
        cursor.execute("""
            SELECT event_number, ranked_date, score, placement, percentile
            FROM song_data
            WHERE LOWER(song_name) = %s AND LOWER(artist_name) = %s
            ORDER BY event_number ASC;
        """, (song_name, artist_name))
        timeline_results = cursor.fetchall()

        timeline_event_numbers = [row[0] for row in timeline_results]
        timeline_scores = [row[2] for row in timeline_results]
        timeline_placements = [row[3] for row in timeline_results]
        timeline_percentiles = [row[4] for row in timeline_results]

        # --- Step 4: Histogram Data ---
        logging.info("Preparing histogram data.")
        bins = np.arange(0.5, 10.5, 0.5)
        histogram_counts, _ = np.histogram(
            [song["ranking"] for song in song_df if song["ranking"] is not None],
            bins=bins
        )
        histogram_bins = [f"{b:.1f}" for b in bins[:-1]]

        # --- Step 5: Render the Page ---
        return render_template(
            "song_page.html",
            song_title=rep["song_name"],
            artist_name=artist_name,
            album_name=album_name,
            album_cover_url=album_cover_url,
            album_link=f"/artist/{artist_name}/album/{album_name}/{album_id}",
            track_number="N/A",  # Could be added with further SQL queries for track numbers
            song_length="N/A",  # Could fetch duration from Spotify data
            release_date=release_date,
            times_ranked=len(song_results),
            highest_score=max([song["ranking"] for song in song_df if song["ranking"] is not None], default=None),
            lowest_score=min([song["ranking"] for song in song_df if song["ranking"] is not None], default=None),
            song_global_rank=song_global_rank,
            song_percentile="N/A",  # Could be added
            artist_rank=artist_rank,
            timeline_dates=[],
            timeline_scores=timeline_scores,
            histogram_bins=histogram_bins,
            histogram_counts=histogram_counts.tolist(),
            current_score=timeline_scores[-1] if timeline_scores else None,
            timeline_event_numbers=timeline_event_numbers,
            timeline_placements=timeline_placements,
            timeline_percentiles=timeline_percentiles,
            timelines_scores=timeline_scores
        )

    except Exception as e:
        logging.error(f"Error in /song endpoint: {e}")
        abort(500, description="An error occurred while processing the song page.")

def is_live_album(album_tracks):
    NON_LIVE_TERMS = {'remaster', 'remastered', 'mix', 'mono', 'edit', 'version'}
    live_count = 0
    for t in album_tracks:
        name = t['name'].lower()
        if '-' in name:
            after_dash = name.split('-', 1)[1].strip()
            if re.match(r'(live(\s|$))', after_dash):
                live_count += 1
                continue
            words = after_dash.split()
            if words:
                first_term = words[0]
                if (first_term not in NON_LIVE_TERMS and
                    re.search(r'\b\d{4}\b', after_dash)):
                    live_count += 1
    return live_count >= 0.8 * len(album_tracks) if album_tracks else False
@app.route("/delete_album", methods=["POST"])
def delete_album():
    artist_name = request.form.get("artist_name")
    album_id = request.form.get("album_id")
    from album_blocklist import add_to_blocklist
    add_to_blocklist(artist_name, album_id)
    return jsonify(success=True)
@app.route("/load_albums_by_artist", methods=["GET", "POST"])
def load_albums_by_artist_route():
    """
    This route loads albums for a given artist, filters out live and duplicate albums,
    calculates rerank statuses, and groups album editions for front-end rendering.
    """
    import re
    from datetime import datetime, timedelta
    import json

    artist_name = request.form.get("artist_name") or request.args.get("artist_name")

    if not artist_name:
        flash("Artist name not provided. Please search for an artist.")
        return redirect(url_for('index'))

    logging.info(f"\n--- LOADING ALBUM LIST FOR ARTIST: {artist_name} ---")

    try:
        # --- Step 1: Fetch Albums from the Database ---
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.album_id, a.album_name, a.artist_name, a.spotify_album_id, a.release_date,
                   a.album_cover_url, a.score_history, a.times_ranked, a.last_ranked_date,
                   pa.prelim_rank, pa.paused
            FROM 'Current Positions' a
            LEFT JOIN prelim_album_ranks pa ON a.album_id = pa.album_id
            WHERE LOWER(a.artist_name) = %s;
        """, [artist_name.strip().lower()])
        album_metadata_results = cursor.fetchall()

        # Parse album metadata from query results
        album_metadata = {}
        for row in album_metadata_results:
            album_id = str(row[0]).strip()
            album_metadata[album_id] = {
                "album_id": album_id,
                "album_name": row[1],
                "artist_name": row[2],
                "spotify_album_id": row[3],
                "release_date": row[4],
                "album_cover_url": row[5],
                "score_history": row[6],
                "times_ranked": row[7],
                "last_ranked_date": row[8],
                "prelim_rank": row[9],
                "paused": row[10],
            }

        # --- Step 2: Filter and Deduplicate Albums ---
        # Fetch albums and tracks for the artist (assuming a tracks table exists)
        cursor.execute("""
            SELECT t.spotify_album_id, t.track_name
            FROM 'Current Positions' t
            JOIN album_averages a ON t.spotify_album_id = a.spotify_album_id
            WHERE LOWER(a.artist_name) = %s;
        """, [artist_name.strip().lower()])
        album_tracks = cursor.fetchall()

        # Map album tracks for processing
        album_tracks_map = {}
        artist_albums = set()
        for row in album_tracks:
            album_id = row[0]
            track_name = row[1]
            if album_id not in album_tracks_map:
                album_tracks_map[album_id] = []
            album_tracks_map[album_id].append({"name": track_name.strip()})
            artist_albums.add(album_id)

        # Filter out live albums
        filtered_albums = [
            album_metadata[album_id]
            for album_id, tracks in album_tracks_map.items()
            if not is_live_album(tracks) and album_id in album_metadata
        ]

        # Deduplicate by track overlap
        unique_albums = deduplicate_by_track_overlap(filtered_albums)

        # --- Step 3: Enrich Data with Rerank and Streak Status ---
        grouped_albums = {}
        today = datetime.now()
        for album_data in unique_albums:
            album_id = album_data["album_id"]
            metadata = album_metadata.get(album_id, {})

            # Calculate rerank status
            rerank_status = "none"
            last_ranked_date = metadata.get("last_ranked_date")
            if last_ranked_date:
                times_ranked = int(metadata.get("times_ranked", 0))
                last_ranked_date = pd.to_datetime(last_ranked_date)
                days_to_add = 45 if times_ranked > 1 else 15
                next_rerank_date = last_ranked_date + timedelta(days=days_to_add)

                if next_rerank_date < today:
                    rerank_status = "overdue"
                elif (next_rerank_date - today).days <= 5:
                    rerank_status = "due"

            # Calculate streak status
            streak_status = "none"
            history = metadata.get("score_history")
            if history:
                try:
                    history = json.loads(history)
                    streak_status = calculate_streak(history)
                except (json.JSONDecodeError, TypeError):
                    logging.warning(f"Could not calculate streak for album: {album_id}")

            # Group albums by their base name (e.g., strip out "(Deluxe)" suffixes)
            base_name = re.sub(r'[\s\-]*(\[[^\]]*\]|\([^\)]*\))[\s\-]*$', '', metadata.get("album_name", "")).strip()
            grouped_albums.setdefault(base_name, []).append({
                "id": album_id,
                "full_name": metadata.get("album_name"),
                "image": metadata.get("album_cover_url"),
                "average_score": metadata.get("average_score"),
                "weighted_average_score": metadata.get("weighted_average_score"),
                "times_ranked": metadata.get("times_ranked"),
                "last_ranked_date": metadata.get("last_ranked_date"),
                "has_prelim_ranks": metadata.get("prelim_rank") not in [None, "", "0", "None"],
                "rerank_status": rerank_status,
                "streak_status": streak_status,
            })

        # --- Step 4: Render the Template ---
        return render_template("select_album.html", artist_name=artist_name, grouped_albums=grouped_albums)

    except Exception as e:
        logging.error(f"Error in load_albums_by_artist_route for artist {artist_name}: {e}", exc_info=True)
        flash("Could not load album list for that artist.", "error")
        return redirect(url_for("profile_page"))

@app.route("/ranking_page")
def ranking_page():
    """
    Route to load the ranking page for albums.
    Loads ranked songs grouped into bins for display.
    """
    try:
        logging.info("Fetching ranked songs for the ranking page...")
        conn = get_db_connection()
        cursor = conn.cursor()

        # Step 1: Fetch all ranked songs and statistics from the database
        cursor.execute("""
            SELECT s.song_id, s.song_name, s.artist_name, s.album_name,
                   s.ranking, s.rank_group, s.spotify_album_id, a.album_cover_url
            FROM 'Current Positions' s
            LEFT JOIN album_averages a ON s.spotify_album_id = a.spotify_album_id
            WHERE s.ranking IS NOT NULL
            ORDER BY s.ranking DESC;
        """)
        ranked_songs = cursor.fetchall()

        # Step 2: Group songs into rank bins
        group_bins = {f"{i / 2:.1f}": [] for i in range(1, 21)}  # Rank groups for 0.5 to 10.0
        group_bins['I'] = {'excellent': [], 'average': [], 'bad': []}  # Special group 'I'

        for row in ranked_songs:
            song_data = {
                'song_id': row[0],
                'song_name': row[1],
                'artist_name': row[2],
                'album_name': row[3],
                'ranking': row[4],
                'rank_group': row[5],
                'spotify_album_id': row[6],
                'album_cover_url': row[7],
            }

            rank_group = str(song_data['rank_group']).strip()
            if rank_group == 'I':
                score = song_data['ranking']
                category = 'average'
                if score == 3.0:
                    category = 'excellent'
                elif score == 1.0:
                    category = 'bad'
                group_bins['I'][category].append(song_data)
            elif rank_group in group_bins:
                group_bins[rank_group].append(song_data)

        # Step 3: Render the page
        return render_template("album.html", group_bins=group_bins)

    except Exception as e:
        logging.error(f"Error loading ranking page: {e}", exc_info=True)
        flash("An error occurred while loading the ranking page.", "error")
        return redirect(url_for('profile'))


@app.route("/view_album", methods=["POST", "GET"])
def view_album():
    """
    View an album's metadata and rankings, replacing Google Sheets logic with Supabase SQL queries.
    """
    try:
        album_id = request.form.get("album_id") or request.args.get("album_id")
        if not album_id:
            flash("Missing album ID.", "warning")
            return redirect(url_for('index'))

        logging.info(f"--- VIEW ALBUM START (Album ID: {album_id}) ---")
        conn = get_db_connection()
        cursor = conn.cursor()

        # 1. Fetch Album Metadata from Database
        cursor.execute("""
            SELECT album_name, artist_name, album_cover_url, release_date, score_history,
                   times_ranked, last_ranked_date
            FROM 'Re-Ranking and Song History (Album Averages)'
            WHERE album_id = %s;
        """, (album_id,))
        album_metadata = cursor.fetchone()
        if not album_metadata:
            flash(f"No album found for ID {album_id}.", "warning")
            return redirect(url_for('index'))

        album_data = {
            'album_name': album_metadata[0],
            'artist_name': album_metadata[1],
            'album_cover_url': album_metadata[2] or "",
            'release_date': album_metadata[3] or "Unknown",
            'score_history': json.loads(album_metadata[4]) if album_metadata[4] else [],
            'times_ranked': album_metadata[5] or 0,
            'last_ranked_date': album_metadata[6]
        }

        is_rerank_mode = album_data['times_ranked'] > 0
        logging.info(f"DEBUG: Re-rank mode for album '{album_data['album_name']}': {is_rerank_mode}")

        # 2. Fetch All Ranked Songs from Database for This Album
        cursor.execute("""
            SELECT song_id, song_name, ranking, rank_group, ranked_date, percentile, placement
            FROM 'Current Positions'
            WHERE spotify_album_id = %s
            ORDER BY ranking DESC;
        """, (album_id,))
        album_songs = cursor.fetchall()

        album_data['songs'] = [
            {
                'song_id': row[0],
                'song_name': row[1],
                'ranking': row[2],
                'rank_group': row[3],
                'ranked_date': row[4],
                'percentile': row[5],
                'placement': row[6]
            }
            for row in album_songs
        ]

        # 3. Fetch All Ranked Songs (Global and Other Albums) for the Leaderboard
        cursor.execute("""
            SELECT s.song_id, s.song_name, s.ranking, s.rank_group, s.spotify_album_id, s.album_name, 
                   s.artist_name, a.album_cover_url
            FROM 'Current Positions' s
            LEFT JOIN album_averages a ON s.spotify_album_id = a.spotify_album_id
            WHERE s.spotify_album_id <> %s
            ORDER BY s.ranking DESC;
        """, (album_id,))
        other_album_songs = cursor.fetchall()

        # Cache album covers for other albums
        album_covers_cache = {
            row[4]: row[7] for row in other_album_songs if row[4] and row[7]
        }

        # 4. Global Rank Groups Organization
        rank_groups_for_js = {f"{i / 2:.1f}": [] for i in range(1, 21)}  # e.g., "0.5", "1.0", ..., "10.0"
        rank_groups_for_js['I'] = {'excellent': [], 'average': [], 'bad': []}  # Special group I

        for row in other_album_songs:
            try:
                rank_group = row[3]
                rank_group_val = f"{float(rank_group):.1f}" if rank_group.replace('.', '', 1).isdigit() else rank_group

                song_data = {
                    'song_id': row[0],
                    'song_name': row[1],
                    'rank_group': rank_group_val,
                    'calculated_score': row[2],
                    'album_id': row[4],
                    'album_name': row[5],
                    'artist_name': row[6],
                    'album_cover_url': album_covers_cache.get(row[4])
                }

                if rank_group == 'I':
                    score = song_data['calculated_score']
                    category = 'average'
                    if score == 3.0:
                        category = 'excellent'
                    elif score == 1.0:
                        category = 'bad'
                    rank_groups_for_js['I'][category].append(song_data)
                elif rank_group_val in rank_groups_for_js:
                    rank_groups_for_js[rank_group_val].append(song_data)

            except Exception as e:
                logging.warning(f"Error grouping song: {row} - {e}")

        # 5. Fetch Preliminary Ranks
        prelim_ranks = {}
        try:
            cursor.execute("""
                SELECT album_id, song_id, prelim_rank
                FROM 'Preliminary Ranks'
                WHERE album_id = %s;
            """, (album_id,))
            prelim_results = cursor.fetchall()

            for row in prelim_results:
                prelim_ranks[row[1]] = row[2]  # Map song_id -> prelim_rank
        except Exception as e:
            logging.warning(f"Error fetching preliminary ranks: {e}")

        # 6. Songs for Left Panel
        songs_for_left_panel = []
        for song in album_data['songs']:
            song_id = song['song_id']
            song_data = {
                **song,
                'prelim_rank': prelim_ranks.get(song_id, ''),
                'existing_score': song['ranking']
            }
            songs_for_left_panel.append(song_data)

        album_data['songs'] = songs_for_left_panel

        # 7. Prepare Album Data for Template
        album_data_for_template = {
            **album_data,
            'album_id': album_id,
            'is_rerank_mode': is_rerank_mode
        }

        logging.info(f"Album ID processed for ranking: {album_id}")
        return render_template('album.html', album=album_data_for_template, rank_groups=rank_groups_for_js)

    except Exception as e:
        logging.error(f"Error in view_album: {e}", exc_info=True)
        flash(f"An unexpected error occurred: {e}", "error")
        return redirect(url_for('profile'))


@app.route("/finalize_rankings", methods=["POST"])
def finalize_rankings():
    """
    Finalize the rankings for a given set of songs.
    Replaces interactions with Google Sheets by inserting/updating into the database.
    """
    try:
        data = request.get_json()  # JSON payload containing rank groups and corresponding songs
        if not data:
            return "Invalid data", 400

        valid_ranks = {str(r) for r in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]}
        for rank_group in data:
            if str(rank_group) not in valid_ranks:
                return f"Invalid rank group for final: {rank_group}", 400

        rows_to_insert = []
        for rank, songs in data.items():
            for position, song_name in enumerate(songs):
                rows_to_insert.append({
                    "song_name": song_name,
                    "ranking": float(rank),
                    "position_in_group": int(position),
                    "ranking_status": "final"
                })
        conn = get_db_connection()
        cursor = conn.cursor()
        # Insert rows into the `song_data` table
        if rows_to_insert:
            cursor.executemany("""
                INSERT INTO song_data (song_name, ranking, position_in_group, ranking_status)
                VALUES (%(song_name)s, %(ranking)s, %(position_in_group)s, %(ranking_status)s)
                ON CONFLICT (song_name, ranking_status)
                DO UPDATE SET ranking = EXCLUDED.ranking, position_in_group = EXCLUDED.position_in_group;
            """, rows_to_insert)
            conn.commit()

        return jsonify({"status": "success"}), 200
    except Exception as e:
        logging.error(f"Error in /finalize_rankings: {e}", exc_info=True)
        return "An error occurred while finalizing rankings.", 500
@app.route("/get_ranked_songs")
def get_ranked_songs():
    """
    Return a JSON array of songs already assigned to the clicked rank‐group.
    Searches for paused rows first, and falls back to finalized rows.
    """
    album_name = request.args.get("album_name", "").strip()
    artist_name = request.args.get("artist_name", "").strip()
    try:
        rank = float(request.args.get("rank", "0"))
    except ValueError:
        return jsonify({'songs': []})

    try:
        # Normalize input values
        album_key = album_name.lower()
        artist_key = artist_name.lower()
        conn = get_db_connection()
        cursor = conn.cursor()

        # First, search for paused rows
        cursor.execute("""
            SELECT song_name
            FROM song_data
            WHERE LOWER(album_name) = %s AND LOWER(artist_name) = %s
            AND ranking_status = 'paused'
            AND ranking BETWEEN %s AND %s;
        """, (album_key, artist_key, rank - 0.25, rank + 0.25))
        paused_songs = [row[0] for row in cursor.fetchall()]

        # If no paused rows, fall back to finalized rows
        if not paused_songs:
            cursor.execute("""
                SELECT song_name
                FROM song_data
                WHERE LOWER(album_name) = %s AND LOWER(artist_name) = %s
                AND ranking_status = 'final'
                AND ranking BETWEEN %s AND %s;
            """, (album_key, artist_key, rank - 0.25, rank + 0.25))
            paused_songs = [row[0] for row in cursor.fetchall()]

        return jsonify({'songs': paused_songs})
    except Exception as e:
        logging.error(f"Error in /get_ranked_songs: {e}", exc_info=True)
        return jsonify({'songs': []}), 500
@app.route("/save_album", methods=["POST"])
def save_album():
    """
    Save paused rankings for an album. Replaces Google Sheets logic with database queries (Supabase SQL).
    """
    try:
        # Step 1: Validate Ranking Status
        status = request.form.get("Ranking Status")
        if status != "paused":
            return "Only paused rankings can be saved here.", 400

        # Step 2: Parse Request Data
        album_name = request.form.get("album_name")
        artist_name = request.form.get("artist_name")
        prelim_ranks = {key.replace("prelim_rank_", ""): float(value)
                        for key, value in request.form.items() if key.startswith("prelim_rank_")}

        if not album_name or not artist_name or not prelim_ranks:
            return "Missing album name, artist name, or preliminary ranks.", 400

        logging.info(f"Saving paused rankings for album='{album_name}', artist='{artist_name}'.")

        # Normalize keys and prepare SQL-safe inputs
        album_name = album_name.strip()
        artist_name = artist_name.strip()
        prelim_ranks = [
            {
                "album_name": album_name,
                "artist_name": artist_name,
                "song_name": song_name,
                "ranking": rank,
                "ranking_status": "paused",
                "ranked_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rank_group": ""  # Preliminary rows have no rank group
            }
            for song_name, rank in prelim_ranks.items()
        ]
        conn = get_db_connection()
        cursor = conn.cursor()

        # Step 3: Remove Existing Paused Rows for This Album and Artist
        cursor.execute("""
            DELETE FROM song_data
            WHERE LOWER(album_name) = %s
              AND LOWER(artist_name) = %s
              AND ranking_status = 'paused';
        """, (album_name.lower(), artist_name.lower()))
        conn.commit()

        # Step 4: Insert New Paused Rows
        cursor.executemany("""
            INSERT INTO song_data (album_name, artist_name, song_name, ranking, ranking_status, ranked_date, rank_group)
            VALUES (%(album_name)s, %(artist_name)s, %(song_name)s, %(ranking)s, %(ranking_status)s, %(ranked_date)s, %(rank_group)s);
        """, prelim_ranks)
        conn.commit()

        return "Paused rankings saved successfully."

    except Exception as e:
        logging.error(f"Error saving paused rankings for album '{album_name}', artist '{artist_name}': {e}", exc_info=True)
        return "An error occurred while saving paused rankings.", 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)