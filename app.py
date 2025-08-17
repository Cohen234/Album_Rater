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

# --- Google Sheets Setup ---
creds_info = json.loads(os.environ['GOOGLE_SERVICE_ACCOUNT_JSON'])
creds = Credentials.from_service_account_info(
    creds_info,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)
client = gspread.authorize(creds)

SPREADSHEET_ID = '15E4b-DWSYP9AzbAzSviqkW-jEOktbimPlmhNIs_d5jc'
SHEET_NAME = "Sheet1"
album_averages_sheet_name = "Album Averages" # Renamed for clarity and consistency
PRELIM_SHEET_NAME = "Preliminary Ranks"

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

def get_album_averages_df(client_gspread, spreadsheet_id, sheet_name):
    try:
        sheet = client_gspread.open_by_key(spreadsheet_id).worksheet(sheet_name)
        logging.debug(f"Successfully opened sheet '{sheet_name}' for Album Averages.")
    except Exception as e:
        logging.error(f"Failed to open sheet '{sheet_name}': {e}", exc_info=True)
        if isinstance(e, gspread.exceptions.WorksheetNotFound):
            logging.info(f"Worksheet '{sheet_name}' not found, attempting to create.")
            try:
                # Ensure 'last_ranked_date' is in the initial header
                header = ['album_id', 'album_name', 'artist_name', 'average_score', 'weighted_average_score',
                          'original_weighted_score', 'previous_weighted_score', 'times_ranked', 'last_ranked_date', 'rerank_history']
                sheet = client_gspread.open_by_key(spreadsheet_id).add_worksheet(title=sheet_name, rows=1,
                                                                                 cols=len(header))
                sheet.append_row(header)
                logging.info(f"Created new sheet: '{sheet_name}'")
            except Exception as create_e:
                logging.critical(f"CRITICAL ERROR: Could not create sheet '{sheet_name}': {create_e}", exc_info=True)
                raise create_e  # Re-raise to stop execution if sheet creation fails
        raise e  # Re-raise the original error if it's not WorksheetNotFound or creation fails

        # ...
    df = get_as_dataframe(sheet, evaluate_formulas=False)

    # THE FIX: This list now contains all the columns your app uses.
    expected_cols = ['album_id', 'album_name', 'artist_name', 'average_score', 'weighted_average_score',
                     'original_weighted_score', 'previous_weighted_score', 'times_ranked',
                     'last_ranked_date', 'rerank_history', 'score_history', 'album_cover_url']

    # THE FIX: Add the if/else block to handle an empty sheet
    if df.empty:
        return pd.DataFrame(columns=expected_cols)
    else:
        for col in expected_cols:
            if col not in df.columns:
                # Initialize both history columns as an empty JSON array string
                df[col] = pd.NA if col not in ['rerank_history', 'score_history'] else '[]'

    # Now we can safely convert types
    for col in ['average_score', 'weighted_average_score', 'original_weighted_score', 'previous_weighted_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['times_ranked'] = pd.to_numeric(df['times_ranked'], errors='coerce').fillna(0).astype(int)

    df = df.fillna({
        'rerank_history': '[]',
        'score_history': '[]',
        'album_cover_url': ''
    })
    return df
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
    """Fetches release dates for a list of album IDs."""
    release_dates = {}
    if not album_ids:
        return release_dates
    # Spotify allows max 20 IDs per batch call
    album_ids = [str(aid) for aid in album_ids]
    for i in range(0, len(album_ids), 20):
        try:
            batch = album_ids[i:i + 20]
            albums_info = sp_instance.albums(batch)
            for album in albums_info['albums']:
                if album:
                    release_dates[album['id']] = album.get('release_date')
        except Exception as e:
            logging.error(f"Could not fetch album release dates batch: {e}")
    return release_dates
def get_dominant_color(image_url):
    try:
        response = requests.get(image_url)
        color_thief = ColorThief(BytesIO(response.content))
        rgb = color_thief.get_color(quality=1)
        return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
    except Exception as e:

        return "#ffffff"
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
    # This function uses `client`, so it must be defined after `client` is initialized
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    return sheet.get_all_records()


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

    try:
        logging.info(f"--- Loading Artist Stats Page for: {artist_name} ---")

        # 1. --- Load All Base Data ---
        main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        all_songs_df = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("")
        all_albums_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)

        # --- Normalize columns: spaces to underscores everywhere ---
        all_songs_df = standardize_columns(all_songs_df)
        all_albums_df = standardize_columns(all_albums_df)

        # --- Type conversions ---
        all_songs_df['Ranking'] = pd.to_numeric(all_songs_df['Ranking'], errors='coerce')
        all_albums_df['weighted_average_score'] = pd.to_numeric(all_albums_df['weighted_average_score'], errors='coerce')

        # --- Keep only FINAL rows ---
        all_songs_df = all_songs_df[all_songs_df['Ranking_Status'].astype(str).str.lower() == 'final']

        # --- Remove duplicates: keep only the latest by Ranked_Date ---
        all_songs_df = all_songs_df.sort_values('Ranked_Date').drop_duplicates(['Song_Name', 'Artist_Name'], keep='last')

        # --- Filter for valid song/artist/ranking ---
        all_songs_df = all_songs_df[
            (all_songs_df['Song_Name'].astype(str).str.strip() != "") &
            (all_songs_df['Artist_Name'].astype(str).str.strip() != "") &
            (all_songs_df['Ranking'].notnull())
        ]

        all_albums_df = all_albums_df[
            (all_albums_df['album_name'].astype(str).str.strip() != "") &
            (all_albums_df['artist_name'].astype(str).str.strip() != "") &
            (all_albums_df['weighted_average_score'].notnull())
        ]

        # --- Assign Universal Rank *after* deduplication ---
        all_songs_df = all_songs_df.sort_values(by='Ranking', ascending=False)
        all_songs_df['Universal_Rank'] = range(1, len(all_songs_df) + 1)

        # --- Duplicate check for debugging ---
        dupes = all_songs_df[all_songs_df.duplicated(subset=['Song_Name', 'Artist_Name'], keep=False)]

        # --- Sort albums and assign Global Rank ---
        all_albums_df = all_albums_df.sort_values(by='weighted_average_score', ascending=False)
        all_albums_df['Global_Rank'] = range(1, len(all_albums_df) + 1)
        artist_query = artist_name.lower().strip()
        # --- Filter for the current artist ---
        artist_songs_df = all_songs_df[
            all_songs_df['Artist_Name'].astype(str).str.lower().str.split(',')
            .apply(lambda artists: artist_query in [a.strip() for a in artists])
        ].copy()

        # Works for comma-separated lists, allowing arbitrary spaces
        artist_albums_df = all_albums_df[
            all_albums_df['artist_name'].astype(str).str.lower().str.split(',')
            .apply(lambda artists: artist_query in [a.strip() for a in artists])
        ].copy()
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
        print(artist_albums_df[['album_id', 'album_name', 'times_ranked']])
        print("visible_album_ids:", visible_album_ids)
        print("mastery_points:", mastery_points, "max:", max_mastery_points)

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

        # 1. Clean names in all DataFrames (do this as early as possible)
        all_songs_df['album_name_clean'] = all_songs_df['Album_Name'].astype(str).str.strip().str.lower()
        # ...other setup...

        # Assign album_name_clean
        all_albums_df['album_name_clean'] = all_albums_df['album_name'].astype(str).str.strip().str.lower()

        # Map ranking dates and scores
        album_first_ranked = all_songs_df.groupby('album_name_clean')['Ranked_Date'].min()
        album_first_score = all_songs_df.groupby('album_name_clean')['Ranking'].first()
        all_albums_df['first_ranked_date'] = all_albums_df['album_name_clean'].map(album_first_ranked)
        all_albums_df['first_score'] = all_albums_df['album_name_clean'].map(album_first_score)
        all_albums_df['first_ranked_date'] = pd.to_datetime(all_albums_df['first_ranked_date'], errors='coerce')

        # Filter albums to only those ever ranked
        all_albums_df = all_albums_df[all_albums_df['first_ranked_date'].notnull()]

        # Filter for current artist (multi-artist logic)
        artist_albums_df = all_albums_df[
            all_albums_df['artist_name'].astype(str).str.lower().str.split(',')
            .apply(lambda artists: artist_query in [a.strip() for a in artists])
        ].copy()
        artist_albums_df['album_name_clean'] = artist_albums_df['album_name'].astype(str).str.strip().str.lower()

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
        artist_songs_df['Rank_Group_Str'] = artist_songs_df['Rank_Group'].astype(str)
        polar_data_series = pd.Series(index=all_rank_groups + ['I'], dtype=int).fillna(0)
        song_counts = artist_songs_df['Rank_Group_Str'].value_counts()
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
            top_song_cover = top_song_row.get('album_cover_url', '')
            top_song_link = url_for(
                'album_page',
                artist_name=artist_name,
                album_name=quote_plus(top_song_row['album_name']),
                album_id=top_song_row['album_id']
            ) if top_song_row.get('album_id') else "#"
            low_song_name = low_song_row['Song_Name']
            low_song_score = low_song_row['Ranking']
            low_song_cover = low_song_row.get('album_cover_url', '')
            low_song_link = url_for(
                'album_page',
                artist_name=artist_name,
                album_name=quote_plus(low_song_row['album_name']),
                album_id=low_song_row['album_id']
            ) if low_song_row.get('album_id') else "#"
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
            row['Song Name'] = clean_title(row.get('Song_Name', ''))  # Ensure this key is present
            row['Universal Rank'] = row.get('Universal_Rank', '')
            row['Artist Rank'] = row.get('Artist_Rank', '')
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
        return f"An error occurred: {e}", 500

from flask import abort
from urllib.parse import unquote

@app.route("/artist/<artist_name>/album/<path:album_name>/<album_id>")
def album_page(artist_name, album_name, album_id):
    album_name = unquote(album_name)

    album_data = get_album_data(artist_name, album_name, album_id)
    if not album_data:
        print("ALBUM DATA NOT FOUND!")
        abort(404)


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
    try:
        logging.info(f"Received album_id: {album_id}")
        # 1. Load data
        main_df = get_as_dataframe(client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)).fillna("")
        logging.info(f"main_df columns: {main_df.columns.tolist()}; shape: {main_df.shape}")
        averages_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)
        logging.info(f"averages_df columns: {averages_df.columns.tolist()}; shape: {averages_df.shape}")

        # 2. Find the specific album's data
        album_stats = averages_df[averages_df['album_id'].astype(str) == str(album_id)]
        logging.info(f"album_stats shape: {album_stats.shape}")
        if album_stats.empty:
            return jsonify({'error': 'Album not found in averages sheet.'}), 404

        album_stats = album_stats.iloc[0]
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

        placement_series = averages_df.index[averages_df['album_id'].astype(str) == str(album_id)]
        leaderboard_placement = int(placement_series[0] + 1) if not placement_series.empty else 'N/A'

        last_ranked_date = pd.to_datetime(album_stats.get('last_ranked_date'), errors='coerce')
        times_ranked_val = pd.to_numeric(album_stats.get('times_ranked'), errors='coerce')
        times_ranked = 0 if pd.isna(times_ranked_val) else int(times_ranked_val)
        next_rerank_date = 'N/A'
        if pd.notna(last_ranked_date):
            days_to_add = 45 if times_ranked > 1 else 15
            next_rerank_date = (last_ranked_date + pd.Timedelta(days=days_to_add)).strftime('%Y-%m-%d')

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
            'rerank_history': rerank_history  # Pass the history to the frontend
        }
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error in get_album_stats for {album_id}: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred: {e}'}), 500


@app.route("/submit_rankings", methods=["POST"])
def submit_rankings():
    global sp, client
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
                prelim_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(PRELIM_SHEET_NAME)
                prelim_df = get_as_dataframe(prelim_sheet, evaluate_formulas=False).fillna("")
            except gspread.exceptions.WorksheetNotFound:
                prelim_sheet = client.open_by_key(SPREADSHEET_ID).add_worksheet(title=PRELIM_SHEET_NAME, rows=1, cols=8)
                prelim_sheet.append_row(
                    ['album_id', 'album_name', 'artist_name', 'album_cover_url', 'song_id', 'song_name', 'prelim_rank',
                     'timestamp'])
                prelim_df = pd.DataFrame()

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
            set_with_dataframe(prelim_sheet, final_prelim_df, include_index=False, resize=True)
            logging.info(f"Successfully saved draft for album {album_id}.")

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
                averages_df_before = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)
                if not averages_df_before.empty:
                    old_album_data = averages_df_before[averages_df_before['album_id'].astype(str) == str(album_id)]
                    if not old_album_data.empty:
                        old_score = old_album_data.iloc[0]['weighted_average_score']
                        # Sort to find old placement
                        averages_df_before.sort_values(by='weighted_average_score', ascending=False, inplace=True)
                        averages_df_before.reset_index(drop=True, inplace=True)
                        old_placement_series = averages_df_before.index[averages_df_before['album_id'].astype(str) == str(album_id)]
                        old_placement = int(old_placement_series[0] + 1) if not old_placement_series.empty else 1

            # --- 4. Update Google Sheets with New Final Rankings ---
            main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
            main_df = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("")

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

                # Try to preserve the original 'Ranked Date' if it exists for this song/album/artist
                existing_row = main_df[
                    (main_df['Spotify Song ID'].astype(str) == song_id) &
                    (main_df['Album Name'] == ranked_song_data.get('album_name')) &
                    (main_df['Artist Name'] == ranked_song_data.get('artist_name'))
                    ]
                if not existing_row.empty and existing_row.iloc[0].get('Ranked Date'):
                    ranked_date = existing_row.iloc[0]['Ranked Date']
                else:
                    ranked_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                new_final_rows_data.append({
                    'Album Name': ranked_song_data.get('album_name'),
                    'Artist Name': ranked_song_data.get('artist_name'),
                    'Spotify Album ID': ranked_song_data.get('album_id'),
                    'Song Name': details.get('name', ranked_song_data.get('song_name')),
                    'Ranking': ranked_song_data.get('calculated_score', 0.0),
                    'Duration (ms)': details.get('duration_ms', 0),
                    'Ranking Status': 'final',
                    'Ranked Date': ranked_date,  # <<---------------------- FIXED
                    'Position In Group': str(ranked_song_data.get('position_in_group', '')),
                    'Rank Group': str(ranked_song_data.get('rank_group')),
                    'Spotify Song ID': song_id,
                })

            final_main_df = pd.concat([main_df_filtered, pd.DataFrame(new_final_rows_data)],
                                      ignore_index=True) if new_final_rows_data else main_df_filtered
            set_with_dataframe(main_sheet, final_main_df, include_index=False, resize=True)
            logging.info("Updated main ranking sheet.")

            # Recalculate and Update Album Averages Sheet
            df_for_calc = final_main_df.copy()
            df_for_calc['Ranking'] = pd.to_numeric(df_for_calc['Ranking'], errors='coerce')
            df_for_calc['Duration (ms)'] = pd.to_numeric(df_for_calc['Duration (ms)'], errors='coerce')
            df_for_calc.dropna(subset=['Ranking', 'Duration (ms)'], inplace=True)
            df_for_calc_no_interludes = df_for_calc[df_for_calc['Rank Group'].astype(str) != 'I']

            if not df_for_calc_no_interludes.empty:
                simple_averages = df_for_calc_no_interludes.groupby('Spotify Album ID')['Ranking'].mean().round(6)

                def weighted_avg(group):
                    total_duration = group['Duration (ms)'].sum()
                    return ((group['Ranking'] * group['Duration (ms)']).sum() / total_duration) if total_duration > 0 else \
                    group['Ranking'].mean()

                weighted_averages = df_for_calc_no_interludes.groupby('Spotify Album ID').apply(weighted_avg).round(6)
                sorted_scores = weighted_averages.sort_values(ascending=False)
                new_placement = sorted_scores.index.get_loc(album_id) + 1

                total_albums = len(sorted_scores)

                album_averages_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)
                album_info_map = df_for_calc[['Spotify Album ID', 'Album Name', 'Artist Name']].drop_duplicates(
                    'Spotify Album ID').set_index('Spotify Album ID').to_dict('index')

                # THE FIX: This loop now updates the score history for ALL albums
                for album_id_to_update, new_weighted_avg in weighted_averages.items():
                    new_simple_avg = simple_averages.get(album_id_to_update)
                    existing_rows = album_averages_df[album_averages_df['album_id'].astype(str) == str(album_id_to_update)]

                    if not existing_rows.empty:
                        idx = existing_rows.index[0]

                        # --- Update Score History for EVERY album ---
                        try:
                            history_str = album_averages_df.at[idx, 'score_history']
                            score_history = json.loads(history_str) if history_str and pd.notna(history_str) else []
                            score_history.append(float(new_weighted_avg))
                            album_averages_df.at[idx, 'score_history'] = json.dumps(score_history)
                        except (json.JSONDecodeError, TypeError) as e:
                            logging.error(f"Error updating score_history for {album_id_to_update}: {e}")

                        # --- Logic for the SUBMITTED album ---
                        if album_id_to_update == album_id:
                            album_averages_df.at[idx, 'previous_weighted_score'] = new_weighted_avg
                            album_averages_df.at[idx, 'times_ranked'] = int(
                                album_averages_df.at[idx, 'times_ranked'] or 0) + 1

                            # Update history with the NEW score
                            if is_rerank:
                                try:
                                    history_str = album_averages_df.at[idx, 'rerank_history']
                                    rerank_history = json.loads(history_str) if history_str and pd.notna(
                                        history_str) else []
                                    # Append the new score that was just calculated
                                    rerank_history.append({
                                        'date': datetime.now().strftime('%Y-%m-%d'),
                                        'score': float(new_weighted_avg),
                                        # THE FIX: Add the placement at the time of the event
                                        'placement': f"{new_placement}/{total_albums}"
                                    })
                                    album_averages_df.at[idx, 'rerank_history'] = json.dumps(rerank_history)
                                except (json.JSONDecodeError, TypeError) as e:
                                    logging.error(f"Error updating rerank history for {album_id}: {e}")

                            # Now, update the main scores and the last ranked date
                        album_averages_df.at[idx, 'average_score'] = new_simple_avg
                        album_averages_df.at[idx, 'weighted_average_score'] = new_weighted_avg
                        album_averages_df.at[idx, 'last_ranked_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


                    else:
                        # Logic for adding a brand new album
                        info = album_info_map.get(album_id_to_update)
                        if info:
                            # THE FIX: Create the initial history event without placement data,
                            # as it's not available yet. It will be added on the first re-rank.
                            initial_history = [{
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'score': float(new_weighted_avg)
                            }]
                            new_row = pd.DataFrame([{
                                'album_id': album_id_to_update, 'album_name': info['Album Name'],
                                'artist_name': info['Artist Name'], 'average_score': new_simple_avg,
                                'weighted_average_score': new_weighted_avg,
                                'original_weighted_score': new_weighted_avg,
                                'previous_weighted_score': new_weighted_avg, 'times_ranked': 1,
                                'last_ranked_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'rerank_history': json.dumps(initial_history),
                                'score_history': json.dumps([float(new_weighted_avg)]),
                                'album_cover_url': album_cover_url
                            }])
                            album_averages_df = pd.concat([album_averages_df, new_row], ignore_index=True)

                set_with_dataframe(client.open_by_key(SPREADSHEET_ID).worksheet(album_averages_sheet_name),
                                   album_averages_df, include_index=False, resize=True)
                logging.info("Successfully recalculated and saved all averages.")

            # --- 5. Get "AFTER" data for the animation ---
            averages_df_after = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)
            averages_df_after['weighted_average_score'] = pd.to_numeric(averages_df_after['weighted_average_score'],
                                                                        errors='coerce')
            averages_df_after.dropna(subset=['weighted_average_score'], inplace=True)
            averages_df_after.sort_values(by='weighted_average_score', ascending=False, inplace=True)
            averages_df_after.reset_index(drop=True, inplace=True)

            new_album_data = averages_df_after[averages_df_after['album_id'].astype(str) == str(album_id)]
            if new_album_data.empty:
                return jsonify({'status': 'error', 'message': 'Could not find album after ranking.'}), 500

            new_score = float(new_album_data.iloc[0]['weighted_average_score'])
            times_ranked = int(new_album_data.iloc[0]['times_ranked'])
            new_placement_series = averages_df_after.index[averages_df_after['album_id'].astype(str) == str(album_id)]
            new_placement = int(new_placement_series[0] + 1) if not new_placement_series.empty else 1
            total_albums = len(averages_df_after)
            dominant_color = get_dominant_color(album_cover_url)
            print("Returning animation_data:", {
                'album_name': album_name,
                'artist_name': artist_name,
                'album_cover_url': album_cover_url,
                'final_score': new_score,
                'final_rank': new_placement,
                'total_albums': total_albums,
                'dominant_color': dominant_color,
                'album_id': album_id
            })

            # --- 6. Return the correct JSON for the frontend ---
            if is_rerank:
                return jsonify({
                    'status': 'success',
                    'rerank_animation_data': {
                        'album_name': album_name, 'artist_name': artist_name, 'album_cover_url': album_cover_url,
                        'old_score': old_score, 'new_score': new_score,
                        'old_placement': old_placement, 'new_placement': new_placement,
                        'total_albums': total_albums, 'times_ranked': times_ranked,
                        'dominant_color': dominant_color
                    }
                })
            else:
                return jsonify({
                    'status': 'success',
                    'animation_data': {
                        'album_name': album_name,
                        'artist_name': artist_name,
                        'album_cover_url': album_cover_url,
                        'final_score': new_score,
                        'final_rank': new_placement,
                        'total_albums': total_albums,
                        'dominant_color': dominant_color,
                        'album_id': album_id  # <--- ADD THIS LINE
                    }
                })
        

    except Exception as e:
        logging.critical(f"\n🔥 CRITICAL ERROR in /submit_rankings: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f"An unexpected error occurred: {e}"}), 500

def get_album_data(artist_name, album_name, album_id):
    import pandas as pd
    import json


    def format_seconds(seconds):
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h}:{m:02}:{s:02}"
        else:
            return f"{m}:{s:02}"

    # Load dataframes
    main_df = get_as_dataframe(client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)).fillna("")

    averages_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)


    # Clean for matching (album names may have case/space differences)
    album_name_clean = album_name.strip().lower()
    artist_name_clean = artist_name.strip().lower()

    # Find the album row
    album_row = averages_df[
        (averages_df['album_id'].astype(str) == str(album_id)) &
        (averages_df['album_name'].str.strip().str.lower() == album_name_clean) &
        (averages_df['artist_name'].str.strip().str.lower() == artist_name_clean)
        ]
    if album_row.empty:
        return None
    album_row = album_row.iloc[0]
    album_id = album_row['album_id']
    album_cover_url = album_row.get("album_cover_url", "")

    # Sort all albums by score to assign global ranks
    averages_df = averages_df.sort_values(by='weighted_average_score', ascending=False).reset_index(drop=True)
    placement_series = averages_df.index[averages_df['album_id'].astype(str) == str(album_id)]
    global_album_rank = int(placement_series[0] + 1) if not placement_series.empty else None

    # Get all songs for this album
    album_songs_df = main_df[
        (main_df['Album Name'].str.strip().str.lower() == album_name_clean) &
        (main_df['Artist Name'].str.strip().str.lower() == artist_name_clean) &
        (main_df['Ranking Status'].astype(str).str.lower() == 'final')  # <-- add this filter
        ].copy()
    actual_songs_df = album_songs_df[album_songs_df['Rank Group'] != "I"].copy()
    interlude_songs_df = album_songs_df[album_songs_df['Rank Group'] == "I"].copy()
    actual_songs_df['Ranking'] = pd.to_numeric(actual_songs_df['Ranking'], errors='coerce')
    actual_songs_df = actual_songs_df[actual_songs_df['Ranking'].notnull()]


    release_date = None
    album_length = ""
    album_length_sec = 0
    try:
        album_info = load_album_data(sp, album_id)
        track_order_map = {}
        if album_info and 'songs' in album_info:
            for i, song in enumerate(album_info['songs']):
                song_name = song['song_name'].strip().lower()
                track_order_map[song_name] = i + 1

        album_songs_df['track_order'] = album_songs_df['Song Name'].str.strip().str.lower().map(track_order_map)

        if album_songs_df['track_order'].isnull().any():
            if 'Position In Group' in album_songs_df.columns:
                album_songs_df['track_order'] = album_songs_df['track_order'].fillna(
                    album_songs_df['Position In Group'])
            else:
                album_songs_df['track_order'] = album_songs_df['track_order'].fillna(
                    pd.Series(range(1, len(album_songs_df) + 1), index=album_songs_df.index))

        album_songs_df = album_songs_df.sort_values('track_order')

        release_date = album_info.get('release_date', None)
        # Album length calculation from Spotify...
        if album_info.get('songs'):
            total_duration_ms = 0
            for song in album_info['songs']:
                if 'duration_ms' in song:
                    total_duration_ms += int(song['duration_ms'])
            album_length_sec = total_duration_ms // 1000
            album_length = format_seconds(album_length_sec)
        if not album_cover_url and album_info.get('album_cover_url'):
            album_cover_url = album_info.get('album_cover_url')
    except Exception:
        pass

    # ---- ADD THIS Fallback for Release Date ----
    if not release_date or release_date == "":
        # Try album_row
        release_date = (
                album_row.get('release_date', "") or
                album_row.get('Release_Date', "") or
                album_row.get('releaseDate', "")
        )
        # Try release date lookup from artist page logic
        if not release_date:
            release_dates_map = get_album_release_dates(sp, [album_id])
            release_date = release_dates_map.get(album_id, "")  # Use spreadsheet value if present

    # Fallback album length calculation from your main sheet if not found above
    if not album_length or album_length_sec == 0:
        album_songs_df['Duration (ms)'] = pd.to_numeric(album_songs_df.get('Duration (ms)', 0), errors='coerce').fillna(0)
        total_duration_ms = album_songs_df['Duration (ms)'].sum()
        album_length_sec = int(total_duration_ms // 1000)
        album_length = format_seconds(album_length_sec) if album_length_sec else ""

    # Song stats
    album_songs_df['Ranking'] = pd.to_numeric(album_songs_df['Ranking'], errors='coerce')
    album_songs_df = album_songs_df[album_songs_df['Ranking'].notnull()]

    avg_song_score = actual_songs_df['Ranking'].mean() if not actual_songs_df.empty else 0
    median_song_score = actual_songs_df['Ranking'].median() if not actual_songs_df.empty else 0
    std_song_score = actual_songs_df['Ranking'].std() if not actual_songs_df.empty else 0

    # Top/lowest 3 songs
    top_3_songs = actual_songs_df.sort_values('Ranking', ascending=False).head(3)[['Song Name', 'Ranking']].to_dict(
        'records')
    for song in top_3_songs:
        song['title'] = song.pop('Song Name')
        song['score'] = song.pop('Ranking')
    lowest_row = actual_songs_df.sort_values('Ranking', ascending=True).head(1)
    lowest_song = {'title': lowest_row.iloc[0]['Song Name'],
                   'score': lowest_row.iloc[0]['Ranking']} if not lowest_row.empty else None

    most_improved_song = {'title': '', 'delta': 0}
    worst_improved_song = None
    max_delta = float('-inf')
    min_delta = float('inf')
    for song_name, group in album_songs_df.groupby('Song Name'):
        group_sorted = group.sort_values('Ranked Date')
        if len(group_sorted) > 1:
            first_rank = group_sorted.iloc[0]['Ranking']
            last_rank = group_sorted.iloc[-1]['Ranking']
            delta = last_rank - first_rank
            if delta > max_delta:
                max_delta = delta
                most_improved_song = {'title': song_name, 'delta': delta}
            if delta < min_delta:
                min_delta = delta
                worst_improved_song = {'title': song_name, 'delta': delta}
    if 'Duration (ms)' in album_songs_df.columns:
        album_songs_df['duration_ms'] = pd.to_numeric(album_songs_df['Duration (ms)'], errors='coerce').fillna(180000)
    elif 'duration_ms' in album_songs_df.columns:
        album_songs_df['duration_ms'] = pd.to_numeric(album_songs_df['duration_ms'], errors='coerce').fillna(180000)
    else:
        album_songs_df['duration_ms'] = 180000  # fallback: 3min per song
    album_songs_df['duration_sec'] = album_songs_df['duration_ms'] / 1000
    print(album_songs_df[['Song Name', 'duration_ms', 'duration_sec']])

    # Sort by Position In Group or track number, fallback to index

    # Calculate song start and midpoint times
    song_starts = []
    current_time = 0
    for idx, row in album_songs_df.iterrows():
        song_starts.append(current_time)
        duration_sec = row['duration_sec'] or 0
        current_time += duration_sec
    all_songs_df = main_df.copy()
    all_songs_df['Ranking'] = pd.to_numeric(all_songs_df['Ranking'], errors='coerce')
    all_songs_df = all_songs_df[all_songs_df['Ranking'].notnull()]
    all_songs_df = all_songs_df.sort_values('Ranking', ascending=False).reset_index(drop=True)
    all_songs_df['global_rank'] = all_songs_df.index + 1
    album_songs = []
    for (idx, row), start_sec in zip(album_songs_df.iterrows(), song_starts):
        is_interlude = (row.get('Rank Group', row.get('Rank Group', "")) == "I")
        song_name = row['Song Name']
        track_number = int(row.get('Position In Group', idx + 1))
        # Find global rank for this song (and artist to be safe)
        song_match = all_songs_df[
            (all_songs_df['Song Name'] == song_name) &
            (all_songs_df['Artist Name'].str.strip().str.lower() == artist_name_clean)
            ]
        global_rank = int(song_match.iloc[0]['global_rank']) if not song_match.empty else None

        # Delta in the last 7 days (your logic)
        history = album_songs_df[
            (album_songs_df['Song Name'] == song_name)
        ].sort_values('Ranked Date')
        delta_7d = 0.0
        score_history = []
        if len(history) > 1:
            last_7d = history[history['Ranked Date'] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]
            old = history[history['Ranked Date'] < (pd.Timestamp.now() - pd.Timedelta(days=7))]
            delta_7d = (last_7d['Ranking'].mean() if not last_7d.empty else row['Ranking']) - (
                old['Ranking'].mean() if not old.empty else row['Ranking'])
            score_history = list(zip(history['Ranked Date'].astype(str), history['Ranking']))

        album_songs.append({
            'track_number': int(row['track_order']),
            'title': song_name,
            'score': row['Ranking'],
            'delta_7d': delta_7d,
            'score_history_str': "; ".join([f"{d}: {s:.2f}" for d, s in score_history]),
            'start_min': start_sec/60,
            'length': row['duration_sec'],
            'global_rank': global_rank,
            'is_interlude': is_interlude,
        })
    artist_songs_df = main_df[
        (main_df['Artist Name'].str.strip().str.lower() == artist_name_clean)
    ]
    artist_songs_df['Ranking'] = pd.to_numeric(artist_songs_df['Ranking'], errors='coerce')
    artist_actual_songs_df = artist_songs_df[artist_songs_df['Rank Group'] != "I"].copy()
    artist_actual_songs_df = artist_actual_songs_df[artist_actual_songs_df['Ranking'].notnull()]
    artist_avg_song_score = artist_actual_songs_df['Ranking'].mean() if not artist_actual_songs_df.empty else 0

    main_df['Ranking'] = pd.to_numeric(main_df['Ranking'], errors='coerce')
    global_actual_songs_df = main_df[main_df['Rank Group'] != "I"].copy()
    global_actual_songs_df = global_actual_songs_df[global_actual_songs_df['Ranking'].notnull()]
    global_avg_song_score = global_actual_songs_df['Ranking'].mean() if not global_actual_songs_df.empty else 0

    # Album ranking timeline
    rerank_history = album_row.get('rerank_history', '[]')
    try:
        timeline = json.loads(rerank_history)
    except Exception:
        timeline = []
    album_ranking_timeline = [{'date': x['date'], 'rank': x.get('placement')} for x in timeline if 'date' in x]
    print("Album songs times:")
    for song in album_songs:
        print(song['title'], song['start_min'], song['score'])
    # Calculate last song's start in minutes (already stored)
    last_song_start_min = album_songs[-1]['start_min']
    # Calculate last song's duration in minutes
    last_song_duration_min = album_songs_df.iloc[-1]['duration_sec'] / 60 if not album_songs_df.empty else 0
    # The end of the album in minutes
    last_song_end_min = last_song_start_min + last_song_duration_min



    album_data = {
        'album_name': album_row['album_name'],
        'artist_name': album_row['artist_name'],
        'album_cover_url': album_cover_url,
        'release_date': release_date or "Unknown",  # Provide something always
        'album_length': album_length or "",
        'album_length_sec': album_length_sec or 0,
        'album_score': album_row.get('weighted_average_score', 0),
        'avg_song_score': avg_song_score,
        'median_song_score': median_song_score,
        'std_song_score': std_song_score,  # Set as appropriate or remove if not needed
        'top_3_songs': top_3_songs,
        'lowest_song': lowest_song,
        'most_improved_song': most_improved_song,
        'worst_improved_song': worst_improved_song,
        'artist_avg_song_score': artist_avg_song_score,
        'global_avg_song_score': global_avg_song_score,
        'album_ranking_timeline': album_ranking_timeline,
        'album_ranking_delta': 0,
        'last_song_end_min':last_song_end_min,
        "global_album_rank": global_album_rank,
        'album_songs': album_songs,
    }
    return album_data
@app.route("/search_albums", methods=["GET"])
def search_albums():
    """
    AJAX endpoint for album search.
    Accepts ?q=search_term and returns up to 10 albums with: album_id, album_name, artist_name, album_cover_url
    """
    query = request.args.get("q", "").strip().lower()
    if not query or len(query) < 2:
        return jsonify([])  # Empty for short/no query

    # Load album averages df
    all_albums_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)
    all_albums_df = all_albums_df.dropna(subset=["album_id", "album_name", "artist_name"])

    # Match by album_name or artist_name substring (case insensitive)
    mask = (
        all_albums_df["album_name"].astype(str).str.lower().str.contains(query)
        | all_albums_df["artist_name"].astype(str).str.lower().str.contains(query)
    )
    results = all_albums_df[mask].head(10)  # Limit to top 10 results

    albums = []
    for _, row in results.iterrows():
        albums.append({
            "album_id": str(row["album_id"]),
            "album_name": row["album_name"],
            "artist_name": row["artist_name"],
            "album_cover_url": row.get("album_cover_url", ""),
        })
    return jsonify(albums)
from urllib.parse import quote_plus

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

@app.route("/compare_albums", methods=["GET"])
def compare_albums():
    try:
        album_ids = request.args.getlist("album_ids")
        print("Compare called with:", album_ids)
        all_data = []
        colors = ["#1DB954", "#e74c3c", "#3498db", "#ffd700"]

        averages_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)
        print("Averages DF shape:", averages_df.shape)
        for i, album_id in enumerate(album_ids[:4]):
            album_row = averages_df[averages_df['album_id'].astype(str) == str(album_id)]
            if album_row.empty:
                print(f"Album ID not found: {album_id}")
                continue
            album_row = album_row.iloc[0]
            artist_name = album_row['artist_name']
            album_name = album_row['album_name']

            album_data = get_album_data(artist_name, album_name, album_id)
            if not album_data:
                print(f"get_album_data failed for: {album_name} ({album_id})")
                continue

            # --- Fix: Calculate each song's start_min ---
            # Assume album_data['album_songs'] is a list of dicts with 'length' (as "min:sec" string or seconds int)
            songs_with_start = []
            cumulative_seconds = 0
            for song in album_data['album_songs']:
                # If length is a string "3:22"
                if isinstance(song['length'], str) and ':' in song['length']:
                    mins, secs = map(int, song['length'].split(":"))
                    song_seconds = mins * 60 + secs
                else:
                    # If it's already in seconds
                    song_seconds = int(song['length'])

                start_min = cumulative_seconds / 60
                # Add start_min to song dict
                song_with_start = song.copy()
                song_with_start['start_min'] = start_min
                songs_with_start.append(song_with_start)
                cumulative_seconds += song_seconds

            # Filter out interludes for graph and stats
            main_songs = [song for song in songs_with_start if not song.get('is_interlude', False)]

            # Runtime graph points: only main songs
            points = [
                {
                    "x": song['start_min'],
                    "y": song['score'],
                    "song": song['title'],
                    "album": album_data['album_name'],
                }
                for song in main_songs
            ]

            # Boxplot data: only main songs
            song_scores = [song['score'] for song in main_songs]

            # Best/worst song: only main songs
            best_song = max(main_songs, key=lambda s: s['score']) if main_songs else None
            worst_song = min(main_songs, key=lambda s: s['score']) if main_songs else None

            # Package everything for frontend
            all_data.append({
                "id": album_id,
                "name": album_data['album_name'],
                "artist": album_data['artist_name'],
                "album_cover_url": album_data['album_cover_url'],
                "release_date": album_data['release_date'],
                "album_score": album_data['album_score'],
                "avg_song_score": album_data['avg_song_score'],
                "placement": album_data['global_album_rank'],
                "std_dev": album_data['std_song_score'],
                "length": album_data['album_length'],
                "color": colors[i % len(colors)],
                "points": points,
                "album_length_sec": album_data['album_length_sec'],
                "song_scores": song_scores,
                "best_song": {"title": best_song['title'], "score": best_song['score']} if best_song else {"title": "", "score": None},
                "worst_song": {"title": worst_song['title'], "score": worst_song['score']} if worst_song else {"title": "", "score": None},
            })
        print("Compare result:", all_data)
        return jsonify({"albums": all_data})
    except Exception as e:
        print(f"Error in /compare_albums: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/rerank_success')
def rerank_success():
    # Get all the data passed from the redirect
    album_name = request.args.get('album_name')
    artist_name = request.args.get('artist_name')
    album_cover_url = request.args.get('album_cover_url')
    old_score = float(request.args.get('old_score', 0))
    new_score = float(request.args.get('new_score', 0))
    old_placement = int(request.args.get('old_placement', 1))
    new_placement = int(request.args.get('new_placement', 1))
    total_albums = int(request.args.get('total_albums', 1))
    times_ranked = int(request.args.get('times_ranked', 2))
    dominant_color = request.args.get('dominant_color', '#121212')

    # Determine colors for the scores
    old_score_color = 'green' if old_score >= 7 else 'yellow' if old_score >= 4 else 'red'
    new_score_color = 'green' if new_score >= 7 else 'yellow' if new_score >= 4 else 'red'

    return render_template(
        'rerank_success.html',
        album_name=album_name, artist_name=artist_name, album_cover_url=album_cover_url,
        old_score=old_score, new_score=new_score, old_placement=old_placement,
        new_placement=new_placement, total_albums=total_albums, times_ranked=times_ranked,
        dominant_color=dominant_color, old_score_color=old_score_color, new_score_color=new_score_color
    )
@app.route('/')
def index():
    return render_template('index.html')
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
    artist_name = request.form.get('artist_name')
    if artist_name:
        # Redirect to the main artist dashboard
        return redirect(url_for('artist_page_v2', artist_name=artist_name))
    else:
        flash("Please enter an artist name.")
        return redirect(url_for('index'))
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
    artist_name = request.form.get("artist_name") or request.args.get("artist_name")

    if not artist_name:
        flash("Artist name not provided. Please search for an artist.")
        return redirect(url_for('index'))

    logging.info(f"\n--- LOADING ALBUM LIST FOR ARTIST: {artist_name} ---")
    try:
        albums_from_spotify = get_albums_by_artist(sp, artist_name)

        # Fetch tracks and filter out live albums
        filtered_albums = []
        for album_data in albums_from_spotify:
            full_name = album_data.get("name")
            album_id_spotify = album_data.get("id")
            try:
                tracks = sp.album_tracks(album_id_spotify)['items']
                if is_live_album(tracks):
                    continue  # Skip "live" albums
                album_data['tracks'] = tracks
                filtered_albums.append(album_data)
            except Exception as e:
                logging.warning(f"Could not fetch tracks for {full_name}: {e}")

        # Remove compilations by track overlap
        studio_albums = deduplicate_by_track_overlap(filtered_albums)
        # Add this after deduplication:
        from album_blocklist import load_blocklist_for_artist

        blocklist = load_blocklist_for_artist(artist_name)
        studio_albums = [a for a in studio_albums if a['id'] not in blocklist]

        try:
            album_averages_df = get_album_averages_df(client, SPREADSHEET_ID, "Album Averages")
        except Exception as e:
            logging.error(f"Error loading Album Averages DataFrame: {e}", exc_info=True)
            flash(f"Error loading album averages data: {e}", "error")
            return redirect(url_for('index'))

        prelim_ranked_albums_ids = set()
        try:
            prelim_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(PRELIM_SHEET_NAME)
            prelim_sheet_data = get_as_dataframe(prelim_sheet, evaluate_formulas=False).fillna("")
            if not prelim_sheet_data.empty and 'artist_name' in prelim_sheet_data.columns:
                current_artist_prelim_ranks = prelim_sheet_data[
                    prelim_sheet_data["artist_name"].astype(str).str.strip().str.lower() == artist_name.strip().lower()
                    ]
                for _, row in current_artist_prelim_ranks.iterrows():
                    album_id_p = str(row.get('album_id', '')).strip()
                    prelim_rank_value = row.get('prelim_rank')
                    if album_id_p and str(prelim_rank_value).strip() not in ["", "0", "0.0", "None"]:
                        prelim_ranked_albums_ids.add(album_id_p)
        except gspread.exceptions.WorksheetNotFound:
            logging.warning(f"Sheet '{PRELIM_SHEET_NAME}' not found. Cannot check for paused albums.")
        except Exception as e:
            logging.error(f"Error loading preliminary ranks: {e}", exc_info=True)

        # Create a dictionary for quick lookup of averages/times ranked
        album_metadata = {}
        if not album_averages_df.empty:
            for _, row in album_averages_df.iterrows():
                album_id_from_sheet = str(row.get("album_id", "")).strip()
                if album_id_from_sheet:
                    album_metadata[album_id_from_sheet] = row.to_dict()

        grouped_albums = {}
        today = datetime.now()
        for album_data in studio_albums:  # <--- USE studio_albums HERE!
            full_name = album_data.get("name")
            album_id_spotify = album_data.get("id")

            # Create a "base name" by removing phrases in parentheses like (Deluxe), (Remastered), etc.
            base_name = full_name
            base_name = re.sub(r'[\s\-]*(\[[^\]]*\]|\([^\)]*\))[\s\-]*$', '', base_name).strip()
            while re.search(r'(\[[^\]]*\]|\([^\)]*\))[\s\-]*$', base_name):
                base_name = re.sub(r'[\s\-]*(\[[^\]]*\]|\([^\)]*\))[\s\-]*$', '', base_name).strip()

            # Get stats for this specific edition
            metadata = album_metadata.get(album_id_spotify, {})
            has_prelim_ranks = album_id_spotify in prelim_ranked_albums_ids
            rerank_status = 'none'
            if metadata.get('last_ranked_date') and pd.notna(metadata.get('last_ranked_date')):
                last_ranked_date = pd.to_datetime(metadata['last_ranked_date'])
                times_ranked = int(metadata.get('times_ranked', 0))
                days_to_add = 45 if times_ranked > 1 else 15
                next_rerank_date = last_ranked_date + pd.Timedelta(days=days_to_add)

                if next_rerank_date < today:
                    rerank_status = 'overdue'
                elif (next_rerank_date - today).days <= 5:
                    rerank_status = 'due'

            streak_status = 'none'
            if metadata:
                try:
                    history = json.loads(metadata.get('score_history', '[]'))
                    streak_status = calculate_streak(history)
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logging.warning(f"Could not calculate streak for {metadata.get('album_id')}: {e}")
                    streak_status = 'none'

            edition_data = {
                "id": album_id_spotify,
                "full_name": full_name,
                "image": album_data.get("image"),
                "average_score": metadata.get("average_score"),
                "weighted_average_score": metadata.get("weighted_average_score"),
                "times_ranked": metadata.get("times_ranked"),
                "last_ranked_date": metadata.get("last_ranked_date"),
                "has_prelim_ranks": has_prelim_ranks,
                "rerank_status": rerank_status,
                "streak_status": streak_status
            }

            if base_name not in grouped_albums:
                grouped_albums[base_name] = []
            grouped_albums[base_name].append(edition_data)

        return render_template("select_album.html", artist_name=artist_name, grouped_albums=grouped_albums)

    except Exception as e:
        logging.error(f"Error in load_albums_by_artist_route for {artist_name}: {e}", exc_info=True)
        flash("Could not load album list for that artist.", "error")
        return redirect(url_for('index'))

@app.route("/ranking_page")
def ranking_page():
    sheet_rows = load_google_sheet_data()
    group_bins = group_ranked_songs(sheet_rows)
    return render_template("album.html", group_bins=group_bins)


@app.route("/view_album", methods=["POST", "GET"])
def view_album():
    global sp
    try:
        album_id = request.form.get("album_id") or request.args.get("album_id")
        if not album_id:
            flash("Missing album ID.", "warning")
            return redirect(url_for('index'))

        print(f"\n--- VIEW ALBUM START (Album ID: {album_id}) ---")

        # 1. Fetch all necessary data upfront
        album_data = load_album_data(sp, album_id)
        main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        all_final_ranks_df = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("")
        album_averages_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)

        is_rerank_mode = False
        if not album_averages_df.empty and 'album_id' in album_averages_df.columns:
            album_stats = album_averages_df[album_averages_df['album_id'].astype(str) == str(album_id)]
            if not album_stats.empty and album_stats.iloc[0]['times_ranked'] > 0:
                is_rerank_mode = True

        print(f"DEBUG: Re-rank mode for '{album_data['album_name']}': {is_rerank_mode}")

        # In view_album function
        # FIX #2: Add .copy() to prevent SettingWithCopyWarning
        if not all_final_ranks_df.empty and 'Spotify Album ID' in all_final_ranks_df.columns:
            other_albums_df = all_final_ranks_df[all_final_ranks_df['Spotify Album ID'] != album_id].copy()
        else:
            other_albums_df = pd.DataFrame()

            # 2. CRITICAL: Ensure 'Ranking' column is numeric and sort the ENTIRE DataFrame.
        if not other_albums_df.empty:
            other_albums_df['Ranking'] = pd.to_numeric(other_albums_df['Ranking'], errors='coerce')
            sorted_other_albums_df = other_albums_df.sort_values(by='Ranking', ascending=False)
        else:
            sorted_other_albums_df = pd.DataFrame()

        album_covers_cache = {}
        if not sorted_other_albums_df.empty:
            unique_album_ids = [str(aid) for aid in sorted_other_albums_df['Spotify Album ID'].unique() if aid]
            if unique_album_ids:
                for i in range(0, len(unique_album_ids), 20):
                    batch = unique_album_ids[i:i + 20]
                    try:
                        albums_info = sp.albums(batch)
                        for info in albums_info['albums']:
                            if info and info['images']:
                                album_covers_cache[info['id']] = info['images'][-1]['url']
                    except Exception as e:
                        print(f"WARNING: Could not fetch album covers batch: {e}")

        # 3. Iterate through the PRE-SORTED DataFrame to build the groups.
        rank_groups_for_js = {f"{i / 2:.1f}": [] for i in range(1, 21)}
        rank_groups_for_js['I'] = {'excellent': [], 'average': [], 'bad': []}

        for _, row in sorted_other_albums_df.iterrows():
            try:
                rank_group_from_sheet = str(row.get('Rank Group', '')).strip()
                rank_group = rank_group_from_sheet
                try:
                    rank_group_val = float(rank_group_from_sheet)
                    rank_group = f"{rank_group_val:.1f}"
                except (ValueError, TypeError):
                    pass

                song_data = {
                    'song_id': str(row.get('Spotify Song ID')),
                    'song_name': str(row.get('Song Name')),
                    'rank_group': rank_group,
                    'calculated_score': float(row.get('Ranking', 0.0)),
                    'album_id': str(row.get('Spotify Album ID', '')),
                    'album_name': str(row.get('Album Name')),
                    'artist_name': str(row.get('Artist Name')),
                    'album_cover_url': album_covers_cache.get(str(row.get('Spotify Album ID', '')))
                }
                if rank_group == 'I':
                    score = song_data['calculated_score']
                    category = 'average'
                    if score == 3.0:
                        category = 'excellent'
                    elif score == 1.0:
                        category = 'bad'
                    rank_groups_for_js['I'][category].append(song_data)
                elif rank_group in rank_groups_for_js:
                    rank_groups_for_js[rank_group].append(song_data)
            except Exception as e:
                print(f"WARNING: Error parsing row for JS: {row.to_dict()} - {e}")

        def normalize_song_name(name):
            # Remove parentheticals and dashes, lowercase
            name = re.sub(r'\(.*?\)', '', name)
            name = re.sub(r'-.*$', '', name)
            return name.strip().lower()

        ranked_song_name_artist = set()
        ranked_song_scores = {}  # <--- Add this

        if not all_final_ranks_df.empty:
            for _, row in all_final_ranks_df.iterrows():
                song_name = str(row.get('Song Name', '')).strip()
                artist_name = str(row.get('Artist Name', '')).strip()
                if song_name and artist_name:
                    normalized = (normalize_song_name(song_name), artist_name.lower())
                    ranked_song_name_artist.add(normalized)
                    ranked_song_scores[normalized] = row.get('Ranking')  # <--- Add this for score lookup

        # 5. Prepare the left panel (songs for the current album)
        songs_for_left_panel = []
        if is_rerank_mode:
            all_final_ranks_df['Ranking'] = pd.to_numeric(all_final_ranks_df['Ranking'], errors='coerce')
            global_leaderboard = all_final_ranks_df.sort_values(by='Ranking', ascending=False).reset_index(drop=True)
            current_album_previous_ranks = all_final_ranks_df[all_final_ranks_df['Spotify Album ID'] == album_id]
            for song in album_data['songs']:
                song_id = str(song['song_id'])
                previous_rank = "N/A"
                global_placement_text = ""
                song_rank_info = current_album_previous_ranks[
                    current_album_previous_ranks['Spotify Song ID'] == song_id]
                if not song_rank_info.empty:
                    rank_group = song_rank_info.iloc[0].get('Rank Group')
                    rank_value = float(song_rank_info.iloc[0].get('Ranking'))
                    if rank_group == 'I':
                        if rank_value == 3.0:
                            previous_rank = 'Excellent'
                        elif rank_value == 2.0:
                            previous_rank = 'Average'
                        else:
                            previous_rank = 'Bad'
                    else:
                        previous_rank = f"{rank_value:.2f}"
                    placement_series = global_leaderboard.index[global_leaderboard['Spotify Song ID'] == song_id]
                    if not placement_series.empty:
                        placement = int(placement_series[0] + 1)
                        global_placement_text = get_ordinal_suffix(placement)
                songs_for_left_panel.append({**song, 'previous_rank': previous_rank, 'global_placement': global_placement_text})
        else:
            # For initial ranking, load prelims and check for globally ranked songs
            existing_prelim_ranks = {}
            try:
                prelim_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(PRELIM_SHEET_NAME)
                prelim_sheet_data = get_as_dataframe(prelim_sheet, evaluate_formulas=False).fillna("")
                if "album_id" in prelim_sheet_data.columns and not prelim_sheet_data.empty:
                    current_album_prelims = prelim_sheet_data[
                        prelim_sheet_data["album_id"].astype(str) == str(album_id)]
                    for _, row in current_album_prelims.iterrows():
                        if row.get('prelim_rank'):
                            existing_prelim_ranks[str(row.get('song_id'))] = float(row.get('prelim_rank'))
            except gspread.exceptions.WorksheetNotFound:
                print(f"WARNING: Prelim sheet not found, cannot load prelim ranks.")

            # Check which songs are already globally ranked
            globally_ranked_ids = set()
            for group_key, songs in rank_groups_for_js.items():
                if group_key == 'I':
                    for category_songs in songs.values():
                        for s in category_songs:
                            globally_ranked_ids.add(s['song_id'])
                else:
                    for s in songs:
                        globally_ranked_ids.add(s['song_id'])

            for song in album_data['songs']:
                song_id = str(song['song_id'])
                song_name = song.get('song_name', '').strip()
                artist_name = album_data.get('artist_name', '').strip()
                norm = (normalize_song_name(song_name), artist_name.lower())
                already_ranked = norm in ranked_song_name_artist
                existing_score = ranked_song_scores.get(norm, '')

                songs_for_left_panel.append({
                    **song,
                    'already_ranked': already_ranked,
                    'existing_score': existing_score,
                    'prelim_rank': existing_prelim_ranks.get(song_id, '')
                })

        album_data_for_template = {**album_data, 'album_id': album_id, 'songs': songs_for_left_panel,
                                   'is_rerank_mode': is_rerank_mode}
        print("Album ID received for ranking:", album_id)
        return render_template('album.html', album=album_data_for_template, rank_groups=rank_groups_for_js)

    except Exception as e:
        traceback.print_exc()
        flash(f"An unexpected error occurred: {e}", "error")
        return redirect(url_for('index'))


@app.route("/finalize_rankings", methods=["POST"])
def finalize_rankings():
    data = request.get_json()
    if not data:
        return "Invalid data", 400

    print("Received data in finalize_rankings:", data)

    valid_ranks = {str(r) for r in [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]}

    for rank_group in data:
        if str(rank_group) not in valid_ranks:
            return f"Invalid rank group for final: {rank_group}", 400
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    existing_df = get_as_dataframe(sheet).fillna("")

    rows_to_append = []
    for rank, songs in data.items():
        for position, song_name in enumerate(songs):
            rows_to_append.append({
                "Song Name": song_name,
                "Ranking": rank,
                "Position In Group": position,
                "Ranking Status": "finalized"
            })

    if rows_to_append:
        new_df = pd.DataFrame(rows_to_append)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        set_with_dataframe(sheet, updated_df)

    return jsonify({"status": "success"}), 200
@app.route("/get_ranked_songs")
def get_ranked_songs():
    """
    Return a JSON array of songs already assigned to the clicked rank‐group.
    We look first for “paused” rows in that numeric window; if none, show “final” rows.
    """
    album_name  = request.args.get("album_name", "").strip()
    artist_name = request.args.get("artist_name", "").strip()
    try:
        rank = float(request.args.get("rank", "0"))
    except ValueError:
        return jsonify({'songs': []})

    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    # Use get_all_records (faster than get_as_dataframe) then build a small DataFrame
    raw = sheet.get_all_records()
    df = pd.DataFrame(raw).fillna("")
    if df.empty:
        return jsonify({'songs': []})

    # Normalize and convert Ranking to numeric
    df["Album Name"]  = df["Album Name"].astype(str).str.strip().str.lower()
    df["Artist Name"] = df["Artist Name"].astype(str).str.strip().str.lower()
    df["Ranking"]     = pd.to_numeric(df["Ranking"], errors="coerce")
    df["Ranking Status"] = df["Ranking Status"].fillna("")

    album_key  = album_name.lower()
    artist_key = artist_name.lower()

    # First, try to find any “paused” rows whose Ranking sits within [rank−0.25, rank+0.25]
    mask_paused = (
        (df["Album Name"] == album_key) &
        (df["Artist Name"] == artist_key) &
        (df["Ranking Status"] == "paused") &
        (df["Ranking"] >= (rank - 0.25)) &
        (df["Ranking"] <= (rank + 0.25))
    )
    songs = df.loc[mask_paused, "Song Name"].tolist()

    # If none in paused, fall back to finalized
    if not songs:
        mask_final = (
            (df["Album Name"] == album_key) &
            (df["Artist Name"] == artist_key) &
            (df["Ranking Status"] == "final") &
            (df["Ranking"] >= (rank - 0.25)) &
            (df["Ranking"] <= (rank + 0.25))
        )
        songs = df.loc[mask_final, "Song Name"].tolist()

    return jsonify({'songs': songs})
@app.route("/save_album", methods=["POST"])
def save_album():
    status = request.form.get("Ranking Status")
    if status != "paused":
        return "Only paused rankings can be saved here.", 400

    album_name = request.form.get("album_name")
    artist_name = request.form.get("artist_name")
    prelim_ranks = {key.replace("prelim_rank_", ""): float(value)
                    for key, value in request.form.items() if key.startswith("prelim_rank_")}

    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    df_existing = get_as_dataframe(sheet, evaluate_formulas=True).fillna("")

    # Normalize keys for filtering
    album_key = album_name.strip().lower()
    artist_key = artist_name.strip().lower()
    df_existing["Album Name"] = df_existing["Album Name"].str.strip().str.lower()
    df_existing["Artist Name"] = df_existing["Artist Name"].str.strip().str.lower()

    # Remove existing paused rows for this album and artist
    mask = ~(
        (df_existing["Album Name"] == album_key) &
        (df_existing["Artist Name"] == artist_key) &
        (df_existing["Ranking Status"] == "paused")
    )
    df_filtered = df_existing[mask]

    # Create new paused rows with prelim ranks
    new_rows = []
    for song_name, rank in prelim_ranks.items():
        new_rows.append({
            "Album Name": album_name,
            "Artist Name": artist_name,
            "Song Name": song_name,
            "Ranking": rank,
            "Ranking Status": "paused",
            "Ranked Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rank_group": ""  # empty because preliminary
        })

    df_new = pd.DataFrame(new_rows)
    df_updated = pd.concat([df_filtered, df_new], ignore_index=True)

    # Clear sheet and write updated df
    sheet.clear()
    set_with_dataframe(sheet, df_updated)

    return "Paused rankings saved successfully."


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)