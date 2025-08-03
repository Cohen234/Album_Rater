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
    for i in range(0, len(album_ids), 50):
        try:
            albums_info = sp_instance.albums(album_ids[i:i + 50])
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
        print(f"ERROR: Failed to get dominant color for {image_url}: {e}")
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

@app.route("/artist/<string:artist_name>")
def artist_page_v2(artist_name):
    import numpy as np
    import pandas as pd
    import math
    import json
    from jinja2 import Undefined

    try:
        logging.info(f"--- Loading Artist Stats Page for: {artist_name} ---")

        # 1. --- Load All Base Data ---
        main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        all_songs_df = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("")
        all_albums_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)

        # --- Normalize columns: spaces to underscores everywhere ---
        all_songs_df.columns = [c.replace(' ', '_') for c in all_songs_df.columns]
        all_albums_df.columns = [c.replace(' ', '_') for c in all_albums_df.columns]

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
        print("Duplicates:\n", dupes[['Song_Name', 'Artist_Name', 'Ranking_Status', 'Ranked_Date']])
        print("Num duplicates:", len(dupes))

        # --- Sort albums and assign Global Rank ---
        all_albums_df = all_albums_df.sort_values(by='weighted_average_score', ascending=False)
        all_albums_df['Global_Rank'] = range(1, len(all_albums_df) + 1)

        # --- Filter for the current artist ---
        artist_songs_df = all_songs_df[
            all_songs_df['Artist_Name'].astype(str).str.lower() == artist_name.lower()
        ].copy()
        artist_albums_df = all_albums_df[
            all_albums_df['artist_name'].astype(str).str.lower() == artist_name.lower()
        ].copy()
        if artist_songs_df.empty and artist_albums_df.empty:
            return redirect(url_for('load_albums_by_artist_route', artist_name=artist_name))

        artist_songs_df.columns = [c.replace(' ', '_') for c in artist_songs_df.columns]

        # 2. --- Calculate New Stats ----

        # ARTIST MASTERY
        total_spotify_albums = len(get_albums_by_artist(artist_name))
        ranked_albums_count = len(artist_albums_df)
        mastery_points = artist_albums_df['times_ranked'].clip(upper=3).sum() if 'times_ranked' in artist_albums_df else 0
        max_mastery_points = total_spotify_albums * 3
        mastery_percentage = (mastery_points / max_mastery_points) * 100 if max_mastery_points > 0 else 0

        # LEADERBOARD POINTS
        total_songs = len(all_songs_df)
        total_albums = len(all_albums_df)
        song_points = artist_songs_df['Universal_Rank'].apply(lambda x: total_songs - x + 1).sum() if not artist_songs_df.empty else 0
        album_points = ((total_albums - artist_albums_df['Global_Rank'] + 1) * 10).sum() if not artist_albums_df.empty else 0
        total_leaderboard_points = song_points + album_points

        # ARTIST SCORE
        album_percentile = ((total_albums - artist_albums_df['Global_Rank'].mean()) / total_albums) * 100 if total_albums > 0 and not artist_albums_df.empty else 0
        song_percentile = ((total_songs - artist_songs_df['Universal_Rank'].mean()) / total_songs) * 100 if total_songs > 0 and not artist_songs_df.empty else 0
        artist_score = (album_percentile * 0.6) + (song_percentile * 0.4) if ranked_albums_count > 0 else 0
        album_first_ranked = all_songs_df.groupby('Album_Name')['Ranked_Date'].min()
        album_first_score = all_songs_df.groupby('Album_Name')['Ranking'].first()
        all_albums_df['album_name_clean'] = all_albums_df['album_name'].astype(str).str.strip().str.lower()
        all_albums_df['first_ranked_date'] = all_albums_df['album_name_clean'].map(album_first_ranked)
        all_albums_df['first_score'] = all_albums_df['album_name_clean'].map(album_first_score)
        all_albums_df['first_ranked_date'] = pd.to_datetime(all_albums_df['first_ranked_date'], errors='coerce')

        def get_album_placement_on_rank_date(album_id, rank_date, all_albums_df):
            # Only include albums ranked on or before this date
            eligible = all_albums_df[all_albums_df['first_ranked_date'] <= rank_date].copy()
            eligible = eligible.sort_values('weighted_average_score', ascending=False).reset_index(drop=True)
            try:
                placement = eligible[eligible['album_id'] == album_id].index[0] + 1
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
                'score': row.get('first_score'),
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

        ranking_era_data = {
            'datasets': [{
                'label': 'Album Score',
                'data': [
                    {
                        'x': row['release_date'],
                        'y': row['weighted_average_score'],
                        'label': row['album_name'],
                        'image': row.get('album_cover_url', '')
                    }
                    for _, row in era_history_data.iterrows() if
                    pd.notna(row.get('release_date')) and pd.notna(row.get('weighted_average_score'))
                ],
                'borderColor': 'rgba(29, 185, 84, 1)',
                'tension': 0.4,
            }]
        }
        for d in ranking_era_data['datasets'][0]['data']:
            d['label'] = clean_title(d['label'])


        # --- Prepare Leaderboard and other stats ---
        artist_songs_df.sort_values(by='Ranking', ascending=False, inplace=True)
        artist_songs_df['Artist_Rank'] = range(1, len(artist_songs_df) + 1)
        artist_albums_df.sort_values(by='weighted_average_score', ascending=False, inplace=True)
        artist_albums_df['Artist_Rank'] = range(1, len(artist_albums_df) + 1)
        artist_average_score = artist_albums_df['weighted_average_score'].mean() if not artist_albums_df.empty else 0

        # For polar chart
        all_rank_groups = [f"{i / 2:.1f}" for i in range(2, 21)]
        artist_songs_df['Rank_Group_Str'] = artist_songs_df['Rank_Group'].astype(str) if 'Rank_Group' in artist_songs_df else ""
        song_counts = artist_songs_df['Rank_Group_Str'].value_counts() if 'Rank_Group_Str' in artist_songs_df else pd.Series()
        polar_data_series = pd.Series(index=all_rank_groups, dtype=int).fillna(0)
        polar_data_series.update(song_counts)
        polar_data_series = polar_data_series.sort_index(key=lambda x: pd.to_numeric(x))
        polar_chart_data = {
            'labels': polar_data_series.index.tolist(),
            'data': polar_data_series.values.tolist()
        }

        print("Filtered all_songs_df shape:", all_songs_df.shape)
        print("First 5 rows:\n", all_songs_df.head())
        print("Filtered artist_songs_df shape:", artist_songs_df.shape)

        average_song_score = artist_songs_df['Ranking'].mean() if not artist_songs_df.empty else 0
        median_song_score = artist_songs_df['Ranking'].median() if not artist_songs_df.empty else 0
        std_song_score = artist_songs_df['Ranking'].std() if not artist_songs_df.empty else 0

        # Highest and lowest ranked songs
        if not artist_songs_df.empty:
            top_song_row = artist_songs_df.loc[artist_songs_df['Ranking'].idxmax()]
            low_song_row = artist_songs_df.loc[artist_songs_df['Ranking'].idxmin()]
            top_song_name = top_song_row['Song_Name']
            top_song_score = top_song_row['Ranking']
            top_song_cover = top_song_row.get('album_cover_url', '')
            top_song_link = url_for('view_album', album_id=top_song_row.get('Spotify_Album_ID', '')) if top_song_row.get('Spotify_Album_ID') else "#"
            low_song_name = low_song_row['Song_Name']
            low_song_score = low_song_row['Ranking']
            low_song_cover = low_song_row.get('album_cover_url', '')
            low_song_link = url_for('view_album', album_id=low_song_row.get('Spotify_Album_ID', '')) if low_song_row.get('Spotify_Album_ID') else "#"
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
            top_album_link = url_for('view_album', album_id=top_album_row.get('album_id', '')) if top_album_row.get('album_id') else "#"
            low_album_name = low_album_row['album_name']
            low_album_score = low_album_row['weighted_average_score']
            low_album_cover = low_album_row.get('album_cover_url', '')
            low_album_link = url_for('view_album', album_id=low_album_row.get('album_id', '')) if low_album_row.get('album_id') else "#"
        else:
            top_album_name = top_album_score = top_album_cover = top_album_link = ''
            low_album_name = low_album_score = low_album_cover = low_album_link = ''

        global_avg_song_score = all_songs_df['Ranking'].mean() if not all_songs_df.empty else 0
        most_improved_song_name = ""
        most_improved_song_delta = 0

        if not artist_songs_df.empty and 'Song_Name' in artist_songs_df.columns and 'Ranking' in artist_songs_df.columns:
            improvement_data = []
            for song_name, group in artist_songs_df.groupby('Song_Name'):
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

        if not artist_songs_df.empty and "Ranked_Date" in artist_songs_df.columns:
            artist_songs_df['Ranked_Date'] = pd.to_datetime(artist_songs_df['Ranked_Date'], errors='coerce')
            if artist_songs_df['Ranked_Date'].nunique() > 2:
                grouped = artist_songs_df.groupby(artist_songs_df['Ranked_Date'].dt.to_period('M'))
                for period, group in grouped:
                    label = str(period)
                    avg_score = safe_float(group['Ranking'].mean())
                    ranking_trajectory_data['labels'].append(label)
                    ranking_trajectory_data['datasets'][0]['data'].append(avg_score)
            else:
                sorted_rows = artist_songs_df.sort_values('Ranked_Date')
                for _, row in sorted_rows.iterrows():
                    label = row['Ranked_Date'].strftime('%b %d, %Y') if pd.notnull(row['Ranked_Date']) else ""
                    avg_score = safe_float(row['Ranking'])
                    ranking_trajectory_data['labels'].append(label)
                    ranking_trajectory_data['datasets'][0]['data'].append(avg_score)
        if not ranking_trajectory_data["labels"]:
            ranking_trajectory_data = {
                "labels": [],
                "datasets": [{
                    "label": "Avg Song Score",
                    "data": [],
                    "borderColor": "rgba(29, 185, 84, 1)",
                    "backgroundColor": "rgba(29, 185, 84, 0.2)",
                }]
            }

        # For song leaderboard
        song_leaderboard_clean = []
        for row in artist_songs_df.to_dict('records'):
            row['Song_Name'] = clean_title(row['Song_Name'])
            song_leaderboard_clean.append(row)
        # For album leaderboard
        album_leaderboard_clean = []
        for row in artist_albums_df.to_dict('records'):
            row['album_name'] = clean_title(row['album_name'])
            album_leaderboard_clean.append(row)

        # --- First/last album ranked info ---
        if not artist_songs_df.empty and 'Ranked_Date' in artist_songs_df.columns:
            artist_songs_df['Ranked_Date'] = pd.to_datetime(artist_songs_df['Ranked_Date'], errors='coerce')
            # Get the first album for which any song was ranked (by earliest song Ranked_Date)
            first_album_row = \
            artist_songs_df.sort_values('Ranked_Date').drop_duplicates('Album_Name', keep='first').iloc[0]
            last_album_row = artist_songs_df.sort_values('Ranked_Date').drop_duplicates('Album_Name', keep='last').iloc[
                0]
            first_album_ranked_name = first_album_row['Album_Name']
            first_album_ranked_date = first_album_row['Ranked_Date'].strftime('%b %d, %Y') if pd.notnull(
                first_album_row['Ranked_Date']) else ""
            last_album_ranked_name = last_album_row['Album_Name']
            last_album_ranked_date = last_album_row['Ranked_Date'].strftime('%b %d, %Y') if pd.notnull(
                last_album_row['Ranked_Date']) else ""
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
        songs_sorted = artist_songs_df.sort_values(['Album_Name', 'Ranked_Date'])
        points = []

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
        recent = artist_songs_df[artist_songs_df['Ranked_Date'] >= (now - pd.Timedelta(days=days))] if not artist_songs_df.empty else pd.DataFrame()
        old = artist_songs_df[artist_songs_df['Ranked_Date'] < (now - pd.Timedelta(days=days))] if not artist_songs_df.empty else pd.DataFrame()
        recent_avg = recent['Ranking'].mean() if not recent.empty else 0
        old_avg = old['Ranking'].mean() if not old.empty else 0
        arrow_delta = recent_avg - old_avg
        arrow_direction = "up" if arrow_delta > 0 else "down" if arrow_delta < 0 else "flat"

        print(songs_sorted['Album_Name'].unique())
        print(album_boundaries)
        print(album_labels)
        print(album_arts)

        return render_template(
            "artist_page_v2.html",
            artist_name=artist_name,
            artist_mastery=mastery_percentage,
            leaderboard_points=total_leaderboard_points,
            artist_average_score=artist_average_score,
            ranking_era_data=ranking_era_data,
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
            album_arts=album_arts
        )
    except Exception as e:
        logging.critical(f"🔥 CRITICAL ERROR loading artist page for {artist_name}: {e}")
        return f"An error occurred: {e}", 500



@app.route('/get_album_stats/<album_id>')
def get_album_stats(album_id):
    try:
        # 1. Load data
        main_df = get_as_dataframe(client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)).fillna("")
        averages_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)

        # 2. Find the specific album's data
        album_stats = averages_df[averages_df['album_id'] == album_id]
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

        placement_series = averages_df.index[averages_df['album_id'] == album_id]
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
                    old_album_data = averages_df_before[averages_df_before['album_id'] == album_id]
                    if not old_album_data.empty:
                        old_score = old_album_data.iloc[0]['weighted_average_score']
                        # Sort to find old placement
                        averages_df_before.sort_values(by='weighted_average_score', ascending=False, inplace=True)
                        averages_df_before.reset_index(drop=True, inplace=True)
                        old_placement_series = averages_df_before.index[averages_df_before['album_id'] == album_id]
                        old_placement = int(old_placement_series[0] + 1) if not old_placement_series.empty else 1

            # --- 4. Update Google Sheets with New Final Rankings ---
            main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
            main_df = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("")

            all_song_ids = [s.get('song_id') for s in all_ranked_songs_from_js if s.get('song_id')]
            song_details_map = {}
            if sp and all_song_ids:
                try:
                    for i in range(0, len(all_song_ids), 50):
                        tracks_info = sp.tracks(all_song_ids[i:i + 50])
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
                new_final_rows_data.append({
                    'Album Name': ranked_song_data.get('album_name'), 'Artist Name': ranked_song_data.get('artist_name'),
                    'Spotify Album ID': ranked_song_data.get('album_id'),
                    'Song Name': details.get('name', ranked_song_data.get('song_name')),
                    'Ranking': ranked_song_data.get('calculated_score', 0.0),
                    'Duration (ms)': details.get('duration_ms', 0),
                    'Ranking Status': 'final', 'Ranked Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Position In Group': str(ranked_song_data.get('position_in_group', '')),
                    'Rank Group': str(ranked_song_data.get('rank_group')), 'Spotify Song ID': song_id,
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
                    existing_rows = album_averages_df[album_averages_df['album_id'] == album_id_to_update]

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

            new_album_data = averages_df_after[averages_df_after['album_id'] == album_id]
            if new_album_data.empty:
                return jsonify({'status': 'error', 'message': 'Could not find album after ranking.'}), 500

            new_score = float(new_album_data.iloc[0]['weighted_average_score'])
            times_ranked = int(new_album_data.iloc[0]['times_ranked'])
            new_placement_series = averages_df_after.index[averages_df_after['album_id'] == album_id]
            new_placement = int(new_placement_series[0] + 1) if not new_placement_series.empty else 1
            total_albums = len(averages_df_after)
            dominant_color = get_dominant_color(album_cover_url)

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
                        'album_name': album_name, 'artist_name': artist_name, 'album_cover_url': album_cover_url,
                        'final_score': new_score, 'final_rank': new_placement,
                        'total_albums': total_albums, 'dominant_color': dominant_color
                    }
                })

    except Exception as e:
        logging.critical(f"\n🔥 CRITICAL ERROR in /submit_rankings: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f"An unexpected error occurred: {e}"}), 500

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
    # Get all the data passed from the redirect
    album_name = request.args.get('album_name')
    artist_name = request.args.get('artist_name')
    album_cover_url = request.args.get('album_cover_url')
    final_score = float(request.args.get('final_score', 0))
    final_rank = int(request.args.get('final_rank', 1))
    total_albums = int(request.args.get('total_albums', 1))
    dominant_color = request.args.get('dominant_color', '#121212')

    # Determine the color for the score text
    if final_score >= 7:
        score_color = 'green'
    elif final_score >= 4:
        score_color = 'yellow'
    else:
        score_color = 'red'

    # The animation should start counting from the bottom rank
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
        start_rank=start_rank
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


@app.route("/load_albums_by_artist", methods=["GET", "POST"])
def load_albums_by_artist_route():
    artist_name = request.form.get("artist_name") or request.args.get("artist_name")

    if not artist_name:
        flash("Artist name not provided. Please search for an artist.")
        return redirect(url_for('index'))

    logging.info(f"\n--- LOADING ALBUM LIST FOR ARTIST: {artist_name} ---")
    try:
        albums_from_spotify = get_albums_by_artist(artist_name) # Renamed variable for clarity

        # --- DEBUG: Raw albums from Spotify API ---
        logging.debug("Raw albums from Spotify API:")  # Changed print to logging.debug
        if albums_from_spotify:
            for i, album_data in enumerate(albums_from_spotify[:3]):
                logging.debug(f"  Album {i + 1}: {album_data}")
        else:
            logging.debug("  No albums returned from Spotify API.")
        # --- END DEBUG ---

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
                    # THE FIX: Use .to_dict() to include ALL columns from the sheet,
                    # including 'rerank_history' and 'original_weighted_score'.
                    album_metadata[album_id_from_sheet] = row.to_dict()

        grouped_albums = {}
        today = datetime.now()
        for album_data in albums_from_spotify:
            full_name = album_data.get("name")
            album_id_spotify = album_data.get("id")

            # Create a "base name" by removing phrases in parentheses like (Deluxe), (Remastered), etc.
            base_name = re.sub(r'\s*\([^)]*\)$', '', full_name).strip()

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

            # --- NEW: Streak Logic ---
            streak_status = 'none'
            if metadata:
                try:
                    # THE FIX: Use the new 'score_history' column
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
                "rerank_status": rerank_status,  # <-- New flag for date icon
                "streak_status": streak_status
            }

            # If we haven't seen this base name before, create a new list for it
            if base_name not in grouped_albums:
                grouped_albums[base_name] = []

            # Add the current edition to the list for its base name
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
            album_stats = album_averages_df[album_averages_df['album_id'] == album_id]
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
            unique_album_ids = [aid for aid in sorted_other_albums_df['Spotify Album ID'].unique() if aid]
            if unique_album_ids:
                for i in range(0, len(unique_album_ids), 50):
                    try:
                        albums_info = sp.albums(unique_album_ids[i:i + 50])
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
                songs_for_left_panel.append({
                    **song,
                    'already_ranked': song_id in globally_ranked_ids,
                    'prelim_rank': existing_prelim_ranks.get(song_id, '')
                })

        album_data_for_template = {**album_data, 'album_id': album_id, 'songs': songs_for_left_panel,
                                   'is_rerank_mode': is_rerank_mode}
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