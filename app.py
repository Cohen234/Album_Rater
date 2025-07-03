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
                          'original_weighted_score', 'previous_weighted_score', 'times_ranked', 'last_ranked_date']
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
                     'original_weighted_score', 'previous_weighted_score', 'times_ranked', 'last_ranked_date']

    # THE FIX: Add the if/else block to handle an empty sheet
    if df.empty:
        df = pd.DataFrame(columns=expected_cols)
    else:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA

    # Now we can safely convert types
    for col in ['average_score', 'weighted_average_score', 'original_weighted_score', 'previous_weighted_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['times_ranked'] = pd.to_numeric(df['times_ranked'], errors='coerce').fillna(0).astype(int)

    return df.fillna("")
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

def get_dominant_color(image_url):
    try:
        response = requests.get(image_url)
        color_thief = ColorThief(BytesIO(response.content))
        rgb = color_thief.get_color(quality=1)
        return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
    except Exception as e:
        print(f"ERROR: Failed to get dominant color for {image_url}: {e}")
        return "#ffffff"


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


@app.route("/artist/<string:artist_name>")
def artist_page(artist_name):
    """
    This route ONLY displays the artist's statistics dashboard.
    """
    try:
        logging.info(f"--- Loading Artist Stats Page for: {artist_name} ---")

        # All the logic to get data for the leaderboards, stats cards, and
        # pie chart is correct and belongs here.
        main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        all_songs_df = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("")
        all_albums_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)

        # Filter data for the CURRENT artist
        all_songs_df['Ranking'] = pd.to_numeric(all_songs_df['Ranking'], errors='coerce')
        artist_songs_df = all_songs_df[all_songs_df['Artist Name'].str.lower() == artist_name.lower()]
        artist_albums_df = all_albums_df[all_albums_df['artist_name'].str.lower() == artist_name.lower()]

        # Calculate Overall Stats
        avg_song_score = artist_songs_df['Ranking'].mean()
        avg_album_score = artist_albums_df['average_score'].mean()
        artist_stats = {
            'avg_song_score': f"{avg_song_score:.2f}" if pd.notna(avg_song_score) else "N/A",
            'avg_album_score': f"{avg_album_score:.2f}" if pd.notna(avg_album_score) else "N/A",
            'total_songs_ranked': len(artist_songs_df)
        }

        # Prepare Pie Chart Data
        pie_data = artist_songs_df['Rank Group'].value_counts().reset_index()
        pie_data.columns = ['rank_group', 'count']
        pie_chart_data = {
            'labels': pie_data['rank_group'].tolist(),
            'data': pie_data['count'].tolist()
        }

        # Prepare Leaderboards
        artist_song_leaderboard = artist_songs_df.sort_values(by='Ranking', ascending=False).to_dict('records')
        universal_song_leaderboard = all_songs_df.sort_values(by='Ranking', ascending=False).head(100).to_dict('records')
        album_leaderboard = artist_albums_df.sort_values(by='weighted_average_score', ascending=False).to_dict('records')

        return render_template(
            "artist_page.html",
            artist_name=artist_name,
            artist_stats=artist_stats,
            pie_chart_data=pie_chart_data,
            artist_song_leaderboard=artist_song_leaderboard,
            universal_song_leaderboard=universal_song_leaderboard,
            album_leaderboard=album_leaderboard
        )

    except Exception as e:
        logging.critical(f"🔥 CRITICAL ERROR loading artist page for {artist_name}: {e}", exc_info=True)
        flash("Could not load the page for that artist.", "error")
        return redirect(url_for('index'))



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

        # 3. Find Best/Worst Songs
        album_songs_df = main_df[
            (main_df['Spotify Album ID'] == album_id) & (main_df['Rank Group'].astype(str) != 'I')].copy()
        album_songs_df['Ranking'] = pd.to_numeric(album_songs_df['Ranking'], errors='coerce')

        best_song = album_songs_df.loc[album_songs_df['Ranking'].idxmax()] if not album_songs_df.empty else None
        worst_song = album_songs_df.loc[album_songs_df['Ranking'].idxmin()] if not album_songs_df.empty else None

        # 4. Find Leaderboard Placement
        averages_df['weighted_average_score'] = pd.to_numeric(averages_df['weighted_average_score'], errors='coerce')
        averages_df.dropna(subset=['weighted_average_score'], inplace=True)
        averages_df.sort_values(by='weighted_average_score', ascending=False, inplace=True)
        averages_df.reset_index(drop=True, inplace=True)

        placement_series = averages_df.index[averages_df['album_id'] == album_id]
        leaderboard_placement = placement_series[0] + 1 if not placement_series.empty else 'N/A'

        current_score = pd.to_numeric(album_stats.get('weighted_average_score'), errors='coerce')
        original_score = pd.to_numeric(album_stats.get('original_weighted_score'), errors='coerce')

        change_from_original = (current_score - original_score) if pd.notna(current_score) and pd.notna(
            original_score) else 0

        last_ranked_date = pd.to_datetime(album_stats.get('last_ranked_date'), errors='coerce')
        times_ranked = pd.to_numeric(album_stats.get('times_ranked'), errors='coerce').fillna(0)
        next_rerank_date = 'N/A'
        if pd.notna(last_ranked_date):
            days_to_add = 45 if times_ranked > 1 else 15
            next_rerank_date = (last_ranked_date + pd.Timedelta(days=days_to_add)).strftime('%Y-%m-%d')

        response_data = {
            'original_score': f"{original_score:.2f}" if pd.notna(original_score) else 'N/A',
            'best_song': {'name': best_song['Song Name'],
                          'score': f"{best_song['Ranking']:.2f}"} if best_song is not None else {'name': 'N/A',
                                                                                                 'score': ''},
            'worst_song': {'name': worst_song['Song Name'],
                           'score': f"{worst_song['Ranking']:.2f}"} if worst_song is not None else {'name': 'N/A',
                                                                                                    'score': ''},
            'leaderboard_placement': leaderboard_placement,
            'change_from_original': f"{change_from_original:+.2f}",
            'next_rerank_date': next_rerank_date
        }
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error in get_album_stats for {album_id}: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

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

        logging.info(f"\n--- SUBMIT RANKINGS START for Album ID: {album_id} ---")

        # --- 2. Update Main Ranking Sheet (Sheet1) ---
        main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        main_df = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("")

        # Fetch song details (including duration) for all submitted songs in one batch
        all_song_ids = [s.get('song_id') for s in all_ranked_songs_from_js if s.get('song_id')]
        song_details_map = {}
        if sp and all_song_ids:
            try:
                for i in range(0, len(all_song_ids), 50):
                    tracks_info = sp.tracks(all_song_ids[i:i + 50])
                    for track in tracks_info['tracks']:
                        if track:
                            song_details_map[track['id']] = {'name': track['name'], 'duration_ms': track['duration_ms']}
            except Exception as e:
                logging.error(f"Failed to fetch batch track details from Spotify: {e}")

        # Remove old versions of all songs being submitted
        submitted_song_ids = {str(s.get('song_id')) for s in all_ranked_songs_from_js}
        if 'Spotify Song ID' in main_df.columns and submitted_song_ids:
            main_df_filtered = main_df[~main_df['Spotify Song ID'].astype(str).isin(submitted_song_ids)]
        else:
            main_df_filtered = main_df

        # Prepare new rows with updated data
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

        # Combine old and new data and write to sheet
        final_main_df = main_df_filtered
        if new_final_rows_data:
            new_final_df = pd.DataFrame(new_final_rows_data)
            final_main_df = pd.concat([main_df_filtered, new_final_df], ignore_index=True)
        set_with_dataframe(main_sheet, final_main_df, include_index=False, resize=True)
        logging.info(f"Wrote {len(final_main_df)} rows to main ranking sheet.")
        if prelim_ranks_from_js:
            try:
                prelim_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(PRELIM_SHEET_NAME)
                prelim_df = get_as_dataframe(prelim_sheet, evaluate_formulas=False).fillna("")
            except gspread.exceptions.WorksheetNotFound:
                logging.warning(f"'{PRELIM_SHEET_NAME}' not found. It will be created.")
                prelim_sheet = None  # Ensure sheet is None so it's created later
                prelim_df = pd.DataFrame()

            prelim_df_filtered = prelim_df[
                prelim_df['album_id'].astype(str) != str(album_id)] if 'album_id' in prelim_df.columns else prelim_df

            new_prelim_rows = []
            for prelim_data in prelim_ranks_from_js:
                new_prelim_rows.append({
                    'album_id': album_id, 'album_name': album_name, 'artist_name': artist_name,
                    'album_cover_url': album_cover_url, 'song_id': prelim_data.get('song_id'),
                    'song_name': song_details_map.get(prelim_data.get('song_id'), "Unknown"),
                    'prelim_rank': prelim_data.get('prelim_rank'),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

            final_prelim_df = pd.concat([prelim_df_filtered, pd.DataFrame(new_prelim_rows)],
                                        ignore_index=True) if new_prelim_rows else prelim_df_filtered

            if prelim_sheet is None:
                prelim_sheet = client.open_by_key(SPREADSHEET_ID).add_worksheet(title=PRELIM_SHEET_NAME, rows=1, cols=8)
                prelim_sheet.append_row(list(final_prelim_df.columns))

            set_with_dataframe(prelim_sheet, final_prelim_df, include_index=False, resize=True)
            logging.info("Successfully updated preliminary ranks.")

        # --- 3. Recalculate ALL Album Averages (Simple and Weighted) ---
        if submission_status == 'final':
            logging.info("Recalculating all album averages...")

            df_for_calc = final_main_df.copy()
            df_for_calc['Ranking'] = pd.to_numeric(df_for_calc['Ranking'], errors='coerce')
            df_for_calc['Duration (ms)'] = pd.to_numeric(df_for_calc['Duration (ms)'], errors='coerce')
            df_for_calc.dropna(subset=['Ranking', 'Duration (ms)'], inplace=True)
            df_for_calc_no_interludes = df_for_calc[df_for_calc['Rank Group'].astype(str) != 'I']

            if not df_for_calc_no_interludes.empty:
                simple_averages = df_for_calc_no_interludes.groupby('Spotify Album ID')['Ranking'].mean().round(2)

                def weighted_avg(group):
                    total_duration = group['Duration (ms)'].sum()
                    return ((group['Ranking'] * group[
                        'Duration (ms)']).sum() / total_duration) if total_duration > 0 else group['Ranking'].mean()

                weighted_averages = df_for_calc_no_interludes.groupby('Spotify Album ID').apply(weighted_avg).round(2)

                album_averages_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)
                album_info_map = df_for_calc[['Spotify Album ID', 'Album Name', 'Artist Name']].drop_duplicates(
                    'Spotify Album ID').set_index('Spotify Album ID').to_dict('index')

                newly_created_rows = []
                for album_id_to_update in set(simple_averages.index):
                    new_simple_avg = simple_averages.get(album_id_to_update)
                    new_weighted_avg = weighted_averages.get(album_id_to_update)

                    existing_rows = album_averages_df[album_averages_df['album_id'] == album_id_to_update]

                    if not existing_rows.empty:
                        idx = existing_rows.index[0]
                        current_weighted_score = album_averages_df.at[idx, 'weighted_average_score']
                        album_averages_df.at[idx, 'previous_weighted_score'] = current_weighted_score if pd.notna(
                            current_weighted_score) else new_weighted_avg
                        album_averages_df.at[idx, 'average_score'] = new_simple_avg
                        album_averages_df.at[idx, 'weighted_average_score'] = new_weighted_avg
                        if album_id_to_update == album_id:
                            album_averages_df.at[idx, 'times_ranked'] += 1
                            album_averages_df.at[idx, 'last_ranked_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        info = album_info_map.get(album_id_to_update)
                        if info:
                            newly_created_rows.append({
                                'album_id': album_id_to_update, 'album_name': info['Album Name'],
                                'artist_name': info['Artist Name'],
                                'average_score': new_simple_avg, 'weighted_average_score': new_weighted_avg,
                                'original_weighted_score': new_weighted_avg,
                                'previous_weighted_score': new_weighted_avg,
                                'times_ranked': 1, 'last_ranked_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })

                if newly_created_rows:
                    album_averages_df = pd.concat([album_averages_df, pd.DataFrame(newly_created_rows)],
                                                  ignore_index=True)

                set_with_dataframe(client.open_by_key(SPREADSHEET_ID).worksheet(album_averages_sheet_name),
                                   album_averages_df, include_index=False, resize=True)
                logging.info("Successfully recalculated and saved all averages.")

        # --- 4. Update Preliminary Ranks Sheet ---


        logging.info(f"--- SUBMIT RANKINGS END (Success) ---\n")
        return jsonify({'status': 'success', 'message': 'Rankings submitted successfully!'})

    except Exception as e:
        logging.critical(f"\n🔥 CRITICAL ERROR in /submit_rankings: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': f"An unexpected error occurred: {e}"}), 500
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/search', methods=['POST'])
def search_artist():
    artist_name = request.form.get('artist_name')
    if artist_name:
        # Redirect to the main artist dashboard
        return redirect(url_for('artist_page', artist_name=artist_name))
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


        # Create a dictionary for quick lookup of averages/times ranked
        album_metadata = {}
        if not album_averages_df.empty:
            for _, row in album_averages_df.iterrows():
                album_id_from_sheet = str(row.get("album_id", "")).strip()
                if album_id_from_sheet:
                    # This corrected version gets BOTH scores
                    album_metadata[album_id_from_sheet] = {
                        "average_score": row.get("average_score"),
                        "weighted_average_score": row.get("weighted_average_score"),  # <-- This is the essential new line
                        "times_ranked": row.get("times_ranked", 0),
                        "last_ranked_date": row.get("last_ranked_date", "")
                    }

        grouped_albums = {}
        for album_data in albums_from_spotify:
            full_name = album_data.get("name")
            album_id_spotify = album_data.get("id")

            # Create a "base name" by removing phrases in parentheses like (Deluxe), (Remastered), etc.
            base_name = re.sub(r'\s*\([^)]*\)$', '', full_name).strip()

            # Get stats for this specific edition
            metadata = album_metadata.get(album_id_spotify, {})

            edition_data = {
                "id": album_id_spotify,
                "full_name": full_name,
                "image": album_data.get("image"),
                "average_score": metadata.get("average_score"),
                "weighted_average_score": metadata.get("weighted_average_score"),
                "times_ranked": metadata.get("times_ranked"),
                "last_ranked_date": metadata.get("last_ranked_date"),
            }

            # If we haven't seen this base name before, create a new list for it
            if base_name not in grouped_albums:
                grouped_albums[base_name] = []

            # Add the current edition to the list for its base name
            grouped_albums[base_name].append(edition_data)

        logging.debug(f"DEBUG: Loaded {len(album_metadata)} album metadata entries from sheet into dict.")
        prelim_sheet_name = "Preliminary Ranks"
        prelim_ranked_albums_ids = set()
        try:
            prelim_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(prelim_sheet_name)
            prelim_sheet_data = get_as_dataframe(prelim_sheet, evaluate_formulas=False).fillna("")

            # CORRECTED: Check for column existence before using it
            if not prelim_sheet_data.empty and 'artist_name' in prelim_sheet_data.columns:
                current_artist_prelim_ranks = prelim_sheet_data[
                    prelim_sheet_data["artist_name"].astype(str).str.strip().str.lower() == artist_name.strip().lower()
                    ]
                for _, row in current_artist_prelim_ranks.iterrows():
                    album_id_p = str(row.get('album_id', '')).strip()
                    prelim_rank_value = row.get('prelim_rank')
                    if album_id_p and str(prelim_rank_value).strip() not in ["", "0", "0.0", "None"]:
                        prelim_ranked_albums_ids.add(album_id_p)

            logging.debug(f"Found {len(prelim_ranked_albums_ids)} albums with preliminary ranks for '{artist_name}'.")

        except gspread.exceptions.WorksheetNotFound:
            logging.warning(
                f"Preliminary Ranks sheet '{prelim_sheet_name}' not found. No prelim rank check for pause icon.")
        except Exception as e:
            logging.error(f"ERROR: Error loading preliminary ranks for pause icon: {e}", exc_info=True)

        # Prepare albums for template, adding average score and times ranked
        albums_for_template = []
        for album_data in albums_from_spotify:
            album_id_spotify = album_data.get("id")  # Get the Spotify Album ID
            album_name_spotify = album_data.get("name")
            artist_name_current_album = artist_name  # The artist name for this context

            logging.debug(f"DEBUG: Processing Spotify album: ID='{album_id_spotify}', Name='{album_name_spotify}'")

            # --- FIX: Lookup metadata using album_id_spotify ---
            metadata = album_metadata.get(album_id_spotify, {})
            # --- END FIX ---

            has_prelim_ranks = album_id_spotify in prelim_ranked_albums_ids  # Check using album_id

            logging.debug(
                f"DEBUG: Album '{album_name_spotify}' (ID: {album_id_spotify}) - Metadata: {metadata}, Has Prelim Ranks: {has_prelim_ranks}")

            albums_for_template.append({
                "album_name": album_name_spotify,
                "artist_name": artist_name_current_album,
                "image": album_data.get("image"),
                "id": album_id_spotify,
                "average_score": metadata.get("average_score"),
                "times_ranked": metadata.get("times_ranked"),
                "last_ranked_date": metadata.get("last_ranked_date"),
                "url": album_data.get("url"),
                "has_prelim_ranks": has_prelim_ranks
            })
        logging.debug(
            f"DEBUG: Prepared {len(albums_for_template)} albums for album.html with metadata, prelim status, and averages.")

    # Pass the enriched list to the template
        return render_template("select_album.html", artist_name=artist_name, albums=albums_for_template,  grouped_albums=grouped_albums)

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
            current_album_previous_ranks = all_final_ranks_df[all_final_ranks_df['Spotify Album ID'] == album_id]
            for song in album_data['songs']:
                song_id = str(song['song_id'])
                previous_rank = "N/A"
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
                songs_for_left_panel.append({**song, 'previous_rank': previous_rank})
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
