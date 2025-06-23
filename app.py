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
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))
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
                header_cols = ['album_id', 'album_name', 'artist_name', 'average_score', 'times_ranked',
                               'last_ranked_date']
                sheet = client_gspread.open_by_key(spreadsheet_id).add_worksheet(sheet_name, rows=1,
                                                                                 cols=len(header_cols))
                sheet.append_row(header_cols)  # Write the header row
                logging.info(f"Successfully created sheet '{sheet_name}' with header: {header_cols}")
            except Exception as create_e:
                logging.critical(f"CRITICAL ERROR: Could not create sheet '{sheet_name}': {create_e}", exc_info=True)
                raise create_e  # Re-raise to stop execution if sheet creation fails
        raise e  # Re-raise the original error if it's not WorksheetNotFound or creation fails

    df = get_as_dataframe(sheet, evaluate_formulas=True)  # Use your existing get_as_dataframe helper

    expected_cols = ['album_id', 'album_name', 'artist_name', 'average_score', 'times_ranked', 'last_ranked_date']
    if df.empty:
        logging.debug(f"Album Averages DataFrame is empty. Initializing with expected columns: {expected_cols}")
        df = pd.DataFrame(columns=expected_cols)
    else:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None  # Add missing columns
                logging.warning(f"Added missing column '{col}' to Album Averages DataFrame.")

    # --- CRITICAL: Force 'times_ranked' and 'average_score' to numeric types ---
    # Convert 'times_ranked' to numeric, errors become NaN, fill NaN with 0, then convert to int
    df['times_ranked'] = pd.to_numeric(df['times_ranked'], errors='coerce').fillna(0).astype(int)
    # Convert 'average_score' to numeric, errors become NaN
    df['average_score'] = pd.to_numeric(df['average_score'], errors='coerce')

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


@app.route("/submit_rankings", methods=["POST"])
def submit_rankings():
    global sp, client  # Ensure access to global Spotify and GSheets client

    try:
        album_name = request.form.get("album_name")
        artist_name = request.form.get("artist_name")
        album_id = request.form.get("album_id")
        submission_status = request.form.get("status", "final")
        album_cover_url = request.form.get('album_cover_url')

        logging.info(f"\n--- SUBMIT RANKINGS START ---")
        logging.info(
            f"submit_rankings called for Album: '{album_name}' (ID: {album_id}), Artist: '{artist_name}', Status: '{submission_status}'")

        all_ranked_data_json = request.form.get("all_ranked_data", "[]")
        prelim_rank_data_json = request.form.get("prelim_rank_data", "{}")

        try:
            all_ranked_songs_from_js = json.loads(all_ranked_data_json)
            for song in all_ranked_songs_from_js:
                song['calculated_score'] = float(song.get('calculated_score', 0.0))
        except json.JSONDecodeError as e:
            flash(f"Error parsing ranked songs data: {e}", "error")
            logging.error(f"JSONDecodeError on all_ranked_data: {e}", exc_info=True)
            return redirect(url_for('index'))

        try:
            prelim_ranks_from_js = json.loads(prelim_rank_data_json)
            for song_id, score in prelim_ranks_from_js.items():
                prelim_ranks_from_js[song_id] = float(score if score is not None else 0.0)
        except json.JSONDecodeError as e:
            flash(f"Error parsing prelim ranks data: {e}", "error")
            logging.error(f"JSONDecodeError on prelim_rank_data: {e}", exc_info=True)
            return redirect(url_for('index'))

        logging.debug(f"Parsed all_ranked_songs_from_js ({len(all_ranked_songs_from_js)} songs).")
        logging.debug(f"Parsed prelim_ranks_from_js ({len(prelim_ranks_from_js)} entries).")

        # --- Fetch Song Names (centralized for both main and prelim sheets) ---
        spotify_tracks_for_album = {}
        try:
            if sp:  # Check if sp is initialized
                album_spotify_data = sp.album(album_id)
                for track in album_spotify_data['tracks']['items']:
                    spotify_tracks_for_album[track['id']] = track['name']
            else:
                logging.warning("Spotify client (sp) not initialized. Cannot fetch accurate track names from Spotify.")
                # Fallback to submitted data for song names
                spotify_tracks_for_album.update(
                    {str(s.get('song_id')): s.get('song_name', f"Unknown Song {str(s.get('song_id', 'N/A'))}") for s in
                     all_ranked_songs_from_js})
                spotify_tracks_for_album.update(
                    {song_id: f"Unknown Song {song_id}" for song_id in prelim_ranks_from_js.keys() if
                     song_id not in spotify_tracks_for_album})
        except Exception as e:
            logging.warning(
                f"Could not fetch Spotify tracks for album {album_id}: {e}. Falling back to submission data for song names.",
                exc_info=True)
            # Fallback for final ranks:
            spotify_tracks_for_album.update(
                {str(s.get('song_id')): s.get('song_name', f"Unknown Song {str(s.get('song_id', 'N/A'))}") for s in
                 all_ranked_songs_from_js})
            # Fallback for prelim ranks:
            spotify_tracks_for_album.update(
                {song_id: f"Unknown Song {song_id}" for song_id in prelim_ranks_from_js.keys() if
                 song_id not in spotify_tracks_for_album})

        # --- Main Ranking Sheet Operations ---
        main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        main_df = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("")

        # Delete existing rows for this album in main sheet using DataFrame filtering
        if 'Spotify Album ID' in main_df.columns:
            initial_main_rows = len(main_df)
            main_df_filtered = main_df[main_df['Spotify Album ID'].astype(str) != str(album_id)]
            rows_deleted_main = initial_main_rows - len(main_df_filtered)
            if rows_deleted_main > 0:
                set_with_dataframe(main_sheet, main_df_filtered, include_index=False)
                logging.info(
                    f"Deleted {rows_deleted_main} existing FINAL rank rows for album '{album_name}' (ID: {album_id}) from main sheet.")
            else:
                logging.debug(
                    f"No existing FINAL rank rows found for album '{album_name}' (ID: {album_id}) to delete from main sheet.")
        else:
            logging.warning(
                "Main sheet does not have 'Spotify Album ID' column. Cannot delete existing entries reliably.")

        # Prepare and append new final rank rows to main sheet
        new_final_rows_data = []
        if all_ranked_songs_from_js:
            for ranked_song_data in all_ranked_songs_from_js:
                song_id = str(ranked_song_data['song_id'])
                song_name = spotify_tracks_for_album.get(song_id, f"Unknown Song {song_id}")
                prelim_rank_val = ranked_song_data.get('prelim_rank', '')

                new_final_rows_data.append({
                    'Album Name': album_name,
                    'Artist Name': artist_name,
                    'Spotify Album ID': album_id,
                    'Song Name': song_name,
                    'Ranking': f"{ranked_song_data['calculated_score']:.2f}",
                    'Ranking Status': 'final',
                    'Ranked Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Position In Group': str(ranked_song_data['rank_position']),
                    'Rank Group': f"{float(ranked_song_data['rank_group']):.1f}",
                    'Spotify Song ID': song_id,
                    'Preliminary Rank': prelim_rank_val
                })

            if new_final_rows_data:
                # Convert to DataFrame to handle column order and append
                new_final_df = pd.DataFrame(new_final_rows_data)

                # Get current main sheet header to ensure correct column order for append
                # This ensures we append correctly even if the sheet was manually altered
                current_main_sheet_header = main_sheet.row_values(1)

                # Reorder new_final_df columns to match current_main_sheet_header
                # Fill missing columns in new_final_df with None, or drop extra columns
                cols_to_append = [col for col in current_main_sheet_header if col in new_final_df.columns]
                missing_cols_in_new_data = [col for col in current_main_sheet_header if col not in new_final_df.columns]

                if missing_cols_in_new_data:
                    logging.warning(
                        f"Missing columns in new final rank data that exist in main sheet: {missing_cols_in_new_data}. Appending as empty.")
                    for col in missing_cols_in_new_data:
                        new_final_df[col] = None  # Add missing columns

                new_final_df = new_final_df[
                    cols_to_append + missing_cols_in_new_data].to_numpy().tolist()  # Convert to list of lists for append_rows

                main_sheet.append_rows(new_final_df, value_input_option='USER_ENTERED')
                logging.info(f"Appended {len(new_final_rows_data)} FINAL rank rows to main sheet.")
        else:
            logging.info("No new FINAL rank rows to append to main sheet.")

        # --- PRELIMINARY RANKING SHEET OPERATIONS ---
        prelim_sheet = None
        try:
            prelim_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(PRELIM_SHEET_NAME)
            logging.debug(f"Successfully opened Preliminary Ranks sheet: '{PRELIM_SHEET_NAME}'")
        except gspread.exceptions.WorksheetNotFound:
            logging.warning(f"Preliminary Ranks worksheet '{PRELIM_SHEET_NAME}' not found. Attempting to create it.")
            prelim_sheet_header_cols = [
                'album_id', 'album_name', 'artist_name', 'album_cover_url',
                'song_id', 'song_name', 'prelim_rank', 'timestamp'
            ]
            try:
                prelim_sheet = client.open_by_key(SPREADSHEET_ID).add_worksheet(PRELIM_SHEET_NAME, rows=1,
                                                                                cols=len(prelim_sheet_header_cols))
                prelim_sheet.append_row(prelim_sheet_header_cols)
                logging.info(
                    f"Successfully created new Preliminary Ranks sheet with header: {prelim_sheet_header_cols}")
            except Exception as e:
                logging.critical(f"CRITICAL ERROR: Failed to create Preliminary Ranks sheet '{PRELIM_SHEET_NAME}': {e}",
                                 exc_info=True)
                flash(f"Critical error: Failed to create Preliminary Ranks sheet. Check permissions. {e}", "error")
                return redirect(url_for('index'))
        except Exception as e:
            logging.error(f"Could not open Preliminary Ranks sheet '{PRELIM_SHEET_NAME}': {e}", exc_info=True)
            flash(f"Error accessing preliminary ranks sheet: {e}", "error")
            return redirect(url_for('index'))

        # Get existing prelim data as DataFrame
        prelim_df = get_as_dataframe(prelim_sheet, evaluate_formulas=False).fillna("")

        # Delete existing preliminary ranks for this album
        if 'album_id' in prelim_df.columns:
            initial_prelim_rows = len(prelim_df)
            prelim_df_filtered = prelim_df[prelim_df['album_id'].astype(str) != str(album_id)]
            rows_deleted_prelim = initial_prelim_rows - len(prelim_df_filtered)
            if rows_deleted_prelim > 0:
                set_with_dataframe(prelim_sheet, prelim_df_filtered, include_index=False)
                logging.info(
                    f"Deleted {rows_deleted_prelim} existing PRELIMINARY rank rows for album '{album_name}' (ID: {album_id}).")
            else:
                logging.debug(
                    f"No existing PRELIMINARY rank rows found for album '{album_name}' (ID: {album_id}) to delete.")
        else:
            logging.warning(
                "Preliminary Ranks sheet does not have 'album_id' column. Cannot delete existing entries reliably.")

        # Append new preliminary ranks if any were submitted
        new_prelim_rows_data = []
        if prelim_ranks_from_js:
            for song_id, prelim_rank_value in prelim_ranks_from_js.items():
                song_name = spotify_tracks_for_album.get(song_id, f"Unknown Song (ID: {song_id})")
                new_prelim_rows_data.append({
                    'album_id': album_id,
                    'album_name': album_name,
                    'artist_name': artist_name,
                    'album_cover_url': album_cover_url,
                    'song_id': song_id,
                    'song_name': song_name,
                    'prelim_rank': prelim_rank_value,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

            if new_prelim_rows_data:
                new_prelim_df = pd.DataFrame(new_prelim_rows_data)
                # Append to existing DataFrame to maintain structure then write back
                combined_prelim_df = pd.concat([prelim_df_filtered, new_prelim_df], ignore_index=True)
                set_with_dataframe(prelim_sheet, combined_prelim_df, include_index=False)
                logging.info(f"Appended {len(new_prelim_rows_data)} PRELIMINARY rank rows to Preliminary Ranks sheet.")
        else:
            logging.info("No new PRELIMINARY rank rows to append to Preliminary Ranks sheet.")

        # --- START: Album Averages Sheet Logic ---
        if submission_status == 'final' and all_ranked_songs_from_js:
            logging.info("Entering FINAL ranking logic for Album Averages sheet.")

            total_score = sum(s.get('calculated_score', 0) for s in all_ranked_songs_from_js)
            num_ranked_songs = len(all_ranked_songs_from_js)
            average_album_score = round(total_score / num_ranked_songs, 2) if num_ranked_songs > 0 else 0

            logging.info(
                f"Calculated Album Average Score: {average_album_score} (Total: {total_score}, Count: {num_ranked_songs})")

            # This call will now return a DataFrame with 'times_ranked' as int and 'last_ranked_date' column ensured
            album_averages_df = get_album_averages_df(client, SPREADSHEET_ID,
                                                      album_averages_sheet_name)  # Ensure album_averages_sheet_name is defined (e.g., "Album Averages")

            matching_album_row_df = album_averages_df[album_averages_df['album_id'].astype(str) == str(album_id)]

            if not matching_album_row_df.empty:
                row_index_in_df = matching_album_row_df.index[0]
                times_ranked_current = album_averages_df.loc[row_index_in_df, 'times_ranked']
                logging.debug(
                    f"DEBUG: Current 'times_ranked' read from DF (should be int/0): '{times_ranked_current}' for album ID {album_id}")

                # --- THIS IS THE CHANGE: Remove try-except, direct addition ---
                times_ranked_new = times_ranked_current + 1
                # --- END CHANGE ---

                album_averages_df.loc[row_index_in_df, 'average_score'] = average_album_score
                album_averages_df.loc[row_index_in_df, 'times_ranked'] = times_ranked_new
                album_averages_df.loc[row_index_in_df, 'last_ranked_date'] = datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S')

                logging.info(
                    f"Updated existing row for album '{album_name}' (ID: {album_id}). New Times Ranked: {times_ranked_new}")
            else:
                new_row_data = {
                    'album_id': album_id,
                    'album_name': album_name,
                    'artist_name': artist_name,
                    'average_score': average_album_score,
                    'times_ranked': 1,  # Always 1 for a truly new entry
                    'last_ranked_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                new_row_df = pd.DataFrame([new_row_data], columns=album_averages_df.columns)
                album_averages_df = pd.concat([album_averages_df, new_row_df], ignore_index=True)
                logging.info(f"Added new entry for album '{album_name}' (ID: {album_id}) to Album Averages sheet.")

            album_averages_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(album_averages_sheet_name)
            set_with_dataframe(album_averages_sheet, album_averages_df, include_index=False, include_column_header=True)
            logging.info(f"Successfully wrote updated Album Averages DataFrame to sheet.")
        else:
            logging.info(
                "Skipping Album Averages update: Submission status is not 'final' OR no ranked songs submitted.")
        # --- END: Album Averages Sheet Logic ---

        flash('Rankings submitted successfully!', "success")
        logging.info(f"--- SUBMIT RANKINGS END (Redirecting to album) ---\n")

        return redirect(url_for('load_albums_by_artist_route', artist_name=artist_name))

    except Exception as e:
        logging.critical(f"\n🔥 CRITICAL ERROR in /submit_rankings route: {e}", exc_info=True)
        flash(f"An unexpected error occurred during submission: {e}", "error")
        return redirect(url_for('index'))
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/load_albums_by_artist", methods=["GET", "POST"])
def load_albums_by_artist_route():
    artist_name = None # Initialize artist_name

    if request.method == "POST":
        # For initial search (from a form)
        artist_name = request.form["artist_name"]
    elif request.method == "GET":
        # For redirect after ranking (from url_for passing it as a query param)
        artist_name = request.args.get("artist_name")

    if not artist_name:
        # Handle cases where artist_name isn't found (e.g., direct GET without param)
        flash("Artist name not provided. Please search for an artist.")
        return redirect(url_for('index')) # Redirect to your home/search page

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
    logging.debug(f"*** DEBUG: Album Averages DataFrame IMMEDIATELY after get_as_dataframe ***")
    logging.debug(f"*** DEBUG: DataFrame Empty: {album_averages_df.empty}")
    logging.debug(f"*** DEBUG: DataFrame Shape: {album_averages_df.shape}")
    logging.debug(f"*** DEBUG: DataFrame Columns: {album_averages_df.columns.tolist()}")
    logging.debug(f"*** DEBUG: DataFrame Head (first 10 rows): \n{album_averages_df.head(10).to_string()}")
    logging.debug(f"*** DEBUG: DataFrame Tail (last 10 rows): \n{album_averages_df.tail(10).to_string()}")
    logging.debug(
        f"*** DEBUG: DataFrame Values (as list of lists, first 5 rows): {album_averages_df.values.tolist()[:5]}")

    # Create a dictionary for quick lookup of averages/times ranked
    album_metadata = {}
    processed_rows_count = 0
    if not album_averages_df.empty: # Added check for empty df
        for _, row in album_averages_df.iterrows():
            processed_rows_count += 1
            # --- FIX: Use 'album_id' as key and correct column names from the sheet ---
            album_id_from_sheet = str(row.get("album_id", "")).strip()  # Use 'album_id'
            average_score_from_sheet = row.get("average_score", None)  # Use 'average_score'
            times_ranked_from_sheet = row.get("times_ranked", 0)
            last_ranked_date_from_sheet = row.get("last_ranked_date", "")
            logging.debug(f"DEBUG: Raw row from Album Averages sheet (loop): {row.to_dict()}")
            logging.debug(
                f"DEBUG: Processing sheet row in loop: ID='{album_id_from_sheet}', Avg='{average_score_from_sheet}', Times='{times_ranked_from_sheet}'")

            if album_id_from_sheet:  # Only add if album_id is not empty
                album_metadata[album_id_from_sheet] = {
                    "average_score": average_score_from_sheet,
                    "times_ranked": times_ranked_from_sheet,
                    "last_ranked_date": last_ranked_date_from_sheet
                }
                logging.debug(
                    f"DEBUG: Stored in album_metadata[{album_id_from_sheet}]: Avg={album_metadata[album_id_from_sheet]['average_score']}, Times={album_metadata[album_id_from_sheet]['times_ranked']}")
    logging.debug(f"DEBUG: Completed processing {processed_rows_count} rows from Album Averages DataFrame (loop).")
    logging.debug(f"DEBUG: Loaded {len(album_metadata)} album metadata entries from sheet into dict.")
    prelim_sheet_name = "Preliminary Ranks"
    prelim_ranked_albums_ids = set()  # Store album_ids for consistency
    try:
        prelim_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(prelim_sheet_name)
        prelim_sheet_data = get_as_dataframe(prelim_sheet, evaluate_formulas=False).fillna("")

        if not prelim_sheet_data.empty:
            current_artist_prelim_ranks = prelim_sheet_data[
                prelim_sheet_data["artist_name"].astype(str).str.strip().str.lower() == artist_name.strip().lower()
                ]
            for _, row in current_artist_prelim_ranks.iterrows():
                # --- FIX: Use 'album_id' for prelim ranks lookup as well ---
                album_id_p = str(row.get('album_id', '')).strip()
                prelim_rank_value = row.get('prelim_rank')
                if album_id_p and prelim_rank_value not in ["", None, 0, "0", "0.0"]:
                    prelim_ranked_albums_ids.add(album_id_p)
                # --- END FIX ---
        logging.debug(
            f"DEBUG: Found {len(prelim_ranked_albums_ids)} albums with preliminary ranks for '{artist_name}'.")

    except gspread.exceptions.WorksheetNotFound:
        logging.warning(
            f"WARNING: Preliminary Ranks sheet '{prelim_sheet_name}' not found. No prelim rank check for pause icon.")
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
    return render_template("select_album.html", artist_name=artist_name, albums=albums_for_template)
@app.route("/ranking_page")
def ranking_page():
    sheet_rows = load_google_sheet_data()
    group_bins = group_ranked_songs(sheet_rows)
    return render_template("album.html", group_bins=group_bins)


@app.route("/save_global_rankings", methods=["POST"])
def save_global_rankings():
    try:
        data = request.json
        global_ranked_data = data.get('global_ranked_data', [])
        prelim_rank_data = data.get('prelim_rank_data', [])
        current_album_id = data.get('current_album_id')  # Will be passed from JS
        # Added to get album/artist name directly from frontend if no songs ranked globally
        album_name_from_frontend = data.get('album_name_from_frontend')
        artist_name_from_frontend = data.get('artist_name_from_frontend') # Assuming get_gsheet_client() is defined

        # --- Handle FINAL Ranks (Main Ranking Sheet) ---
        main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

        # Always read all existing data to filter, then rewrite. This is safer than just appending.
        # This approach ensures that if a song is unranked from anywhere, it's removed.
        all_existing_main_df = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("")

        # Filter out all existing rows that belong to the current album being saved
        # This assumes 'Spotify Album ID' is the column name in your main sheet
        updated_main_df = all_existing_main_df[
            all_existing_main_df['Spotify Album ID'].astype(str) != str(current_album_id)
            ]

        if global_ranked_data:
            global_ranked_df = pd.DataFrame(global_ranked_data)

            global_ranked_df = global_ranked_df.rename(columns={
                'song_id': 'Spotify Song ID',
                'song_name': 'Song Name',
                'rank_group': 'Rank Group',
                'ranking': 'Ranking',
                'position_in_group': 'Position In Group',
                'album_id': 'Spotify Album ID',
                'album_name': 'Album Name',
                'artist_name': 'Artist Name',
                'ranking_status': 'Ranking Status'
            })

            final_sheet_columns = [
                'Spotify Album ID', 'Album Name', 'Artist Name',
                'Spotify Song ID', 'Song Name', 'Rank Group',
                'Ranking', 'Position In Group', 'Ranking Status'
            ]
            global_ranked_df = global_ranked_df[final_sheet_columns]

            # Append the new global_ranked_data for the current album
            updated_main_df = pd.concat([updated_main_df, global_ranked_df], ignore_index=True)

        # Write the combined (filtered existing + new current album data) back to the main sheet
        main_sheet.clear()  # Clears headers too
        set_with_dataframe(main_sheet, updated_main_df, include_index=False)
        logging.debug(f"Successfully updated main sheet with {len(updated_main_df)} global final ranked songs.")

        # --- Handle PRELIMINARY Ranks (Preliminary Ranks Sheet) ---
        prelim_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(PRELIM_SHEET_NAME)
        existing_prelim_df = get_as_dataframe(prelim_sheet, evaluate_formulas=False).fillna("")

        # Filter out old prelim ranks for the current album before adding new ones
        updated_prelim_df_list_for_df = []
        if not existing_prelim_df.empty and current_album_id:
            # Keep prelim ranks that are NOT from the current album being ranked
            updated_prelim_df_list_for_df = existing_prelim_df[
                existing_prelim_df["album_id"].astype(str) != str(current_album_id)
                ]

        # Convert new prelim_rank_data to DataFrame and concat
        if prelim_rank_data:
            new_prelim_df = pd.DataFrame(prelim_rank_data)
            # Ensure column order matches your prelim sheet exactly
            prelim_sheet_columns = [
                'song_id', 'album_id', 'album_name', 'artist_name', 'prelim_rank'
            ]
            new_prelim_df = new_prelim_df[prelim_sheet_columns]

            if not updated_prelim_df_list_for_df.empty:
                final_prelim_df = pd.concat([updated_prelim_df_list_for_df, new_prelim_df], ignore_index=True)
            else:
                final_prelim_df = new_prelim_df  # No existing prelims to concat with
        else:
            final_prelim_df = updated_prelim_df_list_for_df  # Only existing prelims (not current album)

        # Write the combined prelim data back to the prelim sheet
        prelim_sheet.clear()
        if not final_prelim_df.empty:  # Only write if there's data to write
            set_with_dataframe(prelim_sheet, final_prelim_df, include_index=False)
            logging.debug(f"Successfully updated prelim sheet with {len(final_prelim_df)} preliminary ranked songs.")
        else:
            logging.debug("No preliminary ranked songs to save. Preliminary sheet cleared.")

        # --- NEW: Calculate and Save Album Averages (THIS BLOCK WAS INDENTED INCORRECTLY) ---
        # This block now runs always, after song rankings are handled.
            # --- CORRECTED: Calculate and Save Album Averages ---
        album_averages_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(album_averages_sheet_name)
        album_averages_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)

        # Calculate the average score from the CURRENT submission
        total_score = 0
        song_count = 0
        album_name_for_update = album_name_from_frontend
        artist_name_for_update = artist_name_from_frontend

        for song in global_ranked_data:
            if str(song.get('album_id')) == str(current_album_id):
                total_score += song.get('ranking', 0)
                song_count += 1
                album_name_for_update = song.get('album_name', album_name_for_update)
                artist_name_for_update = song.get('artist_name', artist_name_for_update)

        current_average_score = round(total_score / song_count, 2) if song_count > 0 else 0

        # Find if this album already has an entry
        matching_album_row = album_averages_df[album_averages_df['album_id'].astype(str) == str(current_album_id)]

        if not matching_album_row.empty:
            # Album exists, UPDATE its row
            row_index = matching_album_row.index[0]

            # Safely get current times_ranked, default to 0 if not a number
            current_times_ranked = pd.to_numeric(album_averages_df.loc[row_index, 'times_ranked'],
                                                 errors='coerce').fillna(0).astype(int)

            # Increment the rank count by 1
            new_times_ranked = current_times_ranked + 1

            # Update the DataFrame
            album_averages_df.loc[row_index, 'average_score'] = current_average_score
            album_averages_df.loc[row_index, 'times_ranked'] = new_times_ranked
            album_averages_df.loc[row_index, 'last_ranked_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logging.info(f"Updated album '{album_name_for_update}'. New Times Ranked: {new_times_ranked}")

        else:
            # Album is new, ADD a new row
            new_row_data = {
                'album_id': current_album_id,
                'album_name': album_name_for_update,
                'artist_name': artist_name_for_update,
                'average_score': current_average_score,
                'times_ranked': 1,  # It's the first time, so it's 1
                'last_ranked_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            new_row_df = pd.DataFrame([new_row_data])
            album_averages_df = pd.concat([album_averages_df, new_row_df], ignore_index=True)
            logging.info(f"Added new album '{album_name_for_update}' to averages. Times Ranked: 1")

        # Write the entire updated DataFrame back to the sheet
        set_with_dataframe(album_averages_sheet, album_averages_df, include_index=False)
        logging.debug(f"Successfully wrote updated Album Averages to sheet for album {current_album_id}.")

        return jsonify({'status': 'success', 'message': 'Global, Preliminary, and Album Averages saved successfully!'})

    except Exception as e:
        import traceback
        logging.error("\n🔥 ERROR in /save_global_rankings route:")  # Use logging instead of print
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'An unexpected error occurred: {e}'}), 500

@app.route("/view_album", methods=["POST", "GET"])
def view_album():
    global sp

    try:
        # Get album details from form (POST) or URL parameters (GET)
        # This handles redirects from /submit_rankings (GET) and direct album selection (POST)
        if request.method == 'POST':
            album_id = request.form.get("album_id")
            album_name = request.form.get("album_name")
            artist_name = request.form.get("artist_name")
            album_cover_url = request.form.get("album_cover_url")
        else: # GET request
            album_id = request.args.get("album_id")
            album_name = request.args.get("album_name")
            artist_name = request.args.get("artist_name")
            album_cover_url = request.args.get("album_cover_url")

        # Basic validation: Redirect to index if essential album info is missing
        if not album_id or not album_name or not artist_name or not album_cover_url:
            flash("Missing album details. Please select an album properly.", "warning")
            print(f"DEBUG: Redirecting to index due to missing album details. ID: {album_id}, Name: {album_name}, Artist: {artist_name}, Cover: {album_cover_url}")
            return redirect(url_for('index'))

        print(f"\n--- VIEW ALBUM START ---")
        print(f"DEBUG: Received album_id: {album_id}")
        print(f"DEBUG: Received album_name: {album_name}")
        print(f"DEBUG: Received artist_name: {artist_name}")

        # Fetch album data from Spotify (using the external spotify_logic.py function)
        print(f"DEBUG: Loading album data from Spotify for ID: {album_id}")
        try:
            # Check if the global 'sp' object is initialized
            if sp is None:
                raise Exception("Spotify client (sp) is not initialized. Cannot fetch album data.")

            album_data = load_album_data(sp, album_id) # Pass sp as the first argument
        except Exception as e:
            flash(f"Error loading album data from Spotify: {e}", "error")
            print(f"ERROR: Error loading album data from Spotify for ID {album_id}: {e}")
            return redirect(url_for('index'))

        # Prepare base album data for template
        album_data_for_template = {
            'album_name': album_data['album_name'],
            'artist_name': album_data['artist_name'],
            'album_cover_url': album_data['album_cover_url'],
            'url': album_data.get('url', ''), # Use .get for safety
            'album_id': album_id,
            'songs': album_data['songs'], # This is the full list of all songs from Spotify
            'bg_color': album_data.get('bg_color', '#121212') # Default background if not provided
        }
        print(f"DEBUG: Extracted for template: Name='{album_data_for_template['album_name']}', Artist='{album_data_for_template['artist_name']}', Cover='{album_data_for_template['album_cover_url']}'")


        # --- Load existing Final Ranks from Main Ranking Sheet ---
        main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        print(f"DEBUG: Accessing main ranking sheet: '{SHEET_NAME}'.")
        # Fetch all data; evaluation formulas=False as we're just reading raw data for display
        main_sheet_data = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("")

        # Filter for ONLY FINAL rankings (across ALL albums)
        # Ensure 'Ranking Status' column exists and is handled gracefully
        if "Ranking Status" in main_sheet_data.columns:
            all_final_ranks = main_sheet_data[
                main_sheet_data["Ranking Status"].astype(str).str.lower() == "final"
                ]
        else:
            print("WARNING: 'Ranking Status' column not found in main sheet. Assuming all entries are final.")
            all_final_ranks = main_sheet_data  # Treat all as final if column missing

        print(f"DEBUG: Found {len(all_final_ranks)} existing FINAL entries in sheet (from all albums).")

        # --- Populate `rank_groups_for_js` with ALL globally ranked songs ---
        rank_groups_for_js = {f"{i / 2:.1f}": [] for i in range(1, 21)}  # Initialize all empty groups (0.5 to 10.0)

        for _, row in all_final_ranks.iterrows():  # Iterate through ALL final ranks
            try:
                # Ensure data types are correct and handle potential missing columns with .get()
                rank_group_key = f"{float(row.get('Rank Group', 0.0)):.1f}"  # Default to 0.0 if missing
                song_score = float(row.get('Ranking', 0.0))
                rank_position = int(row.get('Position In Group', 0))  # Default to 0 if missing
                song_id = str(row.get('Spotify Song ID', ''))
                song_name = str(row.get('Song Name', ''))

                # --- IMPORTANT: Capture album/artist info for the globally ranked song ---
                ranked_album_id = str(row.get('Spotify Album ID', ''))
                ranked_album_name = str(row.get('Album Name', ''))
                ranked_artist_name = str(row.get('Artist Name', ''))

                song_data = {
                    'song_id': song_id,
                    'song_name': song_name,
                    'rank_group': rank_group_key,
                    'calculated_score': song_score,
                    'rank_position': rank_position,  # Store original position to sort
                    'album_id': ranked_album_id,  # New: Album ID of the ranked song
                    'album_name': ranked_album_name,  # New: Album Name of the ranked song
                    'artist_name': ranked_artist_name  # New: Artist Name of the ranked song
                }

                if rank_group_key in rank_groups_for_js:
                    rank_groups_for_js[rank_group_key].append(song_data)
                else:
                    print(
                        f"WARNING: Found unexpected rank group '{rank_group_key}' in sheet for song {song_name}. Skipping.")

            except (ValueError, KeyError, TypeError) as e:
                print(
                    f"WARNING: Error parsing existing FINAL ranked song data for JS (Main Sheet - Global): {row.to_dict()} - {e}")

        # Sort songs within each group by their `rank_position` for correct display order
        for group_key in rank_groups_for_js:
            rank_groups_for_js[group_key].sort(key=lambda x: x.get('rank_position', 0))  # Use .get for safety

        total_songs_in_groups = sum(len(v) for v in rank_groups_for_js.values())
        print(
            f"DEBUG: Populated rank_groups_for_js with {total_songs_in_groups} FINAL ranked songs (from ALL albums for right panel).")

        # --- Load existing Preliminary Ranks from "Preliminary Ranks" Sheet ---
        # This part remains mostly the same, as preliminary ranks are still song-specific
        # and only relevant for the current album's unranked songs on the left panel.
        existing_prelim_ranks = {}  # To store song_id: prelim_rank_value for the current album
        prelim_sheet_name = "Preliminary Ranks"  # Assuming this is defined
        try:
            prelim_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(prelim_sheet_name)
            print(f"DEBUG: Accessing preliminary ranks sheet: '{prelim_sheet_name}'.")
            prelim_sheet_data = get_as_dataframe(prelim_sheet, evaluate_formulas=False).fillna("")

            # Filter for current album's preliminary ranks
            # Ensure 'album_id' column exists
            if "album_id" in prelim_sheet_data.columns:
                current_album_prelim_ranks = prelim_sheet_data[
                    (prelim_sheet_data["album_id"].astype(str) == str(album_id)) &
                    (prelim_sheet_data["prelim_rank"] != "")  # Ensure prelim_rank is not empty/null
                    ]
            else:
                print("WARNING: 'album_id' column not found in preliminary sheet. Cannot filter prelim ranks.")
                current_album_prelim_ranks = pd.DataFrame()  # Empty DataFrame if column missing

            for _, row in current_album_prelim_ranks.iterrows():
                song_id = str(row.get('song_id'))
                prelim_rank_value = row.get('prelim_rank')
                try:
                    existing_prelim_ranks[song_id] = float(prelim_rank_value)
                except (ValueError, TypeError):
                    print(
                        f"WARNING: Invalid prelim_rank value '{prelim_rank_value}' for song {song_id} in prelim sheet. Skipping.")
            print(f"DEBUG: Loaded {len(existing_prelim_ranks)} preliminary rank entries for album {album_name}.")
        except gspread.exceptions.WorksheetNotFound:
            print(
                f"WARNING: Preliminary Ranks sheet '{prelim_sheet_name}' not found. No prelim ranks loaded for {album_name}.")
        except Exception as e:
            print(f"ERROR: Error loading preliminary ranks from sheet '{prelim_sheet_name}': {e}")
            flash(f"Error loading preliminary ranks: {e}", "error")

        # --- Prepare `songs` for Left Panel (`album.html`'s `album.songs` loop) ---
        # This list will contain ALL songs from the current Spotify album,
        # but with flags indicating if they are already globally final-ranked.
        all_spotify_songs_with_flags = []
        for song_from_spotify in album_data['songs']:  # Iterate through ALL songs from Spotify API for current album
            song_id = str(song_from_spotify['song_id'])
            song_name = song_from_spotify['song_name']
            is_final_ranked_globally = False

            # Check if this song is present in any of the GLOBAL final rank groups (right panel)
            for group_key in rank_groups_for_js:
                # Use any() with a generator expression for efficiency
                if any(s['song_id'] == song_id for s in rank_groups_for_js[group_key]):
                    is_final_ranked_globally = True
                    break

            all_spotify_songs_with_flags.append({
                'song_name': song_name,
                'song_id': song_id,
                'already_ranked': is_final_ranked_globally,  # This flag tells JS to grey out/hide in left panel
                'prelim_rank': existing_prelim_ranks.get(song_id, '') if not is_final_ranked_globally else ''
            })
        album_data_for_template = {
            'album_name': album_data['album_name'],
            'artist_name': album_data['artist_name'],
            'album_cover_url': album_data['album_cover_url'],
            'url': album_data.get('url', ''),
            'album_id': album_id,
            'songs': all_spotify_songs_with_flags,  # This now includes global ranked flags
            'bg_color': album_data.get('bg_color', '#121212')
        }
        print(
            f"DEBUG: Prepared {len(album_data_for_template['songs'])} songs for left panel (all Spotify songs with global ranked flags).")

        print(f"DEBUG: Final album_data_for_template keys: {album_data_for_template.keys()}")
        print(
            f"DEBUG: Final rank_groups_for_js keys: {rank_groups_for_js.keys()} (Total songs: {total_songs_in_groups})")
        print(f"--- VIEW ALBUM END ---\n")

        return render_template('album.html',
                               album=album_data_for_template,
                               rank_groups=rank_groups_for_js  # Pass the globally populated dictionary for right panel
                               )

    except Exception as e:
        import traceback
        print("\n🔥 ERROR in /view_album route:")
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
