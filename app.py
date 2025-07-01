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


@app.route("/artist/<string:artist_name>")
def artist_page(artist_name):
    try:
        logging.info(f"--- Loading Artist Page for: {artist_name} ---")

        main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        all_songs_df = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("")
        all_albums_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)

        # 2. Filter data for the CURRENT artist
        # Ensure 'Ranking' is numeric for calculations
        all_songs_df['Ranking'] = pd.to_numeric(all_songs_df['Ranking'], errors='coerce')
        # Filter both dataframes
        artist_songs_df = all_songs_df[all_songs_df['Artist Name'].str.lower() == artist_name.lower()]
        artist_albums_df = all_albums_df[all_albums_df['artist_name'].str.lower() == artist_name.lower()]

        # 3. Calculate Overall Stats
        avg_song_score = artist_songs_df['Ranking'].mean()
        avg_album_score = artist_albums_df['average_score'].mean()
        artist_stats = {
            'avg_song_score': f"{avg_song_score:.2f}" if pd.notna(avg_song_score) else "N/A",
            'avg_album_score': f"{avg_album_score:.2f}" if pd.notna(avg_album_score) else "N/A",
            'total_songs_ranked': len(artist_songs_df)
        }

        # 4. Prepare Pie Chart Data
        # Group by the 'Rank Group' column and count songs in each
        pie_data = artist_songs_df['Rank Group'].value_counts().reset_index()
        pie_data.columns = ['rank_group', 'count']
        pie_chart_data = {
            'labels': pie_data['rank_group'].tolist(),
            'data': pie_data['count'].tolist()
        }

        # 5. Prepare Song Leaderboards
        # Artist-Specific Leaderboard
        artist_song_leaderboard = artist_songs_df.sort_values(by='Ranking', ascending=False).to_dict('records')

        # Universal Leaderboard
        all_songs_df_sorted = all_songs_df.sort_values(by='Ranking', ascending=False).reset_index(drop=True)
        universal_song_leaderboard = all_songs_df_sorted.head(100).to_dict('records')  # Show top 100 universal

        # 6. Prepare Album Leaderboard
        album_leaderboard = artist_albums_df.sort_values(by='average_score', ascending=False).to_dict('records')

        # 7. Pass all the data to the template
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

@app.route("/submit_rankings", methods=["POST"])
def submit_rankings():
    global sp, client  # Ensure access to global Spotify and GSheets client

    try:
        # Use request.get_json() to parse data sent from the JavaScript fetch
        data = request.get_json()
        if not data:
            flash("Invalid data received from browser.", "error")
            return redirect(url_for('index'))

        # Now, get all your variables from the parsed 'data' dictionary
        album_name = data.get("album_name")
        artist_name = data.get("artist_name")
        album_id = data.get("album_id")
        submission_status = data.get("status", "final")
        album_cover_url = data.get('album_cover_url')

        # The JSON data is already a list/dict, so no need for json.loads
        all_ranked_songs_from_js = data.get("all_ranked_data", [])
        prelim_ranks_from_js = data.get("prelim_rank_data", {})

        logging.info(f"\n--- SUBMIT RANKINGS START ---")
        logging.info(
            f"submit_rankings called for Album: '{album_name}' (ID: {album_id}), Artist: '{artist_name}', Status: '{submission_status}'")

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
            # --- CORRECTED BACKEND CODE ---
        except Exception as e:
            logging.warning(
                f"Could not fetch Spotify tracks for album {album_id}: {e}. Falling back to submission data for song names.",
                exc_info=True)
            # Fallback for final ranks (this part was already okay)
            spotify_tracks_for_album.update(
                {str(s.get('song_id')): s.get('song_name', f"Unknown Song {str(s.get('song_id', 'N/A'))}") for s in
                 all_ranked_songs_from_js})

            # THE FIX IS HERE: Correctly handle the prelim ranks list
            if prelim_ranks_from_js:
                for prelim_data in prelim_ranks_from_js:
                    song_id = str(prelim_data.get('song_id'))
                    if song_id and song_id not in spotify_tracks_for_album:
                        spotify_tracks_for_album[song_id] = f"Unknown Song {song_id}"

        # --- Main Ranking Sheet Operations ---
        main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        main_df = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("")

        submitted_song_ids = {str(s.get('song_id')) for s in all_ranked_songs_from_js}

        # Filter out ALL rows for the songs that are being re-submitted, regardless of album.
        if 'Spotify Song ID' in main_df.columns and submitted_song_ids:
            # The ~ operator inverts the boolean mask, keeping only rows whose ID is NOT in the set.
            main_df_filtered = main_df[~main_df['Spotify Song ID'].astype(str).isin(submitted_song_ids)]
            logging.info(f"Identified {len(submitted_song_ids)} songs for update. Removing old entries.")
        else:
            # If no songs are submitted or column is missing, just use the original DataFrame.
            main_df_filtered = main_df

        # Now, prepare the new rows from the submitted data.
        new_final_rows_data = []
        for ranked_song_data in all_ranked_songs_from_js:
            new_row = {
                'Album Name': ranked_song_data.get('album_name'),
                'Artist Name': ranked_song_data.get('artist_name'),
                'Spotify Album ID': ranked_song_data.get('album_id'),
                'Song Name': ranked_song_data.get('song_name'),
                'Ranking': ranked_song_data.get('calculated_score', 0.0),
                'Ranking Status': 'final',
                'Ranked Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Position In Group': str(ranked_song_data.get('position_in_group', '')),
                'Rank Group': str(ranked_song_data.get('rank_group')),
                'Spotify Song ID': str(ranked_song_data.get('song_id')),
            }
            new_final_rows_data.append(new_row)

        # Combine the dataframe that has old songs removed with the new updated song data.
        if new_final_rows_data:
            new_final_df = pd.DataFrame(new_final_rows_data)
            final_main_df = pd.concat([main_df_filtered, new_final_df], ignore_index=True)
        else:
            final_main_df = main_df_filtered

        set_with_dataframe(main_sheet, final_main_df, include_index=False, resize=True)
        logging.info(f"Wrote {len(final_main_df)} total rows back to main ranking sheet.")

        # --- PRELIMINARY RANKING SHEET OPERATIONS ---
        # --- CORRECTED PRELIMINARY RANKING SHEET OPERATIONS (Single Write Logic) ---
        prelim_sheet = None
        try:
            prelim_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(PRELIM_SHEET_NAME)
            prelim_df = get_as_dataframe(prelim_sheet, evaluate_formulas=False).fillna("")
        except gspread.exceptions.WorksheetNotFound:
            logging.warning(f"Preliminary Ranks worksheet '{PRELIM_SHEET_NAME}' not found. Will create if needed.")
            prelim_df = pd.DataFrame()  # Start with an empty DataFrame

        # 1. Filter out old prelim ranks for this album IN MEMORY
        if 'album_id' in prelim_df.columns:
            prelim_df_filtered = prelim_df[prelim_df['album_id'].astype(str) != str(album_id)]
        else:
            prelim_df_filtered = prelim_df

        # 2. Prepare new prelim rows by iterating over the LIST correctly
        new_prelim_rows_data = []
        if prelim_ranks_from_js:
            # THE FIX IS HERE: Loop through the list of objects
            for prelim_data in prelim_ranks_from_js:
                song_id = prelim_data.get('song_id')
                prelim_rank_value = prelim_data.get('prelim_rank')
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

        # 3. Combine the filtered old data with the new data
        if new_prelim_rows_data:
            new_prelim_df = pd.DataFrame(new_prelim_rows_data)
            final_prelim_df = pd.concat([prelim_df_filtered, new_prelim_df], ignore_index=True)
            logging.info(f"Prepared {len(new_prelim_rows_data)} PRELIMINARY rank rows.")
        else:
            final_prelim_df = prelim_df_filtered
            logging.info("No new PRELIMINARY rank rows to add.")

        # 4. Write the final, combined DataFrame back to the sheet ONCE
        # (Your logic for creating the sheet if it doesn't exist is good and remains)
        try:
            if prelim_sheet is None:  # If sheet didn't exist, we must create it now
                prelim_sheet_header_cols = ['album_id', 'album_name', 'artist_name', 'album_cover_url', 'song_id',
                                            'song_name', 'prelim_rank', 'timestamp']
                prelim_sheet = client.open_by_key(SPREADSHEET_ID).add_worksheet(PRELIM_SHEET_NAME, rows=1,
                                                                                cols=len(prelim_sheet_header_cols))
                prelim_sheet.append_row(prelim_sheet_header_cols)

            set_with_dataframe(prelim_sheet, final_prelim_df, include_index=False, resize=True)
            logging.info(f"Wrote {len(final_prelim_df)} total rows back to preliminary ranks sheet.")
        except Exception as e:
            logging.critical(f"CRITICAL ERROR: Could not write to or create Preliminary Ranks sheet: {e}",
                             exc_info=True)
            flash(f"Critical error with Preliminary Ranks sheet: {e}", "error")
            # Don't halt the whole submission; the final ranks might have worked.

        if submission_status == 'final' and all_ranked_songs_from_js:
            logging.info("Entering FINAL ranking logic for Album Averages sheet.")

            # THE FIX IS HERE: Filter out interludes BEFORE calculating the average.
            songs_for_average = [
                s for s in all_ranked_songs_from_js if s.get('rank_group') != 'I'
            ]

            # Now, calculate the average using the CLEANED list.
            total_score = sum(s.get('calculated_score', 0) for s in songs_for_average)
            num_ranked_songs = len(songs_for_average)
            average_album_score = round(total_score / num_ranked_songs, 2) if num_ranked_songs > 0 else 0

            logging.info(f"Filtered out interludes. Calculating average from {num_ranked_songs} songs.")
            logging.info(f"Calculated Correct Album Average Score: {average_album_score}")

            # Load the averages sheet
            album_averages_df = get_album_averages_df(client, SPREADSHEET_ID, album_averages_sheet_name)

            # Find the row for the current album
            match_index = album_averages_df.index[
                album_averages_df['album_id'].astype(str) == str(album_id)].tolist()

            if match_index:
                # Album EXISTS. Update its row.
                idx = match_index[0]
                times_ranked_new = album_averages_df.at[idx, 'times_ranked'] + 1
                album_averages_df.at[idx, 'average_score'] = average_album_score
                album_averages_df.at[idx, 'times_ranked'] = times_ranked_new
                album_averages_df.at[idx, 'last_ranked_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logging.info(f"Updated existing row for album '{album_name}'.")
            else:
                # Album is NEW. Append a new row.
                new_row_data = {
                    'album_id': album_id, 'album_name': album_name, 'artist_name': artist_name,
                    'average_score': average_album_score, 'times_ranked': 1,
                    'last_ranked_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                new_row_df = pd.DataFrame([new_row_data])
                album_averages_df = pd.concat([album_averages_df, new_row_df], ignore_index=True)
                logging.info(f"Added new entry for album '{album_name}'.")

            # Write the entire modified DataFrame back to the sheet.
            album_averages_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(album_averages_sheet_name)
            set_with_dataframe(album_averages_sheet, album_averages_df, include_index=False, resize=True)
            logging.info("Successfully wrote updated Album Averages DataFrame to sheet.")
        # --- END: Album Averages Sheet Logic ---

        # --- NEW SUCCESS RESPONSE (JSON) ---
        logging.info(f"--- SUBMIT RANKINGS END (Success) ---\n")

        return jsonify({
            'status': 'success',
            'message': 'Rankings submitted successfully!',
            'artist_name': artist_name  # Send the artist_name back to the browser
        })

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
        artist_name = request.form["artist_name"]
        # Instead of rendering a page here, redirect to the new artist page
        return redirect(url_for('artist_page', artist_name=artist_name))
    elif request.method == "GET":
        # For redirect after ranking (from url_for passing it as a query param)
        artist_name = request.args.get("artist_name")

    if not artist_name:
        # Handle cases where artist_name isn't found (e.g., direct GET without param)
        flash("Artist name not provided. Please search for an artist.")
        return redirect(url_for('index')) # Redirect to your home/search page
    print("\n--- LOADING ALBUM LIST FOR ARTIST:", artist_name, "---")

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
    processed_rows_count = 0
    if not album_averages_df.empty: # Added check for empty df
        for _, row in album_averages_df.iterrows():
            processed_rows_count += 1
            # --- FIX: Use 'album_id' as key and correct column names from the sheet ---
            album_id_from_sheet = str(row.get("album_id", "")).strip()
            print(f"  - Found row with album_id: '{album_id_from_sheet}'")
            # Use 'album_id'
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

        if not all_final_ranks_df.empty and 'Spotify Album ID' in all_final_ranks_df.columns:
            other_albums_df = all_final_ranks_df[all_final_ranks_df['Spotify Album ID'] != album_id]
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
