from flask import Flask, render_template, request, redirect, url_for, jsonify
import gspread
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from google.oauth2.service_account import Credentials
from datetime import datetime
import pandas as pd
from colorthief import ColorThief
import requests
from io import BytesIO
import sys
import os
import json
from google.oauth2 import service_account
from spotify_logic import load_album_data, get_albums_by_artist, extract_album_id
from google.oauth2.service_account import Credentials
# Setup
import json
from google.oauth2.service_account import Credentials
import gspread
import traceback
from collections import Counter
import spotipy # <--- ADD THIS
from spotipy.oauth2 import SpotifyClientCredentials

import os
from dotenv import load_dotenv # New import
load_dotenv()

creds_info = json.loads(os.environ['GOOGLE_SERVICE_ACCOUNT_JSON'])

creds = Credentials.from_service_account_info(
    creds_info,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)

client = gspread.authorize(creds)
SPREADSHEET_ID = '15E4b-DWSYP9AzbAzSviqkW-jEOktbimPlmhNIs_d5jc'
SHEET_NAME = "Sheet1"
album_averages_sheet_name = "Album Averages"

from flask import Flask, request, redirect, url_for, flash, json # Ensure json is imported
from datetime import datetime
# from your_gspread_utils import get_as_dataframe # Make sure this is imported correctly
# from your_main_app import client, SPREADSHEET_ID, SHEET_NAME # Adjust imports as per your setup

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_temporary_dev_key')

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
        except:
            continue  # skip if bad data

    return group_bins
def get_dominant_color(image_url):
    try:
        response = requests.get(image_url)
        color_thief = ColorThief(BytesIO(response.content))
        rgb = color_thief.get_color(quality=1)
        return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
    except Exception:
        return "#ffffff"

def get_album_stats(album_name, artist_name, df=None):
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    if df is None:
        df = get_as_dataframe(sheet, evaluate_formulas=True).fillna("")

    # Normalize keys
    album_key  = album_name.strip().lower()
    artist_key = artist_name.strip().lower()
    df = df.dropna(subset=["Album Name", "Artist Name"])
    df["Album Name"]   = df["Album Name"].astype(str).str.strip().str.lower()
    df["Artist Name"]  = df["Artist Name"].astype(str).str.strip().str.lower()
    df["Ranking Status"] = df["Ranking Status"].fillna("")

    # Filter to this album/artist (ignore session rows)
    album_df = df[
        (df["Album Name"] == album_key) &
        (df["Artist Name"] == artist_key) &
        (df["Song Name"] != "__ALBUM_SESSION__")
    ]
    if album_df.empty:
        return {
            'finalized_rank_count': 0,
            'last_final_avg_rank': None,
            'last_final_rank_date': None,
            'ranking_status': None
        }

    # Is there any paused row?
    paused_exists = album_df["Ranking Status"].str.contains("paused", case=False).any()

    # Finalized rows only:
    finalized_df = album_df[album_df["Ranking Status"] == "final"].copy()
    if finalized_df.empty:
        return {
            'finalized_rank_count': 0,
            'last_final_avg_rank': None,
            'last_final_rank_date': None,
            'ranking_status': "paused" if paused_exists else None
        }

    # Convert Ranking to numeric, compute overall average across *all* final rows
    finalized_df["Ranking"] = pd.to_numeric(finalized_df["Ranking"], errors="coerce")
    avg_rank = round(finalized_df["Ranking"].mean(), 2)

    # Most recent final date
    finalized_df["Ranked Date"] = pd.to_datetime(finalized_df["Ranked Date"], errors="coerce")
    latest_date = finalized_df["Ranked Date"].max()
    formatted_date = latest_date.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(latest_date) else None

    return {
        'finalized_rank_count': len(finalized_df),
        'last_final_avg_rank': avg_rank,
        'last_final_rank_date': formatted_date,
        'ranking_status': "paused" if paused_exists else "final"
    }


from flask import request, redirect, url_for, flash

from datetime import datetime
import json
import pandas as pd


from datetime import datetime
import json
import pandas as pd
# Assuming sp and client are defined globally in your app.py
# from your_gspread_utils import get_as_dataframe # Assuming this is separate

@app.route("/submit_rankings", methods=["POST"])
def submit_rankings():
    # Ensure `sp` is globally accessible
    global sp

    try:
        album_name = request.form.get("album_name")
        artist_name = request.form.get("artist_name")
        album_id = request.form.get("album_id")
        # CRITICAL FIX: Get submission_status from the form, not hardcode
        submission_status = request.form.get("status", "final")
        album_cover_url = request.form.get('album_cover_url') # Ensure this is passed for redirects

        print(f"\n--- SUBMIT RANKINGS START ---")
        print(
            f"DEBUG: submit_rankings called for Album: '{album_name}' (ID: {album_id}), Artist: '{artist_name}', Status: '{submission_status}'")

        all_ranked_data_json = request.form.get("all_ranked_data", "[]")
        prelim_rank_data_json = request.form.get("prelim_rank_data", "{}")

        try:
            all_ranked_songs_from_js = json.loads(all_ranked_data_json)
            # Ensure calculated_score is a float, defaulting to 0.0 if invalid
            for song in all_ranked_songs_from_js:
                score = song.get('calculated_score')
                try:
                    song['calculated_score'] = float(score) if score is not None else 0.0
                except (ValueError, TypeError):
                    song['calculated_score'] = 0.0
        except json.JSONDecodeError as e:
            flash(f"Error parsing ranked songs data: {e}", "error")
            print(f"ERROR: JSONDecodeError on all_ranked_data: {e}")
            return redirect(url_for('index'))

        try:
            prelim_ranks_from_js = json.loads(prelim_rank_data_json)
            # Ensure preliminary scores are floats, defaulting to 0.0 if invalid
            for song_id, score in prelim_ranks_from_js.items():
                try:
                    prelim_ranks_from_js[song_id] = float(score) if score is not None else 0.0
                except (ValueError, TypeError):
                    prelim_ranks_from_js[song_id] = 0.0
        except json.JSONDecodeError as e:
            flash(f"Error parsing prelim ranks data: {e}", "error")
            print(f"ERROR: JSONDecodeError on prelim_rank_data: {e}")
            return redirect(url_for('index'))

        print(
            f"DEBUG: Parsed all_ranked_songs_from_js ({len(all_ranked_songs_from_js)} songs): {all_ranked_songs_from_js[:2] if all_ranked_songs_from_js else '[]'}")
        print(
            f"DEBUG: Parsed prelim_ranks_from_js ({len(prelim_ranks_from_js)} entries): {list(prelim_ranks_from_js.items())[:2] if prelim_ranks_from_js else '[]'}")


        # --- Fetch Song Names (centralized for both main and prelim sheets) ---
        spotify_tracks_for_album = {}
        album_spotify_data = {'tracks': {'items': []}} # Initialize to prevent UnboundLocalError
        try:
            if sp is not None:
                album_spotify_data = sp.album(album_id)
                for track in album_spotify_data['tracks']['items']:
                    spotify_tracks_for_album[track['id']] = track['name']
            else:
                raise Exception("Spotify client (sp) not initialized.")
        except Exception as e:
            print(f"WARNING: Could not fetch Spotify tracks for album {album_id}: {e}. Falling back to submission data for song names.")
            # Fallback for final ranks:
            spotify_tracks_for_album.update({
                str(s.get('song_id')): s.get('song_name', f"Unknown Song {str(s.get('song_id', 'N/A'))}")
                for s in all_ranked_songs_from_js
            })
            # Fallback for prelim ranks:
            spotify_tracks_for_album.update({
                song_id: f"Unknown Song {song_id}"
                for song_id in prelim_ranks_from_js.keys()
                if song_id not in spotify_tracks_for_album
            })


        # --- Main Ranking Sheet Operations ---
        main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

        # 1. DELETE ALL EXISTING ROWS FOR THIS ALBUM IN MAIN SHEET
        all_main_sheet_values = main_sheet.get_all_values()
        main_sheet_header = all_main_sheet_values[0]
        main_sheet_data_rows = all_main_sheet_values[1:]

        main_rows_to_delete_indices = []
        try:
            album_id_col_idx_main = main_sheet_header.index("Spotify Album ID")
        except ValueError:
            print("ERROR: 'Spotify Album ID' column not found in main sheet header. Cannot reliably delete rows.")
            flash("Configuration error: 'Spotify Album ID' column missing in sheet.", "error")
            return redirect(url_for('index'))

        for i, row in enumerate(main_sheet_data_rows):
            if len(row) > album_id_col_idx_main and str(row[album_id_col_idx_main]) == str(album_id):
                main_rows_to_delete_indices.append(i + 2) # +2 for 1-based indexing and header

        main_rows_to_delete_indices.sort(reverse=True)
        if main_rows_to_delete_indices:
            for idx in main_rows_to_delete_indices:
                main_sheet.delete_rows(idx)
            print(f"DEBUG: Deleted {len(main_rows_to_delete_indices)} existing FINAL rank rows for album '{album_name}' (ID: {album_id}).")
        else:
            print(f"DEBUG: No existing FINAL rank rows found for album '{album_name}' (ID: {album_id}) to delete.")

        # 2. PREPARE AND APPEND NEW FINAL RANK ROWS TO MAIN SHEET
        new_final_rows_for_sheet = []
        column_names_main_sheet = [
            'Album Name', 'Artist Name', 'Spotify Album ID', 'Song Name', 'Ranking',
            'Ranking Status', 'Ranked Date', 'Position In Group', 'Rank Group',
            'Spotify Song ID', 'Preliminary Rank'
        ]

        if all_ranked_songs_from_js:
            for ranked_song_data in all_ranked_songs_from_js:
                song_id = str(ranked_song_data['song_id'])
                song_name = spotify_tracks_for_album.get(song_id, f"Unknown Song {song_id}") # Use centralized lookup
                prelim_rank_val = ranked_song_data.get('prelim_rank', '') # Use the prelim rank passed from JS if exists

                row_values_dict = {
                    'Album Name': album_name,
                    'Artist Name': artist_name,
                    'Spotify Album ID': album_id,
                    'Song Name': song_name,
                    'Spotify Song ID': song_id,
                    'Ranking': f"{ranked_song_data['calculated_score']:.2f}",
                    'Ranking Status': 'final', # Always 'final' for songs in all_ranked_songs_from_js
                    'Ranked Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Position In Group': str(ranked_song_data['rank_position']),
                    'Rank Group': f"{float(ranked_song_data['rank_group']):.1f}",
                    'Preliminary Rank': prelim_rank_val
                }
                ordered_row = [row_values_dict.get(col, '') for col in column_names_main_sheet]
                new_final_rows_for_sheet.append(ordered_row)

            if new_final_rows_for_sheet:
                main_sheet.append_rows(new_final_rows_for_sheet, value_input_option='USER_ENTERED')
                print(f"DEBUG: Appended {len(new_final_rows_for_sheet)} FINAL rank rows to main sheet.")
        else:
            print("DEBUG: No new FINAL rank rows to append to main sheet.")


        # --- PRELIMINARY RANKING SHEET OPERATIONS (NEW SECTION) ---
        prelim_sheet_name = "Preliminary Ranks"
        prelim_sheet_header_cols = [
            'album_id', 'album_name', 'artist_name', 'album_cover_url',
            'song_id', 'song_name', 'prelim_rank', 'timestamp'
        ]
        try:
            prelim_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(prelim_sheet_name)
            print(f"DEBUG: Successfully opened Preliminary Ranks sheet: '{prelim_sheet_name}'")
        except gspread.exceptions.WorksheetNotFound:
            print(f"ERROR: Preliminary Ranks worksheet '{prelim_sheet_name}' not found. Attempting to create it.")
            try:
                prelim_sheet = client.open_by_key(SPREADSHEET_ID).add_worksheet(prelim_sheet_name, rows=1, cols=len(prelim_sheet_header_cols))
                prelim_sheet.append_row(prelim_sheet_header_cols)
                print(f"DEBUG: Successfully created new Preliminary Ranks sheet with header: {prelim_sheet_header_cols}")
            except Exception as e:
                print(f"CRITICAL ERROR: Failed to create Preliminary Ranks sheet '{prelim_sheet_name}': {e}")
                flash(f"Critical error: Failed to create Preliminary Ranks sheet. Check permissions. {e}", "error")
                return redirect(url_for('index'))
        except Exception as e:
            print(f"ERROR: Could not open Preliminary Ranks sheet '{prelim_sheet_name}': {e}")
            flash(f"Error accessing preliminary ranks sheet: {e}", "error")
            return redirect(url_for('index'))

        # Delete existing preliminary ranks for this album
        all_prelim_sheet_values = prelim_sheet.get_all_values()
        prelim_sheet_header_actual = all_prelim_sheet_values[0] if all_prelim_sheet_values else []
        prelim_sheet_data_rows = all_prelim_sheet_values[1:] if all_prelim_sheet_values else []

        prelim_rows_to_delete_indices = []
        try:
            album_id_col_idx_prelim = prelim_sheet_header_actual.index("album_id")
        except ValueError:
            print("ERROR: 'album_id' column not found in Preliminary Ranks sheet header. Cannot reliably delete rows.")
            flash("Configuration error: 'album_id' column missing in Preliminary Ranks sheet header.", "error")
            return redirect(url_for('index'))

        for i, row in enumerate(prelim_sheet_data_rows):
            if len(row) > album_id_col_idx_prelim and str(row[album_id_col_idx_prelim]) == str(album_id):
                prelim_rows_to_delete_indices.append(i + 2) # +2 for 1-based indexing and header

        prelim_rows_to_delete_indices.sort(reverse=True)
        if prelim_rows_to_delete_indices:
            for idx in prelim_rows_to_delete_indices:
                prelim_sheet.delete_rows(idx)
            print(f"DEBUG: Deleted {len(prelim_rows_to_delete_indices)} existing PRELIMINARY rank rows for album '{album_name}' (ID: {album_id}).")
        else:
            print(f"DEBUG: No existing PRELIMINARY rank rows found for album '{album_name}' (ID: {album_id}) to delete.")

        # Append new preliminary ranks if any were submitted
        new_prelim_rows_for_sheet = []
        if prelim_ranks_from_js:
            for song_id, prelim_rank_value in prelim_ranks_from_js.items():
                song_name = spotify_tracks_for_album.get(song_id, f"Unknown Song (ID: {song_id})")
                new_prelim_rows_for_sheet.append([
                    album_id,
                    album_name,
                    artist_name,
                    album_cover_url,
                    song_id,
                    song_name,
                    prelim_rank_value,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ])
            if new_prelim_rows_for_sheet:
                prelim_sheet.append_rows(new_prelim_rows_for_sheet, value_input_option='USER_ENTERED')
                print(f"DEBUG: Appended {len(new_prelim_rows_for_sheet)} PRELIMINARY rank rows to Preliminary Ranks sheet.")
        else:
            print("DEBUG: No new PRELIMINARY rank rows to append to Preliminary Ranks sheet.")


        # --- START: Album Averages Sheet Logic ---
        # This logic correctly depends on 'final' status and actual ranked songs
        if submission_status == 'final' and all_ranked_songs_from_js:
            print("DEBUG: Entering FINAL ranking logic for Album Averages sheet.")

            total_score = sum(s.get('calculated_score', 0) for s in all_ranked_songs_from_js)
            num_ranked_songs = len(all_ranked_songs_from_js)
            average_album_score = round(total_score / num_ranked_songs, 2) if num_ranked_songs > 0 else 0

            print(
                f"DEBUG: Calculated Album Average Score: {average_album_score} (Total: {total_score}, Count: {num_ranked_songs})")

            album_averages_sheet_name = "Album Averages"
            try:
                album_averages_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(album_averages_sheet_name)
                print(f"DEBUG: Successfully opened Album Averages sheet: '{album_averages_sheet_name}'")
            except Exception as e:
                print(f"ERROR: Could not open Album Averages sheet '{album_averages_sheet_name}': {e}")
                flash(f"Error accessing album averages sheet: {e}", "error")
                return redirect(url_for('load_albums_by_artist_route', artist_name=artist_name))

            album_averages_df = get_as_dataframe(album_averages_sheet, evaluate_formulas=True).fillna("")

            form_album_name_lower = album_name.strip().lower()
            form_artist_name_lower = artist_name.strip().lower()

            matching_album_row = album_averages_df[
                (album_averages_df["Album Name"].astype(str).str.strip().str.lower() == form_album_name_lower) &
                (album_averages_df["Artist Name"].astype(str).str.strip().str.lower() == form_artist_name_lower)
                ]

            if not matching_album_row.empty:
                album_row_idx = matching_album_row.index[0] + 2
                times_ranked = matching_album_row.iloc[0].get("Times Ranked", 0)
                try:
                    times_ranked = int(times_ranked) + 1
                except ValueError:
                    print(
                        f"WARNING: 'Times Ranked' value '{times_ranked}' is not an integer. Resetting to 1 for '{album_name}'.")
                    times_ranked = 1

                print(
                    f"DEBUG: Found existing row for '{album_name}' at sheet row {album_row_idx}. New Times Ranked: {times_ranked}")

                update_cells_avg = []

                # Dynamic column finding for Album Averages sheet (more robust)
                avg_col_idx = -1
                times_col_idx = -1
                date_col_idx = -1
                avg_sheet_header = album_averages_sheet.row_values(1)  # Get header again for dynamic lookup
                try:
                    avg_col_idx = avg_sheet_header.index('Average Score') + 1  # +1 for gspread 1-based indexing
                    times_col_idx = avg_sheet_header.index('Times Ranked') + 1
                    date_col_idx = avg_sheet_header.index('Last Ranked Date') + 1
                except ValueError as ve:
                    print(f"ERROR: Missing expected column in Album Averages sheet header: {ve}")
                    flash(f"Configuration error: Missing column in 'Album Averages' sheet: {ve}", "error")
                    return redirect(url_for('load_albums_by_artist_route', artist_name=artist_name))

                cell_avg = album_averages_sheet.cell(album_row_idx, avg_col_idx)
                if cell_avg.value != str(average_album_score):
                    cell_avg.value = average_album_score
                    update_cells_avg.append(cell_avg)

                cell_times = album_averages_sheet.cell(album_row_idx, times_col_idx)
                if cell_times.value != str(times_ranked):
                    cell_times.value = times_ranked
                    update_cells_avg.append(cell_times)

                cell_date = album_averages_sheet.cell(album_row_idx, date_col_idx)
                current_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if cell_date.value != current_date_str:
                    cell_date.value = current_date_str
                    update_cells_avg.append(cell_date)

                if update_cells_avg:
                    try:
                        album_averages_sheet.update_cells(update_cells_avg)
                        print(
                            f"DEBUG: Batch updated {len(update_cells_avg)} cells in Album Averages sheet for '{album_name}'.")
                    except Exception as e:
                        print(f"ERROR: Failed to batch update cells in Album Averages sheet: {e}")
                        flash(f"Error updating album average: {e}", "error")
            else:
                print(f"DEBUG: No existing row found for '{album_name}'. Appending new row to Album Averages sheet.")
                new_album_row_values = [
                    album_name,
                    artist_name,
                    average_album_score,
                    1,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
                try:
                    album_averages_sheet.append_row(new_album_row_values)
                    print(f"DEBUG: Added new entry: {new_album_row_values} to Album Averages sheet.")
                except Exception as e:
                    print(f"ERROR: Failed to append new row to Album Averages sheet: {e}")
                    flash(f"Error adding new album average: {e}", "error")
        else:
            print(
                "DEBUG: Skipping Album Averages update: Submission status is not 'final' OR no ranked songs submitted.")
        # --- END: Album Averages Sheet Logic ---

        flash('Rankings submitted successfully!', "success")
        print(f"--- SUBMIT RANKINGS END (Redirecting to album) ---\n")

        # Redirect to the view_album page after submission
        return redirect(url_for('view_album', album_id=album_id, album_name=album_name,
                                artist_name=artist_name, album_cover_url=album_cover_url))

    except Exception as e:
        import traceback
        print("\n🔥 ERROR in /submit_rankings route:")
        traceback.print_exc()
        flash(f"An unexpected error occurred during submission: {e}", "error")
        return redirect(url_for('index'))
@app.route('/')
def index():
    return render_template('index.html')
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
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    return sheet.get_all_records()
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
    print("DEBUG: Raw albums from Spotify API:")
    if albums_from_spotify: # Changed from 'albums' to 'albums_from_spotify'
        for i, album_data in enumerate(albums_from_spotify[:3]):
            print(f"  Album {i+1}: {album_data}")
    else:
        print("  No albums returned from Spotify API.")
    # --- END DEBUG ---

    album_averages_sheet_name = "Album Averages"
    album_averages_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(album_averages_sheet_name)
    album_averages_df = get_as_dataframe(album_averages_sheet, evaluate_formulas=True).fillna("")

    # Create a dictionary for quick lookup of averages/times ranked
    album_metadata = {}
    if not album_averages_df.empty: # Added check for empty df
        for _, row in album_averages_df.iterrows():
            album_key = (str(row.get("Album Name", "")).strip().lower(), str(row.get("Artist Name", "")).strip().lower())
            album_metadata[album_key] = {
                "average_score": row.get("Average Score", None),
                "times_ranked": row.get("Times Ranked", 0)
            }
    print(f"DEBUG: Loaded {len(album_metadata)} album metadata entries from sheet.")

    # Prepare albums for template, adding average score and times ranked
    albums_for_template = []
    for album_data in albums_from_spotify: # Iterate through the raw Spotify data
        album_name_lower = album_data.get("name", "").strip().lower() # Use 'name' from Spotify
        # artist_name_lower needs to be from the request as spotify returns it nested in a list
        # For lookup against sheet, use the artist_name from the request/route
        artist_name_for_lookup_lower = artist_name.strip().lower()

        metadata = album_metadata.get((album_name_lower, artist_name_for_lookup_lower), {})

        albums_for_template.append({
            "album_name": album_data.get("name"),       # Mapped from 'name'
            "artist_name": artist_name,                  # Use the artist_name passed to the route
            "image": album_data.get("image"),            # Mapped from 'image'
            "id": album_data.get("id"),                  # Mapped from 'id'
            "average_score": metadata.get("average_score"),
            "times_ranked": metadata.get("times_ranked"),
            "url": album_data.get("url")                 # Mapped from 'url'
        })
    print(f"DEBUG: Prepared {len(albums_for_template)} albums for select_album.html with metadata.")

    # Pass the enriched list to the template
    return render_template("select_album.html", artist_name=artist_name, albums=albums_for_template)
@app.route("/ranking_page")
def ranking_page():
    sheet_rows = load_google_sheet_data()
    group_bins = group_ranked_songs(sheet_rows)
    return render_template("album.html", group_bins=group_bins)


@app.route("/view_album", methods=["POST", "GET"])
def view_album():
    global sp # Ensure sp is accessible

    try:
        # Get album details from form (POST) or URL parameters (GET)
        if request.method == 'POST':
            album_id = request.form.get("album_id")
            album_name = request.form.get("album_name")
            artist_name = request.form.get("artist_name")
            album_cover_url = request.form.get("album_cover_url")
        else: # GET request, likely from a redirect
            album_id = request.args.get("album_id")
            album_name = request.args.get("album_name")
            artist_name = request.args.get("artist_name")
            album_cover_url = request.args.get("album_cover_url")

        if not album_id:
            flash("No album selected.", "warning")
            return redirect(url_for('index'))

        print(f"\n--- VIEW ALBUM START ---")
        print(f"DEBUG: Received album_id from form/args: {album_id}")
        print(f"DEBUG: Received artist_name from form/args: {artist_name}")

        # Fetch album data from Spotify
        print(f"DEBUG: Constructed Spotify URI: spotify:album:{album_id}")
        try:
            if sp is None:
                raise Exception("Spotify client (sp) not initialized.")
            album_data = load_album_data(album_id)
        except Exception as e:
            flash(f"Error loading album data from Spotify: {e}", "error")
            print(f"ERROR: Error loading album data from Spotify: {e}")
            return redirect(url_for('index'))

        # Prepare album data for template
        album_data_for_template = {
            'album_name': album_data['album_name'],
            'artist_name': album_data['artist_name'],
            'album_cover_url': album_data['album_cover_url'],
            'url': album_data.get('url', ''),
            'album_id': album_id,
            'songs': album_data['songs'],
            'bg_color': album_data.get('bg_color', '#121212') # Default background
        }
        print(f"DEBUG: Extracted for template: Name='{album_data_for_template['album_name']}', Artist='{album_data_for_template['artist_name']}', Cover='{album_data_for_template['album_cover_url']}'")

        # --- Load existing rankings (Final and Preliminary) ---
        main_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        print(f"DEBUG: Loaded {main_sheet.row_count} rows from main ranking sheet.")
        main_sheet_data = get_as_dataframe(main_sheet, evaluate_formulas=False).fillna("") # Fetch as DataFrame

        # Filter for current album's final rankings
        current_album_final_ranks = main_sheet_data[
            (main_sheet_data["Spotify Album ID"].astype(str) == str(album_id)) &
            (main_sheet_data["Ranking Status"].astype(str) == "final")
        ]
        print(f"DEBUG: Found {len(current_album_final_ranks)} existing FINAL entries in sheet for '{album_name}' by '{artist_name}'.")

        # --- Populate rank groups for JavaScript (Right Panel) ---
        rank_groups_for_js = {f"{i/2:.1f}": [] for i in range(1, 21)} # Groups from 0.5 to 10.0
        # This will be passed to JS to rebuild the right panel
        for _, row in current_album_final_ranks.iterrows():
            try:
                rank_group = f"{float(row['Rank Group']):.1f}"
                song_score = float(row['Ranking'])
                position_in_group = int(row['Position In Group']) # Ensure this is an int

                song_data = {
                    'song_id': row['Spotify Song ID'],
                    'song_name': row['Song Name'],
                    'rank_group': rank_group,
                    'calculated_score': song_score,
                    'rank_position': position_in_group # Include position for accurate re-ordering
                }
                if rank_group in rank_groups_for_js:
                    rank_groups_for_js[rank_group].append(song_data)
            except (ValueError, KeyError, TypeError) as e:
                print(f"WARNING: Error parsing existing ranked song data for JS: {row} - {e}")

        # Sort songs within each group by their rank_position
        for group_key in rank_groups_for_js:
            rank_groups_for_js[group_key].sort(key=lambda x: x.get('rank_position', 0)) # Sort by position

        # Convert rank_groups_for_js values from list to a stringified JSON for direct JS parsing
        # (This is how your current JS expects it to be rebuilt if you had previous `rank_groups_for_js` logic)
        # However, a dict directly passed to Flask's render_template usually gets JSON-serialized correctly.
        # Let's keep it as a dict and ensure Jinja renders it properly into a JS variable.
        # print(f"DEBUG: Populated rank_groups_for_js with {sum(len(v) for v in rank_groups_for_js.values())} songs.")
        total_songs_in_groups = sum(len(v) for v in rank_groups_for_js.values())
        print(f"DEBUG: Populated rank_groups_for_js with {total_songs_in_groups} songs.")
        # print(f"DEBUG: Final rank_groups_for_js sent to album.html: {rank_groups_for_js.keys()} (Total songs in groups: {total_songs_in_groups})")


        # --- Load existing preliminary ranks for this album ---
        existing_prelim_ranks = {} # To store song_id: prelim_rank_value
        prelim_sheet_name = "Preliminary Ranks"
        try:
            prelim_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(prelim_sheet_name)
            prelim_sheet_data = get_as_dataframe(prelim_sheet, evaluate_formulas=False).fillna("")

            current_album_prelim_ranks = prelim_sheet_data[
                (prelim_sheet_data["album_id"].astype(str) == str(album_id)) &
                (prelim_sheet_data["prelim_rank"] != "") # Ensure prelim_rank is not empty
            ]

            for _, row in current_album_prelim_ranks.iterrows():
                song_id = str(row['song_id'])
                prelim_rank_value = row['prelim_rank']
                try:
                    existing_prelim_ranks[song_id] = float(prelim_rank_value)
                except ValueError:
                    print(f"WARNING: Invalid prelim_rank value '{prelim_rank_value}' for song {song_id}. Skipping.")
            print(f"DEBUG: Loaded {len(existing_prelim_ranks)} preliminary rank entries for album {album_id}.")
        except gspread.exceptions.WorksheetNotFound:
            print(f"WARNING: Preliminary Ranks sheet '{prelim_sheet_name}' not found. No prelim ranks loaded.")
        except Exception as e:
            print(f"ERROR: Error loading preliminary ranks from sheet: {e}")
            flash(f"Error loading preliminary ranks: {e}", "error")


        # --- Prepare songs for Left Panel ---
        songs_for_template = []
        for song in album_data['songs']:
            song_id = str(song['song_id'])
            song_name = song['song_name']
            is_final_ranked = False

            # Check if song is already in a final rank group
            for group_key in rank_groups_for_js:
                if any(s['song_id'] == song_id for s in rank_groups_for_js[group_key]):
                    is_final_ranked = True
                    break

            song_entry = {
                'song_name': song_name,
                'song_id': song_id,
                'already_ranked': is_final_ranked, # 'already_ranked' means it's in a final rank group
                'prelim_rank': '' # Default to empty
            }

            # If not final ranked, check for preliminary rank
            if not is_final_ranked:
                if song_id in existing_prelim_ranks:
                    song_entry['prelim_rank'] = existing_prelim_ranks[song_id]
                    # Also, you might want a status or flag to indicate it has a prelim rank
                    # e.g., 'status': 'preliminary_saved'
            songs_for_template.append(song_entry)

        album_data_for_template['songs'] = songs_for_template
        print(f"DEBUG: Prepared {len(songs_for_template)} songs for left panel.")


        print(f"DEBUG: Final album_data_for_template sent to album.html: {album_data_for_template.keys()}")
        print(f"DEBUG: Final rank_groups_for_js sent to album.html: {rank_groups_for_js.keys()} (Total songs in groups: {total_songs_in_groups})")
        print(f"--- VIEW ALBUM END ---\n")

        return render_template('album.html',
                               album=album_data_for_template,
                               rank_groups=rank_groups_for_js # Pass the populated dictionary
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
