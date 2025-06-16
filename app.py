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


# In app.py

import json
from datetime import datetime
from flask import request, redirect, url_for, flash
# Assuming these are correctly imported from your project's structure
# from your_gspread_utils import get_as_dataframe
# from your_main_app import app, client, SPREADSHEET_ID, SHEET_NAME

import json
from datetime import datetime
from flask import request, redirect, url_for, flash
# Assuming these are correctly imported from your project's structure
# from your_gspread_utils import get_as_dataframe # Make sure this is available
# from your_main_app import app, client, SPREADSHEET_ID, SHEET_NAME # Make sure these are available

@app.route("/submit_rankings", methods=["POST"])
def submit_rankings():
    try:
        album_name = request.form.get("album_name")
        artist_name = request.form.get("artist_name")
        submission_status = request.form.get("status") # This should be 'final' or 'paused'

        # --- DEBUG: Initial Submission Data ---
        print(f"\n--- SUBMIT RANKINGS START ---")
        print(f"DEBUG: submit_rankings called for Album: '{album_name}', Artist: '{artist_name}', Status: '{submission_status}'")

        all_ranked_data_json = request.form.get("all_ranked_data", "[]")
        prelim_rank_data_json = request.form.get("prelim_rank_data", "{}")

        print(f"DEBUG: Raw all_ranked_data_json (first 200 chars): {all_ranked_data_json[:200]}...")
        print(f"DEBUG: Raw prelim_rank_data_json (first 200 chars): {prelim_rank_data_json[:200]}...")

        all_ranked_songs_from_js = []
        if all_ranked_data_json:
            try:
                all_ranked_songs_from_js = json.loads(all_ranked_data_json)
                # Ensure calculated_score is a number for sum
                for song in all_ranked_songs_from_js:
                    if 'calculated_score' in song and isinstance(song['calculated_score'], str):
                        try:
                            song['calculated_score'] = float(song['calculated_score'])
                        except ValueError:
                            song['calculated_score'] = 0 # Default if conversion fails
            except json.JSONDecodeError as e:
                flash(f"Error parsing ranked songs data: {e}")
                print(f"ERROR: JSONDecodeError on all_ranked_data: {e}")
                print(f"Faulty JSON: {all_ranked_data_json}")
                return redirect(url_for('index')) # Redirect to home page on critical error
        print(f"DEBUG: Parsed all_ranked_songs_from_js ({len(all_ranked_songs_from_js)} songs): {all_ranked_songs_from_js[:2]}") # Print first 2 parsed songs

        prelim_ranks_from_js = {}
        if prelim_rank_data_json:
            try:
                prelim_ranks_from_js = json.loads(prelim_rank_data_json)
            except json.JSONDecodeError as e:
                flash(f"Error parsing prelim ranks data: {e}")
                print(f"ERROR: JSONDecodeError on prelim_rank_data: {e}")
                print(f"Faulty JSON: {prelim_rank_data_json}")
                return redirect(url_for('index'))
        print(f"DEBUG: Parsed prelim_ranks_from_js ({len(prelim_ranks_from_js)} entries): {list(prelim_ranks_from_js.items())[:2]}")


        # --- Existing logic for main song sheet (SHEET_NAME) ---
        sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        all_existing_rows = get_as_dataframe(sheet, evaluate_formulas=True).fillna("")

        form_album_name_lower = album_name.strip().lower()
        form_artist_name_lower = artist_name.strip().lower()

        album_artist_filtered_df = all_existing_rows[
            (all_existing_rows["Album Name"].astype(str).str.strip().str.lower() == form_album_name_lower) &
            (all_existing_rows["Artist Name"].astype(str).str.strip().str.lower() == form_artist_name_lower)
        ]

        column_order_in_sheet = [
            'Album Name', 'Artist Name', 'Song Name', 'Ranking', 'Ranking Status',
            'Ranked Date', 'Position In Group', 'Rank Group', 'Spotify Song ID', 'Unnamed: 8'
        ]

        rows_to_update_data = {}
        new_rows_values = []
        submitted_ranked_song_ids = {s.get('song_id') for s in all_ranked_songs_from_js if s.get('song_id')}

        print(f"DEBUG: Submitted ranked song IDs: {submitted_ranked_song_ids}")

        # --- Step 1: Process songs explicitly submitted as ranked/unranked ---
        for song_data in all_ranked_songs_from_js:
            song_name = song_data.get("song_name")
            song_id = str(song_data.get("song_id")) # Ensure string for comparison
            rank_group = song_data.get("rank_group")
            rank_position = song_data.get("rank_position")
            calculated_score = song_data.get("calculated_score")

            if not song_id:
                print(f"WARNING: Skipping song '{song_name}' due to missing Spotify Song ID in submission data.")
                continue

            prelim_rank_val = prelim_ranks_from_js.get(song_id, '')

            existing_row_for_song = album_artist_filtered_df[
                (album_artist_filtered_df["Spotify Song ID"].astype(str) == song_id)
            ]

            row_values = {
                'Song Name': song_name,
                'Album Name': album_name,
                'Artist Name': artist_name,
                'Ranking': calculated_score,
                'Rank Group': rank_group,
                'Position In Group': rank_position,
                'Preliminary Rank': prelim_rank_val, # This assumes a 'Preliminary Rank' column in your main sheet.
                                                     # If 'Unnamed: 8' is for prelim, map it there.
                'Ranking Status': submission_status,
                'Spotify Song ID': song_id,
                'Ranked Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Unnamed: 8': datetime.now().strftime('%Y-%m-%d %H:%M:%S') # General timestamp
            }
            # Note: Ensure 'Preliminary Rank' is either an actual column in your sheet or you map its data
            # to an 'Unnamed: X' column if that's where it's stored. If not, it will be ignored.

            if not existing_row_for_song.empty:
                original_sheet_row_idx = existing_row_for_song.index[0] + 2
                rows_to_update_data[original_sheet_row_idx] = row_values
                print(f"DEBUG: Marking song '{song_name}' (ID: {song_id}) for UPDATE at row {original_sheet_row_idx}. Status: {submission_status}")
            else:
                ordered_values = [row_values.get(col_name, '') for col_name in column_order_in_sheet]
                new_rows_values.append(ordered_values)
                print(f"DEBUG: Marking song '{song_name}' (ID: {song_id}) for NEW row append. Status: {submission_status}")

        # --- Step 2: Process songs from the sheet that were NOT submitted as ranked (e.g., remained unranked, or prelims) ---
        for _, existing_song_row in album_artist_filtered_df.iterrows():
            existing_song_id = str(existing_song_row.get("Spotify Song ID", ""))
            existing_song_name = existing_song_row.get("Song Name")

            if existing_song_id and existing_song_id not in submitted_ranked_song_ids:
                print(f"DEBUG: Song '{existing_song_name}' (ID: {existing_song_id}) was NOT explicitly submitted. Checking for prelim/status.")

                current_prelim_rank = prelim_ranks_from_js.get(existing_song_id, existing_song_row.get("Preliminary Rank", ""))

                row_data_for_update = {
                    'Song Name': existing_song_name,
                    'Album Name': album_name,
                    'Artist Name': artist_name,
                    'Spotify Song ID': existing_song_id,
                    'Ranked Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Unnamed: 8': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                # Default to clearing ranking data unless it's a paused/prelim submission
                row_data_for_update['Ranking'] = ''
                row_data_for_update['Rank Group'] = ''
                row_data_for_update['Position In Group'] = ''

                if current_prelim_rank:
                    row_data_for_update['Preliminary Rank'] = current_prelim_rank
                    row_data_for_update['Ranking Status'] = 'preliminary'
                elif submission_status == 'paused' and existing_song_row.get('Ranking Status') == 'paused':
                    # If we're pausing, and this song was already paused, keep it paused
                    row_data_for_update['Ranking Status'] = 'paused'
                    row_data_for_update['Ranking'] = existing_song_row.get('Ranking', '') # Retain old ranking if present
                    row_data_for_update['Rank Group'] = existing_song_row.get('Rank Group', '')
                    row_data_for_update['Position In Group'] = existing_song_row.get('Position In Group', '')
                    row_data_for_update['Preliminary Rank'] = existing_song_row.get('Preliminary Rank', '')
                else:
                    row_data_for_update['Ranking Status'] = 'unranked'
                    row_data_for_update['Preliminary Rank'] = '' # Clear prelim if unranked

                original_sheet_row_idx = existing_song_row.name + 2
                rows_to_update_data[original_sheet_row_idx] = row_data_for_update
                print(f"DEBUG: Marking unsubmitted song '{existing_song_name}' (ID: {existing_song_id}) for update at row {original_sheet_row_idx}. Status: {row_data_for_update['Ranking Status']}")

        # --- Step 3: Execute updates and appends for main song sheet ---
        if new_rows_values:
            print(f"DEBUG: Appending {len(new_rows_values)} new rows to main sheet.")
            sheet.append_rows(new_rows_values)

        if rows_to_update_data:
            header = sheet.row_values(1)
            cells_for_batch_update = []
            for row_idx, data in rows_to_update_data.items():
                for col_name, value in data.items():
                    try:
                        col_idx = header.index(col_name)
                        col_letter = chr(ord('A') + col_idx)
                        cell = sheet.acell(f"{col_letter}{row_idx}")
                        # Only update if value has changed to reduce API calls
                        if cell.value != str(value): # Compare string values from gspread
                            cell.value = value
                            cells_for_batch_update.append(cell)
                    except ValueError:
                        print(f"WARNING: Column '{col_name}' not found in main sheet header. Skipping for update.")

            if cells_for_batch_update:
                print(f"DEBUG: Performing batch update for {len(cells_for_batch_update)} cells in main sheet.")
                sheet.update_cells(cells_for_batch_update)
            else:
                print("DEBUG: No cells needed batch update in main sheet.")


        # --- START: Album Averages Sheet Logic (THIS IS THE CRITICAL SECTION) ---
        if submission_status == 'final' and all_ranked_songs_from_js:
            print("DEBUG: Entering FINAL ranking logic for Album Averages sheet.")

            total_score = sum(s.get('calculated_score', 0) for s in all_ranked_songs_from_js)
            num_ranked_songs = len(all_ranked_songs_from_js)
            average_album_score = round(total_score / num_ranked_songs, 2) if num_ranked_songs > 0 else 0

            print(f"DEBUG: Calculated Album Average Score: {average_album_score} (Total: {total_score}, Count: {num_ranked_songs})")

            # Load the Album Averages sheet
            album_averages_sheet_name = "Album Averages"
            try:
                album_averages_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(album_averages_sheet_name)
                print(f"DEBUG: Successfully opened Album Averages sheet: '{album_averages_sheet_name}'")
            except Exception as e:
                print(f"ERROR: Could not open Album Averages sheet '{album_averages_sheet_name}': {e}")
                flash(f"Error accessing album averages sheet: {e}")
                return redirect(url_for('load_albums_by_artist_route', artist_name=artist_name))

            album_averages_df = get_as_dataframe(album_averages_sheet, evaluate_formulas=True).fillna("")
            print(f"DEBUG: Loaded Album Averages DataFrame (first 5 rows):\n{album_averages_df.head().to_string()}")

            # Try to find the album's existing entry
            matching_album_row = album_averages_df[
                (album_averages_df["Album Name"].astype(str).str.strip().str.lower() == form_album_name_lower) &
                (album_averages_df["Artist Name"].astype(str).str.strip().str.lower() == form_artist_name_lower)
            ]

            if not matching_album_row.empty:
                # Update existing album average
                album_row_idx = matching_album_row.index[0] + 2 # +2 for header and 0-indexing
                times_ranked = matching_album_row.iloc[0].get("Times Ranked", 0)
                try:
                    times_ranked = int(times_ranked) + 1 # Increment times ranked
                except ValueError:
                    print(f"WARNING: 'Times Ranked' value '{times_ranked}' is not an integer. Resetting to 1 for '{album_name}'.")
                    times_ranked = 1 # Default to 1 if existing value is invalid

                print(f"DEBUG: Found existing row for '{album_name}' at sheet row {album_row_idx}. Current Times Ranked: {times_ranked-1}, New Times Ranked: {times_ranked}")

                update_cells_avg = []
                # Update Average Score (Column C assumed)
                cell_avg = album_averages_sheet.acell(f"C{album_row_idx}")
                if cell_avg.value != str(average_album_score):
                    cell_avg.value = average_album_score
                    update_cells_avg.append(cell_avg)
                    print(f"DEBUG: Updating Avg Score cell C{album_row_idx} to {average_album_score}")
                else:
                    print(f"DEBUG: Avg Score cell C{album_row_idx} already {average_album_score}. No change needed.")

                # Update Times Ranked (Column D assumed)
                cell_times = album_averages_sheet.acell(f"D{album_row_idx}")
                if cell_times.value != str(times_ranked):
                    cell_times.value = times_ranked
                    update_cells_avg.append(cell_times)
                    print(f"DEBUG: Updating Times Ranked cell D{album_row_idx} to {times_ranked}")
                else:
                    print(f"DEBUG: Times Ranked cell D{album_row_idx} already {times_ranked}. No change needed.")

                # Update Last Ranked Date (Column E assumed, or adjust if your sheet is different)
                cell_date = album_averages_sheet.acell(f"E{album_row_idx}")
                current_date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if cell_date.value != current_date_str:
                    cell_date.value = current_date_str
                    update_cells_avg.append(cell_date)
                    print(f"DEBUG: Updating Last Ranked Date cell E{album_row_idx} to {current_date_str}")
                else:
                    print(f"DEBUG: Last Ranked Date cell E{album_row_idx} already {current_date_str}. No change needed.")


                if update_cells_avg:
                    try:
                        album_averages_sheet.update_cells(update_cells_avg)
                        print(f"DEBUG: Batch updated {len(update_cells_avg)} cells in Album Averages sheet for '{album_name}'.")
                    except Exception as e:
                        print(f"ERROR: Failed to batch update cells in Album Averages sheet: {e}")
                        flash(f"Error updating album average: {e}")
                else:
                    print(f"DEBUG: No cells needed batch update in Album Averages sheet for '{album_name}'.")

            else:
                # Add new album average entry
                print(f"DEBUG: No existing row found for '{album_name}'. Appending new row to Album Averages sheet.")
                new_album_row_values = [
                    album_name,
                    artist_name,
                    average_album_score,
                    1, # First time ranked
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Last Ranked Date
                ]
                try:
                    album_averages_sheet.append_row(new_album_row_values)
                    print(f"DEBUG: Added new entry: {new_album_row_values} to Album Averages sheet.")
                except Exception as e:
                    print(f"ERROR: Failed to append new row to Album Averages sheet: {e}")
                    flash(f"Error adding new album average: {e}")
        else:
            print("DEBUG: Skipping Album Averages update: Submission status is not 'final' OR no ranked songs submitted.")
        # --- END: Album Averages Sheet Logic ---

        flash('Rankings submitted successfully!' if submission_status == 'final' else 'Rankings paused and saved.')
        print(f"--- SUBMIT RANKINGS END (Redirecting to artist albums) ---\n")
        return redirect(url_for('load_albums_by_artist_route', artist_name=artist_name))

    except Exception as e:
        import traceback
        print("\n🔥 ERROR in /submit_rankings route:")
        traceback.print_exc()
        flash(f"An unexpected error occurred during submission: {e}")
        return redirect(url_for('index')) # Redirect to home page on any unhandled error
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


@app.route('/view_album', methods=['POST'])
def view_album():
    album_id_from_form = request.form.get("album_id") # Use album_id as primary identifier
    artist_name_from_form = request.form.get("artist_name") # Keep artist name from form for robustness

    print(f"\n--- VIEW ALBUM START ---")
    print(f"DEBUG: Received album_id from form: {album_id_from_form}")
    print(f"DEBUG: Received artist_name from form: {artist_name_from_form}")

    if not album_id_from_form or not artist_name_from_form:
        flash("Missing album ID or artist name.", "error")
        print("ERROR: Missing album_id or artist_name in /view_album POST request.")
        return redirect(url_for('index')) # Redirect to a safe page

    try:
        spotify_uri_to_fetch = f"spotify:album:{album_id_from_form}"
        print(f"DEBUG: Constructed Spotify URI: {spotify_uri_to_fetch}")

        # Fetch album data from Spotify (only once)
        album_data_from_spotify = load_album_data(spotify_uri_to_fetch)
        print(f"DEBUG: Raw data from spotify_logic.load_album_data: {album_data_from_spotify}")

        if not album_data_from_spotify:
            flash("Could not load album data from Spotify. Album not found.", "error")
            print(f"ERROR: No album data returned from Spotify for ID: {album_id_from_form}")
            return redirect(url_for('load_albums_by_artist_route', artist_name=artist_name_from_form)) # Go back to artist albums

        # Extract essential album info for the template
        album_name_final = album_data_from_spotify.get("album_name", "Unknown Album Name")
        artist_name_final = album_data_from_spotify.get("artist_name", artist_name_from_form)
        album_cover_url_final = album_data_from_spotify.get("album_cover_url", "")
        spotify_tracks_list = album_data_from_spotify.get("songs", []) # List of songs from Spotify

        print(f"DEBUG: Extracted for template: Name='{album_name_final}', Artist='{artist_name_final}', Cover='{album_cover_url_final}'")

        # --- Load existing ranking data from the main Google Sheet ---
        main_rankings_sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        all_existing_rows_df = get_as_dataframe(main_rankings_sheet, evaluate_formulas=True).fillna("")
        print(f"DEBUG: Loaded {len(all_existing_rows_df)} rows from main ranking sheet.")

        # Filter sheet data for the current album/artist (case-insensitive and stripped)
        form_album_name_lower = album_name_final.strip().lower()
        form_artist_name_lower = artist_name_final.strip().lower()

        album_artist_filtered_df = all_existing_rows_df[
            (all_existing_rows_df["Album Name"].astype(str).str.strip().str.lower() == form_album_name_lower) &
            (all_existing_rows_df["Artist Name"].astype(str).str.strip().str.lower() == form_artist_name_lower)
        ]
        print(f"DEBUG: Found {len(album_artist_filtered_df)} existing entries in sheet for '{album_name_final}' by '{artist_name_final}'.")


        # --- Build the rank_groups dictionary for the frontend JS ---
        rank_groups_for_js = {f"{(i * 0.5):.1f}": [] for i in range(1, 21)}
        rank_groups_for_js["?"] = [] # Initialize the unranked/preliminary group

        all_ranked_song_ids_from_sheet = set() # Track all songs found in any ranked group
        prelim_ranks_from_sheet = {} # To store prelim ranks if they are in the sheet

        if not album_artist_filtered_df.empty:
            for _, row in album_artist_filtered_df.iterrows():
                song_id = str(row.get("Spotify Song ID", ""))
                ranking_status = str(row.get("Ranking Status", "")).lower()
                rank_group_val = row.get("Rank Group")
                prelim_rank_val = row.get("Preliminary Rank", "") # Get prelim rank from sheet if exists

                # Store prelim rank if found (regardless of status, just for reference)
                if prelim_rank_val:
                    prelim_ranks_from_sheet[song_id] = prelim_rank_val

                # Process songs that are explicitly ranked (final or paused in a group)
                if ranking_status in ["final", "paused"] and rank_group_val is not None:
                    try:
                        rank_group_str = f"{float(rank_group_val):.1f}"
                    except (ValueError, TypeError):
                        rank_group_str = "?" # Fallback if rank_group is not numeric

                    song_data_for_group = {
                        "song_name": row.get("Song Name", ""),
                        "song_id": song_id,
                        "prelim_rank": prelim_rank_val, # Use prelim from sheet
                        "rank_position": row.get("Position In Group", 999),
                        "calculated_score": row.get("Ranking", "")
                    }

                    if rank_group_str in rank_groups_for_js:
                        rank_groups_for_js[rank_group_str].append(song_data_for_group)
                        all_ranked_song_ids_from_sheet.add(song_id)
                    else:
                        # If a song has a rank_group that's not one of our standard buttons, put it in '?'
                        rank_groups_for_js["?"].append(song_data_for_group)
                        all_ranked_song_ids_from_sheet.add(song_id)
                elif ranking_status == "preliminary":
                    # If status is just preliminary, it belongs in the '?' group on load
                    song_data_for_group = {
                        "song_name": row.get("Song Name", ""),
                        "song_id": song_id,
                        "prelim_rank": prelim_rank_val,
                        "rank_position": 999, # Prelims don't have a specific position in group
                        "calculated_score": ""
                    }
                    rank_groups_for_js["?"].append(song_data_for_group)
                    # We might add prelims to `all_ranked_song_ids_from_sheet` if we want them marked
                    # as 'already_ranked' on the left panel, even if not fully ranked.
                    # For now, let's keep `already_ranked` for `final`/`paused` in a group.


        # Sort songs within each group by rank_position
        for group_key in rank_groups_for_js:
            rank_groups_for_js[group_key].sort(key=lambda s: s.get("rank_position", float('inf')))
        print(f"DEBUG: Populated rank_groups_for_js with {sum(len(v) for v in rank_groups_for_js.values())} songs.")
        # print(f"DEBUG: rank_groups_for_js: {rank_groups_for_js}") # Uncomment for full debug


        # --- Prepare the songs list for the left panel of album.html ---
        songs_for_left_panel = []
        for spotify_track in spotify_tracks_list:
            song_name = spotify_track.get("song_name")
            song_id = str(spotify_track.get("song_id"))

            # Check if this Spotify song is already ranked in the sheet
            # A song is "already_ranked" if it's in a final/paused group, or has a prelim_rank
            already_ranked_flag = False
            prelim_value_for_display = ''

            # Check if it's in our collected ranked song IDs (final/paused)
            if song_id in all_ranked_song_ids_from_sheet:
                already_ranked_flag = True
                # If it's ranked, its prelim_rank should be what's in the sheet
                prelim_value_for_display = prelim_ranks_from_sheet.get(song_id, '')
            else:
                # If not explicitly ranked (final/paused), check if it has a prelim rank from sheet
                prelim_value_for_display = prelim_ranks_from_sheet.get(song_id, '')
                if prelim_value_for_display:
                    # If it has a prelim rank but isn't in a main group, it should go to '?' group if not already there
                    # This implies it should appear in the '?' group's display initially.
                    if not any(s.get('song_id') == song_id for s in rank_groups_for_js['?']):
                        rank_groups_for_js['?'].append({
                            "song_name": song_name,
                            "song_id": song_id,
                            "prelim_rank": prelim_value_for_display,
                            "rank_position": 999,
                            "calculated_score": ""
                        })
                    already_ranked_flag = True # Mark as ranked if it has a prelim, as it's "touched"


            songs_for_left_panel.append({
                "song_name": song_name,
                "song_id": song_id,
                "prelim_rank": prelim_value_for_display, # Pass prelim rank from sheet
                "already_ranked": already_ranked_flag # Flag for frontend checkmark
            })
        print(f"DEBUG: Prepared {len(songs_for_left_panel)} songs for left panel.")


        # --- Prepare the final album dictionary for the template ---
        album_data_for_template = {
            "album_name": album_name_final,
            "artist_name": artist_name_final,
            "album_cover_url": album_cover_url_final, # Used by your header image
            "url": spotify_uri_to_fetch, # Passed to hidden input, original Spotify URI
            "songs": songs_for_left_panel, # The enhanced song list for the left panel
            # Note: `album.image` is used in select_album.html, but `album.album_cover_url`
            # is used in album.html. Both are handled correctly here.
        }

        # Get dominant color for background
        bg_color = get_dominant_color(album_cover_url_final)
        album_data_for_template["bg_color"] = bg_color if bg_color else 'rgb(45, 45, 45)'

        print(f"DEBUG: Final album_data_for_template sent to album.html: {album_data_for_template.keys()}")
        print(f"DEBUG: Final rank_groups_for_js sent to album.html: {rank_groups_for_js.keys()} (Total songs: {sum(len(v) for v in rank_groups_for_js.values())})")
        print(f"--- VIEW ALBUM END ---\n")

        return render_template(
            "album.html",
            album=album_data_for_template,
            rank_groups=rank_groups_for_js # <--- THIS IS NOW PASSED
        )

    except Exception as e:
        import traceback
        print("\n🔥 ERROR in /view_album route:")
        traceback.print_exc()
        flash(f"Error loading album: {e}", "error")
        return redirect(url_for('index')) # Fallback to home page on any unhandled error


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
