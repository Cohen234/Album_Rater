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
# from your_gspread_utils import get_as_dataframe # Make sure this is imported
# from your_main_app import app, client, SPREADSHEET_ID, SHEET_NAME # Adjust imports as per your setup
@app.route("/submit_rankings", methods=["POST"])
def submit_rankings():
    try:
        album_name = request.form["album_name"]
        artist_name = request.form["artist_name"]
        submission_status = request.form["status"]

        print(f"DEBUG: Submitting for Album (from form): '{album_name}', Artist (from form): '{artist_name}', Status: {submission_status}")

        all_ranked_data_json = request.form.get("all_ranked_data", "[]")
        all_ranked_songs_from_js = json.loads(all_ranked_data_json)
        print(f"DEBUG: Received all_ranked_songs_from_js (parsed): {all_ranked_songs_from_js}")


        prelim_rank_data_json = request.form.get("prelim_rank_data", "{}")
        prelim_ranks_from_js = json.loads(prelim_rank_data_json)
        print(f"DEBUG: Received prelim_ranks_from_js (parsed): {prelim_ranks_from_js}")

        sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

        # Get existing data for the entire sheet to simplify updates.
        # This is less efficient for very large sheets, but simplifies logic.
        # A better way for large sheets is to get all rows, then filter in Python
        # and use `update_cells` on specific cells.
        all_existing_rows = get_as_dataframe(sheet, evaluate_formulas=True).fillna("")
        print(f"DEBUG: Loaded {len(all_existing_rows)} existing rows from sheet.")

        print("DEBUG: Distinct Album/Artist combinations in sheet (actual values):")
        if not all_existing_rows.empty:
            # Loop through unique album/artist pairs found in the sheet
            for _, row in all_existing_rows[['Album Name', 'Artist Name']].drop_duplicates().iterrows():
                print(f"  Sheet: Album='{row['Album Name']}', Artist='{row['Artist Name']}'")
        else:
            print("  Sheet is empty or DataFrame creation failed during get_as_dataframe.")

        form_album_name_lower = album_name.strip().lower()
        form_artist_name_lower = artist_name.strip().lower()
        print(f"DEBUG: Form values (stripped, lower) for comparison: Album='{form_album_name_lower}', Artist='{form_artist_name_lower}'")

        # Filter for rows belonging to this specific album/artist in Python
        album_artist_filtered_df = all_existing_rows[
            (all_existing_rows["Album Name"].astype(str).str.strip().str.lower() == form_album_name_lower) &
            (all_existing_rows["Artist Name"].astype(str).str.strip().str.lower() == form_artist_name_lower)
        ]
        print(f"DEBUG: Found {len(album_artist_filtered_df)} existing rows for this album/artist after initial filter.")

        # Prepare data for new/updated rows
        rows_to_update_data = {} # {sheet_row_index: {col_name: value}} for updates
        new_rows_values = []     # [[val1, val2, ...]] for new appends

        # Keep track of Spotify Song IDs that have been explicitly submitted as ranked
        submitted_ranked_song_ids = {s.get('song_id') for s in all_ranked_songs_from_js if s.get('song_id')}
        print(f"DEBUG: Submitted ranked song IDs: {submitted_ranked_song_ids}")

        # --- Step 1: Process songs explicitly submitted as ranked/unranked (from JS all_ranked_data) ---
        for song_data in all_ranked_songs_from_js:
            song_name = song_data.get("song_name")
            song_id = song_data.get("song_id") # Use .get() for safety
            rank_group = song_data.get("rank_group")
            rank_position = song_data.get("rank_position")
            calculated_score = song_data.get("calculated_score")

            if not song_id:
                print(f"WARNING: Skipping song '{song_name}' due to missing Spotify Song ID in submission data.")
                continue # Skip this song if ID is missing

            # Default values for preliminary rank if not explicitly submitted with song_data
            prelim_rank_val = prelim_ranks_from_js.get(song_id, '')

            # Find if this song exists in the filtered DataFrame by ID
            existing_row_for_song = album_artist_filtered_df[
                (album_artist_filtered_df["Spotify Song ID"].astype(str) == str(song_id))
            ]

            row_values = {
                'Song Name': song_name,
                'Album Name': album_name,
                'Artist Name': artist_name,
                'Ranking': calculated_score,
                'Rank Group': rank_group, # Assuming you have a 'Rank Group' column in your sheet now
                'Rank Position': rank_position,
                'Preliminary Rank': prelim_rank_val,
                'Ranking Status': submission_status, # Use the form's overall status
                'Spotify Song ID': song_id,
                'Ranked Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S') # More readable timestamp
            }

            if not existing_row_for_song.empty:
                # Update existing row
                original_sheet_row_idx = existing_row_for_song.index[0] + 2 # +2 for header and 0-indexing
                rows_to_update_data[original_sheet_row_idx] = row_values
                print(f"DEBUG: Marking song '{song_name}' (ID: {song_id}) for UPDATE at row {original_sheet_row_idx}. Values: {row_values}")
            else:
                # Add new row
                # Ensure column order matches your sheet's columns for new rows
                # This list MUST exactly match the order of your columns in the Google Sheet.
                ordered_values = [
                    row_values.get('Album Name', ''),
                    row_values.get('Artist Name', ''),
                    row_values.get('Song Name', ''),
                    row_values.get('Ranking', ''),
                    row_values.get('Rank Group', ''),  # If you added this column in sheet
                    row_values.get('Ranking Status', ''),
                    row_values.get('Ranked Date', ''),
                    row_values.get('Position In Group', ''),
                    row_values.get('Spotify Song ID', ''),
                    row_values.get('Preliminary Rank', '')
                    # Add placeholders ('') for any other columns that exist in your sheet
                    # but are not explicitly handled above. Count your sheet columns!
                ]
                new_rows_values.append(ordered_values)
                print(f"DEBUG: Marking song '{song_name}' (ID: {song_id}) for NEW row append. Values: {ordered_values}")

        # --- Step 2: Process songs from the sheet that were NOT submitted as ranked,
        #            but still belong to this album and might have prelim ranks or 'paused' status ---
        for _, existing_song_row in album_artist_filtered_df.iterrows():
            existing_song_id = existing_song_row.get("Spotify Song ID")
            existing_song_name = existing_song_row.get("Song Name")

            if existing_song_id not in submitted_ranked_song_ids:
                # This song was NOT part of the submitted ranked list for current save.
                # Decide its fate: clear rank, retain prelim, or set to 'unranked'.

                current_prelim_rank = prelim_ranks_from_js.get(existing_song_id, existing_song_row.get("Preliminary Rank", ""))

                row_data_for_update = {
                    'Song Name': existing_song_name,
                    'Album Name': album_name,
                    'Artist Name': artist_name,
                    'Spotify Song ID': existing_song_id,
                    'Ranked Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    # Default to clearing ranking data
                    'Ranking': '',
                    'Rank Group': '',
                    'Rank Position': '',
                    'Ranking Status': '', # Default to unranked
                    'Preliminary Rank': current_prelim_rank # Keep or clear based on new prelim
                }

                if current_prelim_rank:
                    row_data_for_update['Ranking Status'] = 'preliminary' # Or 'paused', but 'preliminary' is clearer
                elif existing_song_row.get('Ranking Status') == 'paused' and submission_status == 'paused':
                    # If it was paused and we're pausing again, retain its paused status but clear rank if not submitted
                    row_data_for_update['Ranking Status'] = 'paused'
                else:
                    # If it's not being submitted as ranked, has no new prelim, and wasn't previously paused (or is being finalized),
                    # then effectively unrank it.
                    row_data_for_update['Ranking Status'] = 'unranked' # Explicitly mark as unranked/cleared

                # Use original sheet row index
                original_sheet_row_idx = existing_song_row.name + 2 # Pandas index + 2
                rows_to_update_data[original_sheet_row_idx] = row_data_for_update
                print(f"DEBUG: Marking existing song '{existing_song_name}' (ID: {existing_song_id}) for UNRANK/PRELIM update at row {original_sheet_row_idx}. Values: {row_data_for_update}")

        # --- Step 3: Execute updates and appends ---
        if new_rows_values:
            print(f"DEBUG: Appending {len(new_rows_values)} new rows to sheet.")
            sheet.append_rows(new_rows_values) # Use append_rows for multiple rows

        if rows_to_update_data:
            # Get the header (column names) from your sheet to ensure correct ordering
            header = sheet.row_values(1) # Assuming first row is header

            cells_for_batch_update = []
            for row_idx, data in rows_to_update_data.items():
                for col_name, value in data.items():
                    try:
                        col_idx = header.index(col_name) # Find column index by name
                        col_letter = chr(ord('A') + col_idx) # Convert index to letter
                        cell = sheet.acell(f"{col_letter}{row_idx}")
                        if cell.value != str(value): # Only update if value has changed
                            cell.value = value
                            cells_for_batch_update.append(cell)
                    except ValueError:
                        print(f"WARNING: Column '{col_name}' not found in sheet header. Skipping.")

            if cells_for_batch_update:
                print(f"DEBUG: Performing batch update for {len(cells_for_batch_update)} cells.")
                sheet.update_cells(cells_for_batch_update)
            else:
                print("DEBUG: No cells needed batch update.")


        flash("Rankings saved successfully!")
        return redirect(url_for('view_album_after_save', album_name=album_name, artist_name=artist_name))

    except Exception as e:
        import traceback
        print("🔥 ERROR in /submit_rankings route:")
        traceback.print_exc()
        return f"Internal server error: {e}", 500


# You might need a new route to redirect to after saving to reload the album.
# This route would fetch the album data again based on name/artist or ID.
# Or, you can make /view_album or /load_album accept GET requests with parameters.
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
@app.route("/load_albums_by_artist", methods=["POST"])
def load_albums_by_artist_route():
    artist_name = request.form["artist_name"]
    albums = get_albums_by_artist(artist_name)

    return render_template("select_album.html", artist_name=artist_name, albums=albums)
@app.route("/ranking_page")
def ranking_page():
    sheet_rows = load_google_sheet_data()
    group_bins = group_ranked_songs(sheet_rows)
    return render_template("album.html", group_bins=group_bins)


@app.route("/load_album", methods=["POST"])
def load_album():
    try:
        # Get the album_id and artist_name from the form (from "Select Album" button)
        album_id_from_form = request.form["album_id"]
        artist_name_from_form = request.form["artist_name"]
        # album_url will NOT be present for this route, so no need to fetch it from request.form

        print(f"DEBUG /load_album: Received album_id: {album_id_from_form}")
        print(f"DEBUG /load_album: Received artist_name: {artist_name_from_form}")

        # CRITICAL FIX: Construct a Spotify URI from the ID for spotify_logic.load_album_data
        # This is the format your spotify_logic.load_album_data's extract_album_id expects
        # to correctly parse an ID.
        spotify_uri_to_fetch = f"spotify:album:{album_id_from_form}"
        print(f"DEBUG /load_album: Constructed Spotify URI for fetch: {spotify_uri_to_fetch}")

        # Call load_album_data ONLY ONCE to get all the album details and songs
        album_data_from_spotify = load_album_data(spotify_uri_to_fetch)
        print(f"DEBUG /load_album: Data from spotify_logic.load_album_data: {album_data_from_spotify}")

        # Extract top-level album info for the template using .get() for robustness
        # Provide sensible fallbacks in case Spotify data is missing (e.g., API error)
        album_name_final = album_data_from_spotify.get("album_name", "Unknown Album Name (API Error)")
        artist_name_final = album_data_from_spotify.get("artist_name", artist_name_from_form)
        album_cover_url_final = album_data_from_spotify.get("album_cover_url", "")
        # Get the songs list from the Spotify data
        spotify_tracks_list = album_data_from_spotify.get("songs", [])

        print(
            f"DEBUG /load_album: Extracted for template: Name='{album_name_final}', Artist='{artist_name_final}', Cover='{album_cover_url_final}'")

        # --- Start of existing logic to merge with Google Sheet data ---

        previously_ranked = load_google_sheet_data()

        # Initialize rank_groups to be passed to JS
        rank_groups_for_js = {f"{(i * 0.5):.1f}": [] for i in range(1, 21)}
        rank_groups_for_js["?"] = []  # Unranked/preliminary group

        # Populate rank_groups_for_js and also track all ranked song IDs for checkmarks
        all_ranked_song_ids = set()  # To easily check if a song is ranked anywhere

        for sheet_row in previously_ranked:
            # Filter for songs belonging to the current album
            if (sheet_row.get("Album Name", "").strip().lower() == album_name_final.strip().lower() and
                    sheet_row.get("Artist Name", "").strip().lower() == artist_name_final.strip().lower()):

                song_data = {
                    "song_name": sheet_row.get("Song Name", ""),
                    "song_id": sheet_row.get("Spotify Song ID", ""),  # Crucial: Ensure this is in your sheet
                    "prelim_rank": sheet_row.get("Preliminary Rank", ""),  # Make sure this column exists
                    "rank_position": sheet_row.get("Rank Position", 999),  # Default to high for sorting
                    # Add any other data needed by the JS, like calculated_score if you want to display it on load
                    "calculated_score": sheet_row.get("Ranking", "")
                }

                rank_group_str = f"{sheet_row.get('Rank Group', '?'):.1f}" if sheet_row.get(
                    'Rank Group') is not None else "?"

                # Add to the appropriate rank group
                if rank_group_str in rank_groups_for_js:
                    rank_groups_for_js[rank_group_str].append(song_data)
                    if sheet_row.get("Ranking Status") in ["paused", "final"]:
                        all_ranked_song_ids.add(song_data["song_id"])  # Track ranked songs by ID
                elif sheet_row.get("Ranking Status") == "paused":  # Songs paused without a specific group go to '?'
                    rank_groups_for_js["?"].append(song_data)
                    all_ranked_song_ids.add(song_data["song_id"])

        # Sort songs within each group by rank_position
        for group_key in rank_groups_for_js:
            rank_groups_for_js[group_key].sort(key=lambda s: s.get("rank_position", float('inf')))

        # Prepare `album.songs` (for the left panel) with `already_ranked` and `prelim_rank`
        songs_for_left_panel = []
        for spotify_track in album_data_from_spotify.get("songs", []):
            song_id = spotify_track.get("song_id")
            song_name = spotify_track.get("song_name")

            # Check if this Spotify song is found anywhere in our loaded ranked data
            already_ranked = song_id in all_ranked_song_ids

            prelim_rank_value = ""
            # Try to find its prelim rank from the sheet data if it's there
            for sheet_row in previously_ranked:
                if (sheet_row.get("Spotify Song ID") == song_id and
                        sheet_row.get("Album Name", "").strip().lower() == album_name_final.strip().lower() and
                        sheet_row.get("Artist Name", "").strip().lower() == artist_name_final.strip().lower()):
                    prelim_rank_value = sheet_row.get("Preliminary Rank", "")
                    break

            songs_for_left_panel.append({
                "song_name": song_name,
                "song_id": song_id,  # Crucial to pass ID to JS
                "prelim_rank": prelim_rank_value,
                "already_ranked": already_ranked
            })

        album_data_for_template = {
            "album_name": album_name_final,
            "artist_name": artist_name_final,
            "album_cover_url": album_cover_url_final,
            "songs": songs_for_left_panel,  # This is the main song list for the left panel
            "url": spotify_uri_to_fetch  # Or the original album_id_from_form
        }

        bg_color = get_dominant_color(album_cover_url_final)
        album_data_for_template["bg_color"] = bg_color

        print(f"DEBUG /load_album: Final album dict sent to template: {album_data_for_template}")
        print(f"DEBUG /load_album: Final rank_groups dict sent to template: {rank_groups_for_js}")

        return render_template("album.html",
                               album=album_data_for_template,
                               rank_groups=rank_groups_for_js)  # Pass rank_groups if needed for initial state

    except Exception as e:
        import traceback
        print("🔥 ERROR in /load_album route:")
        traceback.print_exc()  # This will print the full error traceback
        return f"Internal server error: {e}", 500
@app.route('/view_album', methods=['POST'])
def view_album():
    album_url = request.form.get("album_url")
    print(f"DEBUG: Received album_url: {album_url}")
    if not album_url:
        return "Missing album_url", 400

    try:
        album = load_album_data(album_url)
        print(f"DEBUG: Data from spotify_logic.load_album_data: {album}")

        album_name = album.get("album_name", "ERROR_NO_NAME")
        artist_name = album.get("artist_name", "ERROR_NO_ARTIST")
        album_cover_url = album.get("album_cover_url", "ERROR_NO_COVER")
        print(f"DEBUG: Extracted: Name={album_name}, Artist={artist_name}, Cover={album_cover_url}")

    # Step 1: fetch album info from Spotify
        album = load_album_data(album_url)
        album_name  = album["album_name"]
        artist_name = album["artist_name"]

        # Step 2: load existing rows from Google Sheets
        sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
        df    = get_as_dataframe(sheet, evaluate_formulas=True).fillna("")

        # Normalize for matching
        df["Album Name"]  = df["Album Name"].astype(str).str.strip().str.lower()
        df["Artist Name"] = df["Artist Name"].astype(str).str.strip().str.lower()

        album_key  = album_name.strip().lower()
        artist_key = artist_name.strip().lower()

        # Step 3: collect any “paused” rows for this album/artist
        paused_df = df[
            (df["Album Name"]  == album_key) &
            (df["Artist Name"] == artist_key) &
            (df["Ranking Status"] == "paused")
        ]
        rank_counts = df["Song Name"].value_counts().to_dict()

        songs = []
        for _, row in paused_df.iterrows():
            songs.append({
                "song_name": row["Song Name"],
                "prelim_rank": row["Ranking"],
                "rank_count": row.get("Rank Count", 0)  # Add this line
            })

        # Step 4: if there were no paused rows, fall back to the Spotify tracklist
        if not songs:
            songs = [
                {
                    "song_name": track["song_name"],
                    "prelim_rank": "",
                    "rank_count": 0  # Add this!
                }
                for track in album["songs"]
            ]
        cover_url = album["album_cover_url"]
        bg_color = get_dominant_color(cover_url)
        album_data = {
            "album_name": album_name,
            "artist_name": artist_name,
            "album_cover_url": cover_url,
            "url": album_url,  # if you need url for the “back” form
            "image": cover_url,  # alias if your template still uses album.image
            "songs": songs,
            "bg_color": bg_color
        }
        print(f"DEBUG: Final album_data sent to template: {album_data}")

        # 2. Pass cover_url **and** the other album fields through
        return render_template(
            "album.html", album=album_data
        )
    except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error processing album: {e}", 500


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
