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

creds_info = json.loads(os.environ['GOOGLE_SERVICE_ACCOUNT_JSON'])

creds = Credentials.from_service_account_info(
    creds_info,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)

client = gspread.authorize(creds)
SPREADSHEET_ID = '15E4b-DWSYP9AzbAzSviqkW-jEOktbimPlmhNIs_d5jc'
SHEET_NAME = "Sheet1"

app = Flask(__name__)
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

@app.route("/submit_rankings", methods=["POST"])
def submit_rankings():
    try:
        album_name = request.form["album_name"]
        artist_name = request.form["artist_name"]
        submission_status = request.form["status"]  # 'paused' or 'final'

        # Get all ranked songs data from the hidden input
        all_ranked_data_json = request.form.get("all_ranked_data", "[]")
        all_ranked_songs_from_js = json.loads(all_ranked_data_json)

        # Get all preliminary ranks from the hidden input
        prelim_rank_data_json = request.form.get("prelim_rank_data", "{}")
        prelim_ranks_from_js = json.loads(prelim_rank_data_json)  # {song_id: prelim_value, ...}

        sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

        # Load all existing rows for this album/artist to update them
        # (This avoids fetching the whole sheet if you can filter more efficiently)
        # Assuming your sheet has columns like "Album Name" and "Artist Name"
        existing_rows_df = get_as_dataframe(sheet, evaluate_formulas=True).fillna("")

        # Filter for rows belonging to this specific album/artist
        album_artist_filtered_df = existing_rows_df[
            (existing_rows_df["Album Name"].astype(str).str.strip().str.lower() == album_name.strip().lower()) &
            (existing_rows_df["Artist Name"].astype(str).str.strip().str.lower() == artist_name.strip().lower())
            ]

        # Prepare data for new/updated rows
        rows_to_update = []
        updated_song_ids = set()  # To track which songs we've processed from JS submission

        # Process submitted ranked data
        for song_data in all_ranked_songs_from_js:
            song_name = song_data["song_name"]
            song_id = song_data["song_id"]
            rank_group = song_data["rank_group"]
            rank_position = song_data["rank_position"]
            calculated_score = song_data["calculated_score"]

            # Use the submission_status ('paused' or 'final')
            status_for_sheet = submission_status

            # Find if this song exists in the filtered DataFrame
            existing_song_row = album_artist_filtered_df[
                (album_artist_filtered_df["Spotify Song ID"] == song_id)  # Match by ID if possible
            ]

            if not existing_song_row.empty:
                # Update existing row
                row_index_in_sheet = existing_song_row.index[0] + 2  # +2 for header and 0-indexing
                rows_to_update.append({
                    'row': row_index_in_sheet,
                    'Song Name': song_name,
                    'Album Name': album_name,
                    'Artist Name': artist_name,
                    'Ranking': calculated_score,
                    'Rank Group': rank_group,
                    'Rank Position': rank_position,
                    'Preliminary Rank': prelim_ranks_from_js.get(song_id, ''),  # Take prelim rank if present
                    'Ranking Status': status_for_sheet,
                    'Spotify Song ID': song_id,
                    'Timestamp': datetime.now().isoformat()
                })
            else:
                # Add new row
                rows_to_update.append({
                    'row': 'new',  # Sentinel value for new row
                    'Song Name': song_name,
                    'Album Name': album_name,
                    'Artist Name': artist_name,
                    'Ranking': calculated_score,
                    'Rank Group': rank_group,
                    'Rank Position': rank_position,
                    'Preliminary Rank': prelim_ranks_from_js.get(song_id, ''),
                    'Ranking Status': status_for_sheet,
                    'Spotify Song ID': song_id,
                    'Timestamp': datetime.now().isoformat()
                })
            updated_song_ids.add(song_id)

        # Process songs that were only preliminarily ranked but not dragged into a group
        # Or songs that were previously ranked but are NOT in the `all_ranked_songs_from_js`
        # This handles scenarios where songs might be in the '?' group or just have prelim ranks
        for _, row in album_artist_filtered_df.iterrows():
            song_id = row.get("Spotify Song ID")
            if song_id not in updated_song_ids:  # If this song from sheet was NOT processed above
                current_prelim_rank = prelim_ranks_from_js.get(song_id, row.get("Preliminary Rank", ""))

                # If it has a prelim rank, or it was previously paused, update its status/prelim rank
                if current_prelim_rank or row.get("Ranking Status") == "paused":
                    # Determine its new status. If it had a group but isn't submitted, its rank is "cleared"
                    # Or set to "paused" without a group.
                    new_status = submission_status  # Apply general submission status
                    new_rank_group = row.get("Rank Group", "")  # Keep previous group if any
                    new_rank_pos = row.get("Rank Position", "")
                    new_ranking = row.get("Ranking", "")

                    if submission_status == "final":
                        # If a song was previously paused/ranked but not in the final submission,
                        # you might want to clear its rank or mark it differently.
                        # For now, let's keep its last state if not submitted as final,
                        # unless it explicitly has a new prelim rank
                        if current_prelim_rank:  # If it now has a prelim rank, it's 'paused'
                            new_status = 'paused'
                            new_rank_group = ''  # Clear group
                            new_rank_pos = ''
                            new_ranking = ''
                        else:  # If no prelim rank and not submitted as final, clear it
                            new_status = ''  # Or 'unranked'
                            new_rank_group = ''
                            new_rank_pos = ''
                            new_ranking = ''

                    rows_to_update.append({
                        'row': row.name + 2,  # Pandas index + 2 for sheet row
                        'Song Name': row.get("Song Name"),
                        'Album Name': album_name,
                        'Artist Name': artist_name,
                        'Ranking': new_ranking,
                        'Rank Group': new_rank_group,
                        'Rank Position': new_rank_pos,
                        'Preliminary Rank': current_prelim_rank,
                        'Ranking Status': new_status,
                        'Spotify Song ID': song_id,
                        'Timestamp': datetime.now().isoformat()
                    })

        # --- Batch Update Logic ---
        # Map DataFrame columns to GSpread sheet columns
        # (You'll need a way to map your column names like "Song Name" to their actual column index/letter)
        # This is simplified; you'll need the actual column names from your sheet.
        # Example mapping (adjust based on your actual sheet layout):
        column_mapping = {
            'Song Name': 'A', 'Album Name': 'B', 'Artist Name': 'C',
            'Ranking': 'D', 'Rank Group': 'E', 'Rank Position': 'F',
            'Preliminary Rank': 'G', 'Ranking Status': 'H',
            'Spotify Song ID': 'I', 'Timestamp': 'J'
        }

        # Prepare cells for batch update
        cells_to_update = []
        for row_data in rows_to_update:
            row_num = row_data.pop('row')  # Get the row number or 'new'
            if row_num == 'new':
                # This is a new row, so append it (gspread's append_row)
                # You need to ensure the order of values matches your sheet columns
                new_row_values = [
                    row_data.get('Song Name', ''), row_data.get('Album Name', ''), row_data.get('Artist Name', ''),
                    row_data.get('Ranking', ''), row_data.get('Rank Group', ''), row_data.get('Rank Position', ''),
                    row_data.get('Preliminary Rank', ''), row_data.get('Ranking Status', ''),
                    row_data.get('Spotify Song ID', ''), row_data.get('Timestamp', '')
                ]
                sheet.append_row(new_row_values)
            else:
                # Update existing row cells
                for col_name, col_letter in column_mapping.items():
                    if col_name in row_data:
                        cell = sheet.acell(f"{col_letter}{row_num}")  # Get the cell object
                        cell.value = row_data[col_name]
                        cells_to_update.append(cell)

        if cells_to_update:
            sheet.update_cells(cells_to_update)

        # Consider redirecting back to the album view with a success message
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
