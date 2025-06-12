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
    response = requests.get(image_url)
    color_thief = ColorThief(BytesIO(response.content))
    rgb = color_thief.get_color(quality=1)
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

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
@app.route("/submit_rankings", methods=["POST"])
def submit_rankings():
    """
    Handles both Pause & Final:
      - If status == "paused": read each “prelim_rank_{song}” from request.form and save a paused row per song.
                           Then redirect back to index().
      - If status == "final": read ranking_data (JSON array of song names dragged in order),
                              compute sub‐ranks, drop any old “final” rows for that album/artist,
                              append new final rows, and redirect to index().
    """
    album_name  = request.form.get("album_name", "").strip()
    artist_name = request.form.get("artist_name", "").strip()
    status      = request.form.get("status", "").strip().lower()   # either "paused" or "final"

    # Open the Sheet once
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    df_existing = get_as_dataframe(sheet, evaluate_formulas=True).fillna("")

    # If sheet is empty, create correct columns
    if df_existing.empty:
        df_existing = pd.DataFrame(columns=[
            "Album Name", "Artist Name", "Song Name", "Ranking", "Ranking Status", "Ranked Date"
        ])

    # Normalize text columns
    df_existing["Album Name"]   = df_existing["Album Name"].astype(str).str.strip().str.lower()
    df_existing["Artist Name"]  = df_existing["Artist Name"].astype(str).str.strip().str.lower()
    df_existing["Ranking Status"]= df_existing["Ranking Status"].fillna("")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    album_key  = album_name.lower()
    artist_key = artist_name.lower()

    if status == "paused":
        # 1) Remove any existing “paused” rows for this album/artist
        mask_keep = ~(
            (df_existing["Album Name"] == album_key) &
            (df_existing["Artist Name"] == artist_key) &
            (df_existing["Ranking Status"] == "paused")
        )
        df_filtered = df_existing.loc[mask_keep].copy()

        # 2) Collect all “prelim_rank_{song}” values from the form
        new_rows = []
        for key, val in request.form.items():
            if not key.startswith("prelim_rank_"):
                continue
            song_name = key.replace("prelim_rank_", "")
            try:
                rank_val = float(val)
            except ValueError:
                rank_val = ""
            new_rows.append({
                "Album Name": album_name,
                "Artist Name": artist_name,
                "Song Name": song_name,
                "Ranking": rank_val,
                "Ranking Status": "paused",
                "Ranked Date": now
            })

        df_to_write = pd.concat([df_filtered, pd.DataFrame(new_rows)], ignore_index=True)
        sheet.clear()
        set_with_dataframe(sheet, df_to_write)

        # After pausing, go back to the search screen
        return redirect(url_for("index"))

    elif status == "final":
        # 1) Parse selected_rank and ranking_data (JSON array of song names)
        try:
            selected_rank = float(request.form.get("selected_rank", "").strip())
        except (ValueError, TypeError):
            return "Error: Invalid rank group for final.", 400

        try:
            ranking_data = json.loads(request.form.get("ranking_data", "[]"))
        except (json.JSONDecodeError, TypeError):
            return "Error: Invalid ranking_data JSON.", 400

        # 2) Remove any old “final” rows for this album/artist
        mask_keep = ~(
            (df_existing["Album Name"] == album_key) &
            (df_existing["Artist Name"] == artist_key) &
            (df_existing["Ranking Status"] == "final")
        )
        df_filtered = df_existing.loc[mask_keep].copy()
        df_filtered["Ranking"] = pd.to_numeric(df_filtered["Ranking"], errors="coerce")

        # 3) Build new final rows: distribute songs within [rank - 0.25, rank + 0.25]
        new_rows = []
        sub_step = 0.5 / max(len(ranking_data), 1)
        for i, song_name in enumerate(ranking_data):
            rank_val = selected_rank - 0.25 + (i + 0.5) * sub_step
            new_rows.append({
                "Album Name": album_name,
                "Artist Name": artist_name,
                "Song Name": song_name,
                "Ranking": round(rank_val, 2),
                "Ranking Status": "final",
                "Ranked Date": now
            })

        df_to_write = pd.concat([df_filtered, pd.DataFrame(new_rows)], ignore_index=True)
        sheet.clear()
        set_with_dataframe(sheet, df_to_write)

        return redirect(url_for("index"))

    else:
        return "Error: Unknown status", 400
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
        album_id = request.form["album_id"]
        artist_name = request.form["artist_name"]
        album_data = load_album_data(album_id)

        # If load_album_data returns a single dict with keys like 'songs', fix it here
        if isinstance(album_data, dict) and "songs" in album_data:
            album_songs = album_data["songs"]
        else:
            album_songs = album_data
        previously_ranked = load_google_sheet_data()

        for song in previously_ranked:
            song["id"] = song.get("id") or song.get("track_id")

        ranked_ids = {song["id"] for song in previously_ranked if song["id"]}
        rank_groups = {f"{i/2:.1f}": [] for i in range(2, 21)}
        rank_groups["?"] = []

        for song in previously_ranked:
            group_value = song.get("rank_group")
            if group_value is None:
                continue  # skip if no rank_group — or you could add to "?" group if you prefer
            try:
                group = f"{float(group_value):.1f}"
            except ValueError:
                continue  # skip if not a number
            song["already_ranked"] = True
            if group in rank_groups:
                rank_groups[group].append(song)
        for song in album_songs:
            song["id"] = song.get("id") or song.get("track_id")  # Normalize to ensure "id" exists
            if song["id"] and song["id"] not in ranked_ids:
                song["already_ranked"] = False
                song["rank_position"] = 999
                rank_groups["?"].append(song)


        for group in rank_groups:
            rank_groups[group].sort(key=lambda s: s.get("rank_position", 0))

        album = {
            "album_name": album_songs[0].get("album_name", "Unknown Album") if album_songs else "Unknown Album",
            "album_cover_url": album_songs[0].get("album_cover_url", "") if album_songs else "",
            "artist_name": artist_name,
            "songs": album_songs
        }

        return render_template("album.html",
                               album=album,
                               rank_groups=rank_groups)

    except Exception as e:
        import traceback
        print("🔥 ERROR in /load_album route:")
        traceback.print_exc()
        return f"Internal server error: {e}", 500
@app.route('/view_album', methods=['POST'])
def view_album():
    album_url = request.form.get("album_url")
    if not album_url:
        return "Missing album_url", 400

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
    album_cover_url = album["album_cover_url"]
    try:
        bg_color = get_dominant_color(album_cover_url)
    except Exception:
        bg_color = "#ffffff"

    return render_template("album.html",
        album={
            "album_name":      album_name,
            "artist_name":     artist_name,
            "songs":           songs,
            "album_cover_url": album_cover_url
        },
        bg_color=bg_color
    )

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
