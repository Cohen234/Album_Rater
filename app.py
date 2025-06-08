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

creds_info = json.loads(os.environ['GOOGLE_SERVICE_ACCOUNT_JSON'])

creds = Credentials.from_service_account_info(
    creds_info,
    scopes=["https://www.googleapis.com/auth/spreadsheets"]
)

client = gspread.authorize(creds)
SPREADSHEET_ID = '15E4b-DWSYP9AzbAzSviqkW-jEOktbimPlmhNIs_d5jc'
SHEET_NAME = "Sheet1"

app = Flask(__name__)

def get_dominant_color(image_url):
    response = requests.get(image_url)
    color_thief = ColorThief(BytesIO(response.content))
    rgb = color_thief.get_color(quality=1)
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

def get_album_stats(album_name, artist_name, df=None):
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    if df is None:
        df = get_as_dataframe(sheet, evaluate_formulas=True)

    album_key = album_name.strip().lower()
    artist_key = artist_name.strip().lower()
    df = df.dropna(subset=["Album Name", "Artist Name"])
    df["Album Name"] = df["Album Name"].str.strip().str.lower()
    df["Artist Name"] = df["Artist Name"].str.strip().str.lower()
    df["Ranking Status"] = df["Ranking Status"].fillna("")

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
            '"Ranking Status"': None
        }

    # Check if there is any paused ranking status for this album
    paused_exists = album_df["Ranking Status"].str.contains("paused", case=False).any()

    # Filter finalized ranks
    finalized_df = album_df[album_df["Ranking Status"] == "final"]

    if finalized_df.empty:
        avg_rank = None
        latest_date = None
        finalized_count = 0
    else:
        finalized_df["Ranked Date"] = pd.to_datetime(finalized_df["Ranked Date"], errors="coerce")
        latest_date = finalized_df["Ranked Date"].max()
        latest_session = finalized_df[finalized_df["Ranked Date"] == latest_date]
        avg_rank = latest_session["Ranking"].mean()
        finalized_count = finalized_df["Ranked Date"].nunique()

    return {
        'finalized_rank_count': finalized_count,
        'last_final_avg_rank': round(avg_rank, 2) if avg_rank else None,
        'last_final_rank_date': latest_date.strftime("%Y-%m-%d %H:%M:%S") if latest_date else None,
        '"Ranking Status"': "paused" if paused_exists else "final"
    }
@app.route("/submit_rankings", methods=["POST"])
def submit_rankings():
    album_name = request.form.get("album_name")
    artist_name = request.form.get("artist_name")
    selected_rank_str = request.form.get("selected_rank", "").strip()
    ranking_data_str = request.form.get("ranking_data", "").strip()
    status = request.form.get("Ranking Status")

    # Validate selected_rank
    try:
        selected_rank = float(selected_rank_str)
    except ValueError:
        return "Error: Invalid rank group selected.", 400

    # Validate ranking_data
    try:
        ranking_data = json.loads(ranking_data_str)
    except json.JSONDecodeError:
        return "Error: Invalid song ranking data.", 400

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    songs = []

    # Assign sub-ranks based on order
    sub_step = 0.5 / max(len(ranking_data), 1)
    for i, song_name in enumerate(ranking_data):
        rank = selected_rank - 0.25 + (i + 0.5) * sub_step
        songs.append({
            "Album Name": album_name,
            "Artist Name": artist_name,
            "Song Name": song_name,
            "Ranking": round(rank, 2),
            "Ranking Status": status,
            "Ranked Date": now
        })

    # Google Sheets update logic
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    df_existing = get_as_dataframe(sheet, evaluate_formulas=True).fillna("")
    df_existing["Album Name"] = df_existing["Album Name"].str.strip().str.lower()
    df_existing["Artist Name"] = df_existing["Artist Name"].str.strip().str.lower()

    album_key = album_name.strip().lower()
    artist_key = artist_name.strip().lower()

    # Remove existing entries in this rank group
    mask = ~(
        (df_existing["Album Name"] == album_key) &
        (df_existing["Artist Name"] == artist_key) &
        (df_existing["Ranking Status"] == status) &
        (df_existing["Ranking"] >= selected_rank - 0.25) &
        (df_existing["Ranking"] <= selected_rank + 0.25)
    )
    df_filtered = df_existing[mask]

    df_new = pd.DataFrame(songs)
    final_df = pd.concat([df_filtered, df_new], ignore_index=True)

    sheet.clear()
    set_with_dataframe(sheet, final_df)

    return redirect(url_for("index"))
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_albums_by_artist', methods=['POST'])
def load_albums_by_artist_route():
    artist_name = request.form['artist_name']
    albums = get_albums_by_artist(artist_name)

    for album in albums:
        album["stats"] = get_album_stats(album["name"], artist_name)

    return render_template("select_album.html", artist_name=artist_name, albums=albums)

@app.route("/album")
def view_album():
    album_name = request.args.get("album")
    artist_name = request.args.get("artist")

    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    df = get_as_dataframe(sheet, evaluate_formulas=True).fillna("")

    album_key = album_name.strip().lower()
    artist_key = artist_name.strip().lower()
    df["Album Name"] = df["Album Name"].str.strip().str.lower()
    df["Artist Name"] = df["Artist Name"].str.strip().str.lower()

    album_df = df[(df["Album Name"] == album_key) & (df["Artist Name"] == artist_key)]

    # Get songs and prelim ranks from paused rows, fallback to empty prelim_rank if none
    paused_df = album_df[album_df["Ranking Status"] == "paused"]

    songs = []
    for _, row in paused_df.iterrows():
        songs.append({
            'song_name': row['Song Name'],
            'prelim_rank': row['Ranking']
        })

    # If no paused prelim ranks, fallback to all songs with prelim_rank blank
    if not songs:
        songs_list = album_df['Song Name'].unique()
        songs = [{'song_name': s, 'prelim_rank': ''} for s in songs_list]

    album_cover_url = album_df.iloc[0].get('album_cover_url', '') if not album_df.empty else ''
    bg_color = album_df.iloc[0].get('bg_color', '#ffffff') if not album_df.empty else '#ffffff'

    return render_template("album_ui.html", album={
        'album_name': album_name,
        'artist_name': artist_name,
        'songs': songs,
        'album_cover_url': album_cover_url
    }, bg_color=bg_color)
@app.route("/get_ranked_songs")
def get_ranked_songs():
    album_name = request.args.get("album_name")
    artist_name = request.args.get("artist_name")
    rank = request.args.get("rank")
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)


    df = pd.DataFrame(sheet.get_all_records())
    rank_df = df[(df['Album Name'].str.strip().str.lower() == album_name.strip().lower()) &
        (df['Artist Name'].str.strip().str.lower() == artist_name.strip().lower()) &
        (df['rank_group'].astype(str).str.strip() == rank)]

    # Prioritize paused rankings if present
    songs = rank_df[rank_df["Ranking Status"] != 'final']['song_name'].tolist()
    if not songs:
        songs = rank_df[rank_df["Ranking Status"] == 'final']['song_name'].tolist()

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
