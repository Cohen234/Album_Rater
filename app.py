from flask import Flask, render_template, request, redirect, url_for
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
            'ranking_status': None
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
        'ranking_status': "paused" if paused_exists else "final"
    }

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

@app.route("/view_album", methods=["GET", "POST"])
def view_album():
    album_url = request.form.get("album_url")
    album = load_album_data(album_url)

    # Load paused data
    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    df = get_as_dataframe(sheet, evaluate_formulas=True)
    df = df.dropna(subset=["Album Name", "Artist Name", "Song Name"])

    def normalize(x): return x.strip().lower() if isinstance(x, str) else x
    df["Album Name"] = df["Album Name"].apply(normalize)
    df["Artist Name"] = df["Artist Name"].apply(normalize)

    album_key = normalize(album["album_name"])
    artist_key = normalize(album["artist_name"])

    paused_rows = df[
        (df["Album Name"] == album_key) &
        (df["Artist Name"] == artist_key) &
        (df["Ranking Status"] == "paused")
    ]

    saved_rankings = {
        row["Song Name"]: row["Ranking"]
        for _, row in paused_rows.iterrows()
        if row["Song Name"] not in ["__ALBUM_OVERALL__", "__ALBUM_SESSION__"] and row["Ranking"] != ""
    }

    overall_row = paused_rows[paused_rows["Song Name"] == "__ALBUM_OVERALL__"]
    saved_overall_score = float(overall_row.iloc[0]["Ranking"]) if not overall_row.empty else ""

    bg_color = get_dominant_color(album["album_cover_url"])
    stats = get_album_stats(album["album_name"], album["artist_name"], df)

    return render_template(
        "album.html",
        album=album,
        stats=stats,
        saved_rankings=saved_rankings,
        saved_overall_score=saved_overall_score,
        bg_color=bg_color
    )

@app.route("/save_album", methods=["POST"])
def save_album():
    album_name = request.form.get('album_name')
    artist_name = request.form.get('artist_name')
    status = request.form.get('status')
    overall_score = request.form.get('overall_score')

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    songs = []

    i = 0
    while True:
        ranking = request.form.get(f"ranking_{i}")
        song_name = request.form.get(f"song_name_{i}")
        if not ranking or not song_name:
            break
        try:
            rank_val = float(ranking)
        except ValueError:
            rank_val = ""
        songs.append({
            "Album Name": album_name,
            "Artist Name": artist_name,
            "Song Name": song_name,
            "Ranking": rank_val,
            "Ranking Status": status,
            "Ranked Date": now
        })
        i += 1

    # Add overall score and session marker
    try:
        overall_score_val = float(overall_score) if overall_score else ""
    except ValueError:
        overall_score_val = ""

    songs.append({
        "Album Name": album_name,
        "Artist Name": artist_name,
        "Song Name": "__ALBUM_OVERALL__",
        "Ranking": overall_score_val,
        "Ranking Status": status,
        "Ranked Date": now if overall_score else ""
    })
    songs.append({
        "Album Name": album_name,
        "Artist Name": artist_name,
        "Song Name": "__ALBUM_SESSION__",
        "Ranking": "",
        "Ranking Status": status,
        "Ranked Date": now
    })

    album_df = pd.DataFrame(songs)

    sheet = client.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)
    try:
        df_existing = get_as_dataframe(sheet, evaluate_formulas=True)
        if df_existing.empty:
            df_existing = pd.DataFrame(columns=album_df.columns)
    except Exception:
        df_existing = pd.DataFrame(columns=album_df.columns)

    def normalize(x): return x.strip().lower() if isinstance(x, str) else x
    df_existing["Album Name"] = df_existing["Album Name"].apply(normalize)
    df_existing["Artist Name"] = df_existing["Artist Name"].apply(normalize)

    album_key = normalize(album_name)
    artist_key = normalize(artist_name)

    if status == "paused":
        mask = ~(
            (df_existing["Album Name"] == album_key) &
            (df_existing["Artist Name"] == artist_key) &
            (df_existing["Ranking Status"] == "paused")
        )
    else:
        mask = ~(
            (df_existing["Album Name"] == album_key) &
            (df_existing["Artist Name"] == artist_key) &
            (df_existing["Song Name"] != "__ALBUM_SESSION__")
        )

    df_keep = df_existing[mask]
    new_df = pd.concat([df_keep, album_df], ignore_index=True)
    sheet.clear()
    set_with_dataframe(sheet, new_df)

    if status == "paused":
        print(f"Album '{album_name}' paused and saved. Exiting...")
        sys.exit(0)

    return redirect(url_for("index"))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
