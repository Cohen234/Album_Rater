def save_to_google_sheets(self, status="final"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Validate ranking inputs (same as original)
    all_scores = []
    for i, var in enumerate(self.rank_vars):
        value = var.get().strip()
        if status == "final":
            if not value:
                messagebox.showerror("Missing Ranking", "Please rank every song before submitting.")
                return
            try:
                score = float(value)
                if not (0 <= score <= 10):
                    raise ValueError()
            except ValueError:
                messagebox.showerror(
                    "Invalid Ranking",
                    f"Ranking for song '{self.album_df.at[i, 'Song Name']}' must be a number between 0 and 10."
                )
                return
            all_scores.append(value)
        else:
            if value:
                try:
                    score = float(value)
                    if not (0 <= score <= 10):
                        raise ValueError()
                except ValueError:
                    messagebox.showerror(
                        "Invalid Ranking",
                        f"Ranking for song '{self.album_df.at[i, 'Song Name']}' must be a number between 0 and 10 (or blank if not ranked yet)."
                    )
                    return
            all_scores.append(value)

    overall_score = self.overall_score_var.get().strip()
    if overall_score:
        try:
            overall_num = float(overall_score)
            if not (0 <= overall_num <= 10):
                raise ValueError()
        except ValueError:
            messagebox.showerror("Invalid Overall Score", "Overall score must be a number between 0 and 10.")
            return

    # Update album_df with rankings and dates
    for i, value in enumerate(all_scores):
        self.album_df.at[i, "Ranking"] = value
        self.album_df.at[i, "Ranked Date"] = now if value else ""
        self.album_df.at[i, "Ranking Status"] = status

    album_rows = self.album_df[self.album_df["Song Name"] != "__ALBUM_OVERALL__"].copy()

    # Create overall album row
    overall_row = {col: "" for col in album_rows.columns}
    overall_row["Album Name"] = self.album_name
    overall_row["Artist Name"] = self.artist_name
    overall_row["Song Name"] = "__ALBUM_OVERALL__"
    overall_row["Ranking"] = overall_score
    overall_row["Ranked Date"] = now if overall_score else ""
    overall_row["Ranking Status"] = status
    album_rows = pd.concat([album_rows, pd.DataFrame([overall_row])], ignore_index=True)

    # Create session row
    session_row = {col: "" for col in album_rows.columns}
    session_row["Album Name"] = self.album_name
    session_row["Artist Name"] = self.artist_name
    session_row["Song Name"] = "__ALBUM_SESSION__"
    session_row["Ranking"] = ""
    session_row["Ranked Date"] = now
    session_row["Ranking Status"] = status
    album_rows = pd.concat([album_rows, pd.DataFrame([session_row])], ignore_index=True)

    # Read existing sheet data into df_existing (if sheet empty, this will be empty DataFrame)
    try:
        df_existing = get_as_dataframe(sheet, evaluate_formulas=True)
        if df_existing.empty:
            df_existing = pd.DataFrame(columns=album_rows.columns)
    except Exception:
        # If sheet is empty or error occurs, create empty dataframe with columns
        df_existing = pd.DataFrame(columns=album_rows.columns)

    # Normalize keys function (copy your normalize_key function here or import it)
    def normalize_key(x):
        if isinstance(x, str):
            return x.strip().lower()
        return x

    df_existing['Album Name'] = df_existing['Album Name'].apply(normalize_key)
    df_existing['Artist Name'] = df_existing['Artist Name'].apply(normalize_key)
    album_name_key = normalize_key(self.album_name)
    artist_name_key = normalize_key(self.artist_name)

    # Filter existing rows based on status
    if status == "paused":
        # Remove only paused rows for this album/artist
        mask_not_this_paused = ~(
                (df_existing["Album Name"] == album_name_key) &
                (df_existing["Artist Name"] == artist_name_key) &
                (df_existing["Ranking Status"] == "paused")
        )
        df_keep = df_existing[mask_not_this_paused]
        out_df = pd.concat([df_keep, album_rows], ignore_index=True)
    else:
        # Remove all (except session) rows for this album/artist
        mask_album = ~(
                (df_existing["Album Name"] == album_name_key) &
                (df_existing["Artist Name"] == artist_name_key) &
                (df_existing["Song Name"] != "__ALBUM_SESSION__")
        )
        df_keep = df_existing[mask_album]
        out_df = pd.concat([df_keep, album_rows], ignore_index=True)

    # Clear existing sheet and write updated dataframe
    sheet.clear()
    set_with_dataframe(sheet, out_df)

    # Show messages and navigate UI
    if status == "final":
        messagebox.showinfo("Saved", "Album and score saved to Google Sheets")
        self.build_album_list_screen()
    else:
        messagebox.showinfo("Paused", "Your ranking progress has been paused and saved. You can resume later.")
        self.build_album_list_screen()
