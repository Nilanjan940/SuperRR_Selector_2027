
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SUPERR SELECTOR SEASON 2
# Six-file data model | IPL + T20I | Unique Top 10 + Top 10
# ============================================================

st.set_page_config(
    page_title="SupeRR Selector 2027",
    page_icon="🏏",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

FILES = {
    "balls": BASE / "ball_by_ball.csv",
    "matches": BASE / "matches.csv",
    "competitions": BASE / "competitions.csv",
    "mapping": BASE / "competition_team_player_mapping.csv",
    "playing_eleven": BASE / "playing_eleven.csv",
    "teams": BASE / "teams.csv",
}

st.title("🏏 SupeRR Selector — Season 2")
st.caption("IPL and T20I player evaluation | Pressure performance engine")

# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
}
[data-testid="stMetric"] {
    background-color: #17243a;
    border-radius: 10px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def normalize_id(series):
    """Normalize IDs without converting missing IDs into strings."""
    result = series.astype("string").str.strip()
    result = result.str.replace(r"\.0$", "", regex=True)
    return result.replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "<NA>": pd.NA
    })


def numeric(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def boolean(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def find_col(frame, candidates):
    lookup = {str(c).lower(): c for c in frame.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def get_col(frame, candidates, default=None):
    col = find_col(frame, candidates)

    if col is None:
        return pd.Series(default, index=frame.index)

    return frame[col]


def percentile(series, higher_is_better=True):
    values = pd.to_numeric(series, errors="coerce")

    if values.notna().sum() == 0:
        return pd.Series(50.0, index=series.index)

    result = values.rank(
        pct=True,
        ascending=higher_is_better,
        method="average"
    ) * 100

    return result.fillna(50)


def safe_divide(numerator, denominator):
    return numerator.div(
        denominator.replace(0, np.nan)
    )


def format_number(value):
    if pd.isna(value):
        return "—"
    return f"{value:,.2f}"


# ============================================================
# LOAD SIX FILES
# ============================================================

@st.cache_data(show_spinner=False)
def load_data():

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path.name}"
            )

    data = {
        name: pd.read_csv(path, low_memory=False)
        for name, path in FILES.items()
    }

    return data


try:
    with st.spinner("Loading cricket datasets..."):
        data = load_data()

except Exception as exc:
    st.error(str(exc))
    st.stop()


balls = data["balls"].copy()
matches = data["matches"].copy()
competitions = data["competitions"].copy()
mapping = data["mapping"].copy()
playing_eleven = data["playing_eleven"].copy()
teams = data["teams"].copy()


# ============================================================
# SCHEMA VALIDATION
# ============================================================

required_columns = {
    "balls": [
        "match_id",
        "innings",
        "striker_id",
        "bowler_id",
        "runs",
        "is_wicket",
        "extras_type",
        "extras1_runs",
        "extras_2_runs",
    ],
    "matches": [
        "match_id",
        "comp_id",
    ],
    "competitions": [
        "comp_id",
        "comp_name",
        "match_format",
        "season",
    ],
    "mapping": [
        "comp_id",
        "player_id",
        "player_name",
    ],
    "playing_eleven": [
        "match_id",
        "player_id",
        "team_id",
    ],
}

for file_name, required in required_columns.items():

    frame = data[file_name]

    missing = [
        col for col in required
        if col not in frame.columns
    ]

    if missing:
        st.error(
            f"{file_name}.csv is missing required columns: {missing}"
        )
        st.write("Available columns:", list(frame.columns))
        st.stop()


# ============================================================
# NORMALIZE IDS
# ============================================================

for frame in [
    balls,
    matches,
    playing_eleven
]:
    if "match_id" in frame.columns:
        frame["match_id"] = normalize_id(frame["match_id"])

for frame in [
    matches,
    competitions,
    mapping
]:
    if "comp_id" in frame.columns:
        frame["comp_id"] = normalize_id(frame["comp_id"])

for frame in [
    balls,
    mapping,
    playing_eleven
]:
    if "player_id" in frame.columns:
        frame["player_id"] = normalize_id(frame["player_id"])

for col in [
    "striker_id",
    "non_striker_id",
    "bowler_id",
    "dismissed_player_id"
]:
    if col in balls.columns:
        balls[col] = normalize_id(balls[col])

for frame in [mapping, playing_eleven, teams]:
    if "team_id" in frame.columns:
        frame["team_id"] = normalize_id(frame["team_id"])


# ============================================================
# COMPETITION CLASSIFICATION
# ============================================================

competitions["comp_name"] = (
    competitions["comp_name"]
    .fillna("")
    .astype(str)
)

competitions["match_format"] = (
    competitions["match_format"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)

competitions["season"] = (
    competitions["season"]
    .astype("string")
    .str.strip()
)

comp_name_lower = competitions["comp_name"].str.lower()

competitions["competition_type"] = np.select(
    [
        comp_name_lower.str.contains(
            r"\bipl\b|indian premier league",
            regex=True,
            na=False
        ),
        competitions["match_format"].eq("T20I"),
    ],
    [
        "IPL",
        "T20I"
    ],
    default="Other"
)

# Only IPL and T20I are included.
competitions = competitions[
    competitions["competition_type"].isin(["IPL", "T20I"])
].copy()

if competitions.empty:
    st.error("No IPL or T20I competitions found.")
    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Competition Filters")

available_types = sorted(
    competitions["competition_type"].dropna().unique()
)

selected_types = st.sidebar.multiselect(
    "Competition type",
    available_types,
    default=available_types
)

if not selected_types:
    st.warning("Select at least one competition type.")
    st.stop()

filtered_competitions = competitions[
    competitions["competition_type"].isin(selected_types)
].copy()

available_seasons = sorted(
    filtered_competitions["season"].dropna().unique(),
    key=lambda x: str(x)
)

selected_seasons = st.sidebar.multiselect(
    "Competition season",
    available_seasons,
    default=available_seasons
)

if not selected_seasons:
    st.warning("Select at least one season.")
    st.stop()

filtered_competitions = filtered_competitions[
    filtered_competitions["season"].isin(selected_seasons)
].copy()

if filtered_competitions.empty:
    st.warning("No competitions match these filters.")
    st.stop()

st.sidebar.divider()
st.sidebar.header("Pressure Engine")

pressure_threshold = st.sidebar.slider(
    "High-pressure BPI threshold",
    min_value=50,
    max_value=95,
    value=70,
    step=5
)

reference_rr = st.sidebar.slider(
    "First-innings reference run rate",
    min_value=6.0,
    max_value=12.0,
    value=8.5,
    step=0.5
)

prior_strength = st.sidebar.slider(
    "Sample stabilisation strength",
    min_value=0,
    max_value=300,
    value=100,
    step=10
)

st.sidebar.caption(
    "There is deliberately no minimum pressure-ball qualification "
    "filter. Every eligible player remains rankable."
)


# ============================================================
# SELECT MATCHES AND BALL-BY-BALL DATA
# ============================================================

selected_comp_ids = set(
    filtered_competitions["comp_id"].dropna()
)

selected_matches = matches[
    matches["comp_id"].isin(selected_comp_ids)
].copy()

if selected_matches.empty:
    st.warning("No matches found for the selected filters.")
    st.stop()

selected_match_ids = set(
    selected_matches["match_id"].dropna()
)

df = balls[
    balls["match_id"].isin(selected_match_ids)
].copy()

if df.empty:
    st.error("No ball-by-ball records found for these filters.")
    st.stop()

# Always attach competition metadata through the selected match table.
# This avoids dtype mismatches or duplicate comp_id columns when the
# ball-by-ball source already contains its own comp_id field.
if "comp_id" in df.columns:
    df = df.drop(columns=["comp_id"])

df = df.merge(
    selected_matches[["match_id", "comp_id"]].drop_duplicates("match_id"),
    on="match_id",
    how="left"
).merge(
    filtered_competitions[
        ["comp_id", "comp_name", "competition_type", "season"]
    ],
    on="comp_id",
    how="left"
)


# ============================================================
# PLAYER POOL FROM PLAYING XIs
# ============================================================

xi = playing_eleven[
    playing_eleven["match_id"].isin(selected_match_ids)
].copy()

if "is_played" in xi.columns:

    played_flag = boolean(xi["is_played"])

    # If the column contains at least one affirmative value,
    # use it to exclude non-playing XI entries.
    if played_flag.any():
        xi = xi[played_flag]

xi = xi[
    xi["player_id"].notna()
].copy()

if xi.empty:
    st.error(
        "No eligible players found in playing_eleven.csv "
        "for the selected competitions and seasons."
    )
    st.stop()

# One appearance per player per match.
xi = xi.drop_duplicates(
    ["match_id", "player_id"]
)

xi["batting_order_value"] = numeric(
    get_col(xi, ["batting_order"], np.nan),
    default=np.nan
)

wk_col = find_col(
    xi,
    ["is_wicketkeeper", "is_wk", "wicketkeeper"]
)

if wk_col:
    xi["is_wicketkeeper_flag"] = boolean(xi[wk_col])
else:
    xi["is_wicketkeeper_flag"] = False

player_pool = xi.groupby("player_id").agg(
    appearances=("match_id", "nunique"),
    median_batting_order=("batting_order_value", "median"),
    wicketkeeper=("is_wicketkeeper_flag", "max")
).reset_index()


# ============================================================
# PLAYER NAMES FROM COMPETITION MAPPING
# ============================================================

selected_mapping = mapping[
    mapping["comp_id"].isin(selected_comp_ids)
].copy()

name_master = (
    selected_mapping[
        ["player_id", "player_name"]
    ]
    .dropna(subset=["player_id"])
    .drop_duplicates("player_id", keep="last")
)

# Ball-by-ball names provide an additional fallback.
name_fallbacks = []

for id_col, name_col in [
    ("striker_id", "striker_name"),
    ("non_striker_id", "non_striker_name"),
    ("bowler_id", "bowler_name")
]:

    if id_col in df.columns and name_col in df.columns:

        temp = df[[id_col, name_col]].copy()

        temp.columns = [
            "player_id",
            "player_name"
        ]

        name_fallbacks.append(temp)

if name_fallbacks:

    fallback = pd.concat(
        name_fallbacks,
        ignore_index=True
    )

    fallback = fallback.dropna(
        subset=["player_id", "player_name"]
    )

    fallback = fallback.drop_duplicates(
        "player_id",
        keep="first"
    )

    name_master = pd.concat(
        [name_master, fallback],
        ignore_index=True
    ).drop_duplicates(
        "player_id",
        keep="first"
    )

player_pool = player_pool.merge(
    name_master,
    on="player_id",
    how="left"
)

player_pool["player_name"] = (
    player_pool["player_name"]
    .fillna(
        "Player ID " + player_pool["player_id"].astype(str)
    )
)


# ============================================================
# BALL-BY-BALL MATCH STATE
# ============================================================

df["innings"] = numeric(df["innings"]).astype(int)
df["runs"] = numeric(df["runs"])
df["extras1_runs"] = numeric(df["extras1_runs"])
df["extras_2_runs"] = numeric(df["extras_2_runs"])

df["total_runs"] = (
    df["runs"]
    + df["extras1_runs"]
    + df["extras_2_runs"]
)

df["extras_type_clean"] = (
    df["extras_type"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)

df["is_wide"] = df["extras_type_clean"].str.contains(
    r"wide",
    regex=True,
    na=False
)

df["is_noball"] = df["extras_type_clean"].str.contains(
    r"no.?ball|overstep",
    regex=True,
    na=False
)

df["is_legal"] = ~(
    df["is_wide"] | df["is_noball"]
)

# A wide is not a batter ball faced.
# A no-ball counts as a batter ball faced.
df["batter_ball_faced"] = ~df["is_wide"]

df["is_wicket_flag"] = boolean(df["is_wicket"])

# Prefer source boundary flags; fall back to batter runs.
if "is_four" in df.columns:
    df["is_four_flag"] = boolean(df["is_four"])
else:
    df["is_four_flag"] = df["runs"].eq(4)

if "is_six" in df.columns:
    df["is_six_flag"] = boolean(df["is_six"])
else:
    df["is_six_flag"] = df["runs"].eq(6)

df["is_boundary"] = (
    df["is_four_flag"] | df["is_six_flag"]
)

# Bowler-conceded runs:
# batter runs + wide/no-ball extras.
# Byes and leg-byes are not charged to the bowler.
df["bowler_runs"] = (
    df["runs"]
    + np.where(
        df["is_wide"] | df["is_noball"],
        df["extras1_runs"] + df["extras_2_runs"],
        0
    )
)

dismissal_col = find_col(
    df,
    ["dismissal_type", "wicket_type", "kind"]
)

if dismissal_col:
    df["dismissal_clean"] = (
        df[dismissal_col]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )
else:
    df["dismissal_clean"] = ""

non_bowler_dismissals = {
    "run out",
    "retired hurt",
    "retired out",
    "obstructing the field",
    "handled the ball",
    "retired"
}

df["bowler_credit_wicket"] = (
    df["is_wicket_flag"]
    & ~df["dismissal_clean"].isin(non_bowler_dismissals)
)


# ============================================================
# DELIVERY ORDER
# ============================================================

order_columns = [
    "match_id",
    "innings"
]

for col in [
    "innings_ball_num",
    "overid",
    "over_ball_num"
]:
    if col in df.columns:
        df[col] = numeric(df[col], default=0)
        order_columns.append(col)

df = df.sort_values(
    order_columns,
    kind="stable"
).reset_index(drop=True)

group_keys = ["match_id", "innings"]


# ============================================================
# FIRST-INNINGS TOTAL AND SECOND-INNINGS TARGET
# ============================================================

innings_totals = (
    df.groupby(group_keys, as_index=False)
    .agg(innings_total=("total_runs", "sum"))
)

first_innings = innings_totals[
    innings_totals["innings"] == 1
][["match_id", "innings_total"]].rename(
    columns={"innings_total": "first_total"}
)

df = df.merge(
    first_innings,
    on="match_id",
    how="left"
)

df["target"] = np.where(
    df["innings"] == 2,
    df["first_total"] + 1,
    np.nan
)


# ============================================================
# VECTORISED PRESSURE INDEX
# ============================================================

# Match state BEFORE the current delivery.
df["legal_before"] = (
    df.groupby(group_keys)["is_legal"].cumsum().astype(int)
    - df["is_legal"].astype(int)
)

df["runs_before"] = (
    df.groupby(group_keys)["total_runs"].cumsum()
    - df["total_runs"]
)

df["wickets_before"] = (
    df.groupby(group_keys)["is_wicket_flag"].cumsum()
    - df["is_wicket_flag"].astype(int)
)

df["remaining_balls"] = (
    120 - df["legal_before"]
).clip(lower=1)

df["current_rr"] = np.where(
    df["legal_before"] > 0,
    df["runs_before"] * 6 / df["legal_before"],
    reference_rr
)

df["required_rr"] = np.where(
    df["innings"].eq(2) & df["target"].notna(),
    (
        (df["target"] - df["runs_before"]).clip(lower=0)
        / df["remaining_balls"]
    ) * 6,
    np.nan
)

# First innings: stress when scoring below reference rate.
first_innings_stress = (
    (reference_rr - df["current_rr"])
    / reference_rr
) * 100

# Second innings: stress when required rate exceeds reference rate.
second_innings_stress = (
    (df["required_rr"] - reference_rr)
    / reference_rr
) * 100

df["run_stress"] = np.where(
    df["innings"].eq(2),
    second_innings_stress,
    first_innings_stress
)

df["run_stress"] = df["run_stress"].clip(
    lower=0,
    upper=100
).fillna(0)

df["wicket_stress"] = (
    df["wickets_before"] / 8 * 100
).clip(0, 100)

df["phase_stress"] = (
    df["legal_before"] / 120 * 100
).clip(0, 100)

# Recent momentum from the preceding six deliveries.
# The current delivery is excluded.
df["recent_runs"] = (
    df.groupby(group_keys)["total_runs"]
    .transform(
        lambda s: s.shift(1).rolling(
            6, min_periods=1
        ).sum()
    )
)

df["recent_wickets"] = (
    df.groupby(group_keys)["is_wicket_flag"]
    .transform(
        lambda s: s.shift(1).rolling(
            6, min_periods=1
        ).sum()
    )
)

df["recent_runs"] = df["recent_runs"].fillna(8)
df["recent_wickets"] = df["recent_wickets"].fillna(0)

recent_run_pressure = (
    (8 - df["recent_runs"]) / 8 * 100
).clip(0, 100)

recent_wicket_pressure = (
    df["recent_wickets"] * 35
).clip(0, 100)

df["momentum_stress"] = (
    0.65 * recent_run_pressure
    + 0.35 * recent_wicket_pressure
)

df["BPI"] = (
    0.40 * df["run_stress"]
    + 0.25 * df["wicket_stress"]
    + 0.20 * df["phase_stress"]
    + 0.15 * df["momentum_stress"]
)

df["high_pressure"] = (
    df["BPI"] >= pressure_threshold
)


# ============================================================
# BATTING PERFORMANCE
# ============================================================

bat = df[
    df["striker_id"].notna()
].copy()

bat["pressure_ball"] = (
    bat["high_pressure"]
    & bat["batter_ball_faced"]
)

bat["normal_ball"] = (
    ~bat["high_pressure"]
    & bat["batter_ball_faced"]
)

bat["pressure_runs"] = np.where(
    bat["pressure_ball"],
    bat["runs"],
    0
)

bat["normal_runs"] = np.where(
    bat["normal_ball"],
    bat["runs"],
    0
)

bat["pressure_boundary"] = (
    bat["pressure_ball"] & bat["is_boundary"]
)

bat["pressure_dot"] = (
    bat["pressure_ball"]
    & bat["total_runs"].eq(0)
)

batting = bat.groupby("striker_id").agg(
    batting_balls=("batter_ball_faced", "sum"),
    pressure_balls=("pressure_ball", "sum"),
    normal_balls=("normal_ball", "sum"),
    total_batting_runs=("runs", "sum"),
    pressure_runs=("pressure_runs", "sum"),
    normal_runs=("normal_runs", "sum"),
    pressure_boundaries=("pressure_boundary", "sum"),
    pressure_dots=("pressure_dot", "sum")
).reset_index()

batting = batting.rename(
    columns={"striker_id": "player_id"}
)

# Attribute dismissals to the actual dismissed player.
dismissed_col = find_col(
    df,
    ["dismissed_player_id", "player_out_id", "batter_out_id"]
)

if dismissed_col:

    dismissal_data = df[
        df["is_wicket_flag"]
        & df[dismissed_col].notna()
    ].copy()

    dismissal_data["dismissed_id_clean"] = normalize_id(
        dismissal_data[dismissed_col]
    )

else:

    # Fallback only when the source does not supply dismissed_player_id.
    dismissal_data = df[
        df["is_wicket_flag"]
    ].copy()

    dismissal_data["dismissed_id_clean"] = dismissal_data[
        "striker_id"
    ]

dismissal_data["pressure_dismissal"] = (
    dismissal_data["high_pressure"]
)

dismissal_stats = dismissal_data.groupby(
    "dismissed_id_clean"
).agg(
    pressure_dismissals=("pressure_dismissal", "sum"),
    total_dismissals=("is_wicket_flag", "sum")
).reset_index()

dismissal_stats = dismissal_stats.rename(
    columns={"dismissed_id_clean": "player_id"}
)

batting["player_id"] = normalize_id(
    batting["player_id"]
)

batting = batting.merge(
    dismissal_stats,
    on="player_id",
    how="left"
)

batting["pressure_dismissals"] = (
    numeric(batting["pressure_dismissals"])
)

batting["total_dismissals"] = (
    numeric(batting["total_dismissals"])
)

batting["normal_sr"] = (
    safe_divide(
        batting["normal_runs"],
        batting["normal_balls"]
    ) * 100
)

batting["pressure_sr"] = (
    safe_divide(
        batting["pressure_runs"],
        batting["pressure_balls"]
    ) * 100
)

batting["overall_sr"] = (
    safe_divide(
        batting["total_batting_runs"],
        batting["batting_balls"]
    ) * 100
)

# Fallback to overall strike rate if pressure exposure is absent.
batting["pressure_sr"] = batting["pressure_sr"].fillna(
    batting["overall_sr"]
)

batting["sr_delta"] = (
    batting["pressure_sr"]
    - batting["normal_sr"].fillna(batting["pressure_sr"])
)

batting["boundary_rate"] = (
    safe_divide(
        batting["pressure_boundaries"],
        batting["pressure_balls"]
    ) * 100
)

batting["dot_rate"] = (
    safe_divide(
        batting["pressure_dots"],
        batting["pressure_balls"]
    ) * 100
)

batting["dismissal_rate"] = (
    safe_divide(
        batting["total_dismissals"],
        batting["batting_balls"]
    ) * 100
)

batting["dismissal_avoidance"] = (
    100 - batting["dismissal_rate"]
).clip(0, 100)

k = prior_strength

batting["adjusted_pressure_sr"] = (
    batting["pressure_runs"]
    + (batting["normal_sr"].fillna(
        batting["overall_sr"].fillna(0)
    ) / 100) * k
) / (
    batting["pressure_balls"] + k
) * 100

batting["adjusted_pressure_sr"] = (
    batting["adjusted_pressure_sr"].fillna(
        batting["overall_sr"]
    )
)

batting["batting_score"] = (
    0.35 * percentile(
        batting["adjusted_pressure_sr"]
    )
    + 0.25 * percentile(
        batting["sr_delta"]
    )
    + 0.15 * percentile(
        batting["boundary_rate"]
    )
    + 0.15 * percentile(
        batting["dismissal_avoidance"]
    )
    + 0.10 * percentile(
        batting["pressure_balls"]
    )
)


# ============================================================
# BOWLING PERFORMANCE
# ============================================================

bowl = df[
    df["bowler_id"].notna()
].copy()

bowl["pressure_legal"] = (
    bowl["high_pressure"] & bowl["is_legal"]
)

bowl["normal_legal"] = (
    ~bowl["high_pressure"] & bowl["is_legal"]
)

bowl["pressure_runs_conceded"] = np.where(
    bowl["high_pressure"],
    bowl["bowler_runs"],
    0
)

bowl["normal_runs_conceded"] = np.where(
    ~bowl["high_pressure"],
    bowl["bowler_runs"],
    0
)

bowl["pressure_wicket"] = (
    bowl["high_pressure"]
    & bowl["bowler_credit_wicket"]
)

bowl["normal_wicket"] = (
    ~bowl["high_pressure"]
    & bowl["bowler_credit_wicket"]
)

bowl["pressure_dot"] = (
    bowl["high_pressure"]
    & bowl["is_legal"]
    & bowl["total_runs"].eq(0)
)

bowl["pressure_boundary"] = (
    bowl["high_pressure"]
    & bowl["is_boundary"]
)

bowling = bowl.groupby("bowler_id").agg(
    bowling_balls=("is_legal", "sum"),
    pressure_balls=("pressure_legal", "sum"),
    normal_balls=("normal_legal", "sum"),
    total_bowler_runs=("bowler_runs", "sum"),
    pressure_runs_conceded=("pressure_runs_conceded", "sum"),
    normal_runs_conceded=("normal_runs_conceded", "sum"),
    pressure_wickets=("pressure_wicket", "sum"),
    normal_wickets=("normal_wicket", "sum"),
    pressure_dots=("pressure_dot", "sum"),
    pressure_boundaries=("pressure_boundary", "sum")
).reset_index()

bowling = bowling.rename(
    columns={"bowler_id": "player_id"}
)

bowling["player_id"] = normalize_id(
    bowling["player_id"]
)

bowling["overall_economy"] = (
    safe_divide(
        bowling["total_bowler_runs"],
        bowling["bowling_balls"] / 6
    )
)

bowling["normal_economy"] = (
    safe_divide(
        bowling["normal_runs_conceded"],
        bowling["normal_balls"] / 6
    )
)

bowling["pressure_economy"] = (
    safe_divide(
        bowling["pressure_runs_conceded"],
        bowling["pressure_balls"] / 6
    )
)

bowling["pressure_economy"] = (
    bowling["pressure_economy"].fillna(
        bowling["overall_economy"]
    )
)

bowling["economy_improvement"] = (
    bowling["normal_economy"]
    - bowling["pressure_economy"]
)

bowling["wicket_rate"] = (
    safe_divide(
        bowling["pressure_wickets"],
        bowling["pressure_balls"]
    ) * 100
)

bowling["overall_wicket_rate"] = (
    safe_divide(
        bowling["pressure_wickets"] + bowling["normal_wickets"],
        bowling["bowling_balls"]
    ) * 100
)

bowling["wicket_rate"] = (
    bowling["wicket_rate"].fillna(
        bowling["overall_wicket_rate"]
    )
)

bowling["dot_rate"] = (
    safe_divide(
        bowling["pressure_dots"],
        bowling["pressure_balls"]
    ) * 100
)

bowling["boundary_rate"] = (
    safe_divide(
        bowling["pressure_boundaries"],
        bowling["pressure_balls"]
    ) * 100
)

bowling["adjusted_pressure_economy"] = (
    bowling["pressure_runs_conceded"]
    + (
        bowling["normal_economy"].fillna(
            bowling["overall_economy"].fillna(0)
        ) / 6
    ) * k
) / (
    (bowling["pressure_balls"] + k) / 6
)

bowling["adjusted_pressure_economy"] = (
    bowling["adjusted_pressure_economy"].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(bowling["overall_economy"])
)

bowling["dot_rate"] = bowling["dot_rate"].fillna(0)
bowling["boundary_rate"] = bowling["boundary_rate"].fillna(0)

bowling["bowling_score"] = (
    0.30 * percentile(
        bowling["adjusted_pressure_economy"],
        higher_is_better=False
    )
    + 0.25 * percentile(
        bowling["wicket_rate"]
    )
    + 0.20 * percentile(
        bowling["dot_rate"]
    )
    + 0.15 * percentile(
        bowling["boundary_rate"],
        higher_is_better=False
    )
    + 0.10 * percentile(
        bowling["pressure_balls"]
    )
)


# ============================================================
# COMBINE PLAYER PERFORMANCE
# ============================================================

player_pool["player_id"] = normalize_id(
    player_pool["player_id"]
)

player_pool = player_pool.merge(
    batting,
    on="player_id",
    how="left"
)

player_pool = player_pool.merge(
    bowling,
    on="player_id",
    how="left",
    suffixes=("_bat", "_bowl")
)

numeric_columns = [
    "batting_balls",
    "pressure_balls_bat",
    "bowling_balls",
    "pressure_balls_bowl",
    "total_batting_runs",
    "total_bowler_runs",
    "batting_score",
    "bowling_score",
]

for col in numeric_columns:
    if col in player_pool.columns:
        player_pool[col] = numeric(
            player_pool[col]
        )

# Restore the two independent exposure columns.
player_pool["batting_balls"] = numeric(
    player_pool.get("batting_balls", 0)
)

player_pool["bowling_balls"] = numeric(
    player_pool.get("bowling_balls", 0)
)

player_pool["batting_balls_per_appearance"] = (
    player_pool["batting_balls"]
    / player_pool["appearances"].clip(lower=1)
)

player_pool["bowling_balls_per_appearance"] = (
    player_pool["bowling_balls"]
    / player_pool["appearances"].clip(lower=1)
)


# ============================================================
# INFER PRIMARY DISCIPLINE
# ============================================================

# There is no manually supplied role column.
# Use actual batting and bowling participation.

bat_activity = player_pool[
    "batting_balls_per_appearance"
]

bowl_activity = player_pool[
    "bowling_balls_per_appearance"
]

player_pool["primary_role"] = np.where(
    bat_activity >= bowl_activity,
    "Batter",
    "Bowler"
)

# A wicketkeeper is placed in the batting category.
player_pool.loc[
    player_pool["wicketkeeper"],
    "primary_role"
] = "Batter"

# If there is no recorded batting or bowling activity,
# use the player's observed batting-order position.
no_activity = (
    player_pool["batting_balls"].eq(0)
    & player_pool["bowling_balls"].eq(0)
)

player_pool.loc[
    no_activity
    & player_pool["median_batting_order"].le(7),
    "primary_role"
] = "Batter"

player_pool.loc[
    no_activity
    & player_pool["median_batting_order"].gt(7),
    "primary_role"
] = "Bowler"

# No defined role is invented. This is a data-derived classification.
player_pool["role_basis"] = np.select(
    [
        player_pool["wicketkeeper"],
        no_activity,
        bat_activity > bowl_activity,
        bowl_activity > bat_activity
    ],
    [
        "Wicketkeeper classification",
        "Batting order fallback",
        "Higher batting workload",
        "Higher bowling workload"
    ],
    default="Equal workload; batting default"
)


# ============================================================
# UNIQUE TOP 10 SELECTION
# ============================================================

# Ranking does not require a minimum number of pressure balls.
# The complete selected player pool is considered.

bat_candidates = player_pool[
    player_pool["primary_role"].eq("Batter")
    & (
        player_pool["batting_balls"].gt(0)
        | player_pool["median_batting_order"].notna()
        | player_pool["wicketkeeper"]
    )
].copy()

bowl_candidates = player_pool[
    player_pool["primary_role"].eq("Bowler")
    & (
        player_pool["bowling_balls"].gt(0)
        | player_pool["median_batting_order"].notna()
    )
].copy()

# If one role has fewer than ten players, move dual-skilled
# players from the other group where their actual workload
# supports the destination discipline.

def rebalance_roles(pool, bat_group, bowl_group):

    bat_group = bat_group.copy()
    bowl_group = bowl_group.copy()

    # Move bowling-classified players who have batting evidence
    # if the batting pool is too small.
    if len(bat_group) < 10:

        shortage = 10 - len(bat_group)

        movable = bowl_group[
            bowl_group["batting_balls"].gt(0)
        ].copy()

        if not movable.empty:

            movable["destination_strength"] = (
                movable["batting_balls_per_appearance"]
                - movable["bowling_balls_per_appearance"]
            )

            movable = movable.sort_values(
                "destination_strength",
                ascending=False
            )

            moved = movable.head(shortage)

            bat_group = pd.concat(
                [bat_group, moved],
                ignore_index=True
            )

            bowl_group = bowl_group[
                ~bowl_group["player_id"].isin(
                    moved["player_id"]
                )
            ]

    # Move batting-classified players who have bowling evidence
    # if the bowling pool is too small.
    if len(bowl_group) < 10:

        shortage = 10 - len(bowl_group)

        movable = bat_group[
            bat_group["bowling_balls"].gt(0)
            & ~bat_group["wicketkeeper"]
        ].copy()

        if not movable.empty:

            movable["destination_strength"] = (
                movable["bowling_balls_per_appearance"]
                - movable["batting_balls_per_appearance"]
            )

            movable = movable.sort_values(
                "destination_strength",
                ascending=False
            )

            moved = movable.head(shortage)

            bowl_group = pd.concat(
                [bowl_group, moved],
                ignore_index=True
            )

            bat_group = bat_group[
                ~bat_group["player_id"].isin(
                    moved["player_id"]
                )
            ]

    return bat_group, bowl_group


bat_candidates, bowl_candidates = rebalance_roles(
    player_pool,
    bat_candidates,
    bowl_candidates
)

# Sort by model score, then actual performance volume.
bat_candidates = bat_candidates.sort_values(
    [
        "batting_score",
        "batting_balls",
        "appearances"
    ],
    ascending=[False, False, False],
    na_position="last"
)

bowl_candidates = bowl_candidates.sort_values(
    [
        "bowling_score",
        "bowling_balls",
        "appearances"
    ],
    ascending=[False, False, False],
    na_position="last"
)

top_batters = bat_candidates.head(10).copy()

# Explicitly exclude every batter selected above.
selected_batter_ids = set(
    top_batters["player_id"].dropna()
)

bowl_candidates = bowl_candidates[
    ~bowl_candidates["player_id"].isin(
        selected_batter_ids
    )
].copy()

top_bowlers = bowl_candidates.head(10).copy()

top_batters.insert(
    0,
    "Rank",
    range(1, len(top_batters) + 1)
)

top_bowlers.insert(
    0,
    "Rank",
    range(1, len(top_bowlers) + 1)
)


# ============================================================
# SUMMARY METRICS
# ============================================================

st.divider()
st.subheader("📊 Selected Dataset")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Competitions",
    f"{len(filtered_competitions):,}"
)

m2.metric(
    "Matches",
    f"{len(selected_matches):,}"
)

m3.metric(
    "Unique XI Players",
    f"{len(player_pool):,}"
)

m4.metric(
    "Ball-by-ball records",
    f"{len(df):,}"
)

st.caption(
    "Selected types: "
    + ", ".join(selected_types)
    + " | Seasons: "
    + ", ".join(map(str, selected_seasons))
)


# ============================================================
# SELECTION STATUS
# ============================================================

st.divider()
st.subheader("🏆 Final Selection")

bat_count = len(top_batters)
bowl_count = len(top_bowlers)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Selected Batters",
    f"{bat_count}/10"
)

c2.metric(
    "Selected Bowlers",
    f"{bowl_count}/10"
)

c3.metric(
    "Total Unique Selections",
    f"{bat_count + bowl_count}/20"
)

if bat_count == 10 and bowl_count == 10:

    st.success(
        "Both lists contain exactly 10 players, with no duplicate selections."
    )

else:

    st.warning(
        "The selected competition/year combination does not currently "
        "provide enough eligible players in one or both disciplines. "
        "The dashboard will not invent player performance or silently "
        "duplicate a player to fill the list."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏏 Top 10 Batters",
    "🎯 Top 10 Bowlers",
    "📋 Full Player Pool",
    "🔍 Player Analysis",
    "🌍 Overseas Audit",
    "📘 Methodology"
])


# ============================================================
# BATTER LEADERBOARD
# ============================================================

with tab1:

    st.subheader("Top 10 Batters")

    if top_batters.empty:

        st.warning("No batting candidates found.")

    else:

        display = top_batters[[
            "Rank",
            "player_name",
            "batting_score",
            "appearances",
            "batting_balls",
            "pressure_runs",
            "pressure_balls_bat",
            "adjusted_pressure_sr",
            "sr_delta",
            "boundary_rate_bat",
            "dismissal_rate",
            "role_basis"
        ]].copy()

        display.columns = [
            "Rank",
            "Player",
            "Batting Score",
            "Appearances",
            "Balls Faced",
            "Pressure Runs",
            "Pressure Balls",
            "Adjusted Pressure SR",
            "SR Difference",
            "Boundary Rate %",
            "Dismissal Rate %",
            "Role Basis"
        ]

        st.dataframe(
            display.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            top_batters.set_index("player_name")[
                "batting_score"
            ]
        )

        st.download_button(
            "Download Top 10 Batters",
            top_batters.to_csv(index=False).encode("utf-8"),
            "top_10_batters.csv",
            "text/csv"
        )


# ============================================================
# BOWLER LEADERBOARD
# ============================================================

with tab2:

    st.subheader("Top 10 Bowlers")

    if top_bowlers.empty:

        st.warning("No bowling candidates found.")

    else:

        display = top_bowlers[[
            "Rank",
            "player_name",
            "bowling_score",
            "appearances",
            "bowling_balls",
            "pressure_runs_conceded",
            "pressure_balls_bowl",
            "adjusted_pressure_economy",
            "pressure_wickets",
            "dot_rate_bowl",
            "boundary_rate_bowl",
            "role_basis"
        ]].copy()

        display.columns = [
            "Rank",
            "Player",
            "Bowling Score",
            "Appearances",
            "Legal Balls Bowled",
            "Pressure Runs Conceded",
            "Pressure Balls",
            "Adjusted Pressure Economy",
            "Pressure Wickets",
            "Dot Ball Rate %",
            "Boundary Rate %",
            "Role Basis"
        ]

        st.dataframe(
            display.round(2),
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            top_bowlers.set_index("player_name")[
                "bowling_score"
            ]
        )

        st.download_button(
            "Download Top 10 Bowlers",
            top_bowlers.to_csv(index=False).encode("utf-8"),
            "top_10_bowlers.csv",
            "text/csv"
        )


# ============================================================
# FULL PLAYER POOL
# ============================================================

with tab3:

    st.subheader("Complete Eligible Player Pool")

    st.write(
        "All players appearing in the selected playing XIs are shown. "
        "Players are not removed for having low pressure exposure."
    )

    pool_display = player_pool[[
        "player_id",
        "player_name",
        "appearances",
        "median_batting_order",
        "batting_balls",
        "bowling_balls",
        "batting_balls_per_appearance",
        "bowling_balls_per_appearance",
        "primary_role",
        "role_basis"
    ]].copy()

    pool_display = pool_display.sort_values(
        ["primary_role", "player_name"]
    )

    st.dataframe(
        pool_display.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Full Player Pool",
        pool_display.to_csv(index=False).encode("utf-8"),
        "full_player_pool.csv",
        "text/csv"
    )


# ============================================================
# INDIVIDUAL PLAYER ANALYSIS
# ============================================================

with tab4:

    st.subheader("Player Performance Analysis")

    available_players = (
        player_pool[
            ["player_id", "player_name"]
        ]
        .drop_duplicates()
        .sort_values("player_name")
    )

    selected_player_name = st.selectbox(
        "Select a player",
        available_players["player_name"].tolist()
    )

    selected_player = available_players[
        available_players["player_name"] == selected_player_name
    ].iloc[0]

    selected_player_id = selected_player["player_id"]

    p_bat = batting[
        batting["player_id"] == selected_player_id
    ]

    p_bowl = bowling[
        bowling["player_id"] == selected_player_id
    ]

    if not p_bat.empty:

        st.markdown("#### Batting")

        row = p_bat.iloc[0]

        a, b, c, d = st.columns(4)

        a.metric("Runs", int(row["total_batting_runs"]))
        b.metric("Balls Faced", int(row["batting_balls"]))
        c.metric("Pressure Runs", int(row["pressure_runs"]))
        d.metric(
            "Pressure SR",
            format_number(row["pressure_sr"])
        )

    if not p_bowl.empty:

        st.markdown("#### Bowling")

        row = p_bowl.iloc[0]

        a, b, c, d = st.columns(4)

        a.metric("Legal Balls", int(row["bowling_balls"]))
        b.metric(
            "Runs Conceded",
            int(row["total_bowler_runs"])
        )
        c.metric(
            "Pressure Wickets",
            int(row["pressure_wickets"])
        )
        d.metric(
            "Pressure Economy",
            format_number(row["pressure_economy"])
        )

    player_deliveries = df[
        (df["striker_id"] == selected_player_id)
        | (df["bowler_id"] == selected_player_id)
    ].copy()

    if not player_deliveries.empty:

        st.markdown("#### Match-level delivery data")

        st.dataframe(
            player_deliveries[[
                "match_id",
                "innings",
                "overid",
                "over_ball_num",
                "BPI",
                "high_pressure",
                "total_runs",
                "is_wicket_flag"
            ]].head(300),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# OVERSEAS AUDIT
# ============================================================

with tab5:

    st.subheader("Overseas Player Audit")

    # No nationality field is assumed to exist.
    # Country/overseas status is not invented from player names.

    st.info(
        "The six supplied datasets do not provide a verified nationality "
        "field in the established schema. Therefore this dashboard does "
        "not guess overseas status."
    )

    st.write(
        "The IPL maximum-four-overseas rule applies to the playing XI, "
        "not to an unrestricted statistical leaderboard."
    )

    st.write(
        "Before building a final overseas-constrained XI, nationality "
        "must be verified from a reliable player source."
    )


# ============================================================
# METHODOLOGY
# ============================================================

with tab6:

    st.subheader("Pressure Index")

    st.latex(
        r"BPI = 0.40R + 0.25W + 0.20P + 0.15M"
    )

    st.write("""
    **R — Run-rate stress**

    First innings: scoring below the reference run rate.

    Second innings: required run rate exceeding the reference rate.

    **W — Wicket depletion**

    Higher wickets lost increase pressure.

    **P — Innings phase**

    Later innings stages receive a higher phase component.

    **M — Recent momentum**

    Low scoring and recent wickets increase pressure.
    """)

    st.divider()

    st.subheader("Batting Score")

    st.write("""
    | Component | Weight |
    |---|---:|
    | Adjusted pressure strike rate | 35% |
    | Pressure vs normal SR difference | 25% |
    | Boundary rate | 15% |
    | Dismissal avoidance | 15% |
    | Pressure exposure | 10% |
    """)

    st.subheader("Bowling Score")

    st.write("""
    | Component | Weight |
    |---|---:|
    | Adjusted pressure economy | 30% |
    | Pressure wicket rate | 25% |
    | Dot-ball rate | 20% |
    | Boundary prevention | 15% |
    | Pressure exposure | 10% |
    """)

    st.divider()

    st.subheader("Role inference")

    st.write("""
    No predefined player-role column is assumed.

    Primary discipline is inferred from:

    - Batting balls per playing-XI appearance.
    - Bowling legal balls per playing-XI appearance.
    - Batting order where neither activity is recorded.
    - Wicketkeeper indicator, where available.

    All-rounders are assigned to one primary discipline only.
    """)

    st.divider()

    st.subheader("Delivery treatment")

    st.write("""
    - Wides do not count as batter balls faced.
    - No-balls count as batter balls faced.
    - Wides and no-balls are not legal bowling balls.
    - Wides and no-ball extras are charged to the bowler.
    - Byes and leg-byes are not charged to the bowler.
    - Run outs and other non-bowler-credit dismissals do not count
      as bowler wickets.
    - Pressure is evaluated before the current delivery.
    - IPL and T20I are included; other competition types are excluded.
    """)

    st.warning(
        "The BPI and scoring weights are model assumptions. "
        "They should be back-tested against actual match outcomes "
        "before treating the rankings as definitive."
    )


# ============================================================
# FINAL DUPLICATE VALIDATION
# ============================================================

st.divider()
st.subheader("Selection Validation")

bat_ids = set(top_batters["player_id"].dropna())
bowl_ids = set(top_bowlers["player_id"].dropna())

overlap = bat_ids.intersection(bowl_ids)

v1, v2, v3 = st.columns(3)

v1.metric(
    "Batter selections",
    len(bat_ids)
)

v2.metric(
    "Bowler selections",
    len(bowl_ids)
)

v3.metric(
    "Duplicate selections",
    len(overlap)
)

if overlap:
    st.error(
        "Duplicate player IDs detected: "
        + ", ".join(map(str, overlap))
    )

elif len(bat_ids) == 10 and len(bowl_ids) == 10:
    st.success(
        "Validated: 10 unique batters and 10 unique bowlers."
    )

else:
    st.warning(
        "The selected dataset does not contain enough eligible "
        "candidates to complete both lists."
    )

st.caption(
    "SupeRR Selector Season 2 | Data-driven player selection"
)
