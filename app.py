import os
import uuid
import sqlite3
import torch
import streamlit as st
from PIL import Image, ExifTags
import pandas as pd
from datetime import datetime
import requests
from transformers.pipelines import pipeline
from transformers import AutoImageProcessor

# ── SETUP ────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("submissions.db", check_same_thread=False)
c = conn.cursor()

# -- Schema (with migrations) ------------------
c.execute("""
CREATE TABLE IF NOT EXISTS submissions (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp  TEXT,
  species    TEXT,
  confidence REAL,
  latitude   REAL,
  longitude  REAL
)
""")
c.execute("PRAGMA table_info(submissions)")
cols = {r[1] for r in c.fetchall()}
for col, ddl in [
    ("image_path",     "ADD COLUMN image_path TEXT"),
    ("validated",      "ADD COLUMN validated INTEGER DEFAULT 0"),
    ("review_comment", "ADD COLUMN review_comment TEXT DEFAULT ''"),
    ("user_id",        "ADD COLUMN user_id TEXT DEFAULT ''"),
]:
    if col not in cols:
        c.execute(f"ALTER TABLE submissions {ddl}")
conn.commit()

# ── SIDEBAR CONTROLS ─────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

# 1) User identity
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
uid = st.sidebar.text_input("👤 Your name or email", value=st.session_state.user_id).strip()
st.session_state.user_id = uid
if not uid:
    st.sidebar.error("Please enter your name/email")
    st.stop()

# 2) Device
device = 0 if torch.cuda.is_available() else -1
st.sidebar.write("🚀 GPU" if device == 0 else "🔧 CPU")

# 3) Model choice
MODEL_OPTIONS = {
    "General Vision Transformer": "google/vit-base-patch16-224",
    "Plant-specific ViT":          "marwaALzaabi/plant-identification-vit"
}
model_label = st.sidebar.selectbox("🖼 Model version", list(MODEL_OPTIONS.keys()))
MODEL = MODEL_OPTIONS[model_label]

@st.cache_resource
def load_classifier(model_name: str, device: int):
    proc = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    return pipeline(
        "image-classification",
        model=model_name,
        image_processor=proc,
        device=device
    )

classifier = load_classifier(MODEL, device)

# ── HELPERS ──────────────────────────────────────────────────────────────────
def get_exif_gps(img: Image.Image):
    try:
        exif = img._getexif()
        if not exif: return None, None
        gps = {}
        for tag, val in exif.items():
            name = ExifTags.TAGS.get(tag, tag)
            if name == "GPSInfo":
                for t, v in val.items():
                    gps[ExifTags.GPSTAGS.get(t, t)] = v
        need = {"GPSLatitude","GPSLatitudeRef","GPSLongitude","GPSLongitudeRef"}
        if not need.issubset(gps): return None, None
        def to_deg(vals):
            try:
                d, m, s = vals
                return d[0]/d[1] + m[0]/m[1]/60 + s[0]/s[1]/3600
            except:
                return None
        lat, lon = to_deg(gps["GPSLatitude"]), to_deg(gps["GPSLongitude"])
        if lat is None or lon is None: return None, None
        if gps["GPSLatitudeRef"] != "N": lat = -lat
        if gps["GPSLongitudeRef"] != "E": lon = -lon
        return lat, lon
    except:
        return None, None

def get_id_suggestions(file) -> tuple[str, float]:
    img = Image.open(file).convert("RGB")
    try:
        preds = classifier(img, top_k=1)
        return preds[0]["label"].replace("_"," "), float(preds[0]["score"])
    except Exception as e:
        st.warning(f"Model error: {e}")
        return "Unknown", 0.0

# ── APP UI ────────────────────────────────────────────────────────────────────
st.title("🌱 OIP Species Identification")

mode = st.sidebar.radio("Mode", ["Citizen", "Expert", "Reports"])

if mode == "Citizen":
    st.header("Citizen: Upload & Submit")
    upload = st.file_uploader("📸 Upload photo", type=["jpg","jpeg","png"])
    if upload:
        img = Image.open(upload)
        st.image(img, use_container_width=True)
        lat, lon = get_exif_gps(img)
        if lat is None:
            with st.expander("No GPS data—enter manually"):
                lat = st.number_input("Latitude", format="%.6f")
                lon = st.number_input("Longitude", format="%.6f")
        else:
            st.write(f"**Location:** {lat:.6f}, {lon:.6f}")
        with st.spinner("Identifying…"):
            species, conf = get_id_suggestions(upload)
        st.markdown(f"**Prediction:** `{species}` ({conf*100:.1f}% confidence)")
        if st.button("✅ Confirm & Log"):
            ts = datetime.now().isoformat()
            path = f"data/{uuid.uuid4().hex}.jpg"
            img.save(path)
            c.execute(
                "INSERT INTO submissions "
                "(timestamp,species,confidence,latitude,longitude,image_path,user_id) "
                "VALUES (?,?,?,?,?,?,?)",
                (ts, species, conf, lat, lon, path, uid)
            )
            conn.commit()
            st.success(f"Sighting logged by **{uid}**!")

elif mode == "Expert":
    st.header("Expert: Validate Submissions")
    df = pd.read_sql("SELECT * FROM submissions WHERE validated=0 ORDER BY timestamp", conn)
    valid = df[df["image_path"].apply(lambda p: isinstance(p,str) and os.path.exists(p))]
    if valid.empty:
        st.info("No pending submissions.")
    else:
        for row in valid.itertuples():
            st.image(Image.open(row.image_path), width=200)
            st.write(f"#{row.id} • {row.timestamp} • by **{row.user_id}**")
            new_sp = st.text_input("Species:", value=row.species, key=f"sp{row.id}")
            note   = st.text_area("Notes:", key=f"nt{row.id}")
            a,b = st.columns(2)
            if a.button("Approve", key=f"ap{row.id}"):
                c.execute("UPDATE submissions SET species=?,validated=1,review_comment=? WHERE id=?",
                          (new_sp,note,row.id))
                conn.commit(); st.success("Approved!")
            if b.button("Reject", key=f"rj{row.id}"):
                c.execute("UPDATE submissions SET validated=2,review_comment=? WHERE id=?",
                          (note,row.id))
                conn.commit(); st.error("Rejected!")
            st.markdown("---")

else:  # Reports
    st.header("Reports: Download, Visualize, Map")
    reports = pd.read_sql("SELECT * FROM submissions WHERE validated=1 ORDER BY id", conn)
    if reports.empty:
        st.info("No validated sightings yet.")
    else:
        st.subheader("Validated Sightings Table")
        st.dataframe(reports, use_container_width=True)

        with st.expander("Edit / Delete"):
            sel = st.selectbox("Record ID", reports.id.tolist())
            if st.button("Delete this record"):
                c.execute("DELETE FROM submissions WHERE id=?", (sel,))
                c.execute("DELETE FROM sqlite_sequence WHERE name='submissions'")
                conn.commit(); st.success(f"Deleted {sel}. Refresh.")
            if st.button("Clear all"):
                c.execute("DELETE FROM submissions")
                c.execute("DELETE FROM sqlite_sequence WHERE name='submissions'")
                conn.commit(); st.success("Cleared all. Refresh.")

        csv = reports.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "sightings.csv", "text/csv")

        st.subheader("Species Frequency")
        st.bar_chart(reports.species.value_counts())

        st.subheader("Sightings Over Time")
        times = pd.to_datetime(reports.timestamp).dt.floor("D").value_counts().sort_index()
        st.line_chart(times)

        st.subheader("Sightings Map")
        coords = reports[["latitude","longitude"]].dropna()
        st.map(coords)
