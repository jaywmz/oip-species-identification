import os
import uuid
import sqlite3
import torch
import streamlit as st
from PIL import Image, ExifTags
import pandas as pd
from datetime import datetime
from transformers import pipeline, AutoImageProcessor

# ── SETUP ────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("submissions.db", check_same_thread=False)
c = conn.cursor()

# Base table
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
# Schema migration
c.execute("PRAGMA table_info(submissions)")
cols = {r[1] for r in c.fetchall()}
if 'image_path' not in cols:      c.execute("ALTER TABLE submissions ADD COLUMN image_path TEXT")
if 'validated' not in cols:       c.execute("ALTER TABLE submissions ADD COLUMN validated INTEGER DEFAULT 0")
if 'review_comment' not in cols:  c.execute("ALTER TABLE submissions ADD COLUMN review_comment TEXT DEFAULT ''")
conn.commit()

# ── MODEL SETUP ───────────────────────────────────────────────────────────────
MODEL = "google/vit-base-patch16-224"
device = 0 if torch.cuda.is_available() else -1
st.sidebar.write("🚀 GPU" if device == 0 else "🔧 CPU")

processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=True)
classifier = pipeline(
    "image-classification",
    model=MODEL,
    image_processor=processor,
    device=device
)

# ── HELPERS ──────────────────────────────────────────────────────────────────
def get_exif_gps(img: Image.Image):
    try:
        exif = img._getexif()
        if not exif:
            return None, None
        gps = {}
        for tag, val in exif.items():
            name = ExifTags.TAGS.get(tag, tag)
            if name == 'GPSInfo':
                for t, v in val.items():
                    gps[ExifTags.GPSTAGS.get(t, t)] = v
        need = {'GPSLatitude','GPSLatitudeRef','GPSLongitude','GPSLongitudeRef'}
        if not need.issubset(gps):
            return None, None
        def to_deg(vals):
            d = vals[0][0]/vals[0][1]
            m = vals[1][0]/vals[1][1]
            s = vals[2][0]/vals[2][1]
            return d + m/60 + s/3600
        lat = to_deg(gps['GPSLatitude'])
        lon = to_deg(gps['GPSLongitude'])
        if gps['GPSLatitudeRef'] != 'N': lat = -lat
        if gps['GPSLongitudeRef'] != 'E': lon = -lon
        return lat, lon
    except Exception:
        return None, None

def get_id_suggestions(file):
    """
    Try the HF model; if it errors (meta-tensor), fall back to a stub.
    """
    img = Image.open(file).convert("RGB")
    try:
        preds = classifier(img, top_k=1)
        return preds[0]['label'].replace('_',' '), float(preds[0]['score'])
    except NotImplementedError:
        st.warning("⚠️ Model inference unavailable; using stub prediction.")
        return "Testus plantus", 0.42
    except Exception as e:
        st.error(f"⚠️ Identification error: {e}")
        return "Unknown", 0.0

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌱 OIP Species Identification")

mode = st.sidebar.radio("Mode", ["Citizen", "Expert", "Reports"])

# — Citizen —
if mode == "Citizen":
    st.markdown("## Citizen: Upload & Submit")
    st.markdown(
        "1. Snap or upload a plant/bug photo.\n"
        "2. Check or enter location.\n"
        "3. Confirm & Log your sighting."
    )
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
                "(timestamp, species, confidence, latitude, longitude, image_path) "
                "VALUES (?,?,?,?,?,?)",
                (ts, species, conf, lat, lon, path)
            )
            conn.commit()
            st.success("📥 Sighting recorded!")

# — Expert —
elif mode == "Expert":
    st.markdown("## Expert: Validate Submissions")
    df = pd.read_sql("SELECT * FROM submissions WHERE validated=0 ORDER BY timestamp", conn)
    valid = df[df['image_path'].apply(lambda p: isinstance(p, str) and os.path.exists(p))]
    if valid.empty:
        st.info("✅ No pending submissions.")
    else:
        for row in valid.itertuples():
            with st.container():
                st.image(Image.open(row.image_path), width=200)
                st.write(f"**#{row.id}** • {row.timestamp}")
                new_sp = st.text_input("Species:", value=row.species, key=f"sp{row.id}")
                note  = st.text_area("Notes:", key=f"nt{row.id}")
                col1, col2 = st.columns(2)
                if col1.button("Approve", key=f"ap{row.id}"):
                    c.execute(
                        "UPDATE submissions SET species=?,validated=1,review_comment=? WHERE id=?",
                        (new_sp, note, row.id)
                    )
                    conn.commit()
                    st.success("Approved!")
                if col2.button("Reject", key=f"rj{row.id}"):
                    c.execute(
                        "UPDATE submissions SET validated=2,review_comment=? WHERE id=?",
                        (note, row.id)
                    )
                    conn.commit()
                    st.error("Rejected!")
                st.markdown("---")

# — Reports + Map —
else:
    st.markdown("## Reports: Download, Visualize & Map")
    reports = pd.read_sql("SELECT * FROM submissions WHERE validated=1", conn)

    if reports.empty:
        st.info("ℹ️ No validated sightings yet.")
    else:
        # CSV export
        csv = reports.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download CSV", csv, "sightings.csv", "text/csv")

        # Species freq
        st.subheader("Species Frequency")
        st.bar_chart(reports['species'].value_counts())

        # Sightings over time
        st.subheader("Sightings Over Time")
        times = (
            pd.to_datetime(reports['timestamp'])
              .dt.floor('D')
              .value_counts()
              .sort_index()
        )
        st.line_chart(times)

        # Interactive map
        st.subheader("Sightings Map")
        map_data = reports[['latitude', 'longitude']].dropna()
        st.map(map_data)
