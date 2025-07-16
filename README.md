# OIP Species Identification

A Streamlit proof-of-concept app for citizen scientists and experts to upload photos of plants & wildlife, get automated ID suggestions, and validate sightings for Glasgow Botanic Gardens' biodiversity records.

## 🚀 Getting Started

### 1. Clone the repo  
```bash
git clone git@github.com:YourOrg/oip-species-identification.git
cd oip-species-identification
```

### 2. Set up environment
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

## 📁 Project Structure
```
.
├─ app.py            # Streamlit entrypoint
├─ requirements.txt  # Python dependencies
├─ README.md         # This file
├─ data/             # (ignored) sample images, CSVs, etc.
├─ models/           # (ignored) saved ML models
└─ .gitignore
```

## 🔧 Next Steps

### Spike the ML/API
- Hook up the iNaturalist API
- Or load a TensorFlow/PyTorch model for on-device inference

### Prototype expert review
- Add a "Validate" button that logs confirmed IDs to a tiny database

### UI polish & deployment
- Tweak the Streamlit layout or port to React/React Native
- Deploy on Streamlit Cloud, Heroku or your own server