# AI Vision Dashboard

- AI Bag Counter (Loading / Unloading detection- Camera + Video Upload)
- Face Authentication (Register + Live Verification)
- Wall Area Estimation (Segmentation-based area calculation- Camera + Video Upload)

---

## Features

### AI Bag Counter
- Detects backpacks, handbags, and suitcases
- Tracks object movement direction
- Counts:
  - Loaded (Left → Right)
  - Unloaded (Right → Left)
- Supports:
  - Live Camera
  - Uploaded Video

---

### Face Authentication
- Register face using uploaded image
- Live camera authentication
- Matches registered face
- Rejects unknown faces

---

### Wall Area Estimation
- Uses segmentation model
- Calculates detected area percentage
- Supports:
  - Live Camera
  - Uploaded Video

---

## How to Run

### Clone the repository

```bash
git clone https://github.com/saadhanag13/AI_Vision_Dashboard.git
```

### Create Virtual Environment (First Time Only)

```bash
python -m venv venv
```

### Activate Virtual Environment

```bash
venv\Scripts\activate #windows
source venv/bin/activate # Mac/Linux
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the UI App

```bash
streamlit run app.py
```