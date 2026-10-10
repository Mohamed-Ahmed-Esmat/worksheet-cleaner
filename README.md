# 📝 Worksheet Cleaner

A Streamlit web application that removes handwritten pen answers from photographed school worksheets, producing clean, printable PDFs.

## How It Works
The app uses OpenCV to process the images:
1. **Blue Pen Removal**: It filters the image in the HSV color space to isolate blue ink, then uses OpenCV's inpainting algorithm to fill the masked regions with the surrounding paper texture.
2. **Auto-Detect**: It isolates colored handwriting and lighter pen marks, distinguishing them from the dark, printed text, and removes them.
3. **Enhancement**: Optionally applies contrast enhancement and slight sharpening to make the photo look more like a scanned document.

## Local Setup

### Prerequisites
- Python 3.8+
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd worksheet-cleaner
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The app uses `opencv-python-headless` to avoid GUI dependencies on servers.)*

### Running the App
Start the Streamlit server:
```bash
streamlit run app.py
```
The app will open in your default web browser at `http://localhost:8501`.

## Deployment to Streamlit Community Cloud

Deploying this app is free and straightforward using Streamlit Community Cloud:

1. Push your code to a public or private GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your repository, branch (e.g., `main`), and the main file path (`app.py`).
5. Click **"Deploy"**.

**Important for Deployment:**
- Streamlit Cloud runs on Linux and requires certain system libraries for OpenCV. This repository includes a `packages.txt` file containing `libgl1-mesa-glx` and `libglib2.0-0`. Streamlit Cloud will automatically detect this file and install the necessary system dependencies.

## Screenshots
*(Add screenshots of your application here)*

## License
MIT License
