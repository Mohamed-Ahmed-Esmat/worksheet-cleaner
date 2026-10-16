# 📝 Worksheet Cleaner

A Streamlit web application that removes handwritten pen answers from photographed school worksheets, producing clean, printable PDFs.

## How It Works
The app has three cleaning engines (pick one in the sidebar):

1. **Detect & Erase (default, free on the Gemini API free tier):** a cheap vision model
   (`gemini-3.5-flash-lite`, called on the Gemini API directly, or via OpenRouter for
   ~$0.002/page) only *finds* the handwriting and
   returns bounding boxes. The erasing happens locally with OpenCV
   (`handwriting_eraser.py`), so the printed questions are never redrawn and cannot be
   changed. Each box is labelled:
   - **answer** (writing in empty space): all ink inside is erased, any pen color or pencil,
     except long printed lines such as table borders.
   - **mark** (circle/underline over printed text): only pen-colored pixels are erased, so the
     printed word underneath stays.
   An optional "scanned look" evens out the lighting so the paper prints white.
2. **AI Regeneration (~$0.13/page with Nano Banana Pro):** an image model redraws the whole
   page without the answers. It can look cleaner on heavily marked pages, but it may alter
   printed text (articles, umlauts, `ß`), so check the output.
3. **Local OpenCV Filter (free, offline):** removes blue or colored ink by color thresholds
   and inpaints it. Works best on high-resolution photos with clearly colored pen.

**Photo quality matters for every engine:** send the original camera photo, not a
WhatsApp-forwarded copy. WhatsApp shrinks pages to ~600px wide and compresses color heavily,
which makes pen and print harder to tell apart.

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

### API Keys
- **Detect & Erase** uses a [Gemini API key](https://aistudio.google.com/apikey) by default.
  It is free within the free-tier rate limits (a 429 error means you hit them; wait and retry).
  On the free tier Google may use requests to improve its products. Use a paid key, or the
  OpenRouter provider, if that matters for your worksheets. Busy-model 503s are retried
  automatically.
- **AI Regeneration**, and Detect & Erase with the OpenRouter provider, use an
  [OpenRouter key](https://openrouter.ai/keys). OpenRouter also works where the Gemini API is
  geo-blocked.

Type the key in the sidebar, or configure it for the app in `.streamlit/secrets.toml` (git-ignored)
or the environment:
```toml
GEMINI_API_KEY = "AIza..."
OPENROUTER_API_KEY = "sk-or-v1-..."
```
A configured key is used server-side and is never sent to the browser, but anyone who can open the
app can then spend it.

### Running the App
Start the Streamlit server:
```bash
streamlit run app.py
```
The app will open in your default web browser at `http://localhost:8501`.

### Running the Tests
```bash
pip install -r requirements-dev.txt
pytest
```
The tests mock OpenRouter, so they need no API key and make no network calls.

## Deployment to Streamlit Community Cloud

Deploying this app is free and straightforward using Streamlit Community Cloud:

1. Push your code to a public or private GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your repository, branch (e.g., `main`), and the main file path (`app.py`).
5. Click **"Deploy"**.

**Important for Deployment:**
- Streamlit Cloud runs on Linux. `packages.txt` lists the one system library OpenCV needs (`libglib2.0-0`); Streamlit Cloud installs it automatically.
- Add `GEMINI_API_KEY` (and/or `OPENROUTER_API_KEY`) under the app's **Secrets** if visitors should not need their own key.

## Screenshots
*(Add screenshots of your application here)*

## License
MIT License
