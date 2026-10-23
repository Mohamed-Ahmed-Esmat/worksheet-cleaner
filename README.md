# 📝 Worksheet Cleaner

A Streamlit web application that removes handwritten pen answers from photographed school worksheets, producing clean, printable PDFs.

## How It Works
The app has four cleaning engines (pick one in the sidebar):

| Engine | Cost per page | Result |
|---|---|---|
| **Re-typeset** (default) | free on the Gemini API free tier (~$0.005 paid) | crisp new page, original layout approximated |
| Detect & Erase | free on the free tier (~$0.002 via OpenRouter) | the photo, with handwriting erased |
| AI Regeneration | ~$0.03 (Nano Banana 2.1), ~$0.13 (Pro) | the photo, redrawn by an image model |
| Local OpenCV Filter | free, offline | the photo, colored ink filtered out |

1. **Re-typeset (default):** a free vision model reads only the *printed* text of the photo and
   returns it as simple HTML (`retypeset.py`), which is rendered as a clean A4 PDF. Handwriting,
   pen circles, shadows and watermarks never make it into the output, and no printed letter is
   redrawn as pixels.
   - Models are tried in order: `gemini-robotics-er-2-preview` (most faithful, ~20 free pages a
     day), then `gemini-3-flash-preview`, then `gemma-4-26b-a4b-it` (free, slower). A model that
     runs out of quota or is overloaded is skipped for the rest of the session.
   - **Double-check (on by default):** a second, different model reads the page too, and every
     line the two readings disagree on is listed for you to check. A reader can silently misread
     a printed word hidden under a student's circle, so check flagged lines before printing.
   - The model's HTML is sanitized (simple formatting tags only, no links, scripts or remote
     resources) before rendering.
   - Not kept: the original fonts, exact positions and watermark. Pictures are shown as
     "[picture]" for now; use Detect & Erase for pages that need them.
2. **Detect & Erase (free on the Gemini API free tier):** a cheap vision model
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
3. **AI Regeneration (~$0.03/page with the default Nano Banana 2.1, ~$0.13 with Nano Banana Pro):**
   an image model redraws the whole
   page without the answers. It can look cleaner on heavily marked pages, but it may alter
   printed text (articles, umlauts, `ß`), so check the output.
4. **Local OpenCV Filter (free, offline):** removes blue or colored ink by color thresholds
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
- **Re-typeset** and **Detect & Erase** use a [Gemini API key](https://aistudio.google.com/apikey) by default.
  It is free within the free-tier rate limits (a 429 error means you hit them; wait and retry).
  On the free tier Google may use requests to improve its products. Use a paid key, or the
  OpenRouter provider for Detect & Erase, if that matters for students' worksheets. Busy-model 503s are retried
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
