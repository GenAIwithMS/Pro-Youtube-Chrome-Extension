# YouTube RAG Chat Premium - Installation Guide

## Quick Start (5 minutes)

### Step 1: Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install fastapi uvicorn pydantic python-dotenv
   ```

3. **Set up your API key:**
   Create a `.env` file in the backend directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   > **Get your free Groq API key at:** https://console.groq.com/keys

4. **Start the server:**
   ```bash
   python test_server.py
   ```
   
   You should see: `Server will be available at: http://localhost:8000`

### Step 2: Chrome Extension Setup

1. **Open Chrome and go to:**
   ```
   chrome://extensions/
   ```

2. **Enable "Developer mode"** (toggle in the top-right corner)

3. **Click "Load unpacked"** and select the `frontend` folder

4. **The extension icon should appear** in your Chrome toolbar

### Step 3: Test the Extension

1. **Go to any YouTube video** (e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ)

2. **Click the extension icon** in your toolbar

3. **The extension will automatically process the video**

4. **Start asking questions** about the video content!

---

## Full Installation (Production Ready)

### Prerequisites

- **Python 3.8+** (Check with `python --version`)
- **Node.js 16+** (Optional, for development)
- **Chrome Browser**
- **Groq API Key** (Free at https://console.groq.com)

### Backend Setup (Full Version)

1. **Clone/Download the project:**
   ```bash
   # If you have the zip file, extract it
   unzip youtube_extension_premium.zip
   cd youtube_extension_premium/backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env file with your API key
   ```

5. **Start the production server:**
   ```bash
   python main.py
   ```

### Chrome Extension Setup (Detailed)

1. **Open Chrome Extensions page:**
   - Type `chrome://extensions/` in the address bar
   - Or go to Chrome Menu → More Tools → Extensions

2. **Enable Developer Mode:**
   - Toggle the "Developer mode" switch in the top-right corner

3. **Load the Extension:**
   - Click "Load unpacked"
   - Navigate to and select the `frontend` folder
   - The extension should appear in your extensions list

4. **Pin the Extension (Optional):**
   - Click the puzzle piece icon in Chrome toolbar
   - Click the pin icon next to "YouTube RAG Chat Premium"

### Verification

1. **Check Backend Health:**
   ```bash
   curl http://localhost:8000/health
   ```
   Should return: `{"status":"healthy",...}`

2. **Test Extension:**
   - Go to any YouTube video
   - Click the extension icon
   - You should see the extension interface

---

## Troubleshooting

### Backend Issues

**"ModuleNotFoundError" when starting server:**
```bash
# Make sure you're in the backend directory
cd backend

# Install missing packages
pip install fastapi uvicorn pydantic python-dotenv

# For full version, install all requirements
pip install -r requirements.txt
```

**"Port 8000 already in use":**
```bash
# Find and kill the process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
python test_server.py --port 8001
```

**"API key not found" errors:**
```bash
# Check your .env file exists and has the correct format
cat .env

# Should contain:
# GROQ_API_KEY=your_actual_api_key_here
```

### Extension Issues

**Extension not loading:**
1. Make sure Developer Mode is enabled
2. Check for errors in Chrome DevTools (F12)
3. Try reloading the extension from chrome://extensions/

**"Backend not available" error:**
1. Ensure the backend server is running on http://localhost:8000
2. Check the browser console for CORS errors
3. Try refreshing the YouTube page

**Extension icon not visible:**
1. Check if the extension is enabled in chrome://extensions/
2. Pin the extension to the toolbar
3. Look for the extension in the Chrome menu (puzzle piece icon)

### Common Issues

**YouTube video not processing:**
- Some videos don't have transcripts available
- Try with a different video that has captions/subtitles
- Check the extension popup for error messages

**Premium features not working:**
- Verify your subscription status in the extension popup
- Check that you've started a trial or upgraded your plan
- Ensure the backend is processing premium features correctly

---

## Development Setup

### Frontend Development

1. **Install Node.js dependencies (optional):**
   ```bash
   cd frontend
   npm install
   ```

2. **Make changes to the extension:**
   - Edit files in the `frontend` directory
   - Reload the extension from chrome://extensions/

### Backend Development

1. **Install development dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8  # Additional dev tools
   ```

2. **Run in development mode:**
   ```bash
   # Enable debug logging
   export LOG_LEVEL=DEBUG
   python main.py
   ```

3. **API Documentation:**
   - Visit http://localhost:8000/docs for interactive API docs
   - Visit http://localhost:8000/redoc for alternative docs

---

## Configuration Options

### Backend Configuration (.env file)

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
HOST=0.0.0.0                    # Server host
PORT=8000                       # Server port
LOG_LEVEL=INFO                  # Logging level (DEBUG, INFO, WARNING, ERROR)
ALLOWED_ORIGINS=*               # CORS allowed origins

# Feature Flags
ENABLE_PREMIUM_FEATURES=true    # Enable premium features
ENABLE_VOICE_FEATURES=true      # Enable voice assistant
ENABLE_ANALYTICS=true           # Enable usage analytics
```

### Extension Configuration

The extension automatically detects the backend URL. If you're running the backend on a different port, update the `API_BASE_URL` in `frontend/popup.js`:

```javascript
const API_BASE_URL = 'http://localhost:8001';  // Change port if needed
```

---

## Getting Help

### Documentation
- **README.md** - Overview and features
- **API Documentation** - http://localhost:8000/docs (when server is running)

### Support
1. Check this installation guide first
2. Review the troubleshooting section
3. Check the browser console for error messages
4. Ensure all prerequisites are installed

### Reporting Issues
When reporting issues, please include:
- Operating system and version
- Python version (`python --version`)
- Chrome version
- Error messages from console/terminal
- Steps to reproduce the issue

---

## Next Steps

Once you have the extension working:

1. **Get a Groq API Key** for full AI functionality
2. **Try the premium features** with a 7-day free trial
3. **Explore different YouTube videos** to test various content types
4. **Check out the knowledge graphs** and timeline features
5. **Customize the extension** for your learning needs

**Enjoy your AI-powered YouTube experience!** 🚀

