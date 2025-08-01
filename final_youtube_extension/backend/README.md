# YouTube RAG Chat Chrome Extension

A Chrome extension that enables users to chat with YouTube video content using advanced RAG (Retrieval-Augmented Generation) technology.

## Features

- **🎯 Automatic Video Detection**: Automatically detects the current YouTube video
- **🔄 Auto-Processing**: Automatically processes video transcripts when you open a video
- **💬 Dark Theme Chat**: Beautiful dark-themed chat interface matching modern design
- **💾 Message History**: Stores conversation history per video
- **🔵 Visual Indicators**: Badge indicators and floating chat button
- **⚡ Real-time Chat**: Instant responses from the RAG backend

## Installation

### 1. Load Extension in Chrome

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" in the top right corner
3. Click "Load unpacked" and select the `youtube-rag-extension` folder
4. The extension icon should appear in your Chrome toolbar

### 2. Start Backend Server

Make sure the FastAPI backend is running:

```bash
cd ../backend
python main.py
```

The backend should be accessible at `http://localhost:8000`

## Usage

1. **Navigate to YouTube**: Go to any YouTube video page
2. **Automatic Processing**: The extension will automatically detect and process the video
3. **Open Chat**: Click the extension icon in the Chrome toolbar
4. **Start Chatting**: Ask questions about the video content in the dark-themed chat interface

## Interface Design

The extension features a modern dark theme that matches the provided screenshot:

- **Dark Background**: Clean dark interface (#1a1a1a)
- **Message Bubbles**: Rounded chat bubbles with proper alignment
- **Blue Accent**: Blue color scheme (#4a9eff) for user messages and buttons
- **Status Indicators**: Real-time status updates at the bottom
- **Responsive Design**: Adapts to different content lengths

## File Structure

```
youtube-rag-extension/
├── manifest.json              # Extension manifest (Manifest V3)
├── popup/
│   ├── popup.html            # Main chat interface
│   └── popup.js              # Chat functionality and API communication
├── content/
│   ├── content.js            # YouTube page integration
│   └── content.css           # Floating chat button styles
├── background/
│   └── background.js         # Background service worker
└── icons/
    ├── icon16.png           # 16x16 icon
    ├── icon48.png           # 48x48 icon
    └── icon128.png          # 128x128 icon
```

## Features in Detail

### Automatic Video Detection
- Detects video ID from YouTube URLs
- Monitors for video changes in single-page application
- Updates badge indicator when on YouTube videos

### Auto-Processing
- Automatically processes videos when detected
- Shows processing status in real-time
- Handles errors gracefully with user feedback

### Chat Interface
- Modern dark theme design
- Message history persistence
- Auto-scrolling to latest messages
- Loading indicators during processing
- Error handling with user-friendly messages

### Backend Integration
- RESTful API communication with FastAPI backend
- Automatic retry logic for failed requests
- Comprehensive error handling
- Status monitoring and feedback

## Permissions

The extension requires these permissions:

- `activeTab`: Access current YouTube tab for video detection
- `storage`: Store conversation history and settings
- `https://www.youtube.com/*`: Access YouTube pages
- `http://localhost:8000/*`: Communicate with local backend

## Development

### Testing the Extension

1. Make changes to extension files
2. Go to `chrome://extensions/`
3. Click the refresh icon on your extension
4. Test the changes on a YouTube video

### Debugging

- Open Chrome DevTools on the popup: Right-click extension icon → "Inspect popup"
- View background script logs: Go to `chrome://extensions/` → Click "background page" link
- Check content script logs: Open DevTools on YouTube page

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Extension not working | Check if developer mode is enabled in Chrome |
| Backend connection failed | Ensure FastAPI server is running on port 8000 |
| Video not detected | Refresh the YouTube page and try again |
| Chat not responding | Check browser console for errors |
| Processing failed | Verify the video has English captions available |

## Browser Compatibility

- **Chrome**: Fully supported (Manifest V3)
- **Edge**: Should work with Chromium-based Edge
- **Firefox**: Not supported (uses different extension format)

## Future Enhancements

- [ ] Support for multiple languages
- [ ] Export conversation history
- [ ] Custom themes and appearance settings
- [ ] Integration with other video platforms
- [ ] Offline mode with cached responses
- [ ] User authentication and cloud sync

