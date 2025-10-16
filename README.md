# YouTube RAG Chat Premium Extension

A powerful Chrome extension that enables AI-powered conversations with YouTube videos using RAG (Retrieval-Augmented Generation) technology, featuring premium subscription tiers with advanced AI capabilities.

## 🚀 Features

### Free Tier
- **Basic Q&A**: Ask questions about any YouTube video (50 chats/day)
- **Transcript Access**: View and search video transcripts
- **Smart Suggestions**: AI-generated follow-up questions

### Pro Tier ($5/month)
- **Unlimited Chat**: No daily limits on conversations
- **🧠 Brain Mode**: Hyperpersonalized AI responses that learn from your interactions
- **⏱️ Timeline-Aware**: Ask questions about specific timestamps in videos
- **👁️ Scene Understanding**: AI analysis of visual content using GPT-4 Vision
- **🎤 Voice Assistant**: Voice input and audio output capabilities
- **📺 Video Overlays**: AR-like annotations and explanations on videos

### Power+ Tier ($15/month)
- **All Pro Features** plus:
- **🕸️ Knowledge Graphs**: Visual concept mapping and relationship visualization
- **🔗 Cross-Video RAG**: Ask questions across multiple videos and playlists
- **⚡ Real-Time Explain**: Auto-pause for complex content with instant explanations
- **📊 Advanced Analytics**: Deep learning insights and progress tracking

### Creator Tier ($25/month)
- **All Power+ Features** plus:
- **❓ Question Generator**: AI-generated engagement questions for your audience
- **💬 Comment Analyzer**: Sentiment analysis and audience feedback insights
- **🎯 SEO Optimizer**: Optimized titles, descriptions, and tags
- **👥 Audience Insights**: Detailed viewer behavior and engagement analytics

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- Chrome Browser
- Groq API Key (for AI model access)

### Backend Setup

1. **Clone and navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the backend directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Start the backend server:**
   ```bash
   python main.py
   ```
   The server will run on `http://localhost:8000`

### Frontend Setup (Chrome Extension)

1. **Open Chrome and navigate to:**
   ```
   chrome://extensions/
   ```

2. **Enable Developer Mode** (toggle in top-right corner)

3. **Click "Load unpacked"** and select the `frontend` folder

4. **The extension will appear in your Chrome toolbar**

## 🎯 Usage

### Getting Started

1. **Navigate to any YouTube video**
2. **Click the extension icon** in your Chrome toolbar
3. **The extension will automatically detect and process the video**
4. **Start asking questions** about the video content

### Basic Chat
- Ask any question about the video content
- Get AI-powered responses based on the transcript
- Receive smart follow-up suggestions

### Premium Features

#### Brain Mode (Pro+)
- Toggle Brain Mode for personalized responses
- AI learns from your interaction patterns
- Contextual follow-up suggestions

#### Timeline-Aware Queries (Pro+)
- Ask questions about specific timestamps: "What happens at 2:30?"
- Get contextual responses for specific video moments
- Visual timeline navigation

#### Voice Assistant (Pro+)
- Click the microphone to start voice input
- Speak your questions naturally
- Receive audio responses

#### Knowledge Graphs (Power+)
- Visualize concept relationships in videos
- Interactive node-based exploration
- Export graphs for study materials

#### Cross-Video RAG (Power+)
- Ask questions across multiple videos
- Compare content from different sources
- Playlist-wide knowledge synthesis

#### Real-Time Explain (Power+)
- Auto-pause at complex concepts
- Instant explanations and clarifications
- Customizable complexity thresholds

#### Creator Tools (Creator)
- Generate engagement questions for your videos
- Analyze comment sentiment and themes
- Optimize content for better reach

## 🔧 API Endpoints

### Core Endpoints
- `POST /process_video` - Process a YouTube video for RAG
- `POST /chat` - Chat with processed video content
- `GET /videos` - List processed videos
- `DELETE /videos/{video_id}` - Remove processed video

### Subscription Management
- `GET /subscription/{user_id}` - Get user subscription info
- `POST /subscription/upgrade` - Upgrade subscription plan
- `POST /subscription/trial` - Start free trial
- `GET /plans` - Get available subscription plans

### Premium Features
- `POST /timeline_query` - Timeline-aware queries (Pro+)
- `POST /voice_query` - Voice input processing (Pro+)
- `GET /knowledge_graph/{video_id}` - Get knowledge graph (Power+)
- `GET /auto_pause_points/{video_id}` - Get auto-pause points (Power+)
- `GET /generated_questions/{video_id}` - Get creator questions (Creator)

## 🏗️ Architecture

### Backend (FastAPI)
- **RAG Pipeline**: LangChain + FAISS for vector storage
- **AI Models**: Groq (Gemma2-9B-IT) for text generation
- **Embeddings**: HuggingFace Sentence Transformers
- **Subscription Management**: File-based user data storage
- **Premium Features**: Modular feature system with access control

### Frontend (Chrome Extension)
- **Popup Interface**: Modern, responsive design with tab system
- **Content Scripts**: YouTube page integration
- **Background Service**: Extension lifecycle management
- **Storage**: Chrome storage API for user data and chat history

### Key Technologies
- **LangChain**: RAG pipeline orchestration
- **FAISS**: Vector similarity search
- **YouTube Transcript API**: Video transcript extraction
- **Chrome Extensions API**: Browser integration
- **FastAPI**: High-performance backend API

## 🔒 Security & Privacy

- **Local Processing**: All AI processing happens on your machine
- **No Data Collection**: User conversations are stored locally
- **API Key Security**: Your API keys never leave your environment
- **Subscription Privacy**: Minimal user data collection for billing

## 🐛 Troubleshooting

### Common Issues

1. **"Backend not available" error:**
   - Ensure the backend server is running on `http://localhost:8000`
   - Check that all Python dependencies are installed
   - Verify your Groq API key is set correctly

2. **"No transcript found" error:**
   - Some videos don't have transcripts available
   - Try with a different video that has captions

3. **Extension not loading:**
   - Ensure Developer Mode is enabled in Chrome
   - Check the console for JavaScript errors
   - Reload the extension from chrome://extensions/

4. **Premium features not working:**
   - Verify your subscription status in the extension popup
   - Check that the backend is processing premium features correctly
   - Ensure you have the required subscription tier

### Debug Mode

Enable debug logging by setting `LOG_LEVEL=DEBUG` in your `.env` file.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support, please:
1. Check the troubleshooting section above
2. Review the GitHub issues
3. Create a new issue with detailed information about your problem

## 🔮 Roadmap

### Upcoming Features
- **Multi-language Support**: Support for non-English videos
- **Mobile App**: React Native mobile application
- **Team Collaboration**: Shared workspaces and annotations
- **Advanced Analytics**: Learning progress tracking and insights
- **Integration APIs**: Connect with note-taking and learning management systems

### Performance Improvements
- **Caching**: Intelligent caching for faster responses
- **Streaming**: Real-time response streaming
- **Offline Mode**: Local model support for offline usage

---

**Made with ❤️ for learners, creators, and knowledge seekers everywhere.**

