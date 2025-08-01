from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import logging
import json
import base64
from datetime import datetime, timedelta
from pathlib import Path

# RAG Pipeline imports
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from Get_transcript import fetch_transcript

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YouTube RAG Chat API with Premium Features",
    description="A FastAPI backend for chatting with YouTube video content using RAG with premium subscription features",
    version="3.0.0"
)

# CORS middleware for allowing frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Subscription Management Classes
class SubscriptionPlan:
    FREE = "free"
    PRO = "pro"
    POWER_PLUS = "power_plus"
    CREATOR = "creator"

class SubscriptionStatus:
    ACTIVE = "active"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    TRIAL = "trial"

class UserSubscription(BaseModel):
    user_id: str
    plan: str
    status: str
    start_date: str
    end_date: Optional[str] = None
    trial_end_date: Optional[str] = None
    features_used_today: Dict[str, int] = {}
    last_reset_date: str
    payment_id: Optional[str] = None

# Pydantic models for request/response
class VideoRequest(BaseModel):
    video_id: str
    user_id: Optional[str] = None

class ChatRequest(BaseModel):
    video_id: str
    query: str
    user_id: str

class ProcessResponse(BaseModel):
    message: str
    video_id: str
    status: str
    timestamp: str
    subscription_info: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    video_id: str
    query: str
    timestamp: str
    premium_features: Optional[Dict[str, Any]] = None
    subscription_info: Optional[Dict[str, Any]] = None
    upgrade_prompt: Optional[Dict[str, Any]] = None

class SubscriptionRequest(BaseModel):
    user_id: str
    plan: str
    payment_id: Optional[str] = None

class TrialRequest(BaseModel):
    user_id: str
    plan: str
    trial_days: int = 7

class TimelineQueryRequest(BaseModel):
    video_id: str
    query: str
    timestamp: Optional[float] = None
    user_id: str

class VoiceQueryRequest(BaseModel):
    video_id: str
    audio_data: str
    user_id: str

class CrossVideoQueryRequest(BaseModel):
    query: str
    video_ids: List[str]
    user_id: str

# Global storage for processed videos and subscriptions
processed_videos: Dict[str, Dict[str, Any]] = {}
user_subscriptions: Dict[str, UserSubscription] = {}

# Storage directories
subscriptions_dir = Path("user_subscriptions")
subscriptions_dir.mkdir(exist_ok=True)

# Initialize embeddings model
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Embeddings model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embeddings model: {e}")
    embeddings = None

# Initialize LLM
try:
    model = ChatGroq(model="gemma2-9b-it", temperature=0.2)
    logger.info("ChatGroq model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChatGroq model: {e}")
    model = None

# Feature access configuration
FEATURE_ACCESS = {
    "basic_chat": {
        "required_plan": SubscriptionPlan.FREE,
        "daily_limit": 50,
        "description": "Basic Q&A with video content",
        "upgrade_message": "Upgrade to Pro for unlimited chat!"
    },
    "brain_mode": {
        "required_plan": SubscriptionPlan.PRO,
        "description": "Hyperpersonalized AI responses",
        "upgrade_message": "🧠 Unlock Brain Mode with Pro subscription! Get personalized responses that learn from your interactions."
    },
    "timeline_aware": {
        "required_plan": SubscriptionPlan.PRO,
        "description": "Ask questions about specific timestamps",
        "upgrade_message": "⏱️ Upgrade to Pro for Timeline-Aware conversations! Ask about specific video moments."
    },
    "scene_understanding": {
        "required_plan": SubscriptionPlan.PRO,
        "description": "Visual content analysis with AI",
        "upgrade_message": "👁️ Unlock Scene Understanding with Pro! Get AI analysis of visual content."
    },
    "voice_assistant": {
        "required_plan": SubscriptionPlan.PRO,
        "description": "Voice input and output",
        "upgrade_message": "🎤 Upgrade to Pro for Voice Assistant! Talk naturally with your AI assistant."
    },
    "video_overlay": {
        "required_plan": SubscriptionPlan.PRO,
        "description": "AR-like video overlays",
        "upgrade_message": "📺 Unlock Video Overlays with Pro! Get AR-like annotations on videos."
    },
    "knowledge_graph": {
        "required_plan": SubscriptionPlan.POWER_PLUS,
        "description": "Visual concept mapping",
        "upgrade_message": "🕸️ Upgrade to Power+ for Knowledge Graphs! Visualize concept relationships."
    },
    "cross_video_rag": {
        "required_plan": SubscriptionPlan.POWER_PLUS,
        "description": "Query across multiple videos",
        "upgrade_message": "🔗 Unlock Cross-Video RAG with Power+! Ask questions across multiple videos."
    },
    "realtime_explain": {
        "required_plan": SubscriptionPlan.POWER_PLUS,
        "description": "Auto-pause for explanations",
        "upgrade_message": "⚡ Upgrade to Power+ for Real-time Explain! Auto-pause for complex content."
    },
    "question_generator": {
        "required_plan": SubscriptionPlan.CREATOR,
        "description": "AI-generated engagement questions",
        "upgrade_message": "❓ Upgrade to Creator for Question Generator! Generate engaging questions for your audience."
    },
    "comment_analyzer": {
        "required_plan": SubscriptionPlan.CREATOR,
        "description": "Comment sentiment analysis",
        "upgrade_message": "💬 Unlock Comment Analyzer with Creator! Analyze audience sentiment and feedback."
    }
}

# Plan configuration
PLAN_CONFIG = {
    SubscriptionPlan.FREE: {
        "price": 0,
        "features": ["basic_chat"],
        "description": "Basic video Q&A"
    },
    SubscriptionPlan.PRO: {
        "price": 5,
        "features": ["basic_chat", "brain_mode", "timeline_aware", "scene_understanding", "voice_assistant", "video_overlay"],
        "description": "Enhanced AI features"
    },
    SubscriptionPlan.POWER_PLUS: {
        "price": 15,
        "features": ["basic_chat", "brain_mode", "timeline_aware", "scene_understanding", "voice_assistant", "video_overlay", "knowledge_graph", "cross_video_rag", "realtime_explain"],
        "description": "Advanced learning tools"
    },
    SubscriptionPlan.CREATOR: {
        "price": 25,
        "features": ["basic_chat", "brain_mode", "timeline_aware", "scene_understanding", "voice_assistant", "video_overlay", "knowledge_graph", "cross_video_rag", "realtime_explain", "question_generator", "comment_analyzer"],
        "description": "Creator tools and analytics"
    }
}

# Prompt template
prompt = PromptTemplate(
    template="""You are a helpful assistant that answers questions based on YouTube video transcripts. 
    
    Instructions:
    - Provide accurate answers based only on the given transcript
    - If the transcript doesn't contain relevant information, say so clearly
    - Include specific details and examples from the video when possible
    - Be conversational and helpful
    - If asked about timestamps, mention that you can reference general sections but not exact times
    
    Transcript:
    {transcript}
    
    Question: {question}
    
    Answer:""",
    input_variables=["transcript", "question"]
)

# Subscription Management Functions
def get_user_subscription(user_id: str) -> UserSubscription:
    """Get user subscription or create free tier"""
    subscription_file = subscriptions_dir / f"{user_id}.json"
    
    if subscription_file.exists():
        with open(subscription_file, 'r') as f:
            data = json.load(f)
            subscription = UserSubscription(**data)
            
            # Check if subscription needs daily reset
            check_daily_reset(subscription)
            
            # Check if subscription expired
            check_subscription_expiry(subscription)
            
            return subscription
    else:
        # Create new free subscription
        subscription = UserSubscription(
            user_id=user_id,
            plan=SubscriptionPlan.FREE,
            status=SubscriptionStatus.ACTIVE,
            start_date=datetime.now().isoformat(),
            features_used_today={},
            last_reset_date=datetime.now().date().isoformat()
        )
        save_user_subscription(subscription)
        return subscription

def save_user_subscription(subscription: UserSubscription):
    """Save user subscription to storage"""
    subscription_file = subscriptions_dir / f"{subscription.user_id}.json"
    
    with open(subscription_file, 'w') as f:
        json.dump(subscription.dict(), f, indent=2)

def check_feature_access(user_id: str, feature_name: str) -> Dict[str, Any]:
    """Check if user has access to a feature"""
    subscription = get_user_subscription(user_id)
    
    if feature_name not in FEATURE_ACCESS:
        return {
            "has_access": False,
            "reason": "Feature not found",
            "upgrade_message": "Feature not available"
        }
    
    feature = FEATURE_ACCESS[feature_name]
    user_plan = subscription.plan
    required_plan = feature["required_plan"]
    
    # Check plan level access
    plan_hierarchy = [SubscriptionPlan.FREE, SubscriptionPlan.PRO, SubscriptionPlan.POWER_PLUS, SubscriptionPlan.CREATOR]
    
    user_plan_level = plan_hierarchy.index(user_plan)
    required_plan_level = plan_hierarchy.index(required_plan)
    
    if user_plan_level < required_plan_level:
        return {
            "has_access": False,
            "reason": "Insufficient plan level",
            "upgrade_message": feature["upgrade_message"],
            "required_plan": required_plan,
            "current_plan": user_plan
        }
    
    # Check daily limits
    if "daily_limit" in feature:
        usage_today = subscription.features_used_today.get(feature_name, 0)
        if usage_today >= feature["daily_limit"]:
            return {
                "has_access": False,
                "reason": "Daily limit exceeded",
                "upgrade_message": f"Daily limit of {feature['daily_limit']} reached. {feature['upgrade_message']}",
                "daily_limit": feature["daily_limit"],
                "usage_today": usage_today
            }
    
    # Check subscription status
    if subscription.status not in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]:
        return {
            "has_access": False,
            "reason": "Subscription inactive",
            "upgrade_message": "Please renew your subscription to continue using premium features."
        }
    
    return {
        "has_access": True,
        "reason": "Access granted",
        "remaining_usage": feature.get("daily_limit", None) and feature["daily_limit"] - subscription.features_used_today.get(feature_name, 0)
    }

def record_feature_usage(user_id: str, feature_name: str):
    """Record feature usage for daily limits"""
    subscription = get_user_subscription(user_id)
    
    if feature_name in subscription.features_used_today:
        subscription.features_used_today[feature_name] += 1
    else:
        subscription.features_used_today[feature_name] = 1
    
    save_user_subscription(subscription)

def check_daily_reset(subscription: UserSubscription):
    """Check if daily usage should be reset"""
    last_reset = datetime.fromisoformat(subscription.last_reset_date).date()
    today = datetime.now().date()
    
    if last_reset < today:
        subscription.features_used_today = {}
        subscription.last_reset_date = today.isoformat()
        save_user_subscription(subscription)

def check_subscription_expiry(subscription: UserSubscription):
    """Check if subscription has expired"""
    now = datetime.now()
    
    # Check trial expiry
    if subscription.status == SubscriptionStatus.TRIAL and subscription.trial_end_date:
        trial_end = datetime.fromisoformat(subscription.trial_end_date)
        if now > trial_end:
            subscription.status = SubscriptionStatus.EXPIRED
            subscription.plan = SubscriptionPlan.FREE
            save_user_subscription(subscription)
    
    # Check subscription expiry
    elif subscription.status == SubscriptionStatus.ACTIVE and subscription.end_date:
        end_date = datetime.fromisoformat(subscription.end_date)
        if now > end_date:
            subscription.status = SubscriptionStatus.EXPIRED
            subscription.plan = SubscriptionPlan.FREE
            save_user_subscription(subscription)

def get_upgrade_suggestions(user_id: str, requested_feature: str = None) -> Dict[str, Any]:
    """Get upgrade suggestions for user"""
    subscription = get_user_subscription(user_id)
    current_plan = subscription.plan
    
    suggestions = []
    
    if requested_feature and requested_feature in FEATURE_ACCESS:
        # Suggest specific plan for requested feature
        required_plan = FEATURE_ACCESS[requested_feature]["required_plan"]
        if required_plan != current_plan:
            plan_info = PLAN_CONFIG[required_plan]
            suggestions.append({
                "plan": required_plan,
                "price": plan_info["price"],
                "description": plan_info["description"],
                "reason": f"Required for {FEATURE_ACCESS[requested_feature]['description']}",
                "features": plan_info["features"]
            })
    else:
        # Suggest next tier upgrade
        plan_hierarchy = [SubscriptionPlan.FREE, SubscriptionPlan.PRO, SubscriptionPlan.POWER_PLUS, SubscriptionPlan.CREATOR]
        
        current_index = plan_hierarchy.index(current_plan)
        
        for i in range(current_index + 1, len(plan_hierarchy)):
            next_plan = plan_hierarchy[i]
            plan_info = PLAN_CONFIG[next_plan]
            
            suggestions.append({
                "plan": next_plan,
                "price": plan_info["price"],
                "description": plan_info["description"],
                "features": plan_info["features"]
            })
    
    return {
        "current_plan": current_plan,
        "suggestions": suggestions,
        "trial_available": subscription.status != SubscriptionStatus.TRIAL
    }

# Premium Feature Functions
def create_brain_mode_response(user_id: str, video_id: str, query: str, base_response: str) -> Dict[str, Any]:
    """Create personalized response using brain mode"""
    try:
        # Simple brain mode implementation - store user preferences and context
        user_context_file = subscriptions_dir / f"{user_id}_context.json"
        
        if user_context_file.exists():
            with open(user_context_file, 'r') as f:
                user_context = json.load(f)
        else:
            user_context = {"interests": [], "learning_style": "general", "previous_queries": []}
        
        # Update user context
        user_context["previous_queries"].append({
            "video_id": video_id,
            "query": query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 queries
        user_context["previous_queries"] = user_context["previous_queries"][-10:]
        
        # Save updated context
        with open(user_context_file, 'w') as f:
            json.dump(user_context, f, indent=2)
        
        # Generate personalized response
        if len(user_context["previous_queries"]) > 1:
            personalized_response = f"{base_response}\n\n💡 **Personalized Insight**: Based on your previous questions, you might also be interested in exploring related concepts in this video."
        else:
            personalized_response = base_response
        
        # Generate follow-up suggestions
        follow_up_suggestions = [
            "Can you explain this concept in more detail?",
            "How does this relate to the main topic?",
            "What are the practical applications of this?"
        ]
        
        return {
            "personalized_response": personalized_response,
            "follow_up_suggestions": follow_up_suggestions,
            "brain_mode_active": True
        }
    except Exception as e:
        logger.error(f"Brain mode error: {e}")
        return {"personalized_response": base_response, "brain_mode_active": False}

def create_timeline_aware_response(video_id: str, query: str, timestamp: float = None) -> Dict[str, Any]:
    """Create timeline-aware response"""
    try:
        if timestamp is not None:
            # Format timestamp
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"{minutes}:{seconds:02d}"
            
            timeline_response = f"**At {time_str}**: {query}\n\nBased on the content around this timestamp, here's what's happening in the video at this moment."
            
            return {
                "timeline_response": timeline_response,
                "timestamp": timestamp,
                "timeline_context": f"Content analysis for timestamp {time_str}"
            }
        else:
            return {"timeline_response": "Please specify a timestamp for timeline-aware queries."}
    except Exception as e:
        logger.error(f"Timeline aware error: {e}")
        return {"timeline_response": "Timeline analysis failed."}

def create_scene_understanding_response(video_id: str, query: str, timestamp: float) -> Dict[str, Any]:
    """Create scene understanding response"""
    try:
        # Simulate scene understanding
        scene_analysis = {
            "visual_elements": ["Text overlay", "Speaker presentation", "Background graphics"],
            "scene_type": "Educational content",
            "complexity_level": "Medium",
            "key_visual_concepts": ["Diagrams", "Charts", "Text highlights"]
        }
        
        visual_response = f"**Visual Analysis**: At this moment in the video, I can see {', '.join(scene_analysis['visual_elements'])}. This appears to be {scene_analysis['scene_type']} with {scene_analysis['complexity_level'].lower()} complexity."
        
        return {
            "visual_response": visual_response,
            "scene_analysis": scene_analysis,
            "visual_context": "AI-powered visual content analysis"
        }
    except Exception as e:
        logger.error(f"Scene understanding error: {e}")
        return {"visual_response": "Visual analysis failed."}

def create_voice_response(video_id: str, audio_data: str, user_id: str) -> Dict[str, Any]:
    """Process voice query"""
    try:
        # Simulate voice processing (in real implementation, use Whisper API)
        # For demo purposes, we'll simulate transcription
        transcribed_text = "What is the main topic of this video?"
        
        # Generate text response (would use the normal chat pipeline)
        text_response = "This video discusses the main concepts and provides detailed explanations of the topic."
        
        # Simulate audio response generation (in real implementation, use TTS)
        audio_response_url = f"/audio_responses/{user_id}_{video_id}_{datetime.now().timestamp()}.wav"
        
        return {
            "transcribed_text": transcribed_text,
            "response": text_response,
            "audio_response_url": audio_response_url,
            "voice_processing_success": True
        }
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        return {"voice_processing_success": False, "error": str(e)}

def create_knowledge_graph(video_id: str, transcript_text: str) -> Dict[str, Any]:
    """Create knowledge graph for video"""
    try:
        # Simulate knowledge graph creation
        concepts = ["Main Topic", "Key Concept 1", "Key Concept 2", "Supporting Idea", "Conclusion"]
        relationships = [
            {"from": "Main Topic", "to": "Key Concept 1", "type": "contains"},
            {"from": "Main Topic", "to": "Key Concept 2", "type": "contains"},
            {"from": "Key Concept 1", "to": "Supporting Idea", "type": "supports"},
            {"from": "Key Concept 2", "to": "Conclusion", "type": "leads_to"}
        ]
        
        knowledge_graph = {
            "video_id": video_id,
            "concepts": concepts,
            "relationships": relationships,
            "created_at": datetime.now().isoformat()
        }
        
        # Save knowledge graph
        kg_file = subscriptions_dir / f"kg_{video_id}.json"
        with open(kg_file, 'w') as f:
            json.dump(knowledge_graph, f, indent=2)
        
        return {
            "knowledge_graph": knowledge_graph,
            "visualization_data": {
                "nodes": [{"id": concept, "label": concept} for concept in concepts],
                "edges": relationships
            }
        }
    except Exception as e:
        logger.error(f"Knowledge graph error: {e}")
        return {"error": str(e)}

def create_auto_pause_points(video_id: str) -> List[Dict[str, Any]]:
    """Create auto-pause points for real-time explain"""
    try:
        # Simulate auto-pause point detection
        pause_points = [
            {
                "timestamp": 30.0,
                "reason": "Complex concept introduction",
                "explanation": "This section introduces a complex concept that may require additional explanation.",
                "complexity_score": 0.8
            },
            {
                "timestamp": 120.0,
                "reason": "Technical terminology",
                "explanation": "Multiple technical terms are introduced here.",
                "complexity_score": 0.7
            },
            {
                "timestamp": 240.0,
                "reason": "Key insight",
                "explanation": "This is a crucial insight that ties together previous concepts.",
                "complexity_score": 0.9
            }
        ]
        
        return pause_points
    except Exception as e:
        logger.error(f"Auto-pause points error: {e}")
        return []

def generate_creator_questions(video_id: str, transcript_text: str) -> Dict[str, Any]:
    """Generate questions for creators"""
    try:
        # Simulate question generation
        questions = {
            "engagement_questions": [
                "What was your biggest takeaway from this video?",
                "Which concept would you like me to explain further?",
                "How would you apply this information in your own work?"
            ],
            "comprehension_questions": [
                "Can you summarize the main points discussed?",
                "What are the key differences between the concepts mentioned?",
                "How do these ideas connect to what we discussed earlier?"
            ],
            "discussion_questions": [
                "What are your thoughts on this approach?",
                "Have you encountered similar situations?",
                "What questions do you still have about this topic?"
            ]
        }
        
        return {
            "video_id": video_id,
            "generated_questions": questions,
            "total_questions": sum(len(q_list) for q_list in questions.values()),
            "created_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Question generation error: {e}")
        return {"error": str(e)}

def analyze_comments(video_id: str, comments: List[str]) -> Dict[str, Any]:
    """Analyze video comments"""
    try:
        # Simulate comment analysis
        analysis = {
            "total_comments": len(comments),
            "sentiment_distribution": {
                "positive": 0.6,
                "neutral": 0.3,
                "negative": 0.1
            },
            "key_themes": ["Educational", "Helpful", "Clear explanation", "Good examples"],
            "engagement_metrics": {
                "average_length": 45,
                "question_count": 12,
                "appreciation_count": 28
            },
            "suggestions": [
                "Viewers appreciate the clear explanations",
                "Some users requested more examples",
                "Overall positive reception"
            ]
        }
        
        return {
            "video_id": video_id,
            "comment_analysis": analysis,
            "analyzed_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Comment analysis error: {e}")
        return {"error": str(e)}

# Core Functions
def extract_transcript(video_id: str) -> str:
    """Extract transcript from YouTube video"""
    try:
        docs = fetch_transcript(video_id)
        return docs
        
    except TranscriptsDisabled:
        logger.error(f"Transcripts disabled for video {video_id}")
        raise HTTPException(status_code=400, detail="Transcripts are disabled for this video")
    except NoTranscriptFound:
        logger.error(f"No transcript found for video {video_id}")
        raise HTTPException(status_code=404, detail="No transcript found for this video")
    except Exception as e:
        logger.error(f"Error extracting transcript for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract transcript: {str(e)}")

def create_vector_store(docs: str, video_id: str) -> FAISS:
    """Create FAISS vector store from documents"""
    try:
        vector_store = FAISS.from_documents(docs, embedding=embeddings)
        logger.info(f"Successfully created vector store for video {video_id}")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector store for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {str(e)}")

def format_docs(retrieved_docs):
    """Format retrieved documents for prompt"""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

# API Endpoints
@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {
        "message": "YouTube RAG Chat API with Premium Features is running",
        "status": "healthy",
        "version": "3.0.0",
        "subscription_system_enabled": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "embeddings_loaded": embeddings is not None,
        "model_loaded": model is not None,
        "processed_videos": len(processed_videos),
        "features_available": list(FEATURE_ACCESS.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/process_video", response_model=ProcessResponse)
async def process_video(request: VideoRequest):
    """Process a YouTube video for RAG chat"""
    video_id = request.video_id.strip()
    user_id = request.user_id or f"anonymous_{video_id}"
    
    if not video_id:
        raise HTTPException(status_code=400, detail="Video ID is required")
    
    logger.info(f"Processing video: {video_id} for user: {user_id}")
    
    # Get user subscription info
    subscription = get_user_subscription(user_id)
    
    try:
        # Check if video is already processed
        if video_id in processed_videos:
            logger.info(f"Video {video_id} already processed")
            return ProcessResponse(
                message=f"Video {video_id} was already processed and is ready for chat",
                video_id=video_id,
                status="already_processed",
                timestamp=datetime.now().isoformat(),
                subscription_info={
                    "plan": subscription.plan,
                    "status": subscription.status
                }
            )
        
        # Extract transcript
        transcript = extract_transcript(video_id)
        
        # Create vector store
        vector_store = create_vector_store(transcript, video_id)
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )
        
        # Create RAG chain
        parallel_chain = RunnableParallel({
            "transcript": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | model | parser
        
        # Process premium features based on subscription
        premium_data = {}
        
        try:
            # Generate knowledge graph (Power+ feature)
            if subscription.plan in [SubscriptionPlan.POWER_PLUS, SubscriptionPlan.CREATOR]:
                transcript_text = " ".join([doc.page_content for doc in transcript])
                knowledge_graph = create_knowledge_graph(video_id, transcript_text)
                premium_data["knowledge_graph"] = knowledge_graph
            
            # Generate questions for creators (Creator feature)
            if subscription.plan == SubscriptionPlan.CREATOR:
                transcript_text = " ".join([doc.page_content for doc in transcript])
                question_set = generate_creator_questions(video_id, transcript_text)
                premium_data["generated_questions"] = question_set
            
            # Create auto-pause points for real-time explain (Power+ feature)
            if subscription.plan in [SubscriptionPlan.POWER_PLUS, SubscriptionPlan.CREATOR]:
                pause_points = create_auto_pause_points(video_id)
                premium_data["auto_pause_points"] = pause_points
            
        except Exception as e:
            logger.error(f"Error processing premium features: {e}")
            premium_data["error"] = "Some premium features failed to process"
        
        # Store processed video data
        processed_videos[video_id] = {
            "vector_store": vector_store,
            "retriever": retriever,
            "chain": main_chain,
            "transcript_length": len(" ".join([doc.page_content for doc in transcript])),
            "processed_at": datetime.now().isoformat(),
            "premium_data": premium_data
        }
        
        logger.info(f"Successfully processed video {video_id}")
        
        return ProcessResponse(
            message=f"Video {video_id} processed successfully. {len(premium_data)} premium features available.",
            video_id=video_id,
            status="processed",
            timestamp=datetime.now().isoformat(),
            subscription_info={
                "plan": subscription.plan,
                "status": subscription.status,
                "features_available": len(premium_data)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with processed video content"""
    video_id = request.video_id.strip()
    query = request.query.strip()
    user_id = request.user_id
    
    if not video_id:
        raise HTTPException(status_code=400, detail="Video ID is required")
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required for subscription checking")
    
    logger.info(f"Chat request for video {video_id} from user {user_id}: {query}")
    
    # Check basic chat access and record usage
    access_result = check_feature_access(user_id, "basic_chat")
    
    if not access_result["has_access"]:
        upgrade_suggestions = get_upgrade_suggestions(user_id, "basic_chat")
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Daily chat limit exceeded",
                "reason": access_result["reason"],
                "upgrade_message": access_result["upgrade_message"],
                "upgrade_suggestions": upgrade_suggestions
            }
        )
    
    try:
        # Check if video is processed
        if video_id not in processed_videos:
            raise HTTPException(
                status_code=404, 
                detail=f"Video {video_id} has not been processed. Please process it first."
            )
        
        # Get the RAG chain for this video
        video_data = processed_videos[video_id]
        chain = video_data["chain"]
        
        if model is None:
            raise HTTPException(status_code=500, detail="Language model not available")
        
        # Record basic chat usage
        record_feature_usage(user_id, "basic_chat")
        
        # Get user subscription for premium features
        subscription = get_user_subscription(user_id)
        
        # Generate base response
        response = chain.invoke(query)
        
        # Enhanced response with premium features (if available)
        premium_features = {}
        upgrade_prompt = None
        
        # Check for timeline query
        timestamp = extract_timestamp_from_query(query)
        if timestamp is not None:
            if subscription.plan in [SubscriptionPlan.PRO, SubscriptionPlan.POWER_PLUS, SubscriptionPlan.CREATOR]:
                timeline_response = create_timeline_aware_response(video_id, query, timestamp)
                premium_features.update(timeline_response)
                response = timeline_response.get("timeline_response", response)
            else:
                upgrade_prompt = {
                    "feature": "Timeline-Aware Queries",
                    "message": "⏱️ Upgrade to Pro for Timeline-Aware conversations! Ask about specific video moments.",
                    "suggestions": get_upgrade_suggestions(user_id, "timeline_aware")
                }
        
        # Use Brain Mode for personalized response (Pro+ feature)
        if subscription.plan in [SubscriptionPlan.PRO, SubscriptionPlan.POWER_PLUS, SubscriptionPlan.CREATOR]:
            brain_mode_result = create_brain_mode_response(user_id, video_id, query, response)
            premium_features.update(brain_mode_result)
            response = brain_mode_result.get("personalized_response", response)
        elif subscription.plan == SubscriptionPlan.FREE and not upgrade_prompt:
            upgrade_prompt = {
                "feature": "Brain Mode",
                "message": "🧠 Upgrade to Pro for personalized AI responses that learn from your interactions!",
                "suggestions": get_upgrade_suggestions(user_id, "brain_mode")
            }
        
        # Get subscription info for response
        subscription_info = {
            "plan": subscription.plan,
            "status": subscription.status,
            "remaining_chats": access_result.get("remaining_usage"),
            "features_available": PLAN_CONFIG[subscription.plan]["features"]
        }
        
        logger.info(f"Generated response for video {video_id}")
        
        return ChatResponse(
            response=response,
            video_id=video_id,
            query=query,
            timestamp=datetime.now().isoformat(),
            premium_features=premium_features if premium_features else None,
            subscription_info=subscription_info,
            upgrade_prompt=upgrade_prompt
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating response for video {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

def extract_timestamp_from_query(query: str) -> Optional[float]:
    """Extract timestamp from query if present"""
    import re
    
    # Look for patterns like "at 2:30", "2:30", "2m30s"
    patterns = [
        r"(?:at\s+)?(\d{1,2}):(\d{2})",  # 2:30 or at 2:30
        r"(?:at\s+)?(\d+)m\s*(\d+)s",    # 2m30s
        r"(?:at\s+)?(\d+):\d{2}:\d{2}"   # 1:02:30
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            if "m.*s" in pattern:
                # Handle minutes and seconds
                return int(match.group(1)) * 60 + int(match.group(2))
            else:
                # Handle mm:ss format
                return int(match.group(1)) * 60 + int(match.group(2))
    
    return None

# Subscription Management Endpoints
@app.get("/subscription/{user_id}")
async def get_subscription(user_id: str):
    """Get user subscription information"""
    subscription = get_user_subscription(user_id)
    
    stats = {
        "user_id": user_id,
        "plan": subscription.plan,
        "status": subscription.status,
        "features_used_today": subscription.features_used_today,
        "plan_features": PLAN_CONFIG[subscription.plan]["features"],
        "subscription_end": subscription.end_date,
        "trial_end": subscription.trial_end_date
    }
    
    return {
        "subscription": subscription.dict(),
        "stats": stats,
        "upgrade_suggestions": get_upgrade_suggestions(user_id),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/subscription/upgrade")
async def upgrade_subscription(request: SubscriptionRequest):
    """Upgrade user subscription"""
    try:
        subscription = get_user_subscription(request.user_id)
        
        subscription.plan = request.plan
        subscription.status = SubscriptionStatus.ACTIVE
        subscription.start_date = datetime.now().isoformat()
        subscription.end_date = (datetime.now() + timedelta(days=30)).isoformat()
        subscription.payment_id = request.payment_id
        
        # Reset daily usage
        subscription.features_used_today = {}
        subscription.last_reset_date = datetime.now().date().isoformat()
        
        save_user_subscription(subscription)
        
        return {
            "message": f"Successfully upgraded to {request.plan}",
            "subscription": subscription.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error upgrading subscription: {e}")
        raise HTTPException(status_code=500, detail="Failed to upgrade subscription")

@app.post("/subscription/trial")
async def start_trial(request: TrialRequest):
    """Start trial subscription"""
    try:
        subscription = get_user_subscription(request.user_id)
        
        subscription.plan = request.plan
        subscription.status = SubscriptionStatus.TRIAL
        subscription.trial_end_date = (datetime.now() + timedelta(days=request.trial_days)).isoformat()
        
        # Reset daily usage
        subscription.features_used_today = {}
        subscription.last_reset_date = datetime.now().date().isoformat()
        
        save_user_subscription(subscription)
        
        return {
            "message": f"Started {request.trial_days}-day trial for {request.plan}",
            "subscription": subscription.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting trial: {e}")
        raise HTTPException(status_code=500, detail="Failed to start trial")

@app.get("/plans")
async def get_subscription_plans():
    """Get available subscription plans"""
    return {
        "plans": {
            "free": {
                "price": 0,
                "features": ["Basic Q&A (50/day)", "Transcript viewing"],
                "description": "Basic video Q&A"
            },
            "pro": {
                "price": 5,
                "features": ["Unlimited chat", "Brain Mode", "Timeline-aware", "Scene understanding", "Voice assistant", "Video overlays"],
                "description": "Enhanced AI features"
            },
            "power_plus": {
                "price": 15,
                "features": ["All Pro features", "Knowledge graphs", "Cross-video RAG", "Real-time explain", "Advanced analytics"],
                "description": "Advanced learning tools"
            },
            "creator": {
                "price": 25,
                "features": ["All Power+ features", "Question generator", "Comment analyzer", "SEO optimizer", "Audience insights"],
                "description": "Creator tools and analytics"
            }
        },
        "timestamp": datetime.now().isoformat()
    }

# Premium Feature Endpoints
@app.post("/timeline_query")
async def timeline_query(request: TimelineQueryRequest):
    """Query video content with timeline awareness (Pro+ feature)"""
    access_result = check_feature_access(request.user_id, "timeline_aware")
    if not access_result["has_access"]:
        raise HTTPException(
            status_code=403, 
            detail={
                "error": "Premium feature access required",
                "feature": "timeline_aware",
                "reason": access_result["reason"],
                "upgrade_message": access_result["upgrade_message"],
                "upgrade_suggestions": get_upgrade_suggestions(request.user_id, "timeline_aware")
            }
        )
    
    try:
        record_feature_usage(request.user_id, "timeline_aware")
        
        response = create_timeline_aware_response(request.video_id, request.query, request.timestamp)
        
        return {
            "video_id": request.video_id,
            "query": request.query,
            "timeline_response": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Timeline query error: {e}")
        raise HTTPException(status_code=500, detail=f"Timeline query failed: {str(e)}")

@app.post("/voice_query")
async def voice_query(request: VoiceQueryRequest):
    """Process voice query (Pro+ feature)"""
    access_result = check_feature_access(request.user_id, "voice_assistant")
    if not access_result["has_access"]:
        raise HTTPException(
            status_code=403, 
            detail={
                "error": "Premium feature access required",
                "feature": "voice_assistant",
                "reason": access_result["reason"],
                "upgrade_message": access_result["upgrade_message"],
                "upgrade_suggestions": get_upgrade_suggestions(request.user_id, "voice_assistant")
            }
        )
    
    try:
        record_feature_usage(request.user_id, "voice_assistant")
        
        response = create_voice_response(request.video_id, request.audio_data, request.user_id)
        
        return {
            "video_id": request.video_id,
            "voice_response": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Voice query error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice query failed: {str(e)}")

@app.get("/knowledge_graph/{video_id}")
async def get_knowledge_graph(video_id: str, user_id: str):
    """Get knowledge graph for a video (Power+ feature)"""
    access_result = check_feature_access(user_id, "knowledge_graph")
    if not access_result["has_access"]:
        raise HTTPException(
            status_code=403, 
            detail={
                "error": "Premium feature access required",
                "feature": "knowledge_graph",
                "reason": access_result["reason"],
                "upgrade_message": access_result["upgrade_message"],
                "upgrade_suggestions": get_upgrade_suggestions(user_id, "knowledge_graph")
            }
        )
    
    try:
        kg_file = subscriptions_dir / f"kg_{video_id}.json"
        
        if not kg_file.exists():
            raise HTTPException(status_code=404, detail="Knowledge graph not found for this video")
        
        with open(kg_file, 'r') as f:
            knowledge_graph = json.load(f)
        
        return {
            "video_id": video_id,
            "knowledge_graph": knowledge_graph,
            "visualization": {
                "nodes": [{"id": concept, "label": concept} for concept in knowledge_graph["concepts"]],
                "edges": knowledge_graph["relationships"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Knowledge graph error: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge graph retrieval failed: {str(e)}")

@app.get("/auto_pause_points/{video_id}")
async def get_auto_pause_points(video_id: str, user_id: str):
    """Get auto-pause points for a video (Power+ feature)"""
    access_result = check_feature_access(user_id, "realtime_explain")
    if not access_result["has_access"]:
        raise HTTPException(
            status_code=403, 
            detail={
                "error": "Premium feature access required",
                "feature": "realtime_explain",
                "reason": access_result["reason"],
                "upgrade_message": access_result["upgrade_message"],
                "upgrade_suggestions": get_upgrade_suggestions(user_id, "realtime_explain")
            }
        )
    
    try:
        pause_points = create_auto_pause_points(video_id)
        
        return {
            "video_id": video_id,
            "pause_points": pause_points,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Auto-pause points error: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-pause points retrieval failed: {str(e)}")

@app.get("/generated_questions/{video_id}")
async def get_generated_questions(video_id: str, user_id: str):
    """Get generated questions for a video (Creator feature)"""
    access_result = check_feature_access(user_id, "question_generator")
    if not access_result["has_access"]:
        raise HTTPException(
            status_code=403, 
            detail={
                "error": "Premium feature access required",
                "feature": "question_generator",
                "reason": access_result["reason"],
                "upgrade_message": access_result["upgrade_message"],
                "upgrade_suggestions": get_upgrade_suggestions(user_id, "question_generator")
            }
        )
    
    try:
        if video_id in processed_videos and "generated_questions" in processed_videos[video_id].get("premium_data", {}):
            questions = processed_videos[video_id]["premium_data"]["generated_questions"]
            return {
                "video_id": video_id,
                "questions": questions,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Generated questions not found for this video")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generated questions error: {e}")
        raise HTTPException(status_code=500, detail=f"Generated questions retrieval failed: {str(e)}")

# Original endpoints
@app.get("/videos")
async def list_processed_videos():
    """List all processed videos"""
    videos = []
    for video_id, data in processed_videos.items():
        video_info = {
            "video_id": video_id,
            "processed_at": data["processed_at"],
            "transcript_length": data["transcript_length"],
            "has_premium_features": "premium_data" in data
        }
        videos.append(video_info)
    
    return {
        "processed_videos": videos,
        "count": len(videos),
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/videos/{video_id}")
async def delete_processed_video(video_id: str):
    """Delete a processed video from memory"""
    if video_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Clean up premium feature data
    try:
        kg_file = subscriptions_dir / f"kg_{video_id}.json"
        if kg_file.exists():
            kg_file.unlink()
    except Exception as e:
        logger.error(f"Error cleaning up premium features for video {video_id}: {e}")
    
    del processed_videos[video_id]
    logger.info(f"Deleted processed video: {video_id}")
    
    return {
        "message": f"Video {video_id} deleted successfully",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

