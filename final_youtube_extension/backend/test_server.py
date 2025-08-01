#!/usr/bin/env python3
"""
Simplified test server for YouTube RAG Chat Extension
This version uses minimal dependencies for testing purposes.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YouTube RAG Chat API - Test Version",
    description="Simplified test version for validation",
    version="3.0.0-test"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Subscription Plans
class SubscriptionPlan:
    FREE = "free"
    PRO = "pro"
    POWER_PLUS = "power_plus"
    CREATOR = "creator"

class SubscriptionStatus:
    ACTIVE = "active"
    EXPIRED = "expired"
    TRIAL = "trial"

# Pydantic models
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

class TrialRequest(BaseModel):
    user_id: str
    plan: str
    trial_days: int = 7

# In-memory storage for testing
processed_videos = {}
user_subscriptions = {}

# Feature access configuration
FEATURE_ACCESS = {
    "basic_chat": {
        "required_plan": SubscriptionPlan.FREE,
        "daily_limit": 50,
        "upgrade_message": "Upgrade to Pro for unlimited chat!"
    },
    "brain_mode": {
        "required_plan": SubscriptionPlan.PRO,
        "upgrade_message": "🧠 Unlock Brain Mode with Pro subscription!"
    },
    "timeline_aware": {
        "required_plan": SubscriptionPlan.PRO,
        "upgrade_message": "⏱️ Upgrade to Pro for Timeline-Aware conversations!"
    },
    "knowledge_graph": {
        "required_plan": SubscriptionPlan.POWER_PLUS,
        "upgrade_message": "🕸️ Upgrade to Power+ for Knowledge Graphs!"
    },
    "question_generator": {
        "required_plan": SubscriptionPlan.CREATOR,
        "upgrade_message": "❓ Upgrade to Creator for Question Generator!"
    }
}

def get_user_subscription(user_id: str) -> Dict[str, Any]:
    """Get or create user subscription"""
    if user_id not in user_subscriptions:
        user_subscriptions[user_id] = {
            "user_id": user_id,
            "plan": SubscriptionPlan.FREE,
            "status": SubscriptionStatus.ACTIVE,
            "start_date": datetime.now().isoformat(),
            "features_used_today": {},
            "last_reset_date": datetime.now().date().isoformat()
        }
    
    subscription = user_subscriptions[user_id]
    
    # Check daily reset
    last_reset = datetime.fromisoformat(subscription["last_reset_date"]).date()
    today = datetime.now().date()
    
    if last_reset < today:
        subscription["features_used_today"] = {}
        subscription["last_reset_date"] = today.isoformat()
    
    return subscription

def check_feature_access(user_id: str, feature_name: str) -> Dict[str, Any]:
    """Check if user has access to a feature"""
    subscription = get_user_subscription(user_id)
    
    if feature_name not in FEATURE_ACCESS:
        return {"has_access": False, "reason": "Feature not found"}
    
    feature = FEATURE_ACCESS[feature_name]
    user_plan = subscription["plan"]
    required_plan = feature["required_plan"]
    
    # Check plan level
    plan_hierarchy = [SubscriptionPlan.FREE, SubscriptionPlan.PRO, SubscriptionPlan.POWER_PLUS, SubscriptionPlan.CREATOR]
    user_level = plan_hierarchy.index(user_plan)
    required_level = plan_hierarchy.index(required_plan)
    
    if user_level < required_level:
        return {
            "has_access": False,
            "reason": "Insufficient plan level",
            "upgrade_message": feature["upgrade_message"]
        }
    
    # Check daily limits
    if "daily_limit" in feature:
        usage = subscription["features_used_today"].get(feature_name, 0)
        if usage >= feature["daily_limit"]:
            return {
                "has_access": False,
                "reason": "Daily limit exceeded",
                "upgrade_message": feature["upgrade_message"]
            }
    
    return {"has_access": True, "reason": "Access granted"}

def record_feature_usage(user_id: str, feature_name: str):
    """Record feature usage"""
    subscription = get_user_subscription(user_id)
    current_usage = subscription["features_used_today"].get(feature_name, 0)
    subscription["features_used_today"][feature_name] = current_usage + 1

@app.get("/")
async def read_root():
    """Health check endpoint"""
    return {
        "message": "YouTube RAG Chat API Test Server is running",
        "status": "healthy",
        "version": "3.0.0-test",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "processed_videos": len(processed_videos),
        "active_users": len(user_subscriptions),
        "features_available": list(FEATURE_ACCESS.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/process_video", response_model=ProcessResponse)
async def process_video(request: VideoRequest):
    """Process a YouTube video (simulated)"""
    video_id = request.video_id.strip()
    user_id = request.user_id or f"anonymous_{video_id}"
    
    if not video_id:
        raise HTTPException(status_code=400, detail="Video ID is required")
    
    logger.info(f"Processing video: {video_id} for user: {user_id}")
    
    subscription = get_user_subscription(user_id)
    
    # Simulate processing
    if video_id not in processed_videos:
        processed_videos[video_id] = {
            "processed_at": datetime.now().isoformat(),
            "transcript_length": 1500,  # Simulated
            "premium_features": {}
        }
        
        # Add premium features based on subscription
        if subscription["plan"] in [SubscriptionPlan.POWER_PLUS, SubscriptionPlan.CREATOR]:
            processed_videos[video_id]["premium_features"]["knowledge_graph"] = True
        
        if subscription["plan"] == SubscriptionPlan.CREATOR:
            processed_videos[video_id]["premium_features"]["generated_questions"] = True
    
    return ProcessResponse(
        message=f"Video {video_id} processed successfully",
        video_id=video_id,
        status="processed",
        timestamp=datetime.now().isoformat(),
        subscription_info={
            "plan": subscription["plan"],
            "status": subscription["status"]
        }
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with processed video content (simulated)"""
    video_id = request.video_id.strip()
    query = request.query.strip()
    user_id = request.user_id
    
    if not video_id or not query or not user_id:
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    logger.info(f"Chat request for video {video_id} from user {user_id}: {query}")
    
    # Check basic chat access
    access_result = check_feature_access(user_id, "basic_chat")
    
    if not access_result["has_access"]:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "Daily chat limit exceeded",
                "reason": access_result["reason"],
                "upgrade_message": access_result["upgrade_message"]
            }
        )
    
    # Check if video is processed
    if video_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Video not processed")
    
    # Record usage
    record_feature_usage(user_id, "basic_chat")
    
    subscription = get_user_subscription(user_id)
    
    # Generate simulated response
    base_response = f"This is a simulated AI response to your question: '{query}'. The video discusses various topics and provides insights related to your query."
    
    # Enhanced response with premium features
    premium_features = {}
    upgrade_prompt = None
    
    # Check for timeline query
    if any(word in query.lower() for word in ["at", "timestamp", "time", "minute", "second"]):
        if subscription["plan"] in [SubscriptionPlan.PRO, SubscriptionPlan.POWER_PLUS, SubscriptionPlan.CREATOR]:
            premium_features["timeline_response"] = "Timeline-aware response generated"
            base_response += "\n\n⏱️ **Timeline Context**: This response includes specific timestamp information."
        else:
            upgrade_prompt = {
                "feature": "Timeline-Aware Queries",
                "message": "⏱️ Upgrade to Pro for Timeline-Aware conversations! Ask about specific video moments."
            }
    
    # Brain Mode enhancement
    if subscription["plan"] in [SubscriptionPlan.PRO, SubscriptionPlan.POWER_PLUS, SubscriptionPlan.CREATOR]:
        premium_features["brain_mode_active"] = True
        premium_features["follow_up_suggestions"] = [
            "Can you explain this concept in more detail?",
            "How does this relate to the main topic?",
            "What are the practical applications?"
        ]
        base_response += "\n\n🧠 **Personalized Insight**: Based on your previous questions, you might find these related concepts interesting."
    elif subscription["plan"] == SubscriptionPlan.FREE and not upgrade_prompt:
        upgrade_prompt = {
            "feature": "Brain Mode",
            "message": "🧠 Upgrade to Pro for personalized AI responses that learn from your interactions!"
        }
    
    subscription_info = {
        "plan": subscription["plan"],
        "status": subscription["status"],
        "remaining_chats": 50 - subscription["features_used_today"].get("basic_chat", 0) if subscription["plan"] == SubscriptionPlan.FREE else None
    }
    
    return ChatResponse(
        response=base_response,
        video_id=video_id,
        query=query,
        timestamp=datetime.now().isoformat(),
        premium_features=premium_features if premium_features else None,
        subscription_info=subscription_info,
        upgrade_prompt=upgrade_prompt
    )

@app.get("/subscription/{user_id}")
async def get_subscription(user_id: str):
    """Get user subscription information"""
    subscription = get_user_subscription(user_id)
    
    return {
        "subscription": subscription,
        "stats": {
            "user_id": user_id,
            "plan": subscription["plan"],
            "status": subscription["status"],
            "features_used_today": subscription["features_used_today"]
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/subscription/trial")
async def start_trial(request: TrialRequest):
    """Start trial subscription"""
    try:
        subscription = get_user_subscription(request.user_id)
        
        subscription["plan"] = request.plan
        subscription["status"] = SubscriptionStatus.TRIAL
        subscription["trial_end_date"] = (datetime.now() + timedelta(days=request.trial_days)).isoformat()
        subscription["features_used_today"] = {}
        subscription["last_reset_date"] = datetime.now().date().isoformat()
        
        return {
            "message": f"Started {request.trial_days}-day trial for {request.plan}",
            "subscription": subscription,
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
                "features": ["Unlimited chat", "Brain Mode", "Timeline-aware", "Scene understanding", "Voice assistant"],
                "description": "Enhanced AI features"
            },
            "power_plus": {
                "price": 15,
                "features": ["All Pro features", "Knowledge graphs", "Cross-video RAG", "Real-time explain"],
                "description": "Advanced learning tools"
            },
            "creator": {
                "price": 25,
                "features": ["All Power+ features", "Question generator", "Comment analyzer", "SEO optimizer"],
                "description": "Creator tools and analytics"
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/videos")
async def list_processed_videos():
    """List all processed videos"""
    videos = []
    for video_id, data in processed_videos.items():
        videos.append({
            "video_id": video_id,
            "processed_at": data["processed_at"],
            "transcript_length": data["transcript_length"],
            "has_premium_features": bool(data.get("premium_features"))
        })
    
    return {
        "processed_videos": videos,
        "count": len(videos),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting YouTube RAG Chat Test Server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

