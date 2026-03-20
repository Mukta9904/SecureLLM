from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone
from app.services.models import ChatRequest, ConfigUpdate, ChatResponse, SecurityDetail, LogEntry
from app.security.scanner import SecureScanner
from app.services.gemini import get_gemini_response # Assuming you have this
from app.services.database import chat_collection, sessions_collection, settings_collection

app = FastAPI(title="SecureLLM System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scanner = SecureScanner()

# --- THE ZERO-LATENCY THRESHOLD TRICK ---
GLOBAL_THRESHOLD = 0.45
GLOBAL_MODEL = "LR_models" # Your default LR folder

@app.on_event("startup")
async def startup_event():
    """Load config from DB and load the ML model into RAM."""
    global GLOBAL_THRESHOLD, GLOBAL_MODEL
    setting = await settings_collection.find_one({"_id": "config"})
    
    if setting:
        GLOBAL_THRESHOLD = setting.get("threshold", 0.45)
        GLOBAL_MODEL = setting.get("model_folder", "LR_models")
    else:
        await settings_collection.insert_one({"_id": "config", "threshold": 0.45, "model_folder": "new_models"})
    
    print(f"⚙️ Boot Configuration -> Threshold: {GLOBAL_THRESHOLD} | Model: {GLOBAL_MODEL}")
    scanner.load_model_from_folder(GLOBAL_MODEL)


# --- SETTINGS ENDPOINTS ---
@app.get("/admin/settings/config")
async def get_config():
    return {"threshold": GLOBAL_THRESHOLD, "model_folder": GLOBAL_MODEL}

@app.post("/admin/settings/config")
async def update_config(update: ConfigUpdate):
    """Admin updates the threshold or model. Hot-reloads instantly."""
    global GLOBAL_THRESHOLD, GLOBAL_MODEL
    
    GLOBAL_THRESHOLD = update.threshold
    
    # Only hot-reload the `.pkl` files if the user actually changed the model dropdown!
    if GLOBAL_MODEL != update.model_folder:
        GLOBAL_MODEL = update.model_folder
        success = scanner.load_model_from_folder(GLOBAL_MODEL)
        if not success:
            return {"error": f"Failed to load models from {GLOBAL_MODEL}"}

    await settings_collection.update_one(
        {"_id": "config"}, 
        {"$set": {"threshold": GLOBAL_THRESHOLD, "model_folder": GLOBAL_MODEL}}, 
        upsert=True
    )
    return {"message": "Configuration applied instantly.", "config": update.dict()}

# --- 2. CHAT ENDPOINTS ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = datetime.now(timezone.utc)
    
    # Pass the ultra-fast global threshold to the scanner
    is_safe, risk_score, triggers = scanner.scan(request.message, threshold=GLOBAL_THRESHOLD)
    
    security_detail = SecurityDetail(
        scanner_name=f"SecureLLM-{GLOBAL_MODEL}",
        is_safe=is_safe,
        risk_score=risk_score,
        triggers=triggers
    )

    if is_safe:
        status = "success"
        bot_reply = get_gemini_response(request.message) # Pass history here if your gemini function supports it
    else:
        status = "blocked"
        bot_reply = "⚠️ Security Alert: Your prompt was blocked by the SecureLLM Firewall due to potential injection patterns."

    # A. Save the specific Security Log
    log_entry = LogEntry(
        session_id=request.session_id,
        user_input=request.message,
        bot_response=bot_reply,
        is_safe=is_safe,
        risk_score=risk_score,
        triggers=triggers,
        timestamp=start_time
    )
    await chat_collection.insert_one(log_entry.dict())

    # B. Update the Chat Session History
    bot_message_entry = {
        "role": "bot", 
        "content": bot_reply, 
        "timestamp": datetime.now(timezone.utc),
        "is_blocked": not is_safe
    }
    
    if not is_safe:
        # Attach the forensic data to the history so the UI can reconstruct the red box
        bot_message_entry["security_log"] = security_detail.dict()

    await sessions_collection.update_one(
        {"session_id": request.session_id},
        {"$push": {
            "messages": {"$each": [
                {"role": "user", "content": request.message, "timestamp": start_time},
                bot_message_entry
            ]}
        }},
        upsert=True
    )
    return ChatResponse(
        status=status,
        bot_reply=bot_reply,
        security_log=security_detail,
        timestamp=start_time,
        session_id=request.session_id
    )

@app.get("/chat/sessions/{session_id}")
async def get_session_history(session_id: str):
    """Frontend calls this on reload to get the chat history back."""
    session = await sessions_collection.find_one({"session_id": session_id})
    if not session:
        return {"messages": []}
    messages = session.get("messages", [])
    
    # --- TIMEZONE FIX ---
    for msg in messages:
        if "timestamp" in msg and msg["timestamp"].tzinfo is None:
            msg["timestamp"] = msg["timestamp"].replace(tzinfo=timezone.utc)
            
    return {"messages": messages}
# --- 3. ADMIN DASHBOARD ENDPOINTS ---
@app.get("/admin/stats")
async def get_dashboard_stats():
    """Returns REAL stats for the dashboard charts"""
    total_requests = await chat_collection.count_documents({})
    blocked_requests = await chat_collection.count_documents({"is_safe": False})
    
    # Calculate real injection rate
    injection_rate = 0.0
    if total_requests > 0:
        injection_rate = round((blocked_requests / total_requests) * 100, 1)

    # MongoDB Magic: Find the most common trigger words automatically
    pipeline = [
        {"$match": {"is_safe": False}},          # Only look at blocked prompts
        {"$unwind": "$triggers"},                # Unpack the list of triggers
        {"$group": {"_id": "$triggers", "count": {"$sum": 1}}}, # Count them
        {"$sort": {"count": -1}},                # Sort highest to lowest
        {"$limit": 4}                            # Get top 4
    ]
    top_triggers_cursor = chat_collection.aggregate(pipeline)
    top_patterns = [{"trigger": doc["_id"], "count": doc["count"]} async for doc in top_triggers_cursor]

    # Get recent logs
    recent_logs = await chat_collection.find().sort("timestamp", -1).limit(10).to_list(10)
    for log in recent_logs:
        log["_id"] = str(log["_id"])
        if "timestamp" in log and log["timestamp"].tzinfo is None:
            log["timestamp"] = log["timestamp"].replace(tzinfo=timezone.utc)

    return {
        "total_requests": total_requests,
        "blocked_count": blocked_requests,
        "safe_count": total_requests - blocked_requests,
        "injection_rate": f"{injection_rate}%",
        "top_patterns": top_patterns,
        "recent_logs": recent_logs
    }
    
@app.get("/chat/sessions")
async def list_all_sessions():
    """Returns a list of all chat sessions for the sidebar."""
    # Grab the most recent 20 sessions, but only fetch the very first message of each to use as the title
    cursor = sessions_collection.find({}, {"session_id": 1, "messages": {"$slice": 1}}).sort("_id", -1).limit(20)
    sessions = await cursor.to_list(length=20)
    
    result = []
    for s in sessions:
        title = "Empty Chat"
        if "messages" in s and len(s["messages"]) > 0:
            # Use the first 30 characters of the first message as the title
            title = s["messages"][0].get("content", "New Chat")[:30] + "..."
            
        result.append({
            "session_id": s.get("session_id"),
            "title": title
        })
        
    return {"sessions": result}