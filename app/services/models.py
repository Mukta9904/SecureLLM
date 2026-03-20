from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone
import uuid

# --- API Models (Frontend <-> Backend) ---
class ChatRequest(BaseModel):
    message: str
    # If the frontend doesn't send a session_id, generate a new one automatically
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class SecurityDetail(BaseModel):
    scanner_name: str
    is_safe: bool
    risk_score: float
    triggers: List[str]

class ChatResponse(BaseModel):
    status: str
    bot_reply: Optional[str] = None
    security_log: SecurityDetail
    timestamp: datetime
    session_id: str

# --- Database Models (Backend <-> MongoDB) ---
class LogEntry(BaseModel):
    session_id: str
    user_input: str
    bot_response: Optional[str]
    is_safe: bool
    risk_score: float
    triggers: List[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ConfigUpdate(BaseModel):
    threshold: float
    model_folder: str 