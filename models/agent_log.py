from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from sqlalchemy.sql import func
from config.database import Base
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum
import uuid



class AgentType(Enum):
    """Enum for different agent types in the system."""
    ENUM_HANDLING = "enum_agent"
    SCHEMA = "schema_agent"
    SELECT_RELEVANT_COLUMN = "select_column_agent"
    SELECT_TABLE = "select_table_agent"
    SQL_QUERY = "sql_query_agent"

class AgentLog(Base):
    __tablename__ = "agent_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    prompt_id = Column(String, nullable=True, index=True)  # Track ask-ai requests
    agent_name = Column(String, nullable=False, index=True)  # Will store AgentType.value
    input_data = Column(Text, nullable=True)
    output_data = Column(Text, nullable=True)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    original_model = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_agent_model', 'agent_name', 'original_model'),
        Index('idx_timestamp_desc', timestamp.desc()),
        Index('idx_prompt_id', 'prompt_id'),
    )


class AgentLogCreate(BaseModel):
    agent_name: str  # Will be AgentType.value
    prompt_id: Optional[str] = None
    input_data: Optional[str] = None
    output_data: Optional[str] = None
    input_tokens: int
    output_tokens: int
    original_model: str


class AgentLogResponse(BaseModel):
    id: str
    prompt_id: Optional[str]
    agent_name: str
    input_data: Optional[str]
    output_data: Optional[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int  # Calculated property
    original_model: str
    timestamp: datetime
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    class Config:
        orm_mode = True


class AgentLogFilter(BaseModel):
    agent_name: Optional[str] = None  # Can be AgentType.value
    prompt_id: Optional[str] = None  # Filter by ask-ai request ID
    original_model: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    page: int = 1
    page_size: int = 50