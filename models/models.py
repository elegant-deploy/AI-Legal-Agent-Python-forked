from pydantic import BaseModel
from typing import List, Any, Optional

# Legal Assistant Models
class LegalQueryRequest(BaseModel):
    question: str

class LegalQueryResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None

class IngestRequest(BaseModel):
    pdf_path: str
    force: Optional[bool] = False

class IngestResponse(BaseModel):
    success: bool
    collection_name: Optional[str] = None
    message: str

class SystemInfoResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    message: Optional[str] = None
    error: Optional[str] = None

class TokenUsage(BaseModel):
    query_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: str
    model: str