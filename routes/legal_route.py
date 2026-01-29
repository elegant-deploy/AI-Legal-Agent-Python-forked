import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.models import LegalQueryRequest, LegalQueryResponse, IngestRequest, IngestResponse, SystemInfoResponse
from controllers.legal_controller import ask_legal_question, ask_legal_question_stream, get_system_info
from controllers.decision_controller import process_decision_query, get_decision_flow_status
from controllers.ingest_controller import ingest_pdf
from helper.token_tracker import get_token_usage_by_query_id, get_all_token_usage, get_total_tokens_used

router = APIRouter()

@router.post("/ask", response_model=LegalQueryResponse)
async def ask_legal(payload: LegalQueryRequest):
    """Ask a legal question to the AI assistant"""
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = await ask_legal_question(payload.question.strip(), payload.context)
    return LegalQueryResponse(**result)


@router.post("/ask/stream")
async def ask_legal_stream(payload: LegalQueryRequest):
    """Ask a legal question; stream the answer as Server-Sent Events. First tokens in ~2–3s instead of full 12s."""
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def sse_events():
        async for event in ask_legal_question_stream(payload.question.strip(), payload.context):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        sse_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_legal_document(payload: IngestRequest):
    """Ingest a PDF document into the vector database"""
    import os
    if not os.path.exists(payload.pdf_path):
        raise HTTPException(status_code=400, detail=f"PDF file not found: {payload.pdf_path}")

    try:
        success, collection_name, message = ingest_pdf(payload.pdf_path, payload.force)
        return IngestResponse(
            success=success,
            collection_name=collection_name,
            message=message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest error: {str(e)}")


@router.get("/collections", response_model=SystemInfoResponse)
async def get_collections():
    """Get information about available collections"""
    result = get_system_info()
    return SystemInfoResponse(**result)


@router.get("/system-info", response_model=SystemInfoResponse)
async def system_info():
    """Get system information including collections and domains"""
    result = get_system_info()
    return SystemInfoResponse(**result)


@router.get("/tokens/{query_id}")
async def get_query_tokens(query_id: str):
    """Get token usage for a specific query ID"""
    token_data = get_token_usage_by_query_id(query_id)
    if token_data:
        return {"success": True, "data": token_data}
    else:
        raise HTTPException(status_code=404, detail="Query ID not found")


@router.get("/tokens")
async def get_all_tokens():
    """Get all token usage records"""
    tokens = get_all_token_usage()
    return {"success": True, "data": tokens, "total": len(tokens)}


@router.get("/tokens/summary")
async def get_token_summary():
    """Get total token usage summary"""
    summary = get_total_tokens_used()
    return {"success": True, "data": summary}


@router.post("/decide", response_model=LegalQueryResponse)
async def make_decision(payload: LegalQueryRequest):
    """Make legal decisions and provide actionable advice"""
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = await process_decision_query(payload.question.strip(), payload.context)
    return LegalQueryResponse(**result)


@router.get("/decision-status")
async def decision_status():
    """Get status of the decision-making flow"""
    result = get_decision_flow_status()
    return result