import json
import os
import asyncio
from datetime import datetime
from typing import Dict, Any
from helper.token_counter import count_tokens
from config.settings import settings

TOKEN_LOG_FILE = "token_usage.json"

def load_token_logs() -> list:
    """Load token usage logs from file"""
    if os.path.exists(TOKEN_LOG_FILE):
        try:
            with open(TOKEN_LOG_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_token_logs(logs: list):
    """Save token usage logs to file"""
    with open(TOKEN_LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

async def track_token_usage_async(query_id: str, input_text: str, output_text: str, model: str = None):
    """Asynchronously track token usage for a query"""
    if not model:
        model = settings.OPENROUTER_MODEL

    input_tokens = count_tokens(input_text, model)
    output_tokens = count_tokens(output_text, model)
    total_tokens = input_tokens + output_tokens

    token_data = {
        "query_id": query_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "timestamp": datetime.now().isoformat(),
        "model": model
    }

    # Load existing logs
    logs = load_token_logs()
    logs.append(token_data)

    # Save asynchronously (in a real app, this would be in a database)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, save_token_logs, logs)

def get_token_usage_by_query_id(query_id: str) -> Dict[str, Any]:
    """Get token usage for a specific query ID"""
    logs = load_token_logs()
    for log in logs:
        if log.get("query_id") == query_id:
            return log
    return None

def get_all_token_usage() -> list:
    """Get all token usage logs"""
    return load_token_logs()

def get_total_tokens_used() -> Dict[str, int]:
    """Get total tokens used across all queries"""
    logs = load_token_logs()
    total_input = sum(log.get("input_tokens", 0) for log in logs)
    total_output = sum(log.get("output_tokens", 0) for log in logs)
    total = sum(log.get("total_tokens", 0) for log in logs)

    return {
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_tokens": total,
        "total_queries": len(logs)
    }