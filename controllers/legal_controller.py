import asyncio
from agent.legal_agent import SmartLegalAssistant

# Initialize the assistant
assistant = SmartLegalAssistant()

async def ask_legal_question(question: str, context=None):
    """Process a legal question using the smart assistant"""
    try:
        # Use the async method directly since we're in an async context
        result = await assistant.__call_async(question, context)
        return {
            "success": True,
            "data": result
        }
    except AttributeError as e:
        if "'SmartLegalAssistant' object has no attribute '__call_async'" in str(e):
            # Fallback to sync method if async method doesn't exist
            print("Falling back to synchronous method...")
            result = assistant(question, context)
            return {
                "success": True,
                "data": result
            }
        else:
            raise
    except Exception as e:
        import traceback
        print(f"Error in ask_legal_question: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e)
        }

def get_system_info():
    """Get information about available collections"""
    from agent.legal_agent import get_all_collections, DOMAIN_KEYWORDS

    try:
        collections = get_all_collections()
        if not collections:
            return {
                "success": False,
                "message": "No collections found. Please ingest some legal documents first."
            }

        # Group collections by domain
        domain_info = {}
        for domain in DOMAIN_KEYWORDS.keys():
            domain_collections = [coll for coll in collections if domain in coll]
            if domain_collections:
                domain_info[domain.upper()] = domain_collections

        return {
            "success": True,
            "data": {
                "total_collections": len(collections),
                "collections": collections,
                "domains": domain_info,
                "supported_domains": list(DOMAIN_KEYWORDS.keys())
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }