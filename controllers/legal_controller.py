from agent.legal_agent import SmartLegalAssistant

# Initialize the assistant
assistant = SmartLegalAssistant()

def ask_legal_question(question: str):
    """Process a legal question using the smart assistant"""
    try:
        result = assistant(question)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
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