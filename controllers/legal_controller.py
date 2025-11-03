import asyncio
from agent.legal_agent import SmartLegalAssistant
from controllers.decision_controller import process_decision_query

# Initialize the assistant
assistant = SmartLegalAssistant()

async def ask_legal_question(question: str, context=None):
    """Process a legal question using the smart assistant with intelligent routing"""
    try:
        # Step 1: Fast intent detection (cached, lightweight)
        from agent.decision_agents import DecisionIntentDetectionAgent
        from agent.legal_agent import OpenRouterLLM

        # Create a lightweight intent detector for fast classification
        llm = OpenRouterLLM()
        intent_detector = DecisionIntentDetectionAgent(llm)

        # Detect intent - this is cached and very fast (< 0.1s for cached results)
        # For repeated queries, this will be instant due to caching
        intent_result = intent_detector.detect_intent(question)

        # Step 2: Route based on intent
        if intent_result['intent'] == 'DECISION_MAKING':
            # Route to decision-making flow
            print(f"Routing to decision-making flow (confidence: {intent_result['confidence']})")
            decision_result = await process_decision_query(question, context)

            if decision_result['success']:
                # Return decision-making response
                return {
                    "success": True,
                    "data": {
                        "result": decision_result['data']['response'],
                        "intent_analysis": decision_result['data']['intent_analysis'],
                        "processing_path": "decision_making",
                        "query_id": decision_result['data']['query_id'],
                        "response_type": "decision_advice"
                    }
                }
            else:
                # Fallback to regular RAG if decision flow fails
                print("Decision flow failed, falling back to RAG")
                pass  # Continue to regular flow

        # Step 3: Regular RAG flow for informational queries or fallback
        print(f"Using RAG flow (intent: {intent_result['intent']})")
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