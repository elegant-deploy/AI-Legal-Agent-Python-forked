import asyncio
import json
import time
from agent.legal_agent import SmartLegalAssistant
from agent.gemini_llm import gemini_llm
from controllers.decision_controller import process_decision_query

# Initialize the assistant
assistant = SmartLegalAssistant()

async def ask_legal_question(question: str, context=None):
    """Process a legal question using the smart assistant with intelligent routing"""
    try:
        # Step 1: Fast intent detection (cached, lightweight)
        from agent.decision_agents import DecisionIntentDetectionAgent
        from agent.legal_agent import primary_llm

        # Create a lightweight intent detector for fast classification
        llm = primary_llm
        intent_detector = DecisionIntentDetectionAgent(llm)

        # Detect intent - this is cached and very fast (< 0.1s for cached results)
        # For repeated queries, this will be instant due to caching
        intent_result = intent_detector.detect_intent(question)
        print(f"🎯 Intent Detection Result: {intent_result['intent']} via {intent_result['method']} (LLM used: {intent_result.get('llm_used', False)})")

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
        # Use the sync method which internally handles async operations
        result = assistant(question, context)
        return {
            "success": True,
            "data": result
        }
    except Exception as e:
        import traceback
        print(f"Error in ask_legal_question: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e)
        }


async def ask_legal_question_stream(question: str, context=None):
    """Run RAG then stream Gemini response. Yields SSE-style dicts: start, chunk, done.
    If cached or early return, yields single 'result' then stops."""
    t0 = time.perf_counter()
    print(f"[{time.perf_counter()-t0:.2f}s] 📥 STREAM REQUEST RECEIVED")
    try:
        from agent.decision_agents import DecisionIntentDetectionAgent
        from agent.legal_agent import primary_llm
        intent_detector = DecisionIntentDetectionAgent(primary_llm)
        intent_result = intent_detector.detect_intent(question)
        print(f"[{time.perf_counter()-t0:.2f}s] 🎯 Intent: {intent_result['intent']} ({intent_result.get('method', '')})")
        if intent_result["intent"] == "DECISION_MAKING":
            decision_result = await process_decision_query(question, context)
            if decision_result.get("success"):
                yield {"type": "result", "success": True, "data": decision_result}
                return
        print(f"[{time.perf_counter()-t0:.2f}s] 📚 Starting RAG prepare (domain + retrieval + prompt)...")
        prep = await assistant.prepare_rag_for_stream(question, context, start_time=t0)
        if prep.get("_cached"):
            yield {"type": "result", "success": True, "data": {"data": prep["_cached"]}}
            return
        if prep.get("_early"):
            yield {"type": "result", "success": True, "data": {"data": prep["_early"]}}
            return
        query_id = prep["query_id"]
        print(f"[{time.perf_counter()-t0:.2f}s] 🤖 RAG done. Calling LLM stream (prompt {len(prep['enhanced_prompt'])} chars)...")
        yield {"type": "start", "query_id": query_id}
        full_text = []
        first_chunk = True
        for chunk in gemini_llm.stream_content(prep["enhanced_prompt"], start_time=t0):
            if first_chunk:
                print(f"[{time.perf_counter()-t0:.2f}s] 📤 FIRST CHUNK FROM LLM (TTFT)")
                first_chunk = False
            full_text.append(chunk)
            yield {"type": "chunk", "text": chunk}
        result_text = "".join(full_text)
        print(f"[{time.perf_counter()-t0:.2f}s] ✅ STREAM DONE (total {len(result_text)} chars)")
        yield {
            "type": "done",
            "query_id": query_id,
            "result": result_text,
            "sources_count": len(prep["top_docs"]),
        }
    except Exception as e:
        print(f"[{time.perf_counter()-t0:.2f}s] ❌ STREAM ERROR: {e}")
        yield {"type": "error", "error": str(e)}


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