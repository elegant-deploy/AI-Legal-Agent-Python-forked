import asyncio
from agent.langgraph_flow import LegalDecisionFlow
from agent.legal_agent import primary_llm

# Initialize the decision flow
llm = primary_llm
decision_flow = LegalDecisionFlow(llm)

async def process_decision_query(question: str, context=None):
    """Process a legal query using the decision-making flow"""
    try:
        result = await decision_flow.process_query(question)

        return {
            "success": result.get('success', False),
            "data": {
                "response": result.get('response', ''),
                "intent_analysis": result.get('intent', {}),
                "processing_path": result.get('processing_path', 'unknown'),
                "metadata": result.get('metadata', {}),
                "query_id": "decision_" + str(hash(question))[:8]
            }
        }
    except Exception as e:
        import traceback
        print(f"Error in process_decision_query: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "data": {
                "response": f"**Error in Decision Processing**\n\nUnable to process your query due to: {str(e)}\n\nPlease try again or use the standard legal assistant.",
                "intent_analysis": {},
                "processing_path": "error",
                "metadata": {"error": str(e)},
                "query_id": "error_" + str(hash(question))[:8]
            }
        }

def get_decision_flow_status():
    """Get status of the decision-making flow"""
    try:
        return {
            "success": True,
            "data": {
                "decision_flow_enabled": True,
                "llm_model": llm.model,
                "status": "operational"
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": {
                "decision_flow_enabled": False,
                "status": "error"
            }
        }