import asyncio
from agent.decision_agents import DecisionIntentDetectionAgent, DecisionMakingAgent, LegalDecisionOrchestrator
from agent.langgraph_flow import LegalDecisionFlow
from agent.legal_agent import OpenRouterLLM

# Test data
INFORMATIONAL_QUERIES = [
    "What is bail under Pakistani law?",
    "Define wrongful restraint",
    "What are the requirements for marriage registration?",
    "Explain section 302 of PPC",
    "What is the penalty for traffic violation?"
]

DECISION_QUERIES = [
    "I received a wrongful traffic ticket, what should I do?",
    "My employer is not paying salary, what are my rights?",
    "I want to file for divorce, what is the procedure?",
    "Someone cheated me, how can I get justice?",
    "I was falsely accused, what should I do next?"
]

def test_decision_intent_detection():
    """Test the intent detection agent"""
    print("Testing Decision Intent Detection...")

    llm = OpenRouterLLM()
    detector = DecisionIntentDetectionAgent(llm)

    # Test informational queries
    print("\nTesting informational queries:")
    for query in INFORMATIONAL_QUERIES[:3]:
        result = detector.detect_intent(query)
        print(f"Query: {query[:50]}...")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']})")
        assert result['intent'] == 'INFORMATIONAL', f"Failed for query: {query}"

    # Test decision queries
    print("\nTesting decision-making queries:")
    for query in DECISION_QUERIES[:3]:
        result = detector.detect_intent(query)
        print(f"Query: {query[:50]}...")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']})")
        assert result['intent'] == 'DECISION_MAKING', f"Failed for query: {query}"

    # Test caching
    print("\nTesting caching:")
    query = "What should I do if I get a ticket?"
    result1 = detector.detect_intent(query)
    result2 = detector.detect_intent(query)
    assert result1 == result2, "Caching failed"
    print("Caching works correctly")

    print("Intent detection tests passed!")

async def test_decision_making_agent():
    """Test the decision making agent"""
    print("\nTesting Decision Making Agent...")

    llm = OpenRouterLLM()
    decision_maker = DecisionMakingAgent(llm)

    query = "I got a traffic ticket for no reason, what should I do?"
    legal_context = []  # Empty context for testing

    result = await decision_maker.make_decision(query, legal_context)

    assert result['success'] is True, "Decision making failed"
    assert 'advice' in result, "No advice in result"
    assert len(result['advice']) > 0, "Empty advice"
    print(f"Generated advice length: {len(result['advice'])} characters")
    print("Decision making agent test passed!")

async def test_langgraph_flow():
    """Test the LangGraph flow"""
    print("\nTesting LangGraph Flow...")

    llm = OpenRouterLLM()
    flow = LegalDecisionFlow(llm)

    # Test decision-making query
    query = "I was wrongly accused of theft, what should I do?"
    result = await flow.process_query(query)

    assert result['success'] is True, "Flow processing failed"
    assert 'response' in result, "No response in result"
    assert 'intent' in result, "No intent in result"
    assert 'processing_path' in result, "No processing path"
    assert len(result['response']) > 0, "Empty response"
    print(f"Processing path: {result['processing_path']}")
    print("LangGraph flow test passed!")

async def test_full_integration():
    """Test the full integration with the legal agent"""
    print("\nTesting Full Integration...")

    from controllers.decision_controller import process_decision_query

    # Test decision-making query using the controller
    query = "My boss fired me without reason, what should I do?"
    result = await process_decision_query(query)

    assert result['success'] is True, "Decision processing failed"
    assert 'data' in result, "No data in response"
    data = result['data']
    assert 'response' in data, "No response in data"
    assert len(data['response']) > 0, "Empty response"
    assert 'intent_analysis' in data, "No intent analysis"
    print(f"Response length: {len(data['response'])} characters")
    print("Full integration test passed!")

async def main():
    """Run all tests"""
    print("Running Decision Flow Tests...")

    try:
        # Run synchronous tests
        test_decision_intent_detection()

        # Run asynchronous tests
        await test_decision_making_agent()
        await test_langgraph_flow()
        await test_full_integration()

        print("\nAll tests passed successfully!")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())