import asyncio
from controllers.legal_controller import ask_legal_question

# Test queries
INFORMATIONAL_QUERIES = [
    "What is bail under Pakistani law?",
    "Define wrongful restraint",
    "What are the requirements for marriage registration?",
    "Explain section 302 of PPC"
]

DECISION_QUERIES = [
    "I received a wrongful traffic ticket, what should I do?",
    "My employer is not paying salary, what are my rights?",
    "I want to file for divorce, what is the procedure?",
    "Someone cheated me, how can I get justice?"
]

async def test_routing_integration():
    """Test the integrated routing in /legal/ask endpoint"""
    print("Testing Integrated Routing in /legal/ask...")

    # Test informational queries
    print("\nTesting informational queries (should use RAG flow):")
    for query in INFORMATIONAL_QUERIES[:2]:
        print(f"\nQuery: {query}")
        result = await ask_legal_question(query)

        if result['success']:
            data = result['data']
            processing_path = data.get('processing_path', 'unknown')
            print(f"Processing path: {processing_path}")
            print(f"Response length: {len(data.get('result', ''))} characters")

            # Should be informational or not specified (default RAG)
            assert processing_path in ['informational', None], f"Expected informational path, got {processing_path}"
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

    # Test decision queries
    print("\nTesting decision queries (should use decision-making flow):")
    for query in DECISION_QUERIES[:2]:
        print(f"\nQuery: {query}")
        result = await ask_legal_question(query)

        if result['success']:
            data = result['data']
            processing_path = data.get('processing_path', 'unknown')
            response_type = data.get('response_type', 'unknown')
            print(f"Processing path: {processing_path}")
            print(f"Response type: {response_type}")
            print(f"Response length: {len(data.get('result', ''))} characters")

            # Should be decision_making
            assert processing_path == 'decision_making', f"Expected decision_making path, got {processing_path}"
            assert response_type == 'decision_advice', f"Expected decision_advice type, got {response_type}"
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")

    print("\n✅ Routing integration test completed!")

async def test_performance():
    """Test performance of the routing system"""
    import time

    print("\nTesting Performance...")

    query = "What should I do if I get a traffic ticket?"
    times = []

    # Test multiple times to check caching performance
    for i in range(3):
        start_time = time.time()
        result = await ask_legal_question(query)
        end_time = time.time()

        response_time = end_time - start_time
        times.append(response_time)
        print(".2f")

        assert result['success'], f"Query {i+1} failed"
        assert result['data']['processing_path'] == 'decision_making', f"Wrong path for query {i+1}"

    avg_time = sum(times) / len(times)
    print(".2f")

    # Should be fast (under 4 seconds average, much faster on cached calls)
    assert avg_time < 4.0, f"Average response time too slow: {avg_time:.2f}s"

    print("✅ Performance test passed!")

async def main():
    """Run all integration tests"""
    print("Running Integration Tests...")

    try:
        await test_routing_integration()
        await test_performance()

        print("\nAll integration tests passed!")

    except Exception as e:
        print(f"\nIntegration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())