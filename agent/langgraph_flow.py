from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from typing import Dict, List, Any, TypedDict
from agent.decision_agents import LegalDecisionOrchestrator
from agent.legal_agent import SmartLegalAssistant
import asyncio

# Define the state structure
class LegalQueryState(TypedDict):
    question: str
    intent_result: Dict
    legal_context: List
    decision_advice: str
    final_response: str
    processing_path: str
    metadata: Dict

# ------------------ LangGraph Flow for Legal Decision Making ------------------
class LegalDecisionFlow:
    """LangGraph-based orchestration for legal decision making"""

    def __init__(self, llm):
        self.llm = llm
        self.orchestrator = LegalDecisionOrchestrator(llm)
        # Avoid circular import - create RAG assistant only when needed
        self._rag_assistant = None

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph flow"""
        workflow = StateGraph(LegalQueryState)

        # Add nodes
        workflow.add_node("intent_detection", self.intent_detection_node)
        workflow.add_node("retrieve_context", self.retrieve_context_node)
        workflow.add_node("decision_making", self.decision_making_node)
        workflow.add_node("rag_response", self.rag_response_node)
        workflow.add_node("format_response", self.format_response_node)

        # Add edges with routing logic
        workflow.add_edge("intent_detection", "retrieve_context")

        # Conditional routing based on intent
        workflow.add_conditional_edges(
            "retrieve_context",
            self.route_based_on_intent,
            {
                "decision_making": "decision_making",
                "informational": "rag_response"
            }
        )

        # Final formatting
        workflow.add_edge("decision_making", "format_response")
        workflow.add_edge("rag_response", "format_response")
        workflow.add_edge("format_response", END)

        # Set entry point
        workflow.set_entry_point("intent_detection")

        return workflow.compile()

    async def intent_detection_node(self, state: LegalQueryState) -> Dict:
        """Node for intent detection"""
        question = state["question"]

        # Use the orchestrator's intent detection
        intent_result = self.orchestrator.intent_detector.detect_intent(question)

        return {
            "intent_result": intent_result,
            "processing_path": "decision_making" if intent_result['intent'] == 'DECISION_MAKING' else "informational"
        }

    async def retrieve_context_node(self, state: LegalQueryState) -> Dict:
        """Node for retrieving legal context"""
        question = state["question"]

        # Use existing RAG system to get context
        try:
            # Get domain and collections using Gemini classification
            from agent.legal_agent import SmartLegalAssistant
            assistant = SmartLegalAssistant()

            # Use the classification agent to get domains
            classification = assistant.classification_agent.classify(question)
            all_domains = classification['all_domains']

            # Get collections for all relevant domains
            from agent.legal_agent import get_all_collections
            collections_to_search = []
            for dom in all_domains:
                domain_collections = [coll for coll in get_all_collections() if dom in coll]
                collections_to_search.extend(domain_collections)

            # Remove duplicates
            collections_to_search = list(set(collections_to_search))

            if not collections_to_search:
                collections_to_search = get_all_collections()

            # Retrieve documents using hybrid search
            all_docs = []
            for collection_name in collections_to_search[:3]:  # Limit to 3 collections for performance
                from agent.legal_agent import get_collection_retriever
                retriever = get_collection_retriever(collection_name, k=3)
                if retriever:
                    docs = await asyncio.get_event_loop().run_in_executor(
                        None, retriever.get_relevant_documents, question
                    )
                    all_docs.extend(docs)

            # Deduplicate
            unique_docs = self._deduplicate_docs(all_docs)

            return {
                "legal_context": unique_docs[:8]  # Limit to 8 docs
            }

        except Exception as e:
            print(f"Context retrieval error: {e}")
            return {
                "legal_context": []
            }

    async def decision_making_node(self, state: LegalQueryState) -> Dict:
        """Node for decision making"""
        question = state["question"]
        legal_context = state.get("legal_context", [])

        # Use decision making agent
        decision_result = await self.orchestrator.decision_maker.make_decision(question, legal_context)

        return {
            "decision_advice": decision_result.get('advice', ''),
            "metadata": {
                "context_used": decision_result.get('context_used', 0),
                "method": decision_result.get('method', 'unknown'),
                "success": decision_result.get('success', False)
            }
        }

    async def rag_response_node(self, state: LegalQueryState) -> Dict:
        """Node for standard RAG response"""
        question = state["question"]
        legal_context = state.get("legal_context", [])

        # Skip RAG processing to avoid rate limits - just return a simple response
        return {
            "final_response": f"**Legal Information Query**\n\n{question}\n\n*Note: Full RAG processing temporarily disabled to avoid API rate limits.*",
            "metadata": {
                "source_documents": 0,
                "searched_collections": 0,
                "detected_domain": "general",
                "rate_limit_avoided": True
            }
        }
    @property
    def rag_assistant(self):
        """Lazy load RAG assistant to avoid circular imports"""
        if self._rag_assistant is None:
            # Import here to avoid circular dependency
            from agent.legal_agent import SmartLegalAssistant
            # Create a fresh instance without decision flow to avoid recursion
            assistant = SmartLegalAssistant.__new__(SmartLegalAssistant)
            assistant.llm = self.llm
            from agent.legal_agent import QueryReformulationAgent, QueryClassificationAgent, RerankingAgent, QueryCache
            assistant.reformulation_agent = QueryReformulationAgent(assistant.llm)
            assistant.classification_agent = QueryClassificationAgent(assistant.llm)
            assistant.reranking_agent = RerankingAgent(assistant.llm)
            assistant.query_cache = QueryCache()
            assistant.decision_flow = None
            assistant.decision_enabled = False
            from langchain_core.prompts import PromptTemplate
            assistant.prompt = PromptTemplate(
                template="""You are a professional Pakistan AI Legal Assistant. Use ONLY the CONTEXT provided to answer the QUESTION.

**INSTRUCTIONS:**
1. Provide a clear, well-structured answer using proper legal terminology
2. Format your response with clear headings, bullet points, and proper spacing
3. If listing multiple items, use proper numbering or bullet points
4. Always reference the specific sections or provisions mentioned in the context
5. If information is incomplete, state what information is available
6. Use formal legal language appropriate for Pakistan's legal system
7. Mention which legal domain this information comes from
8. Consider the conversation history for context, but prioritize the legal context provided

**CONVERSATION HISTORY:**
{history}

**CONTEXT FROM {domain} LAW:**
{context}

**QUESTION:**
{question}

**ANSWER:**
""",
                input_variables=["history", "context", "question", "domain"]
            )
            self._rag_assistant = assistant
        return self._rag_assistant

    async def format_response_node(self, state: LegalQueryState) -> Dict:
        """Node for final response formatting"""
        intent_result = state.get("intent_result", {})
        processing_path = state.get("processing_path", "unknown")

        if processing_path == "decision_making":
            advice = state.get("decision_advice", "")
            metadata = state.get("metadata", {})

            final_response = f"""# 🤖 **Legal Decision-Making Assistant**

## 📋 **Query Analysis**
**Intent Detected:** {intent_result.get('intent', 'UNKNOWN')}
**Confidence:** {intent_result.get('confidence', 'UNKNOWN')}
**Method:** {intent_result.get('method', 'unknown')}

## 💡 **Decision-Making Advice**

{advice}

---
**Processing Details:** Context used: {metadata.get('context_used', 0)} documents | Method: {metadata.get('method', 'unknown')}
"""

        else:
            # Informational response
            rag_response = state.get("final_response", "")
            metadata = state.get("metadata", {})

            final_response = f"""# 📚 **Legal Information Assistant**

## 📋 **Query Analysis**
**Intent Detected:** {intent_result.get('intent', 'UNKNOWN')}
**Confidence:** {intent_result.get('confidence', 'UNKNOWN')}

## 📖 **Legal Information**

{rag_response}

---
**Processing Details:** {metadata.get('source_documents', 0)} source documents | {metadata.get('searched_collections', 0)} collections searched
"""

        return {
            "final_response": final_response
        }

    def route_based_on_intent(self, state: LegalQueryState) -> str:
        """Route to decision making or informational path"""
        intent_result = state.get("intent_result", {})
        intent = intent_result.get('intent', 'INFORMATIONAL')

        if intent == 'DECISION_MAKING':
            return "decision_making"
        else:
            return "informational"

    def _deduplicate_docs(self, documents: List) -> List:
        """Remove duplicate documents"""
        seen = set()
        unique = []

        for doc in documents:
            content_hash = hash(doc.page_content[:200].lower().strip())
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(doc)

        return unique

    async def process_query(self, question: str) -> Dict:
        """Main entry point for processing queries"""
        initial_state = {
            "question": question,
            "intent_result": {},
            "legal_context": [],
            "decision_advice": "",
            "final_response": "",
            "processing_path": "",
            "metadata": {}
        }

        try:
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            return {
                "success": True,
                "response": result.get("final_response", ""),
                "intent": result.get("intent_result", {}),
                "processing_path": result.get("processing_path", ""),
                "metadata": result.get("metadata", {})
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Graph processing failed: {str(e)}",
                "response": f"**Error:** Unable to process your legal query due to: {str(e)}\n\nPlease try again or consult a legal professional.",
                "intent": {},
                "processing_path": "error",
                "metadata": {"error": str(e)}
            }