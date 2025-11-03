import asyncio
import time
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import LLM
from langchain_core.documents import Document
from config.settings import settings
from agent.gemini_llm import gemini_llm

# ------------------ Decision Intent Detection Agent ------------------
class DecisionIntentDetectionAgent:
    """Agent to detect whether a query requires decision-making or is purely informational"""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour TTL

        # Lightweight keyword-based detection for speed
        self.decision_keywords = {
            'should', 'what should i do', 'how to', 'what to do', 'recommend', 'advice',
            'action', 'steps', 'procedure', 'process', 'next steps', 'file', 'appeal',
            'complain', 'report', 'challenge', 'fight', 'defend', 'rights', 'remedy',
            'solution', 'resolve', 'handle', 'deal with', 'respond to', 'against me',
            'wrongful', 'unfair', 'illegal', 'violation', 'breach', 'penalty', 'fine',
            'ticket', 'citation', 'charge', 'accusation', 'complaint', 'lawsuit',
            'court', 'legal action', 'sue', 'claim', 'compensation', 'damages',
            'what are my rights', 'my rights', 'rights', 'entitled to', 'entitled',
            'can i', 'am i entitled', 'do i have right', 'what can i do',
            'file for', 'apply for', 'get divorce', 'divorce procedure', 'divorce process'
        }

        self.informational_keywords = {
            'what is', 'define', 'meaning', 'explain', 'describe', 'definition',
            'section', 'article', 'clause', 'provision', 'law states', 'according to',
            'under', 'penalty for', 'punishment for', 'requirements', 'eligibility'
        }

        self.prompt = PromptTemplate(
            template="""You are a legal intent classification expert. Analyze if the query requires decision-making advice or is purely informational.

**QUERY:** {question}

**CLASSIFICATION TASK:**
- INFORMATIONAL: Pure fact-finding, legal definitions, what the law says (e.g., "What is bail?")
- DECISION-MAKING: Requires advice on what to do, legal actions, remedies, procedures (e.g., "I got a ticket, what should I do?")

**RESPONSE FORMAT:**
Intent: [INFORMATIONAL or DECISION_MAKING]
Confidence: [HIGH/MEDIUM/LOW]
Reason: [brief explanation]

Return only the classification.""",
            input_variables=["question"]
        )

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _is_cached(self, key: str) -> Optional[Dict]:
        """Check if result is cached and not expired"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.cache_ttl:
                return entry['result']
            else:
                del self.cache[key]  # Remove expired
        return None

    def _cache_result(self, key: str, result: Dict):
        """Cache the result"""
        self.cache[key] = {
            'result': result,
            'timestamp': time.time()
        }

    def _lightweight_detect(self, query: str) -> Tuple[str, float]:
        """Fast keyword-based detection for common cases"""
        query_lower = query.lower().strip()

        decision_score = sum(1 for keyword in self.decision_keywords if keyword in query_lower)
        info_score = sum(1 for keyword in self.informational_keywords if keyword in query_lower)

        # Clear decision indicators
        if any(phrase in query_lower for phrase in ['what should i do', 'how to', 'what to do', 'should i']):
            return "DECISION_MAKING", 0.9

        # Clear informational indicators
        if query_lower.startswith(('what is', 'define', 'explain', 'what are')):
            return "INFORMATIONAL", 0.9

        # Score-based decision
        if decision_score > info_score:
            confidence = min(0.8, 0.5 + (decision_score * 0.1))
            return "DECISION_MAKING", confidence
        elif info_score > decision_score:
            confidence = min(0.8, 0.5 + (info_score * 0.1))
            return "INFORMATIONAL", confidence
        else:
            return "UNKNOWN", 0.5  # Need LLM classification

    def detect_intent(self, query: str) -> Dict:
        """Detect if query requires decision-making or is informational"""
        # Check cache first
        cache_key = self._get_cache_key(query)
        cached = self._is_cached(cache_key)
        if cached:
            return cached

        # Fast lightweight detection first
        intent, confidence = self._lightweight_detect(query)

        if confidence >= 0.8:
            # High confidence from keywords, use this
            result = {
                'intent': intent,
                'confidence': 'HIGH' if confidence >= 0.9 else 'MEDIUM',
                'method': 'keyword',
                'reason': f'Keyword-based detection (confidence: {confidence:.2f})'
            }
        else:
            # Use Gemini for uncertain cases (lightweight and free)
            try:
                prompt = self.prompt.format(question=query)
                response = gemini_llm._call(prompt)

                # Parse Gemini response
                intent = "INFORMATIONAL"  # default
                confidence = "MEDIUM"
                reason = "Gemini classification"

                for line in response.split('\n'):
                    line = line.strip()
                    if line.startswith('Intent:'):
                        intent_val = line.split(':', 1)[1].strip().upper()
                        if 'DECISION' in intent_val:
                            intent = 'DECISION_MAKING'
                        else:
                            intent = 'INFORMATIONAL'
                    elif line.startswith('Confidence:'):
                        confidence = line.split(':', 1)[1].strip().upper()
                    elif line.startswith('Reason:'):
                        reason = line.split(':', 1)[1].strip()

                result = {
                    'intent': intent,
                    'confidence': confidence,
                    'method': 'gemini',
                    'reason': reason
                }

            except Exception as e:
                # Fallback to keyword detection if Gemini fails
                result = {
                    'intent': intent if intent != "UNKNOWN" else "INFORMATIONAL",
                    'confidence': 'LOW',
                    'method': 'fallback',
                    'reason': f'Gemini error: {str(e)}'
                }

        # Cache result
        self._cache_result(cache_key, result)
        return result

# ------------------ Decision Making Agent ------------------
class DecisionMakingAgent:
    """Agent to provide structured legal decision-making advice"""

    def __init__(self, llm: LLM):
        self.llm = llm

        self.prompt = PromptTemplate(
            template="""You are an elite Pakistani Legal Decision-Making AI with extensive expertise in Pakistani law. Provide authoritative, actionable legal guidance based ONLY on the provided context.

**CRITICAL LEGAL REQUIREMENTS:**
1. **EVIDENCE-BASED ADVICE**: Every recommendation must cite specific Pakistani laws, sections, or acts
2. **CONSERVATIVE APPROACH**: Always err on the side of caution and recommend professional legal consultation
3. **STRUCTURED ANALYSIS**: Break down complex legal situations into clear, actionable steps
4. **PAKISTANI LEGAL SYSTEM**: Reference appropriate courts, procedures, and legal processes
5. **RISK ASSESSMENT**: Clearly identify potential consequences and mitigation strategies

**USER'S LEGAL SITUATION:**
{question}

**LEGAL CONTEXT FROM PAKISTAN LAW DATABASE:**
{context}

**MANDATORY RESPONSE STRUCTURE:**

### 🎯 **Legal Issue Analysis**
[Precise identification of the legal problem with relevant Pakistani law context]

### 📜 **Applicable Pakistani Laws & Sections**
[Cite specific acts, ordinances, sections with direct quotes where possible]

### ⚖️ **Your Legal Rights & Obligations**
[Clear explanation of user's legal position, rights, and responsibilities under Pakistani law]

### 🚀 **Recommended Legal Actions**
[Numbered step-by-step action plan with specific procedures and deadlines]

### ⚠️ **Critical Legal Warnings & Risks**
[Bullet-pointed list of potential consequences, pitfalls, and what to avoid]

### 🏛️ **Court Procedures & Timeline**
[Specific court processes, filing requirements, and realistic timeframes]

### 📞 **Professional Legal Consultation**
[Strong recommendation to consult qualified Pakistani lawyers with specialization details]

**LEGAL DISCLAIMER:**
This AI analysis is for informational purposes only and does not constitute legal advice. Pakistani law is complex and situation-specific. Always consult qualified legal professionals for personalized guidance.

**AUTHORITY LEVEL:** High confidence based on Pakistani legal database analysis.
""",
            input_variables=["question", "context"]
        )

        self.verification_prompt = PromptTemplate(
            template="""Verify the legal advice below for accuracy and completeness. Check against the provided context.

**ORIGINAL ADVICE:** {advice}

**LEGAL CONTEXT:** {context}

**VERIFICATION TASK:**
1. Are all legal citations accurate and relevant?
2. Does the advice align with Pakistani law?
3. Are there any missing important considerations?
4. Is the advice appropriately cautious?

**VERIFICATION RESULT:**
[APPROVED/MODIFIED/REJECTED]
[Brief explanation and any corrections needed]
""",
            input_variables=["advice", "context"]
        )

    def _extract_structured_info(self, context_docs: List[Document]) -> str:
        """Extract and format legal context for decision-making"""
        context_parts = []

        for doc in context_docs[:5]:  # Limit to top 5 docs
            collection = doc.metadata.get('collection_name', 'Unknown')
            page = doc.metadata.get('page_number', 'N/A')
            domain = doc.metadata.get('domain', 'general')

            context_parts.append(f"[{domain.upper()} LAW - {collection}]\n{doc.page_content}\n[Source: Page {page}]")

        return "\n\n".join(context_parts)

    def _verify_advice(self, advice: str, context: str) -> str:
        """Self-verify the generated advice"""
        try:
            prompt = self.verification_prompt.format(advice=advice, context=context)
            verification = self.llm._call(prompt)

            if "APPROVED" in verification.upper():
                return advice
            elif "MODIFIED" in verification.upper():
                # Extract modified advice if available
                lines = verification.split('\n')
                for i, line in enumerate(lines):
                    if "MODIFIED" in line.upper():
                        return '\n'.join(lines[i+1:]).strip()
            else:
                # Add verification note
                return advice + "\n\n⚠️ **VERIFICATION NOTE:** " + verification.split('\n', 1)[1].strip()

        except Exception as e:
            return advice + f"\n\n⚠️ **VERIFICATION NOTE:** Could not verify advice: {str(e)}"

    async def make_decision(self, question: str, legal_context: List[Document]) -> Dict:
        """Generate structured decision-making advice"""
        try:
            # Extract and format context
            formatted_context = self._extract_structured_info(legal_context)

            # Generate initial advice (skip verification to avoid rate limits)
            prompt = self.prompt.format(question=question, context=formatted_context)
            advice = self.llm._call(prompt)

            return {
                'success': True,
                'advice': advice,
                'context_used': len(legal_context),
                'method': 'structured_decision_making'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Decision making failed: {str(e)}',
                'advice': f"**Error in Decision Analysis**\n\nUnable to provide decision-making advice due to: {str(e)}\n\nPlease consult a qualified legal professional for personalized advice.",
                'context_used': 0,
                'method': 'error_fallback'
            }

# ------------------ Legal Decision Orchestrator ------------------
class LegalDecisionOrchestrator:
    """Orchestrates the decision-making flow with routing"""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.intent_detector = DecisionIntentDetectionAgent(llm)
        self.decision_maker = DecisionMakingAgent(llm)

    async def process_query(self, question: str, legal_context: List[Document]) -> Dict:
        """Main orchestration method"""
        start_time = time.time()

        # Step 1: Detect intent
        intent_result = self.intent_detector.detect_intent(question)

        result = {
            'query': question,
            'intent_analysis': intent_result,
            'processing_time_seconds': 0,
            'path_taken': 'informational' if intent_result['intent'] == 'INFORMATIONAL' else 'decision_making'
        }

        if intent_result['intent'] == 'INFORMATIONAL':
            # Use existing RAG flow - return context for standard processing
            result.update({
                'response_type': 'informational_rag',
                'message': 'Query classified as informational - proceeding with standard RAG retrieval',
                'legal_context': legal_context
            })

        else:
            # Decision-making path
            decision_result = await self.decision_maker.make_decision(question, legal_context)

            result.update({
                'response_type': 'decision_making',
                'decision_advice': decision_result.get('advice', ''),
                'context_used': decision_result.get('context_used', 0),
                'method': decision_result.get('method', 'unknown'),
                'success': decision_result.get('success', False)
            })

        # Calculate processing time
        result['processing_time_seconds'] = time.time() - start_time

        return result