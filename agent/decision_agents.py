import asyncio
import time
import hashlib
import json
import re
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import LLM
from langchain_core.documents import Document
from config.settings import settings
from agent.gemini_llm import gemini_llm

# ============================================================================
# OPTIMIZED MULTI-STAGE INTENT DETECTION (Minimal LLM calls)
# ============================================================================

class OptimizedIntentDetectionAgent:
    """
    Multi-stage intent detection to minimize LLM calls:
    Stage 1 (FASTEST): Regex patterns for clear cases
    Stage 2 (FAST): Comprehensive keyword scoring
    Stage 3 (FALLBACK): Only use LLM for ambiguous cases (~10% of queries)
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self.cache = {}
        self.cache_ttl = 3600
        self.llm_call_count = 0  # Track LLM usage
        self.total_queries = 0

        # ============================================================
        # STAGE 1: Regex Patterns (Highest Precision, Zero Cost)
        # ============================================================
        
        # Decision-making patterns
        self.decision_patterns = [
            # Seeking action/advice
            r'\b(should\s+i|what\s+should|how\s+(can\s+)?i|what\s+to\s+do|what\s+are\s+my\s+options)\b',
            r'\b(need\s+to|have\s+to|must|should)\s+(file|report|sue|appeal|complain|respond)',
            r'\b(step|procedure|process|action)\s+(plan|i\s+should\s+take)',
            # Legal remedies
            r'\b(remedy|compensation|damages|relief|recovery|redress)\b',
            r'\b(how\s+to\s+(file|report|appeal|challenge|fight))\b',
            # Personal situations
            r'\b(i\s+(was|got|received|am).+(ticket|fine|charge|accused|sued))\b',
            r'\b(against\s+me|wrongfully|unfairly|illegally)\b',
            # Rights and entitlements
            r'\b(what\s+are\s+my\s+rights|am\s+i\s+entitled|can\s+i\s+(claim|sue|file))\b',
        ]

        # Informational patterns
        self.informational_patterns = [
            # Pure definitions
            r'^(what\s+is|define|explain|describe|tell\s+me)\s+',
            r'\b(definition\s+of|meaning\s+of)\b',
            # Legal references
            r'\b(section|article|clause|provision|law\s+states|according\s+to|under\s+)',
            r'\b(penalty\s+for|punishment\s+for|punishment\s+is|fine\s+for)\b',
            # Requirements/eligibility
            r'\b(requirements?\s+for|eligibility|qualifications|who\s+can|when\s+can)\b',
            r'\b(what\s+is\s+the.*?(process|procedure|rule|law))\b',
            # List and information requests
            r'\b(can\s+you\s+(list|tell\s+me|give\s+me|provide|explain))\b',
            r'\b(list\s+(of|me)|what\s+are\s+the|types\s+of|categories\s+of)\b',
            # Rights and information seeking
            r'\b(what\s+are\s+my\s+rights|employee\s+rights|corporate\s+rights)\b',
        ]

        # ============================================================
        # STAGE 2: Keyword Sets (Moderate Precision, Instant)
        # ============================================================
        
        self.decision_keywords = {
            # Action verbs
            'should', 'could', 'can i', 'do i', 'am i', 'will i',
            # Guidance seeking
            'advice', 'recommend', 'suggest', 'guidance', 'help',
            # Procedural
            'file', 'appeal', 'report', 'sue', 'claim', 'complaint',
            'challenge', 'fight', 'defend', 'respond', 'reply',
            # Remedy seeking
            'remedy', 'compensation', 'damages', 'relief', 'recover',
            # Rights awareness
            'rights', 'entitled', 'entitled to', 'my rights', 'legal right',
            # Problem resolution
            'resolve', 'handle', 'deal with', 'what to do', 'how to',
            'next steps', 'procedure', 'process', 'action plan',
            # Situations
            'ticket', 'fine', 'charge', 'accused', 'sued', 'against me',
            'wrongful', 'unfair', 'illegal', 'violation', 'breach',
            'dispute', 'conflict', 'problem', 'issue',
            # Specific legal actions
            'divorce', 'custody', 'inheritance', 'inheritance dispute',
            'contract dispute', 'employment issue', 'bail', 'bail hearing',
        }

        self.informational_keywords = {
            # Definition and explanation
            'what is', 'define', 'meaning', 'definition', 'explain',
            'describe', 'difference between', 'tell me about',
            # Legal reference
            'section', 'article', 'clause', 'provision', 'act',
            'ordinance', 'law', 'legal', 'statute', 'code',
            # Factual questions
            'penalty', 'punishment', 'fine', 'jail', 'imprisonment',
            'requirements', 'eligibility', 'who can', 'when can',
            'where to', 'how much', 'how long', 'how many',
            # Informational patterns
            'explain the', 'what are the', 'list of', 'types of',
            'categories of', 'according to', 'under the', 'can you',
            # Information seeking
            'tell me', 'give me', 'provide', 'show me', 'help me understand',
            'list', 'explain', 'describe', 'what is',
            # General inquiry
            'information about', 'details about', 'about the', 'regarding',
        }

        # ============================================================
        # STAGE 3: LLM Prompt (Used Only for Ambiguous Cases)
        # ============================================================
        
        self.llm_prompt = PromptTemplate(
            template="""Classify this legal query in ONE word: INFORMATIONAL or DECISION

Query: {question}

Only respond with one word: INFORMATIONAL or DECISION""",
            input_variables=["question"]
        )

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def _is_cached(self, key: str) -> Optional[Dict]:
        """Check cache"""
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.cache_ttl:
                return entry['result']
            else:
                del self.cache[key]
        return None

    def _cache_result(self, key: str, result: Dict):
        """Cache result"""
        self.cache[key] = {'result': result, 'timestamp': time.time()}

    # ====================================================================
    # STAGE 1: Regex Pattern Matching
    # ====================================================================
    
    def _stage1_regex_detection(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Stage 1: Ultra-fast regex pattern matching
        Returns: (intent, confidence) or None if no clear match
        """
        query_lower = query.lower()

        # Check decision patterns
        for pattern in self.decision_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return ("DECISION_MAKING", 0.95)  # Highest confidence

        # Check informational patterns
        for pattern in self.informational_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return ("INFORMATIONAL", 0.95)

        return None  # No clear pattern match

    # ====================================================================
    # STAGE 2: Keyword Scoring
    # ====================================================================
    
    def _stage2_keyword_scoring(self, query: str) -> Tuple[str, float]:
        """
        Stage 2: Keyword-based scoring
        Returns: (intent, confidence)
        """
        query_lower = query.lower()
        
        decision_score = sum(1 for kw in self.decision_keywords if kw in query_lower)
        info_score = sum(1 for kw in self.informational_keywords if kw in query_lower)

        # Calculate confidence based on score difference
        max_score = max(decision_score, info_score)
        if max_score == 0:
            return ("UNKNOWN", 0.3)  # No keywords found

        if decision_score > info_score:
            # Higher decision score
            confidence = 0.5 + (min(decision_score / 10, 0.4))
            return ("DECISION_MAKING", confidence)
        elif info_score > decision_score:
            # Higher info score
            confidence = 0.5 + (min(info_score / 10, 0.4))
            return ("INFORMATIONAL", confidence)
        else:
            # Equal scores
            return ("UNKNOWN", 0.4)

    # ====================================================================
    # STAGE 3: LLM Fallback (Only for Ambiguous Cases)
    # ====================================================================
    
    def _stage3_llm_classification(self, query: str) -> Tuple[str, float]:
        """
        Stage 3: LLM-based classification (Only when stages 1-2 are uncertain)
        Returns: (intent, confidence)
        """
        try:
            self.llm_call_count += 1
            prompt = self.llm_prompt.format(question=query)
            response = gemini_llm._call(prompt).strip().upper()

            intent = "DECISION_MAKING" if "DECISION" in response else "INFORMATIONAL"
            return (intent, 0.75)  # Medium confidence from LLM

        except Exception as e:
            print(f"⚠️ LLM classification failed: {e}")
            return ("INFORMATIONAL", 0.4)  # Safe fallback

    # ====================================================================
    # MAIN: Multi-Stage Detection Pipeline
    # ====================================================================
    
    def detect_intent(self, query: str) -> Dict:
        """
        Multi-stage intent detection with progressive fallback
        Minimizes LLM calls through cascading detection stages
        """
        self.total_queries += 1
        
        # Check cache
        cache_key = self._get_cache_key(query)
        cached = self._is_cached(cache_key)
        if cached:
            return cached

        # ===== STAGE 1: Regex Patterns =====
        result = self._stage1_regex_detection(query)
        if result:
            intent, confidence = result
            print(f"🔍 Intent detected via REGEX: {intent} (confidence: {confidence})")
            output = {
                'intent': intent,
                'confidence': 'HIGH',
                'method': 'regex_pattern',
                'reason': 'Clear pattern match detected',
                'llm_used': False
            }
            self._cache_result(cache_key, output)
            return output

        # ===== STAGE 2: Keyword Scoring =====
        intent, confidence = self._stage2_keyword_scoring(query)
        if confidence >= 0.65:  # Lower threshold to catch more queries with keyword detection
            print(f"🔍 Intent detected via KEYWORDS: {intent} (confidence: {confidence:.2f})")
            output = {
                'intent': intent,
                'confidence': 'HIGH' if confidence >= 0.8 else 'MEDIUM',
                'method': 'keyword_scoring',
                'reason': f'Keyword analysis (confidence: {confidence:.2f})',
                'llm_used': False
            }
            self._cache_result(cache_key, output)
            return output

        # ===== STAGE 3: LLM Fallback (Only ~10-15% of queries) =====
        print(f"🔍 Intent detection falling back to LLM for ambiguous query")
        intent, confidence = self._stage3_llm_classification(query)
        print(f"🤖 LLM Intent result: {intent} (confidence: {confidence})")
        output = {
            'intent': intent,
            'confidence': 'MEDIUM',
            'method': 'llm_classification',
            'reason': 'LLM-based classification for ambiguous query',
            'llm_used': True
        }
        self._cache_result(cache_key, output)
        return output

    def get_statistics(self) -> Dict:
        """Get detection statistics"""
        return {
            'total_queries': self.total_queries,
            'llm_calls': self.llm_call_count,
            'llm_usage_percentage': (self.llm_call_count / max(self.total_queries, 1)) * 100,
            'cache_size': len(self.cache)
        }


# Keep the old class name for backwards compatibility but use the new implementation
class DecisionIntentDetectionAgent(OptimizedIntentDetectionAgent):
    """Backwards compatible wrapper"""
    pass

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