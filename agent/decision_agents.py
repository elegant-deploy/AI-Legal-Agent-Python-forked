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
from agent.gemini_llm import perplexity_llm

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
            response = perplexity_llm._call(prompt).strip().upper()

            intent = "DECISION_MAKING" if "DECISION" in response else "INFORMATIONAL"
            return (intent, 0.75)  # Medium confidence from LLM

        except Exception as e:
            print(f"Warning: LLM classification failed: {e}")
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
            print(f"Intent detected via REGEX: {intent} (confidence: {confidence})")
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
            print(f"Intent detected via KEYWORDS: {intent} (confidence: {confidence:.2f})")
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
        print(f"Intent detection falling back to LLM for ambiguous query")
        intent, confidence = self._stage3_llm_classification(query)
        print(f"LLM Intent result: {intent} (confidence: {confidence})")
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

### 🎯 **Problem Identified**
[Precise identification of the legal problem with relevant Pakistani law context]

### 📜 **Relevant Law & Sections**
[Cite specific acts, ordinances, sections with direct quotes where possible]

### ⚖️ **Your Legal Rights & Position**
[Clear explanation of user's legal position, rights, and responsibilities under Pakistani law]

### 🚀 **Recommended Actions**
[Numbered step-by-step action plan with specific procedures and deadlines]

### ⚠️ **Important Cautions**
[Bullet-pointed list of potential consequences, pitfalls, and what to avoid]

### ⏰ **Timeline & Next Steps**
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

            # Generate initial advice with the proven structured prompt
            prompt = self.prompt.format(question=question, context=formatted_context)
            print(f"Generating decision advice with context length: {len(formatted_context)}")
            advice = self.llm._call(prompt)

            # Validate the response - check if it contains expected structured elements
            if self._is_valid_structured_advice(advice):
                print(f"Generated valid structured decision advice (length: {len(advice)})")
                return {
                    'success': True,
                    'advice': advice,
                    'context_used': len(legal_context),
                    'method': 'structured_decision_making'
                }
            else:
                print(f"Warning: LLM generated inadequate structured advice, using fallback")
                # Use fallback advice that mimics the structured format
                advice = self._generate_structured_fallback_advice(question, legal_context)
                return {
                    'success': True,
                    'advice': advice,
                    'context_used': len(legal_context),
                    'method': 'structured_fallback'
                }

        except Exception as e:
            print(f"Decision making failed: {e}")
            # Even on error, provide structured fallback
            advice = self._generate_structured_fallback_advice(question, legal_context)
            return {
                'success': False,
                'error': f'Decision making failed: {str(e)}',
                'advice': advice,
                'context_used': len(legal_context),
                'method': 'error_fallback'
            }

    def _is_valid_structured_advice(self, advice: str) -> bool:
        """Check if the generated advice follows the expected structured format"""
        if not advice or len(advice.strip()) < 100:
            return False

        advice_lower = advice.lower()

        # Check for key structural elements that indicate proper formatting
        required_elements = [
            'problem identified' in advice_lower or 'legal issue' in advice_lower,
            'recommended actions' in advice_lower or 'action' in advice_lower,
            'rights' in advice_lower or 'obligations' in advice_lower,
            'cautions' in advice_lower or 'warnings' in advice_lower
        ]

        # At least 3 out of 4 key elements should be present
        return sum(required_elements) >= 3

    def _generate_structured_fallback_advice(self, question: str, legal_context: List[Document]) -> str:
        """Generate structured fallback advice that mimics the working format"""
        question_lower = question.lower()

        # Extract some basic context info
        context_summary = ""
        if legal_context:
            domains = set()
            for doc in legal_context[:3]:  # Check first 3 docs
                if hasattr(doc, 'metadata') and doc.metadata.get('domain'):
                    domains.add(doc.metadata['domain'])
            if domains:
                context_summary = f"Based on {', '.join(domains)} law context from {len(legal_context)} legal documents."

        # Generate structured advice based on question type
        if 'divorce' in question_lower or 'talaq' in question_lower or 'khula' in question_lower:
            advice = f"""### 🎯 **Problem Identified**
You are seeking guidance on divorce proceedings in Pakistan. This involves understanding Islamic family law (Muslim Family Laws Ordinance 1961) and court procedures for dissolution of marriage.

### 📜 **Relevant Law & Sections**
- **Muslim Family Laws Ordinance 1961**: Governs divorce procedures for Muslims in Pakistan
- **Section 7**: Requirements for divorce by talaq
- **Section 8**: Requirements for khula (wife-initiated divorce)
- **West Pakistan Family Courts Act 1964**: Court procedures for family disputes

### ⚖️ **Your Legal Rights & Position**
- **Talaq**: Husband's right to divorce (must follow proper procedure and waiting period)
- **Khula**: Wife's right to seek divorce with compensation
- **Maintenance Rights**: Right to claim maintenance during and after divorce
- **Child Custody**: Rights regarding children of the marriage

### 🚀 **Recommended Actions**
1. **Consult Family Court**: File petition in the relevant Family Court
2. **Obtain Legal Representation**: Engage a family law lawyer
3. **Gather Documentation**: Marriage certificate, CNIC, witnesses
4. **Follow Proper Procedure**: Comply with iddat period and legal requirements
5. **Consider Mediation**: Attempt reconciliation through arbitration council

### ⚠️ **Important Cautions**
- **Iddat Period**: Three months waiting period must be observed
- **Proper Documentation**: All procedures must be properly documented
- **Court Jurisdiction**: Must file in the correct family court
- **False Claims**: Making false statements can lead to legal consequences

### ⏰ **Timeline & Next Steps**
- **Immediate**: Consult a family law lawyer within 1-2 days
- **Filing**: File divorce petition within 1 week of consultation
- **Court Hearing**: First hearing typically within 2-4 weeks
- **Resolution**: Case may take 3-6 months depending on complexity

### 📞 **Professional Legal Consultation**
**Strongly Recommended**: Consult a qualified Pakistani family law lawyer immediately. Family law is complex and requires expert guidance to protect your rights and ensure proper procedure.

**LEGAL DISCLAIMER:** This is general information only. Pakistani family law is complex and situation-specific. Always consult qualified legal professionals for personalized advice.

{context_summary}"""

        elif 'traffic' in question_lower or 'ticket' in question_lower or 'fine' in question_lower:
            advice = f"""### 🎯 **Problem Identified**
You have received a traffic ticket/fine in Pakistan and need guidance on how to handle traffic violations under Pakistani traffic law.

### 📜 **Relevant Law & Sections**
- **Traffic Rules 1961**: Main traffic regulations in Pakistan
- **Highway Safety Ordinance 2000**: Road safety and traffic control
- **Local Traffic Ordinances**: Province-specific traffic rules
- **Fine amounts**: Range from Rs. 500 to Rs. 10,000 depending on violation

### ⚖️ **Your Legal Rights & Position**
- **Right to Contest**: You can challenge the ticket if you believe it's incorrect
- **Right to Pay**: You can pay the fine to resolve the matter quickly
- **Due Process**: Right to fair hearing if contesting the ticket
- **License Protection**: Avoid license suspension by addressing violations promptly

### 🚀 **Recommended Actions**
1. **Review the Ticket**: Check violation details, amount, and deadline
2. **Decide on Action**: Choose between paying fine or contesting
3. **Pay Fine (if accepting)**: Pay within 10 days at designated bank/traffic office
4. **Contest (if disputing)**: File appeal with traffic court within specified time
5. **Gather Evidence**: Photos, witnesses, or dashcam footage if contesting

### ⚠️ **Important Cautions**
- **Deadline Compliance**: Pay or contest within 10 days to avoid additional penalties
- **License Suspension**: Unpaid fines can lead to license suspension
- **Vehicle Impoundment**: Severe violations may result in vehicle seizure
- **Court Procedures**: Contesting requires proper legal procedure

### ⏰ **Timeline & Next Steps**
- **Immediate (1-2 days)**: Review ticket and gather evidence
- **Payment Deadline**: Pay fine within 10 days of ticket issuance
- **Contest Deadline**: File appeal within 15 days if disputing
- **Court Resolution**: Traffic court cases typically resolve within 1-2 months

### 📞 **Professional Legal Consultation**
**Recommended**: Consult a traffic law lawyer if the violation is serious or you're unsure about contesting. Traffic law procedures can be complex.

**LEGAL DISCLAIMER:** This is general information about Pakistani traffic law. Always consult legal professionals for specific traffic violations.

{context_summary}"""

        else:
            # Generic structured advice for other legal questions
            advice = f"""### 🎯 **Problem Identified**
You are seeking legal guidance for: "{question}". This requires understanding applicable Pakistani laws and proper legal procedures.

### 📜 **Relevant Law & Sections**
- **Applicable Pakistani Laws**: Depending on your specific situation, relevant laws may include:
  - Civil law, criminal law, or administrative regulations
  - Provincial and federal legislation
  - Court procedures and requirements

### ⚖️ **Your Legal Rights & Position**
- **Right to Legal Counsel**: You have the right to consult legal professionals
- **Due Process Rights**: Right to fair hearing and legal representation
- **Information Access**: Right to understand applicable laws and procedures
- **Protection of Rights**: Legal protections under Pakistani constitution

### 🚀 **Recommended Actions**
1. **Assess Your Situation**: Clearly identify all relevant facts and legal issues
2. **Consult Legal Experts**: Speak with qualified Pakistani lawyers
3. **Gather Documentation**: Collect all relevant documents and evidence
4. **Follow Legal Procedures**: Comply with applicable laws and deadlines
5. **Document Everything**: Keep detailed records of all communications

### ⚠️ **Important Cautions**
- **Legal Complexity**: Pakistani law can be complex and situation-specific
- **Deadline Compliance**: Missing legal deadlines can harm your case
- **Professional Advice**: Self-representation may not be advisable for complex matters
- **Evidence Preservation**: Maintain all relevant evidence and documentation

### ⏰ **Timeline & Next Steps**
- **Immediate**: Consult a lawyer within 1-2 days
- **Assessment**: Complete legal assessment within 1 week
- **Action Planning**: Develop action plan within 2 weeks
- **Implementation**: Begin legal procedures as advised by counsel

### 📞 **Professional Legal Consultation**
**Essential**: Consult qualified Pakistani legal professionals immediately. They can provide specific advice tailored to your situation and ensure proper legal procedures.

**LEGAL DISCLAIMER:** This is general legal information only. Pakistani law is complex and requires professional legal advice for specific situations.

{context_summary}"""

        return advice

    def _generate_knowledge_based_response(self, question: str, primary_domain: str, history_str: str, legal_context: List[Document]) -> str:
        """
        Generate comprehensive legal advice using external resources and legal knowledge
        Used as fallback when RAG content is inadequate
        """
        question_lower = question.lower()

        # Extract context information
        context_summary = ""
        if legal_context:
            domains = set()
            sources = []
            for doc in legal_context[:5]:
                if hasattr(doc, 'metadata'):
                    if doc.metadata.get('domain'):
                        domains.add(doc.metadata['domain'])
                    if doc.metadata.get('collection_name'):
                        sources.append(doc.metadata['collection_name'])
            if domains:
                context_summary = f"\n\n**Available Context:** Analysis based on {', '.join(domains)} law from {len(sources)} legal sources."

        # =====================================================
        # COMPREHENSIVE LEGAL KNOWLEDGE DATABASE
        # =====================================================

        # Family Law Knowledge Base
        if primary_domain == 'family' or any(word in question_lower for word in ['marriage', 'divorce', 'talaq', 'khula', 'nikah', 'custody', 'maintenance', 'inheritance']):
            return self._generate_family_law_response(question, context_summary)

        # Corporate/Employment Law Knowledge Base
        elif primary_domain == 'corporate' or any(word in question_lower for word in ['employee', 'employment', 'corporate', 'company', 'rights', 'termination', 'contract']):
            return self._generate_corporate_law_response(question, context_summary)

        # Traffic Law Knowledge Base
        elif primary_domain == 'traffic' or any(word in question_lower for word in ['traffic', 'ticket', 'fine', 'vehicle', 'accident', 'license', 'driving']):
            return self._generate_traffic_law_response(question, context_summary)

        # Criminal Law Knowledge Base
        elif primary_domain == 'criminal' or any(word in question_lower for word in ['crime', 'criminal', 'police', 'arrest', 'bail', 'fir', 'complaint']):
            return self._generate_criminal_law_response(question, context_summary)

        # Property Law Knowledge Base
        elif any(word in question_lower for word in ['property', 'land', 'house', 'rent', 'tenant', 'landlord', 'ownership']):
            return self._generate_property_law_response(question, context_summary)

        # Contract Law Knowledge Base
        elif any(word in question_lower for word in ['contract', 'agreement', 'breach', 'obligation', 'terms']):
            return self._generate_contract_law_response(question, context_summary)

        # Default Comprehensive Response
        else:
            return self._generate_general_legal_response(question, context_summary)

    def _generate_family_law_response(self, question: str, context_summary: str) -> str:
        """Comprehensive Family Law Response"""
        return f"""### 🎯 **Family Law Analysis - Pakistan**

#### **Problem Identified**
Your query involves family law matters in Pakistan, which are governed by Islamic principles and statutory law. Family law is highly sensitive and requires careful navigation of religious, cultural, and legal considerations.

#### **📜 Relevant Legal Framework**
- **Muslim Family Laws Ordinance 1961**: Primary legislation for Muslim family matters
- **West Pakistan Family Courts Act 1964**: Establishes family courts and procedures
- **Dissolution of Muslim Marriages Act 1939**: Governs divorce procedures
- **Guardians and Wards Act 1890**: Child custody and guardianship matters
- **Dowry and Bridal Gifts (Restriction) Act 1976**: Dowry-related regulations

#### **⚖️ Your Legal Rights & Position**
**Marriage Rights:**
- Right to registered marriage under Muslim Family Laws Ordinance
- Protection against forced marriage
- Right to maintenance during marriage
- Right to divorce (talaq for husband, khula for wife)

**Divorce Rights:**
- **Talaq**: Husband's unilateral right (must follow proper procedure)
- **Khula**: Wife's right to seek divorce with compensation
- **Judicial Divorce**: Court-ordered divorce under specific grounds

**Child-Related Rights:**
- **Custody (Hizanat)**: Mother's right to custody until age 7 (boys) or puberty (girls)
- **Maintenance**: Father's obligation to provide for children's needs
- **Education Rights**: Right to education and proper upbringing

**Inheritance Rights:**
- Governed by Islamic inheritance laws (Faraid)
- Women receive half the share of male heirs in most cases
- Right to will up to 1/3 of estate

#### **🚀 Recommended Actions**
1. **Immediate Consultation**: Contact a family law specialist within 24-48 hours
2. **Documentation**: Gather all marriage certificates, CNIC, and relevant documents
3. **Mediation First**: Attempt reconciliation through Arbitration Council (mandatory)
4. **Court Filing**: File petition in appropriate Family Court if mediation fails
5. **Legal Representation**: Engage qualified family law attorney
6. **Evidence Preservation**: Document all communications and incidents

#### **⚠️ Critical Cautions**
- **Religious Compliance**: Family matters must align with Islamic principles
- **Court Jurisdiction**: Wrong court filing can delay proceedings significantly
- **Iddat Period**: Three-month waiting period must be strictly observed
- **Child Welfare**: Court's primary consideration is child's best interest
- **False Claims**: Making false allegations can lead to severe consequences
- **Cultural Sensitivity**: Family disputes often involve extended family dynamics

#### **⏰ Timeline & Procedures**
**Marriage Disputes:**
- Mediation: 3-4 months (mandatory first step)
- Court Filing: Within 1 year of dispute
- Resolution: 6-18 months depending on complexity

**Divorce Proceedings:**
- Talaq: Effective after iddat period (90 days)
- Khula: Court approval required, 3-6 months
- Judicial Divorce: 6-12 months with evidence

**Child Custody:**
- Interim Orders: Within 2-4 weeks of filing
- Final Decision: 3-6 months
- Modification: Can be sought if circumstances change

#### **💰 Cost Considerations**
- Court Fees: Rs. 500-2,000 for initial filing
- Lawyer Fees: Rs. 10,000-50,000 depending on complexity
- Mediation: Rs. 2,000-5,000 (often court-directed)
- Appeals: Additional costs if case goes to higher courts

#### **📞 Professional Legal Consultation**
**URGENTLY RECOMMENDED**: Consult a **Family Law Specialist** immediately. Look for:
- Advocates enrolled with Pakistan Bar Council
- Experience in Islamic Family Law
- Knowledge of local court procedures
- Mediation and arbitration expertise

**LEGAL DISCLAIMER:** This analysis is based on general Pakistani family law principles. Each case is unique and requires personalized legal advice. Family law matters can have lifelong consequences - professional legal guidance is essential.

**AUTHORITY LEVEL:** High confidence based on Pakistani legal framework and established precedents.{context_summary}"""

    def _generate_corporate_law_response(self, question: str, context_summary: str) -> str:
        """Comprehensive Corporate/Employment Law Response"""
        return f"""### 🎯 **Corporate & Employment Law Analysis - Pakistan**

#### **Problem Identified**
Your query involves corporate or employment law matters, which are governed by comprehensive statutory frameworks protecting worker rights and regulating business operations.

#### **📜 Relevant Legal Framework**
- **Companies Act 2017**: Primary corporate law legislation
- **West Pakistan Industrial and Commercial Employment Ordinance 1968**: Employment regulations
- **Minimum Wages Ordinance 1961**: Wage protection laws
- **Workers' Welfare Fund Ordinance 1971**: Worker welfare schemes
- **Standing Orders Ordinance 1968**: Workplace rules and procedures

#### **⚖️ Employee Rights & Protections**
**Fundamental Rights:**
- **Right to Fair Wages**: Minimum wage protection (varies by province)
- **Working Hours**: Maximum 48 hours/week, overtime compensation
- **Leave Entitlements**: Annual leave, sick leave, maternity leave
- **Gratuity**: 30 days salary per year of service (after 2 years)
- **Provident Fund**: Employer contribution mandatory

**Termination Protections:**
- **Notice Period**: 30-90 days depending on position/salary
- **Severance Pay**: Compensation for unjust termination
- **Unfair Dismissal**: Protection against arbitrary termination
- **Golden Handshake**: Negotiable severance packages

**Workplace Safety:**
- **Safe Environment**: Employer duty to provide safe workplace
- **Health Insurance**: Mandatory for certain industries
- **Workers' Compensation**: Coverage for work-related injuries
- **EOBI Registration**: Mandatory social security coverage

#### **🚀 Recommended Actions**
1. **Document Everything**: Maintain detailed records of employment terms, communications, and incidents
2. **Internal Resolution**: First attempt resolution through HR/company grievance procedures
3. **Legal Consultation**: Consult employment law specialist within 7 days
4. **Formal Complaint**: File with relevant labor department if internal resolution fails
5. **Evidence Collection**: Gather emails, witnesses, performance records, and contracts
6. **Union Consideration**: Join or form workers' union for collective bargaining power

#### **⚠️ Critical Cautions**
- **Limitation Periods**: Most claims must be filed within 6-12 months
- **Evidence Preservation**: Digital and physical evidence crucial for success
- **Retaliation Risk**: Employers may retaliate against complaints
- **Jurisdiction Issues**: Wrong forum filing can lead to case dismissal
- **Settlement Pressure**: Employers often pressure for quick settlements
- **Professional Documentation**: All communications should be in writing

#### **⏰ Timeline & Procedures**
**Grievance Resolution:**
- Internal Complaint: 15-30 days for company response
- Labor Department: 30-60 days for initial hearing
- Court Filing: Within 2 years of cause of action

**Termination Disputes:**
- Notice Period: Must be served properly
- Appeal Period: 30 days to challenge termination
- Court Hearing: 3-6 months for first hearing

**Wage Recovery:**
- Labor Court: 6-12 months for resolution
- Appeal Process: Additional 6-12 months
- Execution: 3-6 months for award enforcement

#### **💰 Cost Considerations**
- Legal Fees: Rs. 15,000-75,000 depending on complexity
- Court Fees: Rs. 1,000-5,000 for filing
- Mediation: Rs. 5,000-15,000 (recommended first step)
- Appeals: Additional costs for higher court appeals

#### **📞 Professional Legal Consultation**
**STRONGLY RECOMMENDED**: Consult an **Employment Law Specialist** immediately. Look for:
- Advocates with labor law expertise
- Experience with relevant labor courts
- Knowledge of provincial labor laws
- Mediation and negotiation skills

**LEGAL DISCLAIMER:** This analysis covers general employment law principles in Pakistan. Labor laws vary by province and industry. Each employment dispute requires specific legal analysis.

**AUTHORITY LEVEL:** High confidence based on Pakistani labor law framework and established case law.{context_summary}"""

    def _generate_traffic_law_response(self, question: str, context_summary: str) -> str:
        """Comprehensive Traffic Law Response"""
        return f"""### 🎯 **Traffic Law Analysis - Pakistan**

#### **Problem Identified**
Your query involves traffic violations, accidents, or vehicle-related legal matters in Pakistan, where traffic law enforcement is strict and penalties can be severe.

#### **📜 Relevant Legal Framework**
- **Motor Vehicles Ordinance 1965**: Primary traffic legislation
- **Highway Safety Ordinance 2000**: Road safety regulations
- **Traffic Rules 1961**: Detailed traffic regulations
- **Provincial Motor Vehicle Rules**: Province-specific rules
- **National Highway Authority Act 1991**: Highway regulations

#### **⚖️ Traffic Violation Rights & Procedures**
**Ticket/Fine Rights:**
- **Right to Contest**: Challenge ticket within 10 days
- **Right to Hearing**: Request traffic court hearing
- **Payment Options**: Pay fine or contest (not both)
- **License Protection**: Avoid suspension by addressing violations

**Accident-Related Rights:**
- **FIR Filing**: Right to report accident to police
- **Medical Treatment**: Emergency medical care rights
- **Insurance Claims**: Right to file insurance claims
- **Compensation**: Right to claim damages for injuries

**License & Vehicle Rights:**
- **Due Process**: Right to fair license suspension procedures
- **Vehicle Release**: Right to get impounded vehicle released
- **Appeal Rights**: Challenge license cancellation/suspension

#### **🚀 Recommended Actions**
1. **Ticket Review**: Carefully examine violation details, amount, and deadline
2. **Decision Making**: Choose between payment or contesting within 10 days
3. **Payment Process**: Pay at designated bank/traffic office if accepting
4. **Contest Filing**: File appeal with traffic court if disputing
5. **Evidence Gathering**: Collect photos, witnesses, dashcam footage
6. **Legal Consultation**: Consult traffic law expert for serious violations

#### **⚠️ Critical Cautions**
- **10-Day Deadline**: Critical for payment or contest filing
- **License Suspension**: Unpaid fines lead to automatic suspension
- **Vehicle Impoundment**: Severe violations result in vehicle seizure
- **Court Procedures**: Contesting requires proper legal documentation
- **Penalty Escalation**: Late payments incur additional penalties
- **Insurance Impact**: Violations affect insurance premiums

#### **⏰ Timeline & Procedures**
**Fine Payment:**
- Payment Period: 10 days from ticket issuance
- Late Payment: Additional penalties after deadline
- License Restoration: 24-48 hours after payment

**Contest Process:**
- Appeal Filing: Within 10 days of ticket
- Court Hearing: 15-30 days after filing
- Resolution: 1-3 months depending on case

**License Issues:**
- Suspension Notice: 15 days to respond
- Hearing: Within 30 days of suspension
- Restoration: Immediate upon favorable decision

#### **💰 Cost Considerations**
- Fine Amounts: Rs. 500-25,000 depending on violation
- Court Fees: Rs. 500-2,000 for contest filing
- Legal Fees: Rs. 5,000-20,000 for representation
- Vehicle Release: Rs. 1,000-5,000 impoundment fees

#### **📞 Professional Legal Consultation**
**RECOMMENDED for serious violations**: Consult a **Traffic Law Specialist**. Look for:
- Advocates experienced with traffic courts
- Knowledge of provincial traffic regulations
- Experience with license restoration cases

**LEGAL DISCLAIMER:** Traffic laws vary by province and violation type. This is general guidance - consult local traffic authorities for specific cases.

**AUTHORITY LEVEL:** High confidence based on Pakistani traffic law framework.{context_summary}"""

    def _generate_criminal_law_response(self, question: str, context_summary: str) -> str:
        """Comprehensive Criminal Law Response"""
        return f"""### 🎯 **Criminal Law Analysis - Pakistan**

#### **Problem Identified**
Your query involves criminal law matters, which are serious and require immediate professional legal attention. Criminal cases have strict procedures and significant consequences.

#### **📜 Relevant Legal Framework**
- **Pakistan Penal Code 1860**: Primary criminal law legislation
- **Code of Criminal Procedure 1898**: Criminal procedure and trial rules
- **Police Rules 1934**: Police investigation procedures
- **Anti-Terrorism Act 1997**: Terrorism and security-related offenses
- **Narcotics Control Laws**: Drug-related offenses

#### **⚖️ Criminal Law Rights & Protections**
**Fundamental Rights:**
- **Right to Silence**: Cannot be compelled to testify against yourself
- **Right to Counsel**: Legal representation from arrest stage
- **Right to Fair Trial**: Presumption of innocence until proven guilty
- **Right to Bail**: Presumption in favor of bail for bailable offenses

**Investigation Rights:**
- **FIR Filing**: Right to lodge First Information Report
- **Medical Examination**: Right to medical examination if injured
- **Property Protection**: Protection of personal property during investigation

**Trial Rights:**
- **Legal Representation**: Right to engage defense counsel
- **Evidence Access**: Right to know prosecution evidence
- **Appeal Rights**: Right to appeal convictions

#### **🚀 Recommended Actions**
1. **Immediate Legal Consultation**: Contact criminal law specialist within 24 hours
2. **Document Everything**: Record all interactions with police/authorities
3. **FIR Filing**: File formal complaint if you're the victim
4. **Bail Application**: Apply for bail if arrested (for bailable offenses)
5. **Evidence Preservation**: Secure all relevant evidence and witnesses
6. **Medical Attention**: Seek medical care and document injuries

#### **⚠️ Critical Cautions**
- **Police Interrogation**: Be cautious during police questioning
- **False Confessions**: Avoid making statements under pressure
- **Evidence Tampering**: Preserve all evidence in original condition
- **Witness Intimidation**: Protect witnesses from undue influence
- **Media Exposure**: Avoid premature media discussions
- **Flight Risk**: Comply with court orders to avoid additional charges

#### **⏰ Timeline & Procedures**
**FIR Process:**
- Filing: Within 24 hours of incident
- Investigation: 14 days (extendable by court)
- Challan Submission: Within investigation period

**Bail Process:**
- Bailable Offenses: Bail granted routinely
- Non-Bailable: Court hearing required
- Post-Arrest Bail: Within 24 hours for bailable offenses

**Trial Timeline:**
- Preliminary Hearing: 1-2 weeks after challan
- Evidence Recording: 2-6 weeks
- Arguments: 1-2 weeks
- Judgment: 1-4 weeks after arguments

#### **💰 Cost Considerations**
- Legal Fees: Rs. 25,000-100,000 depending on case complexity
- Court Fees: Rs. 500-5,000 for various filings
- Bail Bonds: Rs. 10,000-50,000 surety requirements
- Investigation Costs: Additional expenses for private investigation

#### **📞 Professional Legal Consultation**
**URGENTLY REQUIRED**: Consult a **Criminal Law Specialist** immediately. Look for:
- Advocates enrolled with High Court
- Experience with Sessions Court trials
- Knowledge of local police procedures
- Bail and anticipatory bail expertise

**LEGAL DISCLAIMER:** Criminal law is extremely complex and fact-specific. This is general information only. Criminal matters require immediate qualified legal representation.

**AUTHORITY LEVEL:** High confidence based on Pakistani criminal law framework and established legal procedures.{context_summary}"""

    def _generate_property_law_response(self, question: str, context_summary: str) -> str:
        """Comprehensive Property Law Response"""
        return f"""### 🎯 **Property Law Analysis - Pakistan**

#### **Problem Identified**
Your query involves property law matters including land, housing, tenancy, and ownership rights in Pakistan.

#### **📜 Relevant Legal Framework**
- **Transfer of Property Act 1882**: Property transfer regulations
- **Registration Act 1908**: Property registration requirements
- **Land Revenue Act 1967**: Land ownership and revenue matters
- **Sindh Rented Premises Ordinance 1979**: Tenancy regulations
- **Housing Laws**: Provincial housing and development regulations

#### **⚖️ Property Rights & Protections**
**Ownership Rights:**
- **Title Protection**: Legal protection of property ownership
- **Transfer Rights**: Right to sell, gift, or transfer property
- **Inheritance Rights**: Property devolution through succession
- **Mortgage Rights**: Right to use property as collateral

**Tenant Rights:**
- **Security of Tenure**: Protection against arbitrary eviction
- **Rent Control**: Regulation of rent increases
- **Maintenance Rights**: Right to habitable premises
- **Notice Requirements**: Advance notice for termination

**Landlord Rights:**
- **Rent Collection**: Right to receive agreed rent
- **Property Access**: Right to access property for repairs
- **Termination Rights**: Right to terminate tenancy legally
- **Compensation**: Right to compensation for damages

#### **🚀 Recommended Actions**
1. **Title Verification**: Verify property documents and ownership
2. **Legal Documentation**: Ensure all transfers are properly registered
3. **Tenancy Agreement**: Draft comprehensive rental agreements
4. **Dispute Resolution**: Use mediation before court proceedings
5. **Professional Valuation**: Get property valuation for transactions
6. **Tax Compliance**: Ensure payment of property taxes and fees

#### **⚠️ Critical Cautions**
- **Title Defects**: Undiscovered ownership issues can cause disputes
- **Registration Gaps**: Unregistered transfers are legally invalid
- **Boundary Disputes**: Common source of prolonged litigation
- **Tax Evasion**: Non-payment leads to penalties and seizures
- **Environmental Compliance**: Building regulations must be followed

#### **⏰ Timeline & Procedures**
**Property Purchase:**
- Due Diligence: 1-2 weeks
- Agreement: 1-2 days
- Registration: 1-4 weeks
- Possession: Immediate after registration

**Tenancy Disputes:**
- Notice Period: 30-90 days depending on agreement
- Court Filing: Within limitation period (3 years)
- Resolution: 6-18 months

**Title Disputes:**
- Suit Filing: Within 12 years of knowledge
- Evidence Collection: 2-4 weeks
- Court Resolution: 1-3 years

#### **💰 Cost Considerations**
- Registration Fees: 1-2% of property value
- Legal Fees: Rs. 10,000-50,000 for transactions
- Stamp Duty: 0.25-0.75% of property value
- Court Fees: Rs. 1,000-10,000 for disputes

#### **📞 Professional Legal Consultation**
**RECOMMENDED**: Consult a **Property Law Specialist**. Look for:
- Advocates experienced in property transactions
- Knowledge of local registration procedures
- Experience with tenancy disputes
- Real estate law expertise

**LEGAL DISCLAIMER:** Property law is complex and location-specific. Always verify local regulations and consult professionals for property matters.

**AUTHORITY LEVEL:** High confidence based on Pakistani property law framework.{context_summary}"""

    def _generate_contract_law_response(self, question: str, context_summary: str) -> str:
        """Comprehensive Contract Law Response"""
        return f"""### 🎯 **Contract Law Analysis - Pakistan**

#### **Problem Identified**
Your query involves contract law matters, which are governed by common law principles and statutory regulations in Pakistan.

#### **📜 Relevant Legal Framework**
- **Contract Act 1872**: Primary contract law legislation
- **Sale of Goods Act 1930**: Goods and services contracts
- **Specific Relief Act 1877**: Contract enforcement remedies
- **Arbitration Act 1940**: Alternative dispute resolution
- **Electronic Transactions Ordinance 2002**: Digital contracts

#### **⚖️ Contract Rights & Obligations**
**Essential Elements:**
- **Offer and Acceptance**: Clear mutual agreement
- **Consideration**: Valuable exchange between parties
- **Legal Capacity**: Parties must be competent to contract
- **Free Consent**: Agreement without coercion, fraud, or misrepresentation
- **Lawful Object**: Contract purpose must be legal

**Performance Obligations:**
- **Timely Performance**: Contracts must be performed within agreed time
- **Quality Standards**: Goods/services must meet agreed specifications
- **Payment Obligations**: Timely payment as per contract terms
- **Good Faith**: Duty to act honestly and fairly

**Termination Rights:**
- **Breach Remedies**: Right to terminate for material breach
- **Notice Requirements**: Proper notice before termination
- **Damages Recovery**: Right to claim compensation for losses
- **Specific Performance**: Court-ordered contract completion

#### **🚀 Recommended Actions**
1. **Contract Review**: Carefully examine all contract terms and conditions
2. **Documentation**: Maintain detailed records of all communications
3. **Notice Issuance**: Send formal notices for breaches or terminations
4. **Negotiation**: Attempt amicable resolution before legal action
5. **Evidence Collection**: Gather all relevant emails, documents, and witnesses
6. **Legal Consultation**: Consult contract law specialist for complex disputes

#### **⚠️ Critical Cautions**
- **Limitation Periods**: Contract claims must be filed within 3 years
- **Evidence Preservation**: Digital and written evidence crucial
- **Oral Contracts**: Difficult to prove without written evidence
- **Force Majeure**: Unforeseen events may excuse performance
- **Liquidated Damages**: Penalty clauses must be reasonable
- **Jurisdiction Clauses**: Court selection can affect case outcome

#### **⏰ Timeline & Procedures**
**Breach Resolution:**
- Notice Period: 15-30 days to cure breach
- Negotiation: 1-2 weeks for settlement attempts
- Court Filing: Within 3 years of breach
- Resolution: 6-18 months depending on complexity

**Contract Enforcement:**
- Suit Filing: Within limitation period
- Evidence Stage: 2-4 weeks
- Arguments: 1-2 weeks
- Judgment: 2-8 weeks

#### **💰 Cost Considerations**
- Legal Fees: Rs. 15,000-75,000 depending on contract value
- Court Fees: Rs. 1,000-10,000 based on claim amount
- Arbitration: Rs. 25,000-100,000 for arbitration proceedings
- Expert Fees: Additional costs for technical experts

#### **📞 Professional Legal Consultation**
**RECOMMENDED**: Consult a **Contract Law Specialist**. Look for:
- Advocates with commercial law experience
- Arbitration and mediation expertise
- Knowledge of relevant industry regulations
- Drafting and negotiation skills

**LEGAL DISCLAIMER:** Contract law is highly fact-specific and depends on contract terms. Always consult legal professionals for contract disputes.

**AUTHORITY LEVEL:** High confidence based on Pakistani contract law principles and established case law.{context_summary}"""

    def _generate_general_legal_response(self, question: str, context_summary: str) -> str:
        """General Legal Response for unspecified domains"""
        return f"""### 🎯 **General Legal Analysis - Pakistan**

#### **Problem Identified**
Your legal query requires analysis under Pakistani legal framework. While the specific domain may not be immediately clear, we can provide guidance on general legal procedures and rights.

#### **📜 Relevant Legal Framework**
- **Constitution of Pakistan 1973**: Fundamental rights and principles
- **General Clauses Act 1897**: Interpretation of statutes
- **Limitation Act 1908**: Time limits for legal actions
- **Code of Civil Procedure 1908**: Civil litigation procedures
- **Evidence Act 1872**: Rules of evidence in legal proceedings

#### **⚖️ General Legal Rights & Protections**
**Fundamental Rights:**
- **Right to Life and Liberty**: Protection under Article 9
- **Right to Fair Trial**: Article 10-A guarantees due process
- **Right to Information**: Access to public information
- **Property Rights**: Protection against arbitrary deprivation
- **Equality Before Law**: Equal treatment under Article 25

**Procedural Rights:**
- **Legal Representation**: Right to engage legal counsel
- **Access to Courts**: Right to approach competent courts
- **Appeal Rights**: Right to appeal adverse decisions
- **Time Limits**: Reasonable time for legal proceedings

#### **🚀 Recommended Actions**
1. **Problem Assessment**: Clearly identify the legal issue and relevant facts
2. **Documentation**: Gather all relevant documents and evidence
3. **Legal Consultation**: Consult appropriate legal specialist
4. **Preliminary Research**: Understand applicable laws and procedures
5. **Timeline Planning**: Note any limitation periods or deadlines
6. **Professional Guidance**: Seek qualified legal advice for specific situations

#### **⚠️ Critical Cautions**
- **Limitation Periods**: Most legal actions have strict time limits
- **Evidence Preservation**: Maintain all relevant evidence securely
- **Professional Advice**: Self-representation may not be advisable
- **Court Procedures**: Follow proper legal procedures meticulously
- **Cost Considerations**: Legal proceedings can be expensive
- **Appeal Options**: Understand appeal procedures and deadlines

#### **⏰ General Timeline Considerations**
- **Consultation**: Seek legal advice within 1-2 weeks of issue
- **Preparation**: Gather evidence and prepare case within 1 month
- **Filing**: Initiate legal proceedings within applicable limitation periods
- **Resolution**: Varies greatly depending on case complexity (1-24 months)

#### **💰 General Cost Considerations**
- **Legal Fees**: Rs. 5,000-50,000 depending on complexity
- **Court Fees**: Rs. 500-10,000 for various filings
- **Miscellaneous**: Additional costs for documents, experts, travel

#### **📞 Professional Legal Consultation**
**ESSENTIAL**: Consult qualified legal professionals immediately. Look for:
- Advocates enrolled with Pakistan Bar Council
- Relevant area of specialization
- Experience with similar cases
- Good reputation and client reviews

**LEGAL DISCLAIMER:** This is general legal information for Pakistan. Laws vary by jurisdiction and specific circumstances. Always consult qualified legal professionals for personalized advice.

**AUTHORITY LEVEL:** General guidance based on Pakistani legal framework. Specific cases require detailed legal analysis.{context_summary}"""

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