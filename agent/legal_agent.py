import os
import requests
import chromadb
import uuid
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import LLM
from langchain_core.outputs import LLMResult
from typing import Optional, List, Tuple
from config.settings import settings
from helper.token_tracker import track_token_usage_async
from agent.gemini_llm import gemini_llm, perplexity_llm
import asyncio
import hashlib
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
import time
import re
import openai
from transformers import pipeline
from config.settings import settings

# ------------------ OpenRouter Config ------------------
OPENROUTER_API_URL = settings.OPENROUTER_API_URL
OPENROUTER_MODEL = settings.OPENROUTER_MODEL
OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY

# Domain detection keywords
DOMAIN_KEYWORDS = {
    'traffic': ['traffic', 'updated fines', 'fines', 'fine', 'motor vehicle', 'driving', 'license', 'transport', 'road', 'accident', 'speeding', 'vehicle registration'],
    
    'family': ['family', 'marriage','witness','nikah witness','nikah', 'divorce', 'inheritance', 'guardian', 'child', 'maintenance', 'custody', 'dowry','dower','marital', "alimony", "child support", "adoption", "domestic violence", "family dispute", "nikaah", "mehr", "talaq", "khula", "wasiat","pakistan family law","family court","family act","family ordinance"],
    
    'corporate': ['corporate', 'company', 'business','factories', 'factories act', 'commercial', 'eobi', 'EOBI’s', ' eobi’s ', 'contract', 'partnership', 'incorporation', 'shareholder', 'director', 'board','leaves', 'employee', 'employment', 'labor', 'workplace', 'hr', 'human resources', 'termination', 'hiring', 'firing', 'work hours', 'overtime', 'payroll', 'benefits', 'discrimination', 'harassment', 'workplace safety','pakistan labor law','pakistan employment law','labor court','employment act','industrial relations','maternity', 'paternity', 'casual', 'sick leave', 'annual leave', 'leave policy', 'pension'],
    
    'ppc': [
        # Core PPC terms
        'ppc', 'penal', 'criminal', 'crime', 'offense', 'offence', 'punishment', 'ipc', 'pakistan penal', 'theft', 'jail', 'imprisonment', 'forgery','bail', 'arrest', 'cognizable', 'non-cognizable', 'felony', 'misdemeanor',
        
        # Punishment types
        'death penalty', 'death sentence', 'life imprisonment', 'rigorous imprisonment', 'simple imprisonment', 'solitary confinement', 'fine', 'commutation',
        
        # General exceptions/defenses
        'private defence', 'self defence', 'general exceptions', 'mistake of fact', 'accident', 'consent', 'unsound mind', 'intoxication', 'child offense', 'minor offense',
        
        # Offense categories
        'abetment', 'conspiracy', 'criminal conspiracy', 'attempt', 'attempt to commit',
        
        # Specific offenses from chapters
        'waging war', 'sedition', 'mutiny', 'army offense', 'navy offense', 'air force offense',
        'unlawful assembly', 'rioting', 'affray', 'public tranquility',
        'bribery', 'corruption', 'public servant', 'gratification', 'official act',
        'false evidence', 'perjury', 'fabricating evidence', 'public justice',
        'coin offense', 'counterfeiting', 'government stamp', 'currency note', 'banknote',
        'weights measures', 'false weight', 'false measure',
        'public health', 'public safety', 'nuisance', 'adulteration', 'quarantine',
        'religion offense', 'religious feelings', 'place of worship', 'holy quran', 'prophet',
        'qatl', 'murder', 'homicide', 'hurt', 'grievous hurt', 'diyat', 'qisas', 'arsh',
        'wrongful restraint', 'wrongful confinement', 'criminal force', 'assault',
        'kidnapping', 'abduction', 'slavery', 'forced labour', 'human trafficking',
        'rape', 'sexual offense', 'unnatural offense', 'sexual abuse',
        'theft', 'extortion', 'robbery', 'dacoity', 'hijacking',
        'criminal misappropriation', 'criminal breach trust', 'cheating', 'fraud',
        'mischief', 'criminal trespass', 'house trespass', 'housebreaking',
        'oil gas offense', 'electricity offense', 'tampering', 'theft of energy',
        'forgery', 'false document', 'trade mark', 'property mark',
        'defamation', 'criminal intimidation', 'insult', 'annoyance',
        
        # Legal procedures and concepts
        'sentence', 'punishment', 'conviction', 'previous conviction', 'enhanced punishment',
        'right of private defence', 'extent of private defence', 'commencement of defence'
    ]
}

# ------------------ Custom OpenRouter LLM wrapper ------------------
class OpenRouterLLM(LLM):
    api_key: str = OPENROUTER_API_KEY
    model: str = OPENROUTER_MODEL
    api_url: str = OPENROUTER_API_URL
    request_times: list = []  # Track request timestamps for rate limiting
    max_requests_per_minute: int = 10  # Conservative limit for free tier
    retry_delays: list = [1, 2, 4, 8, 16]  # Exponential backoff in seconds
    max_retries: int = 3

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _enforce_rate_limit(self):
        """Enforce rate limiting to prevent 429 errors"""
        current_time = time.time()

        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]

        # If we've hit the limit, wait
        if len(self.request_times) >= self.max_requests_per_minute:
            oldest_request = min(self.request_times)
            wait_time = 60 - (current_time - oldest_request)
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)

        # Record this request
        self.request_times.append(current_time)

    def _call_with_retry(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call OpenRouter with retry logic"""
        for attempt in range(self.max_retries):
            try:
                print(f"🔴 Calling OpenRouter API (attempt {attempt + 1}/{self.max_retries})...")

                # Enforce rate limiting
                self._enforce_rate_limit()

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 2000
                }

                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)

                if resp.status_code == 429:
                    retry_after = resp.headers.get('retry-after')
                    wait_time = int(retry_after) if retry_after else 5
                    print(f"Rate limited (429). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

                resp.raise_for_status()
                data = resp.json()
                print(f"OpenRouter API call successful on attempt {attempt + 1}")
                return data["choices"][0]["message"]["content"]

            except Exception as e:
                print(f"OpenRouter API error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delays[attempt])
                    continue
                return f"[OpenRouter Error] Failed after {self.max_retries} attempts: {e}"

        return "[OpenRouter Error] Max retries exceeded"

    def _call_with_fallback(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call Gemini first (3 attempts), then fallback to OpenRouter"""
        from agent.gemini_llm import gemini_llm

        # Try Gemini first (3 attempts)
        print("🤖 Generating legal response...")
        gemini_response = gemini_llm._call_with_retry(prompt, stop)
        if not gemini_response.startswith("[Gemini Error]"):
            # print("✅ Gemini API call successful")
            print("✅ Call successful")

            return gemini_response

        print("⚠️ Gemini failed, falling back to OpenRouter...")

        # Fallback to OpenRouter
        return self._call_with_retry(prompt, stop)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Main call method with Gemini-first, OpenRouter-fallback strategy"""
        return self._call_with_fallback(prompt, stop)

# ------------------ Custom OpenRouter LLM wrapper ------------------

# Create singleton instance
openrouter_llm = OpenRouterLLM()

# ------------------ Primary LLM with Perplexity first, OpenRouter fallback ------------------
class PrimaryLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "primary"

    def _call_with_fallback(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call Gemini first, then fallback to Perplexity, then OpenRouter"""
        # Try Gemini first (3 attempts)
        print("🤖 Generating legal response...")
        gemini_response = gemini_llm._call_with_retry(prompt, stop)
        if not gemini_response.startswith("[Gemini Error]"):
            print("✅ Gemini call successful")
            return gemini_response

        print("⚠️ Gemini failed, falling back to Perplexity...")

        # Fallback to Perplexity
        perplexity_response = perplexity_llm._call_with_retry(prompt, stop)
        if not perplexity_response.startswith("[Perplexity Error]"):
            print("✅ Perplexity fallback successful")
            return perplexity_response

        print("⚠️ Perplexity failed, falling back to OpenRouter...")

        # Fallback to OpenRouter
        openrouter_response = openrouter_llm._call_with_retry(prompt, stop)
        if not openrouter_response.startswith("[OpenRouter Error]"):
            print("✅ OpenRouter fallback successful")
            return openrouter_response

        print("❌ All LLMs failed")
        return "[All LLMs Error] Gemini, Perplexity, and OpenRouter all failed"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Main call method with Gemini-first, Perplexity-fallback, OpenRouter-last strategy"""
        return self._call_with_fallback(prompt, stop)

# Create singleton instance
primary_llm = PrimaryLLM()

# ------------------ Chroma Setup ------------------
_chroma_client = None

def create_chroma_client():
    """Create or return cached Chroma Cloud client (avoids ~3s connection cost per request)."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.CloudClient(
            api_key=settings.CHROMA_API_KEY,
            tenant=settings.CHROMA_TENANT,
            database=settings.CHROMA_DATABASE
        )
    return _chroma_client

def get_all_collections():
    """Get all available collections"""
    client = create_chroma_client()
    try:
        collections = client.list_collections()
        return [coll.name for coll in collections]
    except Exception as e:
        print(f"❌ Error getting collections: {e}")
        return []

# Cache vector retrievers per collection
_retriever_cache: dict = {}

def embed_query(query: str):
    """Embed the query using the same API as ingest (Deepinfra or HF) so retrieval matches stored embeddings."""
    from controllers.ingest_controller import get_deepinfra_embeddings, get_hf_embeddings
    use_deepinfra = getattr(settings, "DEEPINFRA_API_KEY", None) and getattr(settings, "DEEPINFRA_API_KEY", "").strip()
    if use_deepinfra:
        api_key = settings.DEEPINFRA_API_KEY
        model = getattr(settings, "DEEPINFRA_EMBED_MODEL", None) or "thenlper/gte-base"
        embs = get_deepinfra_embeddings([query], api_key, model)
    else:
        api_key = settings.HF_API_KEY
        model = settings.HF_EMBED_MODEL
        embs = get_hf_embeddings([query], api_key, model)
    return embs[0] if embs else None

def extract_section_search_queries(question: str) -> List[str]:
    """If the user asks about a specific section (e.g. 'section 302 of PPC'), return extra short queries to improve retrieval of that chunk."""
    extra = []
    # Match "section N" or "Section N" or "s. N" / "sec N", optionally "of PPC/penal"
    m = re.search(r"\bsection\s+(\d+)\b|\b(?:s\.|sec\.?)\s*(\d+)\b|\b(\d{2,4})\s*(?:of\s*)?(?:ppc|penal|pakistan\s*penal)\b", question, re.I)
    if m:
        num = next(g for g in m.groups() if g)
        extra.append(f"section {num}")
        if num not in question[:m.start()] and num not in question[m.end():]:
            extra.append(num)  # bare number helps match "302. Qatl-i-amd" style chunks
    return extra[:2]  # at most 2 extra queries to limit embedding calls


def detect_query_domain(query: str) -> Tuple[str, List[str]]:
    """Detect which domain the query belongs to and return matching collections"""
    query_lower = query.lower()

    # Check for domain keywords
    matched_domains = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in query_lower for keyword in keywords):
            matched_domains.append(domain)

    # Get all collections
    all_collections = get_all_collections()

    if not all_collections:
        return "none", []

    # Find collections matching detected domains
    matching_collections = []
    for domain in matched_domains:
        domain_collections = [coll for coll in all_collections if domain in coll]
        matching_collections.extend(domain_collections)

    # Remove duplicates
    matching_collections = list(set(matching_collections))

    if matching_collections:
        primary_domain = matched_domains[0] if matched_domains else "general"
        return primary_domain, matching_collections
    else:
        # If no specific domain detected, search all collections
        return "general", all_collections

def get_collection_retriever(collection_name: str, k: int = 4):
    """Create a vector-only retriever for a collection (cached per collection). Uses same embedding as ingest."""
    cache_key = (collection_name, k)
    if cache_key in _retriever_cache:
        return _retriever_cache[cache_key]

    client = create_chroma_client()
    try:
        collection = client.get_collection(name=collection_name)

        class VectorRetriever:
            def __init__(self, collection, k=4):
                self.collection = collection
                self.k = k
                self.collection_name = collection_name

            def get_relevant_documents(self, query: str, query_embedding=None):
                from langchain_core.documents import Document
                query_emb = query_embedding if query_embedding is not None else embed_query(query)
                if query_emb is None:
                    print(f"⚠️ Query embedding failed for {self.collection_name}")
                    return []
                try:
                    results = self.collection.query(
                        query_embeddings=[query_emb],
                        n_results=self.k,
                        include=["documents", "metadatas", "distances"],
                    )
                except Exception as e:
                    print(f"Vector search failed for {self.collection_name}: {e}")
                    return []
                documents = []
                if results.get("documents") and results["documents"][0]:
                    distances = results.get("distances", [[]])
                    dist_list = distances[0] if distances else []
                    for i, (doc_text, metadata, doc_id) in enumerate(zip(
                        results["documents"][0],
                        results["metadatas"][0],
                        results["ids"][0],
                    )):
                        dist = dist_list[i] if i < len(dist_list) else float("inf")
                        documents.append(
                            Document(
                                page_content=doc_text,
                                metadata={
                                    **(metadata or {}),
                                    "doc_id": doc_id,
                                    "collection_name": self.collection_name,
                                    "chroma_distance": dist,
                                },
                            )
                        )
                return documents

        retriever = VectorRetriever(collection, k=k)
        _retriever_cache[cache_key] = retriever
        return retriever
    except Exception as e:
        print(f"❌ Failed to load collection {collection_name}: {e}")
        return None

# ------------------ Query Reformulation Agent ------------------
class QueryReformulationAgent:
    def __init__(self, llm):
        self.llm = llm
        # Initialize T5-small for fast, free reformulation
        try:
            self.t5_reformulator = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device=-1  # Use CPU for compatibility, can be changed to 0 for GPU
            )
            print("T5-small reformulator loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load T5-small: {e}. Falling back to LLM.")
            self.t5_reformulator = None

        self.prompt = PromptTemplate(
            template="""You are a query reformulation expert for legal questions. Your task is to:

1. Analyze the user's question for clarity and completeness
2. Reformulate it into multiple precise legal search queries
3. Consider synonyms, legal terminology, and related concepts
4. Generate 2-3 alternative formulations that would improve search results

**ORIGINAL QUESTION:** {question}

**REFORMULATED QUERIES:**
- Query 1: [precise legal formulation]
- Query 2: [alternative formulation]
- Query 3: [broader or related formulation]

Return only the reformulated queries, one per line.""",
            input_variables=["question"]
        )

    def reformulate(self, question: str) -> List[str]:
        """Generate 2 reformulated query variants"""
        try:
            print("Generating 2 query variants...")
            start_time = time.time()

            # Analyze legal context
            legal_context = self._analyze_legal_context(question)

            # Generate rule-based variants
            variants = self._generate_rule_based_variants(question, legal_context)

            # Ensure original question is included
            if question not in variants:
                variants.insert(0, question)

            # Clean and deduplicate - limit to 2 variants total
            final_variants = []
            seen = set()
            for variant in variants[:3]:  # Generate up to 3, then limit to 2
                clean_variant = variant.strip()
                if clean_variant and clean_variant not in seen and len(clean_variant) > 8:
                    seen.add(clean_variant)
                    final_variants.append(clean_variant)

            # Ensure we have at least 2 variants
            result = final_variants[:2]
            if len(result) < 2:
                # If we don't have enough variants, add a simple one
                if question not in result:
                    result.insert(0, question)
                if len(result) < 2:
                    result.append(f"Legal information about {question}")

            elapsed = time.time() - start_time
            print(f"Generated {len(result)} variants in {elapsed:.2f}s")
            return result[:2]

        except Exception as e:
            print(f"Reformulation failed: {e}, using original question")
            return [question, f"Legal information about {question}"]

    def _analyze_legal_context(self, question: str) -> dict:
        """Analyze the legal context of a query"""
        query_lower = question.lower()

        # Detect legal domains
        domains = []
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(keyword in query_lower for keyword in keywords):
                domains.append(domain)

        # Extract section numbers
        import re
        section_matches = re.findall(r'section\s+(\d+)', query_lower, re.IGNORECASE)
        sections = [int(match) for match in section_matches]

        # Check for specific legal terms
        legal_indicators = ['law', 'act', 'ordinance', 'code', 'court', 'provision', 'section', 'article']

        return {
            'domains': domains,
            'sections': sections,
            'has_legal_terms': any(term in query_lower for term in legal_indicators),
            'is_question': question.strip().endswith('?'),
            'word_count': len(question.split())
        }

    def _create_legal_reformulation_prompts(self, question: str, context: dict) -> List[str]:
        """Create targeted prompts for legal reformulation"""
        prompts = []

        # Base reformulation prompt
        prompts.append(f"Rewrite this legal question in different words: {question}")

        # Domain-specific prompts
        if context['domains']:
            primary_domain = context['domains'][0]
            prompts.append(f"Rephrase this {primary_domain} law question: {question}")
            prompts.append(f"Ask this differently about {primary_domain} legal matters: {question}")

        # Section-specific prompts
        if context['sections']:
            section_num = context['sections'][0]
            prompts.append(f"Rephrase question about section {section_num}: {question}")
            prompts.append(f"Ask differently about legal section {section_num}: {question}")

        # General legal prompts
        if context['has_legal_terms']:
            prompts.append(f"Paraphrase this legal query: {question}")
            prompts.append(f"Express this legal question differently: {question}")

        return prompts[:3]  # Limit to 3 prompts for efficiency

    def _is_valid_reformulation(self, text: str, original: str) -> bool:
        """Check if a reformulation is valid and useful"""
        if not text or len(text.strip()) < 5:
            return False

        text_lower = text.lower()
        original_lower = original.lower()

        # Reject if too similar to original (less than 30% different)
        if len(set(text_lower.split()) - set(original_lower.split())) < 0.3 * len(set(original_lower.split())):
            return False

        # Reject nonsensical or too short responses
        if len(text.split()) < 3:
            return False

        # Reject if it starts with strange prefixes
        bad_prefixes = ['t.', 't.i.', 'legal law', 'law legal', 'the the', 'a a', 'an an']
        if any(text_lower.startswith(prefix) for prefix in bad_prefixes):
            return False

        # Reject if it contains too many repeated words
        words = text_lower.split()
        if len(words) > len(set(words)) * 2:  # More than 50% duplicates
            return False

        return True

    def _post_process_variants(self, variants: List[str], original: str, context: dict) -> List[str]:
        """Post-process and rank the generated variants"""
        valid_variants = []

        for variant in variants:
            if self._is_valid_reformulation(variant, original):
                # Score the variant
                score = self._score_variant(variant, original, context)
                valid_variants.append((variant, score))

        # Sort by score and return top variants
        valid_variants.sort(key=lambda x: x[1], reverse=True)
        return [variant for variant, score in valid_variants]

    def _score_variant(self, variant: str, original: str, context: dict) -> float:
        """Score a variant based on quality criteria"""
        score = 0.0
        variant_lower = variant.lower()

        # Prefer variants that maintain legal terminology
        legal_terms = ['law', 'act', 'section', 'provision', 'court', 'legal', 'ordinance', 'code']
        legal_term_count = sum(1 for term in legal_terms if term in variant_lower)
        score += legal_term_count * 0.5

        # Prefer variants that include domain-specific terms
        if context['domains']:
            domain_matches = sum(1 for domain in context['domains'] if domain in variant_lower)
            score += domain_matches * 0.8

        # Prefer variants that keep section numbers
        if context['sections']:
            section_matches = sum(1 for section in context['sections']
                                if f"section {section}" in variant_lower or f"section{section}" in variant_lower)
            score += section_matches * 1.0

        # Prefer variants that are properly formed questions
        if variant.strip().endswith('?'):
            score += 0.3

        # Length appropriateness (not too short, not too long)
        word_count = len(variant.split())
        if 4 <= word_count <= 15:
            score += 0.2
        elif word_count > 20:
            score -= 0.3

        # Diversity from original
        original_words = set(original.lower().split())
        variant_words = set(variant_lower.split())
        overlap_ratio = len(original_words & variant_words) / len(original_words) if original_words else 0
        if 0.3 <= overlap_ratio <= 0.8:  # Good balance of similarity and difference
            score += 0.4

        return score

    def _generate_rule_based_variants(self, question: str, context: dict) -> List[str]:
        """Generate rule-based legal query variants"""
        variants = []
        question_lower = question.lower()

        # Pattern 1: Section-based queries
        if context['sections']:
            section_num = context['sections'][0]
            variants.extend([
                f"What are the provisions of section {section_num}?",
                f"Section {section_num} legal requirements",
                f"Explain section {section_num} of the law"
            ])

        # Pattern 2: Domain-specific reformulations
        elif context['domains']:
            domain = context['domains'][0]

            # Family law patterns
            if domain == 'family':
                if 'nikah' in question_lower or 'marriage' in question_lower:
                    variants.extend([
                        "What are the requirements for marriage in Pakistan?",
                        "Marriage requirements under Pakistani family law",
                        "Conditions for valid nikah in Pakistan"
                    ])
                elif 'divorce' in question_lower:
                    variants.extend([
                        "Divorce procedure in Pakistan",
                        "How to get divorce under Pakistani law",
                        "Talaq requirements in Pakistan"
                    ])
                else:
                    variants.extend([
                        f"Family law provisions for {question}",
                        f"Pakistani family law requirements",
                        f"Family court procedures in Pakistan"
                    ])

            # Corporate law patterns
            elif domain == 'corporate':
                if 'pension' in question_lower or 'eobi' in question_lower:
                    variants.extend([
                        "EOBI pension increase notification",
                        "Revision of EOBI pension amount under labor laws",
                        "EOBI minimum pension rates in Pakistan"
                    ])
                elif 'employee' in question_lower or 'rights' in question_lower:
                    variants.extend([
                        "Employee rights under Pakistani labor law",
                        "Workers rights and protections in Pakistan",
                        "Employment law provisions for employees"
                    ])
                else:
                    variants.extend([
                        f"Corporate law requirements for {question}",
                        f"Company law provisions in Pakistan",
                        f"Business registration requirements"
                    ])

            # Criminal law patterns
            elif domain == 'ppc':
                variants.extend([
                    f"Criminal law provisions for {question}",
                    f"PPC sections related to {question}",
                    f"Pakistani penal code requirements"
                ])

            # Traffic law patterns
            elif domain == 'traffic':
                variants.extend([
                    f"Traffic law violations and penalties",
                    f"Road traffic rules in Pakistan",
                    f"Traffic fines and regulations"
                ])

        # Pattern 3: General legal queries
        else:
            # Requirements-based questions
            if 'requirements' in question_lower or 'conditions' in question_lower:
                variants.extend([
                    f"Legal requirements for {question.replace('requirements', '').replace('conditions', '').strip()}",
                    f"Pakistani law provisions regarding {question}",
                    f"Legal procedure and requirements"
                ])

            # How-to questions
            elif question_lower.startswith(('how', 'what is the process', 'what are the steps')):
                variants.extend([
                    f"Legal procedure for {question}",
                    f"Steps required under Pakistani law",
                    f"Legal requirements and process"
                ])

            # General legal questions
            else:
                variants.extend([
                    f"Pakistani legal provisions for {question}",
                    f"Law and regulations regarding {question}",
                    f"Legal requirements in Pakistan"
                ])

        return variants

    def _create_additional_variant(self, question: str, context: dict, index: int) -> str:
        """Create additional variants when rule-based generation doesn't provide enough"""
        question_lower = question.lower()

        if index == 1:
            # Focus on Pakistani context
            return f"Pakistan law regarding {question}"

        elif index == 2:
            # Focus on legal procedures
            if 'section' in question_lower:
                return f"Legal interpretation of {question}"
            else:
                return f"Legal provisions and requirements for {question}"

        return None

    def _llm_reformulate_fallback(self, question: str) -> List[str]:
        """Fallback LLM-based reformulation when T5 fails"""
        try:
            # Enhanced prompt for better legal query generation
            enhanced_prompt = f"""You are an expert legal query reformulation specialist. Your task is to create multiple precise search queries that will effectively retrieve legal information from document databases.

**ORIGINAL QUESTION:** {question}

**LEGAL REFORMULATION REQUIREMENTS:**
1. **Extract Legal Intent**: Identify if the question seeks specific sections, general concepts, or procedural information
2. **Generate Precise Variants**: Create queries that target exact legal terminology
3. **Include Legal References**: If asking about sections/articles, include both numbered and descriptive forms
4. **Domain-Specific Terms**: Use appropriate legal terminology for the domain (family, corporate, criminal, traffic)
5. **Multiple Search Angles**: Provide different phrasings that might match document content

**REFORMULATED SEARCH QUERIES:**
- Query 1: [Most precise legal formulation - include section numbers if mentioned]
- Query 2: [Alternative legal terminology or broader related concept]
- Query 3: [Procedural or practical angle of the same legal question]

**EXAMPLES:**
For "What is section 5 of family law?":
- Query 1: "section 5 muslim family laws ordinance"
- Query 2: "family law section 5 marriage requirements"
- Query 3: "nikah requirements under section 5"

Return only the reformulated queries, one per line starting with "- Query X:".
"""

            response = self.llm._call(enhanced_prompt)

            # Parse the response to extract queries
            queries = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('- Query') or line.startswith('Query'):
                    # Extract the query text after the colon
                    if ':' in line:
                        query = line.split(':', 1)[1].strip()
                        if query and len(query) > 3:  # Minimum length check
                            queries.append(query)

            # If parsing failed or insufficient queries, create basic variants
            if len(queries) < 2:
                # Create basic legal variants
                import re
                question_lower = question.lower()

                # Extract potential section numbers
                section_matches = re.findall(r'section\s+(\d+)', question_lower, re.IGNORECASE)
                if section_matches:
                    section_num = section_matches[0]
                    queries.extend([
                        question,  # Original
                        f"section {section_num} legal provision",
                        f"legal section {section_num} requirements"
                    ])
                else:
                    # General legal reformulation
                    legal_terms = ['law', 'act', 'ordinance', 'code', 'court']
                    domain_indicators = ['family', 'marriage', 'divorce', 'corporate', 'company', 'traffic', 'criminal', 'ppc']

                    domain = next((d for d in domain_indicators if d in question_lower), 'legal')
                    queries.extend([
                        question,  # Original
                        f"{domain} law {question}",
                        f"pakistan {domain} legal requirements"
                    ])

            # Ensure we have at least the original question
            if question not in queries:
                queries.insert(0, question)

            # Remove duplicates and limit to 3
            seen = set()
            unique_queries = []
            for q in queries:
                if q not in seen:
                    seen.add(q)
                    unique_queries.append(q)

            return unique_queries[:3]

        except Exception as e:
            print(f"Query reformulation error: {e}")
            return [question]

# ------------------ Query Classification Agent ------------------
class QueryClassificationAgent:
    def __init__(self, llm):
        self.llm = llm
        self.domains = list(DOMAIN_KEYWORDS.keys()) + ['general']
        self.prompt = PromptTemplate(
            template="""You are a legal domain classification expert. Analyze the question and classify it into the most relevant legal domain(s).

Available domains: {domains}

**QUESTION:** {question}

**CLASSIFICATION:**
Primary Domain: [most relevant domain]
Confidence: [high/medium/low]
Additional Domains: [comma-separated if applicable]

Return only the classification information.""",
            input_variables=["question", "domains"]
        )

    def classify(self, question: str) -> dict:
        """Classify the query into legal domains using keyword matching first, then LLM as fallback"""
        try:
            question_lower = question.lower()

            # Step 1: Fast keyword-based domain detection
            matched_domains = []
            keyword_confidence_scores = {}

            for domain, keywords in DOMAIN_KEYWORDS.items():
                matches = sum(1 for keyword in keywords if keyword in question_lower)
                if matches > 0:
                    matched_domains.append(domain)
                    keyword_confidence_scores[domain] = matches / len(keywords)  # Normalized score

            # Step 2: Determine confidence based on keyword matching
            if matched_domains:
                # Sort by confidence score
                best_domain = max(matched_domains, key=lambda d: keyword_confidence_scores[d])
                confidence_score = keyword_confidence_scores[best_domain]

                # High confidence if we have strong keyword matches
                if confidence_score >= 0.3 or len(matched_domains) == 1:
                    confidence = "high" if confidence_score >= 0.5 else "medium"
                    print(f"🔍 Keyword-based classification: {best_domain.upper()} (confidence: {confidence})")
                    return {
                        'primary_domain': best_domain,
                        'confidence': confidence,
                        'all_domains': matched_domains
                    }

            # Step 3: Fallback to LLM classification if keyword matching is uncertain
            print("🔄 Keyword matching uncertain, using LLM classification...")
            prompt = self.prompt.format(question=question, domains=', '.join(self.domains))
            response = self.llm._call(prompt)

            # Parse response
            primary_domain = "general"
            confidence = "medium"
            additional_domains = []

            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('Primary Domain:'):
                    primary_domain = line.split(':', 1)[1].strip().lower()
                elif line.startswith('Confidence:'):
                    confidence = line.split(':', 1)[1].strip().lower()
                elif line.startswith('Additional Domains:'):
                    additional = line.split(':', 1)[1].strip()
                    if additional and additional.lower() != 'none':
                        additional_domains = [d.strip().lower() for d in additional.split(',')]

            return {
                'primary_domain': primary_domain,
                'confidence': confidence,
                'all_domains': [primary_domain] + additional_domains
            }

        except Exception as e:
            print(f"Query classification error: {e}")
            return {
                'primary_domain': 'general',
                'confidence': 'low',
                'all_domains': ['general']
            }

# ------------------ Reranking Agent ------------------
class RerankingAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            template="""You are a legal document reranking expert. Evaluate and rerank the provided document excerpts based on their relevance to the question.

**QUESTION:** {question}

**DOCUMENTS TO RANK:**
{documents}

**INSTRUCTIONS:**
1. Score each document from 1-10 based on relevance, accuracy, and legal specificity
2. Consider legal terminology, direct answers, and authoritative content
3. Prioritize documents that directly address the question
4. Return ranked list with scores and brief justification

**RANKED RESULTS:**
1. Document [ID]: Score [X/10] - [brief reason]
2. Document [ID]: Score [X/10] - [brief reason]
...""",
            input_variables=["question", "documents"]
        )

    def rerank(self, question: str, documents: List) -> List:
        """Rerank documents based on relevance using Gemini"""
        if len(documents) <= 3:
            return documents  # No need to rerank small sets

        try:
            # Prepare documents for ranking
            doc_summaries = []
            for i, doc in enumerate(documents[:10]):  # Limit to top 10 for efficiency
                summary = f"Document {i+1}: {doc.page_content[:200]}..."
                doc_summaries.append(summary)

            prompt = self.prompt.format(
                question=question,
                documents='\n'.join(doc_summaries)
            )

            response = self.llm._call(prompt)

            # Parse rankings (simplified - in production, use more robust parsing)
            ranked_indices = []
            for line in response.split('\n'):
                if line.strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                    # Extract document number
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            doc_num = int(parts[1].replace('Document', '').replace(':', '')) - 1
                            if 0 <= doc_num < len(documents):
                                ranked_indices.append(doc_num)
                        except:
                            continue

            # If parsing failed, return original order
            if not ranked_indices:
                return documents

            # Reorder documents based on ranking
            ranked_docs = [documents[i] for i in ranked_indices if i < len(documents)]

            # Add any remaining documents
            remaining = [doc for i, doc in enumerate(documents) if i not in ranked_indices]
            ranked_docs.extend(remaining)

            return ranked_docs

        except Exception as e:
            print(f"Reranking error: {e}")
            return documents

# ------------------ Query Cache ------------------
class QueryCache:
    def __init__(self, max_size=100, ttl_seconds=3600):  # 1 hour TTL
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds

    def _get_cache_key(self, query: str, collections: List[str]) -> str:
        """Generate a unique cache key for the query and collections"""
        key_data = {
            'query': query.lower().strip(),
            'collections': sorted(collections)
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, collections: List[str]) -> Optional[dict]:
        """Get cached result if available and not expired"""
        key = self._get_cache_key(query, collections)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                print("📋 Cache hit! Using cached results.")
                return entry['result']
            else:
                # Expired, remove it
                del self.cache[key]
        return None

    def set(self, query: str, collections: List[str], result: dict):
        """Cache the result"""
        key = self._get_cache_key(query, collections)

        # If cache is full, remove oldest entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]

        self.cache[key] = {
            'result': result,
            'timestamp': time.time()
        }

    def clear(self):
        """Clear all cached results"""
        self.cache.clear()

# ------------------ Smart Multi-Agent Legal Assistant ------------------
class SmartLegalAssistant:
    def __init__(self):
        self.llm = primary_llm
        self.reformulation_agent = QueryReformulationAgent(self.llm)
        self.classification_agent = QueryClassificationAgent(self.llm)
        self.reranking_agent = RerankingAgent(self.llm)
        self.query_cache = QueryCache()

        # Import decision-making capabilities
        self.decision_flow = None
        self.decision_enabled = False

        self.prompt = PromptTemplate(
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

    def _log_rag_context(self, top_docs: List, legal_context: str, max_chunk_preview: int = 280, max_context_log: int = 2400):
        """Log what was retrieved from vector and what context is being sent to the LLM (for debugging)."""
        print("\n" + "=" * 60)
        print("📥 VECTOR RETRIEVED (chunks passed to LLM):")
        for i, doc in enumerate(top_docs):
            src = doc.metadata.get("source", doc.metadata.get("collection_name", "?"))
            page = doc.metadata.get("page_number", "?")
            preview = (doc.page_content or "")[:max_chunk_preview].replace("\n", " ")
            if len(doc.page_content or "") > max_chunk_preview:
                preview += "..."
            print(f"   [{i+1}] source={src} page={page} ({len(doc.page_content or 0)} chars) | \"{preview}\"")
        print("-" * 60)
        print("📤 CONTEXT TO LLM (legal_context):")
        if len(legal_context) <= max_context_log:
            print(legal_context)
        else:
            print(legal_context[:max_context_log] + f"\n... [truncated, total {len(legal_context)} chars]")
        print("=" * 60 + "\n")

    def _create_enhanced_prompt(self, history_str: str, legal_context: str, question: str, domain: str, source_docs: List) -> str:
        """Create an enhanced prompt with better context structuring and accuracy instructions"""

        # Extract key legal references from source documents
        legal_references = []
        section_numbers = []

        for doc in source_docs[:5]:  # Top 5 most relevant
            collection = doc.metadata.get('collection_name', 'Unknown')
            page = doc.metadata.get('page_number', 'N/A')

            # Extract section numbers from content for better context
            content = doc.page_content.lower()
            import re
            found_sections = re.findall(r'section\s+(\d+)', content, re.IGNORECASE)
            if found_sections:
                section_numbers.extend([f"Section {s}" for s in found_sections[:3]])  # Limit to 3 per doc

            legal_references.append(f"- {collection} (Page {page})")

        # Create focused context based on question type
        question_lower = question.lower()
        if 'section' in question_lower and any(char.isdigit() for char in question):
            # Extract section number from question
            section_match = re.search(r'section\s+(\d+)', question_lower, re.IGNORECASE)
            if section_match:
                target_section = section_match.group(1)
                enhanced_context = f"""
**TARGET SECTION:** Section {target_section} of {domain.upper()} Law

**LEGAL CONTEXT (FOCUSED ON SECTION {target_section}):**
{legal_context}

**AVAILABLE SECTIONS IN CONTEXT:** {', '.join(list(set(section_numbers))[:10])}

**SOURCE DOCUMENTS CITED:**
{chr(10).join(legal_references)}

**DOMAIN:** {domain.upper()} LAW - PAKISTAN
"""
            else:
                enhanced_context = f"""
**LEGAL CONTEXT SUMMARY:**
{legal_context}

**AVAILABLE SECTIONS:** {', '.join(list(set(section_numbers))[:10])}

**SOURCE DOCUMENTS CITED:**
{chr(10).join(legal_references)}

**DOMAIN:** {domain.upper()} LAW - PAKISTAN
"""
        else:
            enhanced_context = f"""
**LEGAL CONTEXT SUMMARY:**
{legal_context}

**AVAILABLE SECTIONS:** {', '.join(list(set(section_numbers))[:10])}

**SOURCE DOCUMENTS CITED:**
{chr(10).join(legal_references)}

**DOMAIN:** {domain.upper()} LAW - PAKISTAN
"""

        enhanced_prompt = f"""You are an elite Pakistani Legal AI Assistant with deep expertise in Pakistani law. Your responses must be exceptionally accurate, professional, and authoritative.

**CRITICAL REQUIREMENTS:**
1. **LEGAL ACCURACY FIRST**: Base your answer ONLY on the provided legal context
2. **COMPREHENSIVE ANSWERS**: Provide detailed, thorough explanations with all relevant information from the context
3. **CITATION MANDATORY**: Always cite specific sections, acts, or provisions from the context with exact wording when available
4. **DIRECT ANSWERS**: If the question asks for a specific section, provide its exact content and detailed explanation
5. **STRUCTURED FORMAT**: Use clear headings, numbered lists, bullet points, and professional formatting
6. **PAKISTANI LAW FOCUS**: Reference Pakistani legal system, courts, and procedures with complete details
7. **CONSERVATIVE ADVICE**: When in doubt, recommend consulting qualified legal professionals
8. **EVIDENCE-BASED**: Support all claims with extensive references to the provided legal sources
9. **CLEAN OUTPUT**: Do NOT include any source document names, collection names, page numbers, or internal references in your final answer
10. **COMPLETE INFORMATION**: Cover all aspects of the question with comprehensive details, requirements, procedures, and implications

**CONVERSATION CONTEXT:**
{history_str}

**LEGAL SOURCES AVAILABLE:**
{enhanced_context}

**USER QUESTION:**
{question}

**RESPONSE GUIDELINES:**
- If asking about a specific section, quote it directly and explain its meaning
- Provide complete and accurate information from the context
- Include the exact wording of legal provisions when available
- Do NOT mention document names, collections, or page numbers in the answer
- End with appropriate disclaimers about seeking professional legal advice
- Keep the response clean and professional, as if from a legal expert

**PROFESSIONAL LEGAL RESPONSE:**
"""

        return enhanced_prompt

    def _is_inadequate_response(self, response: str) -> bool:
        """Trigger fallback only when Gemini explicitly says context is missing the info (e.g. bail query)."""
        if not response or len(response.strip()) < 80:
            return True

        response_lower = response.lower()

        # Only when model clearly says "context doesn't have this" → then use knowledge fallback for useful answer
        context_missing_indicators = [
            "the provided context does not contain",
            "does not contain any information about",
            "no information about",
            "no information available in the context",
        ]

        return any(indicator in response_lower for indicator in context_missing_indicators)

    def _generate_knowledge_based_response(self, question: str, domain: str, history_str: str, rag_context: str = "") -> str:
        """Generate response using Gemini's knowledge when RAG response is inadequate"""
        knowledge_prompt = f"""You are an expert Pakistani Legal Assistant with comprehensive knowledge of Pakistani law. The previous RAG search was inadequate, so provide accurate information about the requested legal topic using your legal knowledge, but don't mention anywhere that rag content is inadequate in answer.

**LEGAL DOMAIN:** {domain.upper()} LAW - PAKISTAN

**PREVIOUS RAG CONTEXT (was insufficient):**
{rag_context[:500]}...

**CONVERSATION CONTEXT:**
{history_str}

**USER QUESTION:**
{question}

**RESPONSE REQUIREMENTS:**
1. Provide comprehensive and accurate information about Pakistani law
2. Use proper legal terminology and cite relevant sections/acts
3. Structure the response professionally with clear explanations
4. Include the exact wording of legal provisions when explaining sections
5. Explain legal concepts clearly for general understanding
6. Always recommend consulting qualified legal professionals for specific cases

**IMPORTANT:** Since RAG failed to provide adequate information, use your complete knowledge of Pakistani law to give a thorough, accurate response.

**LEGAL RESPONSE:**
"""

        try:
            response = self.llm._call(knowledge_prompt)
            # Add subtle note that comprehensive legal knowledge was used
            enhanced_response = response + "\n\n---\n*This comprehensive legal information is provided for educational purposes. For your specific situation, please consult the original legal texts or a qualified legal professional.*"
            return enhanced_response
        except Exception as e:
            return f"I apologize, but I'm currently unable to provide information on this legal topic. Please try again later or consult a qualified legal professional for accurate advice. Error: {str(e)}"

    async def search_collections_async(self, queries: List[str], collection_names: List[str], k_per_collection: int = 3, query_embedding=None, start_time: Optional[float] = None):
        """Asynchronously search across multiple collections. When multiple queries (e.g. main + section number), embeds each and uses the right embedding per query."""
        all_docs = []
        t0 = start_time if start_time is not None else time.time()
        # One embedding per query so section-specific queries (e.g. "section 302") pull the right chunks
        if not queries:
            return []
        if query_embedding is not None and not isinstance(query_embedding, list):
            embeddings = [query_embedding] * len(queries)
        elif len(queries) == 1:
            print(f"[{time.perf_counter()-t0:.2f}s] 🧮 Embedding query (API call)...")
            emb = embed_query(queries[0])
            print(f"[{time.perf_counter()-t0:.2f}s] 🧮 Embedding done → querying Chroma...")
            embeddings = [emb] if emb else [None]
        else:
            print(f"[{time.perf_counter()-t0:.2f}s] 🧮 Embedding {len(queries)} queries (parallel API calls)...")
            loop = asyncio.get_event_loop()
            embeddings = await asyncio.gather(*[
                loop.run_in_executor(None, lambda q=q: embed_query(q))
                for q in queries
            ])
            print(f"[{time.perf_counter()-t0:.2f}s] 🧮 Embedding done → querying Chroma...")

        search_tasks = []
        for qi, query in enumerate(queries):
            emb = embeddings[qi] if qi < len(embeddings) else None
            for collection_name in collection_names:
                task = asyncio.create_task(
                    self._search_single_collection_async(collection_name, query, k_per_collection, emb)
                )
                search_tasks.append(task)

        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                print(f"Search error: {result}")
                continue
            if result:
                all_docs.extend(result)

        # Sort by relevance (lower Chroma distance = more relevant), then dedupe so best-ranked copy of each chunk is kept first
        all_docs.sort(key=lambda d: d.metadata.get("chroma_distance", float("inf")))

        # Remove duplicates based on content similarity (keeps first = best distance)
        unique_docs = self._deduplicate_documents(all_docs)
        if start_time is not None:
            print(f"[{time.perf_counter()-t0:.2f}s] 🔍 Chroma done: {len(unique_docs)} unique docs from {len(queries)} query(s)")
        else:
            print(f"   🔍 Multi-query search completed: {len(unique_docs)} unique documents from {len(queries)} queries")
        return unique_docs

    async def _search_single_collection_async(self, collection_name: str, query: str, k: int, query_embedding=None):
        """Search a single collection asynchronously. Uses query_embedding if provided to avoid re-embedding."""
        try:
            retriever = get_collection_retriever(collection_name, k)
            if retriever:
                # Pass query_embedding so we don't call embed_query again per collection
                docs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: retriever.get_relevant_documents(query, query_embedding),
                )
                print(f"   🔍 Searched: {collection_name} with '{query[:50]}...' → Found {len(docs)} documents")
                return docs
        except Exception as e:
            print(f"Error searching {collection_name}: {e}")
        return []

    def _deduplicate_documents(self, documents: List) -> List:
        """Remove duplicate documents based on content similarity"""
        if not documents:
            return documents

        unique_docs = []
        seen_content = set()

        for doc in documents:
            # Create a simple hash of the content (first 200 chars)
            content_hash = hash(doc.page_content[:200].lower().strip())
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)

        return unique_docs

    def search_collections(self, query: str, collection_names: List[str], k_per_collection: int = 3):
        """Legacy synchronous search method for backward compatibility"""
        import asyncio
        import concurrent.futures
        try:
            # Get event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, use ThreadPoolExecutor to run async code
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.search_collections_async([query], collection_names, k_per_collection))
                    return future.result()
            else:
                # Create new loop
                return asyncio.run(self.search_collections_async([query], collection_names, k_per_collection))
        except RuntimeError:
            # Handle case where there's no event loop
            return asyncio.run(self.search_collections_async([query], collection_names, k_per_collection))

    async def __call_async(self, question: str, conversation_history: Optional[List[dict]] = None):
        """Asynchronous call method for better performance with caching"""
        # Generate unique query ID
        query_id = str(uuid.uuid4())

        print("🚀 multi-agent legal analysis...")

        # Check if decision-making is enabled and use LangGraph flow
        if self.decision_enabled and self.decision_flow:
            print("Using Decision-Making Flow...")
            try:
                # First, get basic legal context for decision-making
                basic_context = await self._get_basic_legal_context(question)

                decision_result = await self.decision_flow.process_query(question, legal_context=basic_context)

                if decision_result.get('response_type') == 'decision_making' and decision_result.get('decision_advice'):
                    # Format the decision-making response
                    formatted_response = f"""# 🤖 **Legal Decision-Making Assistant**

## 📋 **Query Analysis**
**Intent Detected:** {decision_result.get('intent_analysis', {}).get('intent', 'DECISION_MAKING')}
**Confidence:** {decision_result.get('intent_analysis', {}).get('confidence', 'HIGH')}
**Method:** {decision_result.get('intent_analysis', {}).get('method', 'structured_decision_making')}

## 💡 **Decision-Making Advice**

{decision_result.get('decision_advice', 'Unable to generate decision-making advice at this time.')}

---
**Processing Details:** Context used: {decision_result.get('context_used', 0)} documents | Method: {decision_result.get('method', 'structured_decision_making')}
"""

                    result = {
                        "result": formatted_response,
                        "source_documents": basic_context,  # Include the context used
                        "searched_collections": [],
                        "detected_domain": "decision_making",
                        "reformulated_queries": [],
                        "query_id": query_id,
                        "intent_analysis": decision_result.get('intent_analysis', {}),
                        "processing_path": decision_result.get('path_taken', 'decision_making')
                    }

                    # Track token usage asynchronously
                    asyncio.create_task(track_token_usage_async(query_id, question, formatted_response))

                    print("Decision-making response generated successfully!")
                    return result
                else:
                    print(f"Decision flow did not return valid advice: {decision_result}")
                    # Fall back to traditional RAG
            except Exception as e:
                print(f"Decision flow error: {e}")
                # Fall back to traditional RAG

        # Traditional RAG flow (fallback or when decision-making disabled)
        print("⚖️ Using Traditional RAG Flow...")

        # Step 1: Search all collections (no routing)
        collections_to_search = get_all_collections()
        domain = detect_query_domain(question)[0]  # for response metadata / primary_domain only
        print(f"📚 Searching all collections: {len(collections_to_search)} (no routing)")

        if not collections_to_search:
            return {
                "result": "❌ No legal document collections found. Please ensure documents have been ingested first.",
                "source_documents": [],
                "searched_collections": [],
                "query_id": query_id
            }

        # Check cache first
        cached_result = self.query_cache.get(question, collections_to_search)
        if cached_result:
            cached_result['query_id'] = query_id  # Update query ID
            return cached_result

        # Step 2: Query Reformulation using Gemini
        # print("🔄 Reformulating query")
        # reformulated_queries = self.reformulation_agent.reformulate(question)
        reformulated_queries = [question]  # Skip reformulation for now
        # For section-number queries (e.g. "section 302 of PPC"), add extra retrieval queries so the right chunk is found
        search_queries = [question] + extract_section_search_queries(question)
        # print(f"📝 Generated {len(reformulated_queries)} query variants")

        # Step 3: Vector search across all collections (single or multi-query for section-specific retrieval)
        print("🔍 Performing vector search across all collections...")
        source_docs = await self.search_collections_async(search_queries, collections_to_search, k_per_collection=10)

        if not source_docs:
            no_result = {
                "result": f"**No Relevant Information Found**\n\nI've searched through all available legal documents but couldn't find specific information about: '{question}'\n\n💡 **Suggestions:**\n• Be more specific about your legal query\n• Check if the relevant legal documents have been uploaded",
                "source_documents": [],
                "searched_collections": collections_to_search,
                "query_id": query_id
            }
            self.query_cache.set(question, collections_to_search, no_result)
            return no_result

        # Step 4: Use top docs directly (no reranking)
        top_n = getattr(settings, "RERANK_N", 12)
        top_docs = source_docs[:top_n]
        print(f"📊 Using top {len(top_docs)} documents from vector search")

        # Step 5: Combine legal context and generate answer
        legal_context_parts = []
        domains_used = set()

        for doc in top_docs:
            doc_domain = doc.metadata.get('domain', 'general')
            domains_used.add(doc_domain)

            legal_context_parts.append(doc.page_content)

        legal_context = "\n\n".join(legal_context_parts)
        self._log_rag_context(top_docs, legal_context)
        primary_domain = domain if domain != "general" else list(domains_used)[0] if domains_used else "general"

        # Format conversation history
        if conversation_history:
            history_str = "\n".join([f"{msg['sender'].capitalize()}: {msg['text']}" for msg in conversation_history])
        else:
            history_str = ""

        # Generate answer with enhanced prompt for better accuracy
        print("🤖 Generating legal response...")

        # Always try RAG first with enhanced prompt
        print("🤖 Generating response using enhanced RAG prompt...")
        enhanced_prompt = self._create_enhanced_prompt(
            history_str, legal_context, question, primary_domain, top_docs
        )
        answer = self.llm._call(enhanced_prompt)

        # Fallback only when first answer explicitly says context is missing info (e.g. bail) → get useful answer
        if self._is_inadequate_response(answer) or len(answer.strip()) < 80:
            print("⚠️ RAG context missing info, using knowledge fallback for useful answer...")
            answer = self._generate_knowledge_based_response(question, primary_domain, history_str, legal_context)
            print("✅ Fallback response generated")
        else:
            print("✅ RAG response adequate")

        # Track token usage asynchronously
        asyncio.create_task(track_token_usage_async(query_id, enhanced_prompt, answer))

        result = {
            "result": answer,
            "source_documents": top_docs,
            "searched_collections": collections_to_search,
            "detected_domain": domain,
            "reformulated_queries": reformulated_queries,
            "query_id": query_id
        }

        # Cache the successful result
        self.query_cache.set(question, collections_to_search, result)

        print("✅ Response generated and cached successfully!")

        return result

    async def prepare_rag_for_stream(self, question: str, conversation_history: Optional[List[dict]] = None, start_time: Optional[float] = None):
        """Run RAG (search all collections, build prompt) without calling the LLM. Returns dict with enhanced_prompt etc., or {'_cached': result} if cache hit."""
        t0 = start_time if start_time is not None else time.time()
        query_id = str(uuid.uuid4())
        collections_to_search = get_all_collections()
        domain = detect_query_domain(question)[0]  # for response metadata only
        print(f"[{time.perf_counter()-t0:.2f}s] 📚 Searching all collections: {len(collections_to_search)} (no routing)")
        if not collections_to_search:
            return {"_early": {"result": "❌ No legal document collections found.", "source_documents": [], "searched_collections": [], "query_id": query_id}}
        cached_result = self.query_cache.get(question, collections_to_search)
        if cached_result:
            print(f"[{time.perf_counter()-t0:.2f}s] 📋 Cache HIT")
            cached_result["query_id"] = query_id
            return {"_cached": cached_result}
        print(f"[{time.perf_counter()-t0:.2f}s] 📋 Cache miss → embedding query & searching Chroma...")
        search_queries = [question] + extract_section_search_queries(question)
        source_docs = await self.search_collections_async(search_queries, collections_to_search, k_per_collection=12, start_time=t0)
        if not source_docs:
            no_result = {
                "result": f"**No Relevant Information Found**\n\nI've searched through all available legal documents but couldn't find specific information about: '{question}'.",
                "source_documents": [], "searched_collections": get_all_collections(), "query_id": query_id
            }
            return {"_early": no_result}
        top_n = getattr(settings, "RERANK_N", 12)
        top_docs = source_docs[:top_n]
        print(f"[{time.perf_counter()-t0:.2f}s] 📄 Got top {len(top_docs)} chunks ({sum(len(d.page_content) for d in top_docs)} chars total)")
        legal_context = "\n\n".join(doc.page_content for doc in top_docs)
        self._log_rag_context(top_docs, legal_context)
        domains_used = set(doc.metadata.get("domain", "general") for doc in top_docs)
        primary_domain = domain if domain != "general" else (list(domains_used)[0] if domains_used else "general")
        history_str = "\n".join([f"{m['sender'].capitalize()}: {m['text']}" for m in conversation_history or []])
        enhanced_prompt = self._create_enhanced_prompt(history_str, legal_context, question, primary_domain, top_docs)
        print(f"[{time.perf_counter()-t0:.2f}s] 📝 Built prompt ({len(enhanced_prompt)} chars) → ready for LLM")
        return {
            "query_id": query_id,
            "enhanced_prompt": enhanced_prompt,
            "top_docs": top_docs,
            "primary_domain": primary_domain,
            "collections_to_search": collections_to_search,
            "domain": domain,
            "reformulated_queries": [question],
        }

    def __call__(self, question: str, conversation_history: Optional[List[dict]] = None):
        """Synchronous call method for backward compatibility"""
        import concurrent.futures
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, use ThreadPoolExecutor to run async code in a separate thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.__call_async(question, conversation_history))
                    return future.result()
            else:
                return loop.run_until_complete(self.__call_async(question, conversation_history))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.__call_async(question, conversation_history))

    async def _get_basic_legal_context(self, question: str) -> List:
        """Get basic legal context for decision-making before full RAG (searches all collections)."""
        try:
            collections_to_search = get_all_collections()
            if not collections_to_search:
                return []

            reformulated_queries = [question]
            basic_docs = await self.search_collections_async(reformulated_queries[:2], collections_to_search, k_per_collection=2)
            return basic_docs[:5]  # Limit to 5 docs for decision-making

        except Exception as e:
            print(f"Warning: Could not get basic legal context: {e}")
            return []

    def _sync_call(self, question: str, conversation_history: Optional[List[dict]] = None):
        """Synchronous fallback method"""
        # Generate unique query ID
        query_id = str(uuid.uuid4())

        print("🚀 Starting legal analysis (sync mode)...")

        # Step 1: Search all collections (no routing)
        collections_to_search = get_all_collections()
        domain = detect_query_domain(question)[0]  # for response metadata only
        print(f"📚 Searching all collections: {len(collections_to_search)} (no routing)")

        if not collections_to_search:
            return {
                "result": "❌ No legal document collections found. Please ensure documents have been ingested first.",
                "source_documents": [],
                "searched_collections": [],
                "query_id": query_id
            }

        # Step 2: Search all collections
        print("🔍 Searching legal databases...")
        source_docs = self.search_collections(question, collections_to_search, k_per_collection=3)

        if not source_docs:
            return {
                "result": f"**No Relevant Information Found**\n\nI've searched through all available legal documents but couldn't find specific information about: '{question}'\n\n💡 **Suggestions:**\n• Be more specific about your legal query\n• Check if the relevant legal documents have been uploaded",
                "source_documents": [],
                "searched_collections": collections_to_search,
                "query_id": query_id
            }

        # Step 3: Combine legal context and generate answer
        legal_context_parts = []
        domains_used = set()

        for doc in source_docs:
            source_info = f"Page {doc.metadata.get('page_number', 'N/A')}"
            collection = doc.metadata.get('collection_name', 'Unknown')
            doc_domain = doc.metadata.get('domain', 'general')
            domains_used.add(doc_domain)

            legal_context_parts.append(f"{doc.page_content}\n[Source: {source_info} | Collection: {collection}]")

        legal_context = "\n\n".join(legal_context_parts)
        primary_domain = domain if domain != "general" else list(domains_used)[0] if domains_used else "general"

        # Format conversation history
        if conversation_history:
            history_str = "\n".join([f"{msg['sender'].capitalize()}: {msg['text']}" for msg in conversation_history])
        else:
            history_str = ""

        # Generate answer
        print("🤖 Generating legal response...")
        formatted_prompt = self.prompt.format(
            history=history_str,
            context=legal_context,
            question=question,
            domain=primary_domain.upper()
        )

        answer = self.llm._call(formatted_prompt)

        # Track token usage asynchronously (don't await to avoid blocking)
        import asyncio
        asyncio.create_task(track_token_usage_async(query_id, formatted_prompt, answer))

        print("✅ Response generated successfully!")

        return {
            "result": answer,
            "source_documents": source_docs,
            "searched_collections": collections_to_search,
            "detected_domain": domain,
            "query_id": query_id
        }