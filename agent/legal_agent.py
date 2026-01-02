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
from rank_bm25 import BM25Okapi
from agent.gemini_llm import gemini_llm, perplexity_llm
import numpy as np
import asyncio
import hashlib
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
import time
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
def create_chroma_client():
    """Create Chroma Cloud client with direct credentials"""
    client = chromadb.CloudClient(
        api_key=settings.CHROMA_API_KEY,
        tenant=settings.CHROMA_TENANT,
        database=settings.CHROMA_DATABASE
    )
    return client

def get_all_collections():
    """Get all available collections"""
    client = create_chroma_client()
    try:
        collections = client.list_collections()
        return [coll.name for coll in collections]
    except Exception as e:
        print(f"❌ Error getting collections: {e}")
        return []

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
    """Create a retriever for a specific collection with hybrid search capabilities"""
    client = create_chroma_client()

    try:
        collection = client.get_collection(name=collection_name)

        class HybridCollectionRetriever:
            def __init__(self, collection, k=4):
                self.collection = collection
                self.k = k
                self.collection_name = collection_name
                self.bm25_index = None
                self.documents = []
                self.doc_ids = []
                self._build_bm25_index()

            def _build_bm25_index(self):
                """Build BM25 index for keyword search"""
                try:
                    # Get all documents from collection with correct include parameters
                    results = self.collection.get(include=['documents', 'metadatas'])
                    if results['documents'] and results['metadatas']:
                        self.documents = results['documents']
                        self.doc_ids = [meta.get('doc_id', f'doc_{i}') for i, meta in enumerate(results['metadatas'])]
                        # Tokenize documents for BM25 with better preprocessing
                        tokenized_docs = []
                        for doc in self.documents:
                            # Clean and tokenize the document
                            tokens = doc.lower().replace('\n', ' ').replace('\t', ' ').split()
                            # Remove very short tokens and common stop words
                            filtered_tokens = [token for token in tokens if len(token) > 2]
                            tokenized_docs.append(filtered_tokens)
                        self.bm25_index = BM25Okapi(tokenized_docs)
                        print(f"Successfully built BM25 index for {self.collection_name} with {len(tokenized_docs)} documents")
                except Exception as e:
                    print(f"Warning: Could not build BM25 index for {self.collection_name}: {e}")
                    self.bm25_index = None

            def _keyword_search(self, query: str, top_k: int = 10):
                """Perform enhanced BM25 keyword search with legal term extraction and exact matching"""
                if not self.bm25_index:
                    return []

                # Enhanced legal query preprocessing
                query_lower = query.lower()
                import re

                # Extract legal-specific terms with better patterns
                legal_patterns = [
                    r'section\s+(\d+)', r'article\s+(\d+)', r'clause\s+(\d+)',
                    r'chapter\s+(\d+)', r'part\s+(\d+)', r'sub-section\s+(\d+)',
                    r'paragraph\s+(\d+)', r'schedule\s+(\d+)'
                ]

                legal_terms = []
                for pattern in legal_patterns:
                    matches = re.findall(pattern, query_lower, re.IGNORECASE)
                    for match in matches:
                        full_match = re.search(pattern, query_lower, re.IGNORECASE)
                        if full_match:
                            legal_terms.append(full_match.group(0))

                # Extract standalone numbers (likely section references)
                numbers = re.findall(r'\b\d+\b', query_lower)
                section_numbers = [num for num in numbers if len(num) <= 4]  # Section numbers are typically short

                # Build comprehensive search terms
                query_tokens = []
                query_tokens.extend(query_lower.split())  # Original tokens
                query_tokens.extend(legal_terms)  # Legal references
                query_tokens.extend(section_numbers)  # Section numbers
                query_tokens.extend([f"section {num}" for num in section_numbers])  # Section prefixes

                # Add legal domain-specific terms
                legal_keywords = ['law', 'act', 'ordinance', 'code', 'court', 'justice', 'legal', 'provision', 'ppc', 'penal', 'criminal']
                query_tokens.extend([kw for kw in legal_keywords if kw in query_lower])

                # Remove duplicates while preserving order
                seen = set()
                query_tokens = [x for x in query_tokens if not (x in seen or seen.add(x))]

                # Remove very short tokens that aren't numbers
                query_tokens = [token for token in query_tokens if len(token) > 1 or token.isdigit()]

                # Get BM25 scores
                bm25_scores = self.bm25_index.get_scores(query_tokens)
                top_indices = np.argsort(bm25_scores)[::-1][:top_k]

                keyword_docs = []
                for idx in top_indices:
                    score = bm25_scores[idx]
                    # Dynamic threshold based on score distribution
                    threshold = 0.01 if len(keyword_docs) < 3 else 0.1  # Lower threshold initially for recall
                    if score > threshold:
                        from langchain_core.documents import Document
                        doc = Document(
                            page_content=self.documents[idx],
                            metadata={
                                'doc_id': self.doc_ids[idx],
                                'collection_name': self.collection_name,
                                'search_score': float(score),
                                'search_type': 'keyword',
                                'matched_tokens': query_tokens,
                                'legal_terms_found': legal_terms,
                                'section_numbers': section_numbers
                            }
                        )
                        keyword_docs.append(doc)

                # If no results with BM25, try exact text matching as fallback
                if not keyword_docs:
                    exact_matches = []
                    for idx, doc_content in enumerate(self.documents):
                        doc_lower = doc_content.lower()
                        # Check for exact legal term matches
                        if any(term.lower() in doc_lower for term in legal_terms):
                            exact_matches.append(idx)
                        # Check for section number matches
                        elif any(f"section {num}" in doc_lower for num in section_numbers):
                            exact_matches.append(idx)
                        elif any(num in doc_lower for num in section_numbers):
                            exact_matches.append(idx)

                    for idx in exact_matches[:top_k]:
                        from langchain_core.documents import Document
                        doc = Document(
                            page_content=self.documents[idx],
                            metadata={
                                'doc_id': self.doc_ids[idx],
                                'collection_name': self.collection_name,
                                'search_score': 1.0,  # High score for exact matches
                                'search_type': 'keyword_exact',
                                'matched_tokens': query_tokens,
                                'legal_terms_found': legal_terms,
                                'section_numbers': section_numbers
                            }
                        )
                        keyword_docs.append(doc)

                return keyword_docs

            def _semantic_search(self, query: str, top_k: int = 10):
                """Perform semantic vector search using BGE-M3 with improved error handling"""
                try:
                    # Embed query using HF API with timeout
                    query_emb = get_hf_embeddings([query], settings.HF_API_KEY, settings.HF_EMBED_MODEL)[0]
                    results = self.collection.query(
                        query_embeddings=[query_emb],
                        n_results=top_k
                    )

                    documents = []
                    if results['documents']:
                        for doc, metadata, doc_id in zip(
                            results['documents'][0],
                            results['metadatas'][0],
                            results['ids'][0]
                        ):
                            from langchain_core.documents import Document
                            document = Document(
                                page_content=doc,
                                metadata={
                                    **metadata,
                                    'doc_id': doc_id,
                                    'collection_name': self.collection_name,
                                    'search_type': 'semantic'
                                }
                            )
                            documents.append(document)
                    return documents
                except Exception as e:
                    print(f"Semantic search failed for {self.collection_name}: {type(e).__name__}: {str(e)}")
                    # Don't fall back here, let the caller handle it
                    return []

            def _hybrid_rerank(self, semantic_docs: List, keyword_docs: List, alpha: float = 0.5):
                """Enhanced hybrid reranking with legal content prioritization"""
                # Create a combined set of unique documents
                all_docs = {}
                doc_scores = {}
                doc_sources = {}
                doc_metadata = {}  # Store additional scoring metadata

                # Add semantic search results with base weight
                for doc in semantic_docs:
                    doc_id = doc.metadata.get('doc_id')
                    all_docs[doc_id] = doc
                    doc_scores[doc_id] = alpha
                    doc_sources[doc_id] = 'semantic'
                    doc_metadata[doc_id] = {'semantic_score': alpha, 'keyword_score': 0}

                # Add keyword search results with enhanced legal scoring
                for doc in keyword_docs:
                    doc_id = doc.metadata.get('doc_id')
                    base_score = doc.metadata.get('search_score', 0)

                    # Enhanced legal content scoring
                    content = doc.page_content.lower()
                    legal_boost = 0
                    precision_boost = 0

                    # Extract legal terms and section numbers from metadata
                    legal_terms = doc.metadata.get('legal_terms_found', [])
                    section_numbers = doc.metadata.get('section_numbers', [])
                    matched_tokens = doc.metadata.get('matched_tokens', [])

                    # Boost for exact legal term matches
                    for term in legal_terms:
                        if term.lower() in content:
                            legal_boost += 0.5  # High boost for legal terms

                    # Boost for section number matches
                    for num in section_numbers:
                        if num in content:
                            precision_boost += 0.8  # Very high boost for section numbers
                        # Also check for "section X" patterns
                        if f"section {num}" in content:
                            precision_boost += 1.0  # Maximum boost for exact section references

                    # Boost for legal keywords
                    legal_keywords = ['shall', 'provided that', 'notwithstanding', 'hereby', 'hereinafter']
                    for keyword in legal_keywords:
                        if keyword in content:
                            legal_boost += 0.1

                    # Calculate final keyword score
                    keyword_score = (1 - alpha) * (base_score + legal_boost + precision_boost)

                    if doc_id in all_docs:
                        # Combine scores for documents found in both searches
                        doc_scores[doc_id] += keyword_score
                        doc_sources[doc_id] = 'hybrid'
                        doc_metadata[doc_id]['keyword_score'] = keyword_score
                        doc_metadata[doc_id]['combined_score'] = doc_scores[doc_id]
                    else:
                        # New document from keyword search
                        all_docs[doc_id] = doc
                        doc_scores[doc_id] = keyword_score
                        doc_sources[doc_id] = 'keyword'
                        doc_metadata[doc_id] = {
                            'semantic_score': 0,
                            'keyword_score': keyword_score,
                            'combined_score': keyword_score
                        }

                # Apply legal relevance filtering - prioritize documents with legal content
                filtered_docs = []
                for doc_id, doc in all_docs.items():
                    score = doc_scores[doc_id]
                    content = doc.page_content.lower()

                    # Minimum relevance threshold
                    if score < 0.1:
                        continue

                    # Boost documents that contain legal structure indicators
                    legal_indicators = ['section', 'article', 'clause', 'provided', 'shall', 'act', 'law', 'ordinance']
                    legal_indicator_count = sum(1 for indicator in legal_indicators if indicator in content)

                    if legal_indicator_count > 0:
                        score += legal_indicator_count * 0.05  # Small boost for legal content

                    # Update final score
                    doc_scores[doc_id] = score
                    doc.metadata['final_score'] = score
                    doc.metadata['search_source'] = doc_sources[doc_id]
                    filtered_docs.append(doc)

                # Sort by final score and return top k
                sorted_docs = sorted(filtered_docs, key=lambda x: doc_scores[x.metadata['doc_id']], reverse=True)
                return sorted_docs[:self.k]

            def get_relevant_documents(self, query):
                """Sequential search: text/regex first, then semantic/hybrid if needed"""
                try:
                    # Step 1: Try keyword search (text/regex) first for speed and accuracy
                    keyword_docs = self._keyword_search(query, top_k=self.k * 2)

                    if keyword_docs:
                        # If keyword search found results, return them (rerank if multiple)
                        if len(keyword_docs) > self.k:
                            # Simple reranking based on score for keyword results
                            keyword_docs.sort(key=lambda x: x.metadata.get('search_score', 0), reverse=True)
                        return keyword_docs[:self.k]

                    # Step 2: If keyword search failed, try semantic search
                    print(f"Keyword search found no results for {self.collection_name}, trying semantic search...")
                    semantic_docs = self._semantic_search(query, top_k=self.k * 2)

                    if semantic_docs:
                        return semantic_docs[:self.k]

                    # Step 3: If both failed, try hybrid as last resort
                    print(f"Semantic search also failed for {self.collection_name}, attempting hybrid fallback...")
                    # Do both searches for hybrid reranking
                    semantic_docs = self._semantic_search(query, top_k=self.k)
                    keyword_docs = self._keyword_search(query, top_k=self.k)

                    if semantic_docs or keyword_docs:
                        combined_docs = self._hybrid_rerank(semantic_docs, keyword_docs)
                        return combined_docs

                    return []

                except Exception as e:
                    print(f"Error in sequential search for {self.collection_name}: {e}")
                    # Ultimate fallback: try keyword search only
                    try:
                        return self._keyword_search(query, top_k=self.k)
                    except:
                        return []

        return HybridCollectionRetriever(collection, k=k)

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

# ------------------ HF Inference Functions ------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_hf_embeddings(texts, api_key, model):
    """Get embeddings from Hugging Face Inference API with retry"""
    url = f"https://api-inference.huggingface.co/embeddings/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"inputs": texts}
    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    return response.json()

def get_reranker_scores_batch(inputs, api_key, model):
    """Get reranker scores from Hugging Face Inference API with fallback models"""
    # Try multiple reranker models in order of preference (prioritizing speed)
    models_to_try = [
        model,  # Primary model from settings (now TinyBERT - fast and lightweight)
        "cross-encoder/ms-marco-MiniLM-L-6-v2",  # MiniLM fallback (still lightweight)
        "BAAI/bge-reranker-base"   # BGE base as final fallback
    ]

    for current_model in models_to_try:
        try:
            url = f"https://router.huggingface.co/hf-inference/models/{current_model}"
            headers = {"Authorization": f"Bearer {api_key}"}
            data = {"inputs": inputs}

            print(f"🔄 Trying reranker model: {current_model}")
            response = requests.post(url, headers=headers, json=data, timeout=60)

            if response.status_code == 404:
                print(f"⚠️ Model {current_model} not found (404), trying next model...")
                continue
            elif response.status_code != 200:
                print(f"⚠️ Reranker API Error for {current_model}: {response.status_code} - {response.text}")
                continue

            response.raise_for_status()
            scores = response.json()
            print(f"✅ Successfully used reranker model: {current_model}")
            return scores  # list of scores

        except Exception as e:
            print(f"⚠️ Error with model {current_model}: {type(e).__name__}: {str(e)}")
            continue

    # If all models fail, raise the last exception
    raise Exception(f"All reranker models failed. Last error: {str(e) if 'e' in locals() else 'Unknown error'}")

def rerank_with_bge(question, docs):
    """Rerank documents using lightweight reranker with optimized performance"""
    if len(docs) <= 3:
        # For small result sets, skip reranking to avoid API calls
        print(f"📊 Small result set ({len(docs)} docs), skipping reranking")
        return docs

    # Limit reranking to configured maximum for speed
    max_docs_to_rerank = min(len(docs), settings.MAX_RERANK_DOCS)
    docs_to_rerank = docs[:max_docs_to_rerank]

    try:
        inputs = []
        for doc in docs_to_rerank:
            # Use simpler input format for lightweight models
            text = f"{question} [SEP] {doc.page_content[:500]}"  # Limit content length for speed
            inputs.append(text)

        print(f"🧮 Computing reranker scores for {len(inputs)} doc pairs (limited to {max_docs_to_rerank} for speed)...")
        scores = get_reranker_scores_batch(inputs, settings.HF_API_KEY, settings.HF_RERANKER_MODEL)

        # Sort docs by scores descending
        doc_score_pairs = list(zip(docs_to_rerank, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return reranked docs plus any remaining docs in original order
        reranked_docs = [doc for doc, score in doc_score_pairs]
        if len(docs) > max_docs_to_rerank:
            reranked_docs.extend(docs[max_docs_to_rerank:])

        print(f"✅ Lightweight reranking completed successfully")
        return reranked_docs

    except Exception as e:
        print(f"⚠️ Lightweight reranker failed: {type(e).__name__}: {str(e)}")
        print(f"📊 Falling back to keyword-based reranking")
        # Fallback to simple keyword-based reranking
        return _fallback_keyword_rerank(question, docs)

def _fallback_keyword_rerank(question, docs):
    """Simple keyword-based reranking as fallback when BGE fails"""
    question_lower = question.lower()
    question_words = set(question_lower.split())

    doc_scores = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        score = 0

        # Count exact word matches
        for word in question_words:
            if len(word) > 2:  # Skip very short words
                score += content_lower.count(word)

        # Boost for legal terms
        legal_terms = ['section', 'article', 'clause', 'act', 'law', 'court', 'case', 'provision']
        for term in legal_terms:
            if term in content_lower:
                score += 0.5

        doc_scores.append((doc, score))

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"✅ Keyword-based reranking completed for {len(docs)} documents")
    return [doc for doc, score in doc_scores]

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
        """Check if the RAG-generated response indicates failure or inadequacy"""
        if not response or len(response.strip()) < 50:
            return True

        response_lower = response.lower()

        # Check for inadequate response indicators
        inadequate_indicators = [
            "unable to provide information",
            "unable to find",
            "no relevant information",
            "No Relevant Information",
            "no information available",
            "i am unable to",
            "i apologize",
            "sorry",
            "cannot provide",
            "not available in the context",
            "the provided context does not contain",
            "does not contain any information",
            "i can provide information on the following sections",
            "i can only provide information on",
            "here are some other sections"
        ]

        return any(indicator in response_lower for indicator in inadequate_indicators)

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

    async def search_collections_async(self, queries: List[str], collection_names: List[str], k_per_collection: int = 3):
        """Asynchronously search across multiple collections with multiple query formulations"""
        all_docs = []

        # Create tasks for parallel search
        search_tasks = []

        for query in queries:
            for collection_name in collection_names:
                task = asyncio.create_task(
                    self._search_single_collection_async(collection_name, query, k_per_collection)
                )
                search_tasks.append(task)

        # Execute all searches in parallel
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                print(f"Search error: {result}")
                continue
            if result:
                all_docs.extend(result)

        # Remove duplicates based on content similarity
        unique_docs = self._deduplicate_documents(all_docs)

        print(f"   🔍 Multi-query search completed: {len(unique_docs)} unique documents from {len(queries)} queries")
        return unique_docs

    async def _search_single_collection_async(self, collection_name: str, query: str, k: int):
        """Search a single collection asynchronously"""
        try:
            retriever = get_collection_retriever(collection_name, k)
            if retriever:
                docs = await asyncio.get_event_loop().run_in_executor(
                    None, retriever.get_relevant_documents, query
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

        # Step 1: Query Classification using Gemini
        # print("🧠 Classifying query domain using Gemini...")
        classification = self.classification_agent.classify(question)
        domain = classification['primary_domain']
        all_domains = classification['all_domains']

        print(f"🎯 Classified Domain: {domain.upper()} (confidence: {classification['confidence']})")

        # Get collections for all relevant domains
        if domain == 'general':
            # For general queries, immediately search all collections in parallel for maximum coverage
            collections_to_search = get_all_collections()
            print(f"📚 General query detected - searching all {len(collections_to_search)} collections in parallel")
        else:
            collections_to_search = []
            for dom in all_domains:
                domain_collections = [coll for coll in get_all_collections() if dom in coll]
                collections_to_search.extend(domain_collections)

            # Remove duplicates
            collections_to_search = list(set(collections_to_search))

            if not collections_to_search:
                return {
                    "result": "❌ No legal document collections found. Please ensure documents have been ingested first.",
                    "source_documents": [],
                    "searched_collections": [],
                    "query_id": query_id
                }

        print(f"📚 Collections to search: {len(collections_to_search)}")

        # Check cache first
        cached_result = self.query_cache.get(question, collections_to_search)
        if cached_result:
            cached_result['query_id'] = query_id  # Update query ID
            return cached_result

        # Step 2: Query Reformulation using Gemini
        print("🔄 Reformulating query")
        reformulated_queries = self.reformulation_agent.reformulate(question)
        print(f"📝 Generated {len(reformulated_queries)} query variants")

        # Step 3: Parallel Multi-Query Search with enhanced retrieval
        print("🔍 Performing enhanced hybrid search across collections...")
        print(f"   Search queries: {reformulated_queries}")
        source_docs = await self.search_collections_async(reformulated_queries, collections_to_search, k_per_collection=4)

        if not source_docs:
            # Fallback: Search all collections with broader search
            print("⚠️  No results in targeted search. Expanding search to all collections...")
            all_collections = get_all_collections()
            source_docs = await self.search_collections_async(reformulated_queries, all_collections, k_per_collection=1)

            if not source_docs:
                no_result = {
                    "result": f"**No Relevant Information Found**\n\nI've searched through all available legal documents but couldn't find specific information about: '{question}'\n\n💡 **Suggestions:**\n• Be more specific about your legal query\n• Check if the relevant legal documents have been uploaded\n• Specify which domain of law you're interested in (Traffic, Family, Corporate)",
                    "source_documents": [],
                    "searched_collections": all_collections,
                    "query_id": query_id
                }
                # Cache negative results too (for shorter time)
                self.query_cache.set(question, collections_to_search, no_result)
                return no_result

        # Step 4: BGE reranking for high accuracy
        print(f"📊 Retrieved {len(source_docs)} total documents, applying BGE reranking...")
        if len(source_docs) > settings.TOP_K:
            # Retrieve top_k by hybrid search, then rerank top_n
            candidate_docs = source_docs[:settings.TOP_K]
            reranked_docs = rerank_with_bge(question, candidate_docs)
            top_docs = reranked_docs[:settings.RERANK_N]
            print(f" 📊  Retrieved top {settings.TOP_K}, reranked to top {len(top_docs)} documents")
        elif len(source_docs) > settings.RERANK_N:
            # Rerank available docs
            reranked_docs = rerank_with_bge(question, source_docs)
            top_docs = reranked_docs[:settings.RERANK_N]
            print(f" 📊  Reranked to top {len(top_docs)} documents")
        else:
            # For smaller result sets, use directly
            top_docs = source_docs[:settings.RERANK_N]
            print(f"Using top {len(top_docs)} documents directly (small result set)")

        # Step 5: Combine legal context and generate answer
        legal_context_parts = []
        domains_used = set()

        for doc in top_docs:
            doc_domain = doc.metadata.get('domain', 'general')
            domains_used.add(doc_domain)

            legal_context_parts.append(doc.page_content)

        legal_context = "\n\n".join(legal_context_parts)
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

        # Enhanced response validation - check for adequacy
        if self._is_inadequate_response(answer) or len(answer.strip()) < 100:
            print("⚠️ RAG content inadequate, using external resources and comprehensive legal knowledge as fallback...")
            answer = self._generate_knowledge_based_response(question, primary_domain, history_str, legal_context)
            print("✅ Fallback response generated using legal knowledge")
        else:
            print("✅ RAG response adequate, proceeding with retrieved content")

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
        """Get basic legal context for decision-making before full RAG"""
        try:
            # Quick domain detection
            domain, collections_to_search = detect_query_domain(question)

            if not collections_to_search:
                return []

            # Get a few documents quickly for context
            reformulated_queries = self.reformulation_agent.reformulate(question)
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

        # Step 1: Detect domain and relevant collections (legacy method)
        print("🧠 Analyzing query domain...")
        domain, collections_to_search = detect_query_domain(question)

        print(f"🎯 Detected Domain: {domain.upper()}")
        print(f"📚 Collections to search: {len(collections_to_search)}")

        if not collections_to_search:
            return {
                "result": "❌ No legal document collections found. Please ensure documents have been ingested first.",
                "source_documents": [],
                "searched_collections": [],
                "query_id": query_id
            }

        # Step 2: Search relevant collections
        print("🔍 Searching legal databases...")
        source_docs = self.search_collections(question, collections_to_search, k_per_collection=3)

        if not source_docs:
            # Fallback: Search all collections with broader search
            print("⚠️  No results in targeted search. Expanding search to all collections...")
            all_collections = get_all_collections()
            source_docs = self.search_collections(question, all_collections, k_per_collection=2)

            if not source_docs:
                return {
                    "result": f"**No Relevant Information Found**\n\nI've searched through all available legal documents but couldn't find specific information about: '{question}'\n\n💡 **Suggestions:**\n• Be more specific about your legal query\n• Check if the relevant legal documents have been uploaded\n• Specify which domain of law you're interested in (Traffic, Family, Corporate)",
                    "source_documents": [],
                    "searched_collections": all_collections,
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