import os
import requests
import chromadb
import uuid
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import LLM
from typing import Optional, List, Tuple
from chromadb.utils import embedding_functions
from config.settings import settings
from helper.token_tracker import track_token_usage_async
from rank_bm25 import BM25Okapi
from agent.gemini_llm import gemini_llm
import numpy as np
import asyncio
import hashlib
import json
from functools import lru_cache
import time

# ------------------ OpenRouter Config ------------------
OPENROUTER_API_URL = settings.OPENROUTER_API_URL
OPENROUTER_MODEL = settings.OPENROUTER_MODEL
OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY

# Domain detection keywords
DOMAIN_KEYWORDS = {
    'traffic': ['traffic', 'updated fines', 'fines', 'fine', 'motor vehicle', 'driving', 'license', 'transport', 'road', 'accident', 'speeding', 'vehicle registration'],
    
    'family': ['family', 'marriage','witness','nikah witness','nikah', 'divorce', 'inheritance', 'guardian', 'child', 'maintenance', 'custody', 'dowry','dower','marital', "alimony", "child support", "adoption", "domestic violence", "family dispute", "nikaah", "mehr", "talaq", "khula", "wasiat","pakistan family law","family court","family act","family ordinance"],
    
    'corporate': ['corporate', 'company', 'business', 'commercial', 'contract', 'partnership', 'incorporation', 'shareholder', 'director', 'board','leaves', 'employee', 'employment', 'labor', 'workplace', 'hr', 'human resources', 'termination', 'hiring', 'firing', 'work hours', 'overtime', 'payroll', 'benefits', 'discrimination', 'harassment', 'workplace safety','pakistan labor law','pakistan employment law','labor court','employment act','industrial relations','maternity', 'paternity', 'casual', 'sick leave', 'annual leave', 'leave policy'],
    
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

    def _call_with_fallback(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call Gemini first (3 attempts), then fallback to OpenRouter"""
        from agent.gemini_llm import gemini_llm

        # Try Gemini first (3 attempts)
        print("🤖 Generating legal response...")
        gemini_response = gemini_llm._call_with_retry(prompt, stop)
        if not gemini_response.startswith("[Gemini Error]"):
            print("✅ Gemini API call successful")
            return gemini_response

        print("⚠️ Gemini failed, falling back to OpenRouter...")

        # Fallback to OpenRouter (1 attempt with rate limiting)
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

        try:
            # Enforce rate limiting
            self._enforce_rate_limit()

            print("Calling OpenRouter API (fallback attempt)...")
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)

            if resp.status_code == 429:
                retry_after = resp.headers.get('retry-after')
                wait_time = int(retry_after) if retry_after else 5
                print(f"Rate limited (429). Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                # One more try after waiting
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)

            resp.raise_for_status()
            data = resp.json()
            print("✅ OpenRouter fallback successful")
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"❌ Both Gemini and OpenRouter failed: {e}")
            return f"[API Error] Both primary and fallback LLM services failed. Please try again later. Error: {e}"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Main call method with Gemini-first, OpenRouter-fallback strategy"""
        return self._call_with_fallback(prompt, stop)

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

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

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
                    # Get all documents from collection
                    results = self.collection.get(include=['documents', 'metadatas', 'ids'])
                    if results['documents']:
                        self.documents = results['documents']
                        self.doc_ids = results['ids']
                        # Tokenize documents for BM25
                        tokenized_docs = [doc.lower().split() for doc in self.documents]
                        self.bm25_index = BM25Okapi(tokenized_docs)
                except Exception as e:
                    print(f"Warning: Could not build BM25 index for {self.collection_name}: {e}")

            def _keyword_search(self, query: str, top_k: int = 10):
                """Perform BM25 keyword search"""
                if not self.bm25_index:
                    return []

                query_tokens = query.lower().split()
                bm25_scores = self.bm25_index.get_scores(query_tokens)
                top_indices = np.argsort(bm25_scores)[::-1][:top_k]

                keyword_docs = []
                for idx in top_indices:
                    if bm25_scores[idx] > 0:  # Only include relevant matches
                        from langchain_core.documents import Document
                        doc = Document(
                            page_content=self.documents[idx],
                            metadata={
                                'doc_id': self.doc_ids[idx],
                                'collection_name': self.collection_name,
                                'search_score': float(bm25_scores[idx]),
                                'search_type': 'keyword'
                            }
                        )
                        keyword_docs.append(doc)
                return keyword_docs

            def _semantic_search(self, query: str, top_k: int = 10):
                """Perform semantic vector search"""
                try:
                    results = self.collection.query(
                        query_texts=[query],
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
                    print(f"Error in semantic search for {self.collection_name}: {e}")
                    return []

            def _hybrid_rerank(self, semantic_docs: List, keyword_docs: List, alpha: float = 0.7):
                """Combine semantic and keyword results with reranking"""
                # Create a combined set of unique documents
                all_docs = {}
                doc_scores = {}

                # Add semantic search results
                for doc in semantic_docs:
                    doc_id = doc.metadata.get('doc_id')
                    all_docs[doc_id] = doc
                    doc_scores[doc_id] = alpha  # Base score for semantic

                # Add keyword search results
                for doc in keyword_docs:
                    doc_id = doc.metadata.get('doc_id')
                    score = doc.metadata.get('search_score', 0)
                    if doc_id in all_docs:
                        # Combine scores for documents found in both
                        doc_scores[doc_id] += (1 - alpha) * score
                    else:
                        # New document from keyword search
                        all_docs[doc_id] = doc
                        doc_scores[doc_id] = (1 - alpha) * score

                # Sort by combined score and return top k
                sorted_docs = sorted(all_docs.values(), key=lambda x: doc_scores[x.metadata['doc_id']], reverse=True)
                return sorted_docs[:self.k]

            def get_relevant_documents(self, query):
                """Hybrid search combining semantic and keyword search"""
                try:
                    # Parallel execution of semantic and keyword search
                    semantic_docs = self._semantic_search(query, top_k=self.k * 2)
                    keyword_docs = self._keyword_search(query, top_k=self.k * 2)

                    # Combine and rerank results
                    combined_docs = self._hybrid_rerank(semantic_docs, keyword_docs)

                    return combined_docs

                except Exception as e:
                    print(f"Error in hybrid search for {self.collection_name}: {e}")
                    # Fallback to semantic search only
                    return self._semantic_search(query, top_k=self.k)

        return HybridCollectionRetriever(collection, k=k)

    except Exception as e:
        print(f"❌ Failed to load collection {collection_name}: {e}")
        return None

# ------------------ Query Reformulation Agent ------------------
class QueryReformulationAgent:
    def __init__(self, llm):
        self.llm = llm
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
        """Reformulate the query into multiple search variants using Gemini"""
        try:
            prompt = self.prompt.format(question=question)
            response = gemini_llm._call(prompt)

            # Parse the response to extract queries
            queries = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('- Query') or line.startswith('Query'):
                    # Extract the query text after the colon
                    if ':' in line:
                        query = line.split(':', 1)[1].strip()
                        if query:
                            queries.append(query)

            # If parsing failed, return original and basic variants
            if not queries:
                queries = [question]

            # Ensure we have at least the original
            if question not in queries:
                queries.insert(0, question)

            return queries[:3]  # Limit to 3 queries

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
        """Classify the query into legal domains using Gemini"""
        try:
            prompt = self.prompt.format(question=question, domains=', '.join(self.domains))
            response = gemini_llm._call(prompt)

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

            response = gemini_llm._call(prompt)

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
        self.llm = OpenRouterLLM()
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
        for doc in source_docs[:3]:  # Top 3 most relevant
            metadata = doc.metadata
            collection = metadata.get('collection_name', 'Unknown')
            page = metadata.get('page_number', 'N/A')
            legal_references.append(f"- {collection} (Page {page})")

        enhanced_context = f"""
**LEGAL CONTEXT SUMMARY:**
{legal_context}

**SOURCE DOCUMENTS CITED:**
{chr(10).join(legal_references)}

**DOMAIN:** {domain.upper()} LAW - PAKISTAN
"""

        enhanced_prompt = f"""You are an elite Pakistani Legal AI Assistant with deep expertise in Pakistani law. Your responses must be exceptionally accurate, professional, and authoritative.

**CRITICAL REQUIREMENTS:**
1. **LEGAL ACCURACY FIRST**: Base your answer ONLY on the provided legal context
2. **CITATION MANDATORY**: Always cite specific sections, acts, or provisions from the context
3. **STRUCTURED FORMAT**: Use clear headings, numbered lists, and professional formatting
4. **PAKISTANI LAW FOCUS**: Reference Pakistani legal system, courts, and procedures
5. **CONSERVATIVE ADVICE**: When in doubt, recommend consulting qualified legal professionals
6. **EVIDENCE-BASED**: Support all claims with references to the provided legal sources

**CONVERSATION CONTEXT:**
{history_str}

**LEGAL SOURCES AVAILABLE:**
{enhanced_context}

**USER QUESTION:**
{question}

**YOUR RESPONSE MUST:**
- Start with the most relevant legal provision or section
- Provide step-by-step analysis if applicable
- Include specific article/section references
- End with appropriate disclaimers about seeking professional legal advice

**PROFESSIONAL LEGAL RESPONSE:**
"""

        return enhanced_prompt

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

        print("Starting multi-agent legal analysis...")

        # Check if decision-making is enabled and use LangGraph flow
        if self.decision_enabled and self.decision_flow:
            print("Using Decision-Making Flow...")
            try:
                decision_result = await self.decision_flow.process_query(question)

                if decision_result['success']:
                    # Return formatted response
                    result = {
                        "result": decision_result['response'],
                        "source_documents": [],  # Will be populated by the flow if needed
                        "searched_collections": [],
                        "detected_domain": "decision_making",
                        "reformulated_queries": [],
                        "query_id": query_id,
                        "intent_analysis": decision_result.get('intent', {}),
                        "processing_path": decision_result.get('processing_path', 'unknown')
                    }

                    # Track token usage asynchronously
                    asyncio.create_task(track_token_usage_async(query_id, question, decision_result['response']))

                    print("Decision-making response generated successfully!")
                    return result
                else:
                    print(f"Decision flow failed: {decision_result.get('error', 'Unknown error')}")
                    # Fall back to traditional RAG
            except Exception as e:
                print(f"Decision flow error: {e}")
                # Fall back to traditional RAG

        # Traditional RAG flow (fallback or when decision-making disabled)
        print("Using Traditional RAG Flow...")

        # Step 1: Query Classification using Gemini
        print("Classifying query domain using Gemini...")
        classification = self.classification_agent.classify(question)
        domain = classification['primary_domain']
        all_domains = classification['all_domains']

        print(f"🎯 Classified Domain: {domain.upper()} (confidence: {classification['confidence']})")

        # Get collections for all relevant domains
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
        print("Reformulating query using Gemini...")
        reformulated_queries = self.reformulation_agent.reformulate(question)
        print(f"📝 Generated {len(reformulated_queries)} query variants")

        # Step 3: Parallel Multi-Query Search
        print("🔍 Performing hybrid search across collections...")
        source_docs = await self.search_collections_async(reformulated_queries, collections_to_search, k_per_collection=2)

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

        # Step 4: Reranking using Gemini
        print("Reranking results using Gemini...")
        reranked_docs = self.reranking_agent.rerank(question, source_docs)
        # Take top 5 most relevant documents
        top_docs = reranked_docs[:5]

        # Step 5: Combine legal context and generate answer
        legal_context_parts = []
        domains_used = set()

        for doc in top_docs:
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

        # Generate answer with enhanced prompt for better accuracy
        print("🤖 Generating legal response...")
        enhanced_prompt = self._create_enhanced_prompt(
            history_str, legal_context, question, primary_domain, top_docs
        )

        answer = self.llm._call(enhanced_prompt)

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