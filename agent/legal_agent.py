import os
import requests
import chromadb
import uuid
from langchain.prompts import PromptTemplate
from langchain_core.language_models import LLM
from typing import Optional, List, Tuple
from chromadb.utils import embedding_functions
from config.settings import settings
from helper.token_tracker import track_token_usage_async

# ------------------ OpenRouter Config ------------------
OPENROUTER_API_URL = settings.OPENROUTER_API_URL
OPENROUTER_MODEL = settings.OPENROUTER_MODEL
OPENROUTER_API_KEY = settings.OPENROUTER_API_KEY

# Domain detection keywords
DOMAIN_KEYWORDS = {
    'traffic': ['traffic', 'motor vehicle', 'driving', 'license', 'transport', 'road', 'accident', 'speeding', 'vehicle registration'],
    'family': ['family', 'marriage','witness','nikah witness', 'divorce', 'inheritance', 'guardian', 'child', 'maintenance', 'custody', 'dowry', 'marital', "alimony", "child support", "adoption", "domestic violence", "family dispute", "nikah", "mehr", "talaq", "khula", "wasiat","pakistan family law","family court","family act","family ordinance"],
    'corporate': ['corporate', 'company', 'business', 'commercial', 'contract', 'partnership', 'incorporation', 'shareholder', 'director', 'board','leaves', 'employee', 'employment', 'labor', 'workplace', 'hr', 'human resources', 'termination', 'hiring', 'firing', 'work hours', 'overtime', 'payroll', 'benefits', 'discrimination', 'harassment', 'workplace safety','pakistan labor law','pakistan employment law','labor court','employment act','industrial relations','maternity', 'paternity', 'casual', 'sick leave', 'annual leave', 'leave policy'],
    'ppc': ['ppc', 'penal', 'criminal', 'crime', 'offense', 'punishment', 'ipc', 'pakistan penal', 'theft', 'jail','imprisonment']
}

# ------------------ Custom OpenRouter LLM wrapper ------------------
class OpenRouterLLM(LLM):
    api_key: str = OPENROUTER_API_KEY
    model: str = OPENROUTER_MODEL
    api_url: str = OPENROUTER_API_URL

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
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
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[OpenRouter Error] {e}"

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
    """Create a retriever for a specific collection"""
    client = create_chroma_client()

    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

        class CollectionRetriever:
            def __init__(self, collection, k=4):
                self.collection = collection
                self.k = k
                self.collection_name = collection_name

            def get_relevant_documents(self, query):
                try:
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=self.k
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
                                    'collection_name': self.collection_name
                                }
                            )
                            documents.append(document)

                    return documents

                except Exception as e:
                    print(f"Error querying collection {self.collection_name}: {e}")
                    return []

        return CollectionRetriever(collection, k=k)

    except Exception as e:
        print(f"❌ Failed to load collection {collection_name}: {e}")
        return None

# ------------------ Smart Multi-Collection QA System ------------------
class SmartLegalAssistant:
    def __init__(self):
        self.llm = OpenRouterLLM()
        self.prompt = PromptTemplate(
            template="""
You are a professional Pakistan AI Legal Assistant. Use ONLY the CONTEXT provided to answer the QUESTION.

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

    def search_collections(self, query: str, collection_names: List[str], k_per_collection: int = 3):
        """Search across multiple collections"""
        all_docs = []

        for collection_name in collection_names:
            retriever = get_collection_retriever(collection_name, k_per_collection)
            if retriever:
                docs = retriever.get_relevant_documents(query)
                all_docs.extend(docs)
                print(f"   🔍 Searched: {collection_name} → Found {len(docs)} documents")

        return all_docs

    def __call__(self, question: str, conversation_history: Optional[List[dict]] = None):
        # Generate unique query ID
        query_id = str(uuid.uuid4())

        # Step 1: Detect domain and relevant collections
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

        return {
            "result": answer,
            "source_documents": source_docs,
            "searched_collections": collections_to_search,
            "detected_domain": domain,
            "query_id": query_id
        }