<p align="center">
  <a href="https://github.com/your-username/pakistan-legal-ai" target="blank"><img src="https://cdn-icons-png.flaticon.com/512/2092/2092132.png" width="120" alt="Pakistan Legal AI Logo" /></a>
</p>

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/your-username/pakistan-legal-ai/blob/main/LICENSE)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-orange)](https://openrouter.ai/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)](https://www.trychroma.com/)

<p align="center">An intelligent AI-powered legal research assistant for Pakistan's legal system, built with FastAPI and modern AI technologies.</p>
<p align="center">
    <a href="https://github.com/your-username/pakistan-legal-ai/issues" target="_blank"><img src="https://img.shields.io/github/issues/your-username/pakistan-legal-ai" alt="GitHub Issues" /></a>
    <a href="https://github.com/your-username/pakistan-legal-ai/stargazers" target="_blank"><img src="https://img.shields.io/github/stars/your-username/pakistan-legal-ai" alt="GitHub Stars" /></a>
    <a href="https://github.com/your-username/pakistan-legal-ai/network" target="_blank"><img src="https://img.shields.io/github/forks/your-username/pakistan-legal-ai" alt="GitHub Forks" /></a>
    <a href="https://github.com/your-username/pakistan-legal-ai/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" /></a>
</p>

## 🏛️ Description

**Pakistan Legal AI Assistant** is a sophisticated AI-powered legal research platform designed specifically for Pakistan's legal system. It leverages state-of-the-art language models and vector databases to provide instant, accurate legal information across multiple domains including Traffic Law, Family Law, Corporate Law, and Pakistan Penal Code (PPC).

The system features intelligent domain detection, multi-collection searching, and a robust API interface for seamless integration with legal applications and services.

## 🚀 Features

- **🤖 Smart Legal Assistant**: AI-powered legal research and Q&A
- **🏷️ Multi-Domain Support**: Traffic, Family, Corporate, and PPC law domains
- **🔍 Intelligent Query Routing**: Automatic domain detection and collection selection
- **📚 Vector-Based Search**: ChromaDB-powered semantic search
- **🌐 RESTful API**: FastAPI-based modern API endpoints
- **📄 PDF Ingestion**: Automated processing of legal documents
- **⚡ High Performance**: Optimized for fast response times
- **🔒 Token Tracking**: Comprehensive usage monitoring
- **☁️ Cloud Ready**: Deployable on any cloud platform

## 🏗️ Project Structure
pakistan-legal-ai/
├── agent/
│ └── legal_agent.py # Core legal assistant logic
├── controllers/
│ ├── legal_controller.py # Legal query handling
│ └── ingest_controller.py # PDF ingestion controller
├── routes/
│ └── legal_route.py # Legal API endpoints
├── helper/
│ ├── token_tracker.py # Token tracking utility
│ └── token_counter.py # Token counting utility
├── utils/
│ └── s3.py # S3 storage utilities
├── requirements.txt # Python dependencies
├── main.py # FastAPI application entry point
└── README.md # Project documentation

text

## 📋 Prerequisites

- Python 3.8 or higher
- ChromaDB Cloud account
- OpenRouter API account
- Required Python packages (see requirements.txt)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pakistan-legal-ai.git
cd pakistan-legal-ai
2. Create Virtual Environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
pip install -r requirements.txt
4. Environment Configuration
Create a .env file in the root directory:

env
CHROMA_API_KEY=your_chroma_api_key
CHROMA_TENANT=your_chroma_tenant
CHROMA_DATABASE=ai-legal-research-assistant
OPENROUTER_API_KEY=your_openrouter_api_key
🚀 Quick Start
1. Ingest Legal Documents
bash
# Ingest Pakistan Penal Code
python -c "
from controllers.ingest_controller import ingest_pdf
ingest_pdf('Pakistan Penal Code.pdf', force=True)
"

# Ingest Traffic Laws
python -c "
from controllers.ingest_controller import ingest_pdf  
ingest_pdf('Traffic Laws.pdf', force=True)
"
2. Start Development Server
bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
3. Access the API
API Documentation: http://localhost:8000/docs

Alternative Docs: http://localhost:8000/redoc

📚 API Endpoints
Legal Research Endpoints
Method	Endpoint	Description
POST	/api/legal/query	Submit legal queries
GET	/api/legal/domains	List supported legal domains
GET	/api/legal/collections	List available document collections
Document Management Endpoints
Method	Endpoint	Description
POST	/api/ingest/pdf	Upload and process PDF documents
GET	/api/ingest/collections	List all collections
DELETE	/api/ingest/collection/{name}	Delete a collection
💡 Usage Examples
Legal Query Example
python
import requests

response = requests.post(
    "http://localhost:8000/api/legal/query",
    json={
        "question": "What are the penalties for speeding under Pakistan traffic laws?",
        "domain": "traffic"  # Optional: auto-detected if not provided
    }
)

print(response.json())
PDF Ingestion Example
python
import requests

with open('Family Laws.pdf', 'rb') as file:
    response = requests.post(
        "http://localhost:8000/api/ingest/pdf",
        files={"file": file},
        data={"force": "true"}
    )

print(response.json())
🏷️ Supported Legal Domains
Domain	Keywords	Description
Traffic Law	traffic, motor vehicle, driving, license, transport	Road and vehicle regulations
Family Law	family, marriage, divorce, inheritance, custody	Personal and family matters
Corporate Law	corporate, company, business, contract, partnership	Business and commercial laws
PPC	ppc, penal code, criminal, crime, punishment	Pakistan Penal Code offenses
🔧 Configuration
ChromaDB Configuration
The project uses ChromaDB Cloud for vector storage. Update your credentials in the environment variables:

python
CHROMA_API_KEY = "your_api_key_here"
CHROMA_TENANT = "your_tenant_id_here" 
CHROMA_DATABASE = "ai-legal-research-assistant"
OpenRouter Configuration
Configure your preferred language model:

python
OPENROUTER_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"
# Alternative models:
# "meta-llama/llama-3-70b-instruct"
# "google/gemini-pro-1.5"
🧪 Testing
Run the test suite to ensure everything is working:

bash
# Run basic functionality tests
python -m pytest tests/ -v

# Test specific components
python -c "
from controllers.legal_controller import test_legal_query
test_legal_query('What is Section 302 of PPC?')
"
🚀 Deployment
Production Deployment
bash
# Install production dependencies
pip install -r requirements.txt

# Start production server
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
Docker Deployment
dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
📊 Monitoring
Token Usage Tracking
The system includes comprehensive token tracking:

python
from helper.token_tracker import TokenTracker

tracker = TokenTracker()
usage = tracker.get_usage_stats()
print(f"Total tokens used: {usage['total_tokens']}")
API Metrics
Monitor API performance and usage through the built-in metrics endpoint:

bash
curl http://localhost:8000/api/metrics
🤝 Contributing
We welcome contributions! Please see our Contributing Guide for details.

Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add some amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🆘 Support
📧 Email: support@pakistanlegalai.com

💬 Discussions: GitHub Discussions

🐛 Bug Reports: GitHub Issues

🙏 Acknowledgments
OpenRouter for LLM API access

ChromaDB for vector database

FastAPI for the web framework

The Pakistan legal community for guidance and support

<div align="center">
Built with ❤️ for the Pakistan Legal Community

https://img.shields.io/twitter/follow/pakistanlegalai?style=social
https://img.shields.io/github/stars/your-username/pakistan-legal-ai?style=social

</div> ```