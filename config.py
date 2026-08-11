import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
LOG_DIR  = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ── Azure OpenAI  ─────────────────────────────────────────────────────────────
# Set AZURE_OPENAI_API_KEY in your .env file.
# Configure your Azure OpenAI endpoint in the .env file.
# Model name must match the deployment name in the Azure project.

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
MODEL = "gpt-4o"          # must match the Azure deployment name

# ── Embeddings (text-embedding-3-small on base Azure endpoint) ────────────────
AZURE_EMBED_ENDPOINT = os.getenv("AZURE_EMBED_ENDPOINT", "")
EMBED_MODEL          = "text-embedding-3-small"
EMBED_API_VERSION    = "2024-02-01"
EMBED_DIM            = 1536

# ── Documentation RAG ─────────────────────────────────────────────────────────
DOC_CHUNKS_PATH  = str(ROOT / "data 2" / "all_chunks.json")
DOC_EMBED_CACHE  = str(DATA_DIR / "doc_embeddings.npy")
DOC_INDEX_CACHE  = str(DATA_DIR / "doc_chunk_ids.json")  # maps embed row → chunk idx

# ── Database / logging ────────────────────────────────────────────────────────
DB_PATH            = str(DATA_DIR / "fugaku.duckdb")
LOG_DB_PATH        = str(LOG_DIR / "conversation_log.db")
MAX_REFLECT_ROUNDS = 3
AGENT_TIMEOUT      = 120
