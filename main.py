# main.py
import os, json, datetime as dt, asyncio, logging
from uuid import uuid4
from typing import Optional, List, Dict
from fastapi import Request, FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse 
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

import asyncpg
from pgvector.asyncpg import register_vector

import httpx # เพิ่มสำหรับเรียก Ollama API แบบ Asynchronous
from huggingface_hub import InferenceClient # ยังคงใช้สำหรับ Embedding model
import numpy as np
from pydantic import BaseModel 

# ─── Logging ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN") # ยังคงใช้สำหรับ Embedding model
POSTGRES_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
SESSION_TTL = int(os.getenv("SESSION_TTL", 1800))

if not HF_TOKEN or not POSTGRES_URL:
    raise RuntimeError("HF_API_TOKEN and DATABASE_URL must be set")

# --- NEW: LLM and Ollama Config ---
OLLAMA_MODEL_NAME = "gemma3:1b" # <--- กำหนดชื่อโมเดล Ollama ที่ใช้
OLLAMA_API_BASE_URL = "http://localhost:11434/api" # <--- Ollama API URL

SYSTEM_PROMPT = """\
You are a helpful Thai online shop assistant. Help customers find products politely, using the provided product data. Always answer in Thai.
When customers ask about products, use the provided product information to give accurate answers about prices, availability, and features.
Be conversational and helpful, but keep responses concise and focused on the customer's needs.
"""
# --- End NEW: LLM and Ollama Config ---

# ─── FastAPI + Templates ───────────────────────────────────────────
app = FastAPI(title="Thai Shop RAG Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Database Pool ─────────────────────────────────────────────────
pg_pool: asyncpg.Pool | None = None

# Global variable to store known product keywords for dynamic filtering
_known_product_keywords: set[str] = set() 

# --- Global variables for Ollama LLM ---
ollama_http_client: Optional[httpx.AsyncClient] = None # <--- Ollama HTTP client

# --- Global variable for Embedding client (still using InferenceClient) ---
embed_client = InferenceClient("BAAI/bge-m3", token=HF_TOKEN)


async def _init_conn(conn: asyncpg.Connection):
    await conn.execute('SET SESSION statement_timeout = 30000')  # 30 seconds
    await register_vector(conn)

@app.on_event("startup")
async def on_startup():
    global pg_pool, _known_product_keywords, ollama_http_client # Declare global variables here
    try:
        pg_pool = await asyncpg.create_pool(
            dsn=POSTGRES_URL,
            min_size=2, 
            max_size=10,
            init=_init_conn,
            statement_cache_size=0,
            command_timeout=60,
            server_settings={
                'application_name': 'ragwfastapi',
                'search_path': 'public',
                'timezone': 'Asia/Bangkok'
            }
        )
        logger.info("Postgres pool created")
        
        await add_sample_data()

        async with pg_pool.acquire() as conn:
            rows_names = await conn.fetch("SELECT DISTINCT product_name FROM products")
            rows_categories = await conn.fetch("SELECT DISTINCT product_category FROM products")

            for row in rows_names:
                for word in str(row['product_name']).lower().split():
                    _known_product_keywords.add(word)
                _known_product_keywords.add(str(row['product_name']).lower())

            for row in rows_categories:
                for word in str(row['product_category']).lower().split():
                    _known_product_keywords.add(word)
                _known_product_keywords.add(str(row['product_category']).lower())
        
        logger.info(f"Populated {len(_known_product_keywords)} known product keywords for filtering.")

        # --- NEW: Initialize Ollama HTTP client on startup ---
        logger.info(f"Initializing Ollama HTTP client for model: {OLLAMA_MODEL_NAME} at {OLLAMA_API_BASE_URL}...")
        ollama_http_client = httpx.AsyncClient(base_url=OLLAMA_API_BASE_URL, timeout=120.0) # 2 minutes timeout
        try:
            # Test Ollama connectivity and model availability
            response = await ollama_http_client.post("/show", json={"name": OLLAMA_MODEL_NAME})
            response.raise_for_status() # Raises an exception for 4xx/5xx responses
            logger.info(f"Ollama model '{OLLAMA_MODEL_NAME}' is available.")
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama model '{OLLAMA_MODEL_NAME}' not found or Ollama server error: {e.response.status_code} - {e.response.text}", exc_info=True)
            raise RuntimeError(f"Ollama model '{OLLAMA_MODEL_NAME}' not available. Please pull it: 'ollama pull {OLLAMA_MODEL_NAME}'")
        except httpx.RequestError as e:
            logger.error(f"Could not connect to Ollama server at {OLLAMA_API_BASE_URL}. Is Ollama running? Error: {e}", exc_info=True)
            raise RuntimeError(f"Ollama server not reachable. Please ensure Ollama is running.")
        except Exception as e:
            logger.error(f"Unexpected error during Ollama client initialization: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize Ollama client.")
        # --- End NEW ---

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

@app.on_event("shutdown")
async def on_shutdown():
    global ollama_http_client
    if pg_pool:
        await pg_pool.close()
        logger.info("Postgres pool closed")
    if ollama_http_client:
        await ollama_http_client.aclose() # Close httpx client gracefully
        logger.info("Ollama HTTP client closed.")

def _embed(text: str) -> List[float]:
    try:
        raw = embed_client.feature_extraction(text, normalize=True, truncate=True)
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            vec = np.mean(raw, axis=0).tolist()
        elif hasattr(raw, "tolist"):
            vec = raw.tolist()
        else:
            vec = list(raw)
        
        if len(vec) < 1024:
            vec += [0.0] * (1024 - len(vec))
        else:
            vec = vec[:1024]
        return vec
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return [0.0] * 1024

async def embed_async(text: str) -> List[float]:
    return await asyncio.to_thread(_embed, text)

async def add_sample_data():
    """Add sample products if database is empty"""
    try:
        async with pg_pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM products")
            if count == 0:
                sample_products = [
                    ("iPhone 14 Pro", "สมารท์โฟน Apple iPhone 14 Pro 256GB สีม่วง กล้องความละเอียดสูง", 35900, 10, "มือถือ"),
                    ("Samsung Galaxy S23", "สมารท์โฟน Samsung Galaxy S23 128GB สีดำ จอ AMOLED", 25900, 15, "มือถือ"),
                    ("MacBook Air M2", "โน้ตบุ๊ค MacBook Air M2 13นิ้ว 8GB RAM 256GB SSD", 42900, 5, "คอมพิวเตอร์"),
                    ("AirPods Pro", "หูฟังไร้สาย Apple AirPods Pro รุ่น 2 พร้อม Active Noise Cancelling", 8900, 20, "หูฟัง"),
                    ("iPad Pro 11", "แท็บเล็ต iPad Pro 11 นิ้ว 128GB Wi-Fi หน้าจอ Liquid Retina", 28900, 8, "แท็บเล็ต"),
                    ("Sony WH-1000XM4", "หูฟัง Sony WH-1000XM4 Wireless Noise Canceling", 12900, 12, "หูฟัง"),
                    ("Dell XPS 13", "โน้ตบุ๊ค Dell XPS 13 Intel i7 16GB RAM 512GB SSD", 45900, 3, "คอมพิวเตอร์"),
                    ("Nintendo Switch", "เครื่องเล่นเกม Nintendo Switch OLED Model", 11900, 7, "เกม"),
                ]
                
                for name, detail, price, qty, category in sample_products:
                    embedding = await embed_async(f"{name} {detail} {category}")
                    await conn.execute(
                        """
                        INSERT INTO products (product_name, product_detail, product_price, product_quantity, product_category, embedding)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        """,
                        name, detail, price, qty, category, embedding
                    )
                logger.info(f"Added {len(sample_products)} sample products")
    except Exception as e:
        logger.error(f"Error adding sample data: {e}")

async def query_products(q: str, k: int = 3) -> str:
    """Query products using vector similarity search, with improved dynamic filtering"""
    try:
        if not q.strip():
            return "กรุณาระบุสินค้าที่ต้องการค้นหา"
        
        vec = await embed_async(q)
        async with pg_pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM products")
            if total == 0:
                return "ยังไม่มีสินค้าในระบบ กรุณาเพิ่มสินค้าก่อน"
            
            filter_parts = ["product_quantity > 0"]
            params = [vec, k]

            query_lower = q.lower()
            
            detected_keywords = []
            for known_keyword in _known_product_keywords:
                if known_keyword in query_lower: 
                    detected_keywords.append(known_keyword)
            
            if detected_keywords:
                keyword_conditions = []
                for keyword in detected_keywords:
                    keyword_conditions.append(f"(product_name ILIKE '%{keyword}%' OR product_category ILIKE '%{keyword}%')")
                
                if keyword_conditions:
                    filter_parts.append("(" + " OR ".join(keyword_conditions) + ")")
                    logger.info(f"Applied dynamic filter based on: {detected_keywords}")
            
            filter_clause = " AND ".join(filter_parts)

            rows = await conn.fetch(
                f"""
                SELECT product_name, product_detail, product_price, product_quantity, product_category
                FROM products
                WHERE {filter_clause}
                ORDER BY embedding <=> $1
                LIMIT $2
                """, *params
            )
        
        if not rows:
            if detected_keywords:
                logger.info("Dynamic filter yielded no results. Retrying query without specific dynamic filters.")
                rows = await conn.fetch(
                    """
                    SELECT product_name, product_detail, product_price, product_quantity, product_category
                    FROM products
                    WHERE product_quantity > 0
                    ORDER BY embedding <=> $1
                    LIMIT $2
                    """, vec, k
                )
                if not rows:
                     return "ไม่พบสินค้าที่เกี่ยวข้องในสต็อก"
            else:
                 return "ไม่พบสินค้าที่เกี่ยวข้องในสต็อก"
        
        products_info = []
        for r in rows:
            product_info = (
                f"สินค้า: {r['product_name']}\n"
                f"รายละเอียด: {r['product_detail']}\n"
                f"ราคา: {r['product_price']:,.0f} บาท\n"
                f"คงเหลือ: {r['product_quantity']} ชิ้น\n"
                f"หมวดหมู่: {r['product_category']}"
            )
            products_info.append(product_info)
        
        return "\n\n".join(products_info)
    
    except Exception as e:
        logger.error(f"Query products error: {e}", exc_info=True)
        return "เกิดข้อผิดพลาดในการค้นหาสินค้า"

# ─── LLM Functions ─────────────────────────────────────────────────
# --- NEW: Function to generate text from Ollama model ---
async def _generate_from_ollama(messages: List[Dict], max_tokens: int, temperature: float) -> str:
    global ollama_http_client
    
    if not ollama_http_client:
        logger.error("Ollama HTTP client not initialized.")
        return ""

    try:
        # Prepare data for Ollama chat API
        json_data = {
            "model": OLLAMA_MODEL_NAME,
            "messages": messages,
            "stream": False, # We want the full response at once
            "options": {
                "num_predict": max_tokens, # Ollama uses num_predict for max tokens
                "temperature": temperature,
                # top_p: Ollama accepts top_p in options
                "top_p": 0.9, # Add top_p here
            }
        }
        
        # Make the asynchronous POST request to Ollama API
        response = await ollama_http_client.post("/chat", json=json_data)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        result = response.json()
        
        if result and result['message'] and result['message']['content']:
            text = result['message']['content'].strip()
            return text
        
        logger.warning(f"Ollama returned empty or invalid response: {result}")
        return ""
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
        return f"Ollama API Error: {e.response.status_code}"
    except httpx.RequestError as e:
        logger.error(f"Ollama network error: {e}", exc_info=True)
        return f"Ollama Network Error: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during Ollama generation: {e}", exc_info=True)
        return "Internal Ollama generation error"

# --- End NEW ---

def build_messages_for_llm(system: str, context: str, history: List[Dict], query: str) -> List[Dict]:
    """Build messages list for LLM with proper formatting for chat method"""
    messages = []
    
    # System message
    messages.append({"role": "system", "content": system})
    
    # Context message
    if context:
        messages.append({"role": "system", "content": f"ข้อมูลสินค้า:\n{context}"})
    
    # Add conversation history (last 6 messages)
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Current user query
    messages.append({"role": "user", "content": query})
    
    return messages

async def call_llm(system: str, context: str, history: List[Dict], query: str) -> Optional[str]:
    """Call LLM (Ollama model) with proper error handling."""
    try:
        messages = build_messages_for_llm(system, context, history, query)
        
        # --- NEW: Call Ollama model via _generate_from_ollama ---
        text = await _generate_from_ollama(
            messages,
            300, # max_tokens
            0.7  # temperature
        )
        # --- End NEW ---
        
        # No more text cleanup for <|eot_id|> here, handled by Ollama or _generate_from_ollama
        
        return text if text else None
        
    except Exception as e:
        logger.error(f"LLM chat error: {e}", exc_info=True)
        return None

def get_fallback_response(user_msg: str, context: str = "") -> str:
    """Provide fallback responses when LLM fails"""
    user_lower = user_msg.lower()
    
    greetings = ["สวัสดี", "หวัดดี", "ครับ", "ค่ะ", "hello", "hi"]
    if any(greeting in user_lower for greeting in greetings):
        return "สวัสดีครับ! ยินดีต้อนรับสู่ร้านค้าออนไลน์ มีอะไรให้ช่วยเหลือไหมครับ?"
    
    product_keywords = ["สินค้า", "ราคา", "แนะนำ", "มี", "ขาย", "หา", "ต้องการ"]
    if any(keyword in user_lower for keyword in product_keywords):
        if context:
            return f"ผมพบสินค้าที่เกี่ยวข้องครับ:\n\n{context}\n\nต้องการรายละเอียดเพิ่มเติมหรือไม่ครับ?"
        else:
            return "กรุณาระบุชื่อหรือประเภทสินค้าที่ต้องการค้นหาให้ชัดเจนกว่านี้ครับ"
    
    return "ขอโทษครับ ผมไม่เข้าใจคำถาม กรุณาลองถามใหม่หรือระบุสินค้าที่ต้องการครับ"

# ใน main.py (เพิ่มฟังก์ชันนี้ต่อจาก SYSTEM_PROMPT หรือใกล้ๆ LLM Functions)
async def rewrite_query_for_rag(history: List[Dict], user_query: str) -> str:
    """Rewrites the user query to include necessary context from chat history for better RAG."""
    
    if not history or "ราคา" not in user_query and "รุ่น" not in user_query:
        return user_query
    
    rewrite_prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant that rewrites a user's short or ambiguous question into a clear, standalone question, leveraging the provided chat history. Ensure the rewritten question contains all necessary context for a search system. If the question is already clear, return it as is. Always output only the rewritten question."},
    ]
    
    for msg in history[-4:]:
        rewrite_prompt_messages.append({"role": msg["role"], "content": msg["content"]})
    
    rewrite_prompt_messages.append({"role": "user", "content": f"Previous conversation context available. Rewrite this question to be clear for searching: '{user_query}'"})

    try:
        # --- NEW: Call Ollama model for rewriting ---
        rewritten_query = await _generate_from_ollama( # ใช้ Ollama generator
            rewrite_prompt_messages,
            50,  # max_tokens for rewritten query
            0.1  # temperature for rewriting
        )
        # --- End NEW ---

        if rewritten_query:
            logger.info(f"Rewritten query from '{user_query}' to '{rewritten_query}'")
            return rewritten_query.strip()
        
    except Exception as e:
        logger.warning(f"Failed to rewrite query: {e}. Using original query.", exc_info=True)
    
    return user_query

# ─── Routes ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with chat interface"""
    try:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "history": [],
        })
    except Exception as e:
        logger.exception("Error in home route:")
        return HTMLResponse("เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์", status_code=500)

class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []

@app.post("/chat", response_class=JSONResponse)
async def chat_api(request: Request, chat_req: ChatRequest):
    """Handle chat messages via API, history managed by client"""
    try:
        message = chat_req.message.strip()
        history = chat_req.history

        if not message:
            return JSONResponse({"response": ""})

        rewritten_query = await rewrite_query_for_rag(history, message)
        
        context = await query_products(rewritten_query, k=3)
        
        ai_response = await call_llm(SYSTEM_PROMPT, context, history, message)
        
        if not ai_response:
            ai_response = get_fallback_response(message, context)
        
        return JSONResponse({"response": ai_response})
        
    except Exception as e:
        logger.exception("Error in chat_api route:")
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดในการประมวลผลคำขอ")

@app.post("/reset")
async def reset_chat_api():
    """No longer resets session, kept for demonstration if needed"""
    return JSONResponse({"status": "ok", "message": "Backend reset (no session to clear)"})

@app.get("/products")
async def list_products():
    """API endpoint to list all products"""
    try:
        async with pg_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, product_name, product_detail, product_price, product_quantity, product_category, created_at
                FROM products
                ORDER BY created_at DESC
                """
            )
        
        products = []
        for row in rows:
            products.append({
                "id": row["id"],
                "name": row["product_name"],
                "detail": row["product_detail"],
                "price": float(row["product_price"]),
                "quantity": row["product_quantity"],
                "category": row["product_category"],
                "created_at": row["created_at"].isoformat()
            })
        
        return {"products": products, "count": len(products)}
        
    except Exception as e:
        logger.exception("Error listing products:")
        raise HTTPException(status_code=500, detail="เกิดข้อผิดพลาดในการดึงข้อมูลสินค้า")

# ─── Health Check ──────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Health check endpoint: checks database, embedding API, and Ollama LLM connections."""
    db_status = "disconnected"
    embed_status = "unreachable"
    chat_status = "unreachable"
    error_details = []

    try:
        # Check database connection
        async with pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            db_status = "connected"
    except Exception as e:
        logger.error(f"Health check: Database connection failed: {e}", exc_info=True)
        db_status = "failed"
        error_details.append(f"Database: {str(e)}")

    try:
        # Check Embedding Client (BAAI/bge-m3)
        # Try embedding a simple text to ensure the service is reachable
        test_embed_text = "test embedding connectivity"
        test_vec = await embed_async(test_embed_text)
        if len(test_vec) == 1024 and any(val != 0.0 for val in test_vec):
            embed_status = "reachable"
        else:
            embed_status = "failed_response"
            error_details.append("Embedding client: Returned zero vector or incorrect dimension.")
    except Exception as e:
        logger.error(f"Health check: Embedding client failed: {e}", exc_info=True)
        embed_status = "failed"
        error_details.append(f"Embedding API: {str(e)}")

    try:
        # Check Ollama Chat Client
        if ollama_http_client: # Check if client is initialized
            test_chat_messages = [{"role": "user", "content": "hello"}]
            test_chat_response = await _generate_from_ollama(
                test_chat_messages,
                10, # max_tokens
                0.1 # temperature
            )
            if test_chat_response and len(test_chat_response) > 0 and not test_chat_response.startswith("Ollama API Error"):
                chat_status = "reachable"
            else:
                chat_status = "failed_response"
                error_details.append(f"Ollama Chat client: Returned empty or invalid response. Response: {test_chat_response}")
        else:
            chat_status = "not_initialized"
            error_details.append("Ollama HTTP client not initialized during startup.")
    except Exception as e:
        logger.error(f"Health check: Ollama Chat client failed: {e}", exc_info=True)
        chat_status = "failed"
        error_details.append(f"Ollama API: {type(e).__name__} - {str(e)}")

    overall_status = "healthy"
    if db_status != "connected" or embed_status != "reachable" or chat_status != "reachable":
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "timestamp": dt.datetime.now().isoformat(),
        "database": db_status,
        "embedding_api": embed_status,
        "chat_api": chat_status,
        "errors": error_details if error_details else "None"
    }

# ─── Run Application ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )