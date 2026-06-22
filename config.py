import os
import logging
import time
import sqlite3
import hashlib
import importlib
try:
    import boto3
except ImportError:
    boto3 = None
import datetime
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import tiktoken
except ImportError:
    tiktoken = None
try:
    langchain_litellm = importlib.import_module("langchain_litellm")
    langchain_globals = importlib.import_module("langchain.globals")
    langchain_community_cache = importlib.import_module("langchain_community.cache")
    ChatLiteLLM = getattr(langchain_litellm, "ChatLiteLLM", None)
    set_llm_cache = getattr(langchain_globals, "set_llm_cache", None)
    SQLiteCache = getattr(langchain_community_cache, "SQLiteCache", None)
    LANGCHAIN_AVAILABLE = bool(ChatLiteLLM and set_llm_cache and SQLiteCache)
except ImportError:
    ChatLiteLLM = None
    set_llm_cache = None
    SQLiteCache = None
    LANGCHAIN_AVAILABLE = False
except Exception:
    ChatLiteLLM = None
    set_llm_cache = None
    SQLiteCache = None
    LANGCHAIN_AVAILABLE = False
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

# Hugging Face client config kept as reference only.
# Uncomment this block if you want to switch back to HF later.
# client = OpenAI(
#     base_url="https://router.huggingface.co/v1",
#     api_key=os.environ["HF_TOKEN"],
# )


def call_llm(
    prompt: str,
    model: str = 'meta.llama3-8b-instruct-v1:0',
    api_key: str = None,
    timeout: int = 60,
    temperature: float = 0.0,
) -> str:
    """
    Call an LLM to convert natural language to SQL.
    Preference order:
      1. AWS Bedrock when `BEDROCK_MODEL_ID` is configured
      2. OpenAI ChatCompletion when OpenAI API key is available
    """

    # Use Bedrock if configured via env var
    bedrock_model = os.getenv('BEDROCK_MODEL_ID')
    if bedrock_model:
        # Override to Claude 3 Haiku for superior SQL generation reasoning, and avoid Opus on-demand failure
        if bedrock_model in ['meta.llama3-8b-instruct-v1:0', 'anthropic.claude-opus-4-8']:
            bedrock_model = 'anthropic.claude-3-haiku-20240307-v1:0'
        logger.info("Using AWS Bedrock backend | model=%s prompt_chars=%s", bedrock_model, len(prompt))
        if boto3 is None:
            raise RuntimeError('boto3 package not installed')
        try:
            start = time.perf_counter()
            bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION') or None,
            )

            messages = [{
                "role": "user",
                "content": [{"text": prompt}]
            }]

            response = bedrock_client.converse(
                modelId=bedrock_model,
                messages=messages,
                inferenceConfig={
                    'temperature': temperature,
                },
            )

            # Extract response text robustly
            output_text = ""
            out = response.get('output') or {}
            msg = out.get('message') or {}
            content = msg.get('content') or []
            if content and isinstance(content, list):
                first = content[0]
                if isinstance(first, dict):
                    output_text = first.get('text') or first.get('body') or ""

            logger.info(
                "Bedrock response received | elapsed_ms=%.2f response_chars=%s",
                (time.perf_counter() - start) * 1000,
                len(output_text or ""),
            )

            return (output_text or "").strip()
        except Exception as e:
            logger.exception("Bedrock call failed: %s", e)
            raise RuntimeError(f'Bedrock call failed: {e}')

    # Fallback to OpenAI
    if OpenAI is None:
        raise RuntimeError('openai package not installed and Bedrock not configured')
    resolved_key = api_key or os.getenv('OPENAI_API_KEY')
    if not resolved_key:
        raise RuntimeError('OPENAI_API_KEY not set and no api_key provided')

    logger.info("Using OpenAI backend | model=%s prompt_chars=%s", model, len(prompt))
    start = time.perf_counter()
    openai_client = OpenAI(api_key=resolved_key)
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=timeout,
    )
    text = resp.choices[0].message.content
    logger.info(
        "OpenAI response received | elapsed_ms=%.2f response_chars=%s",
        (time.perf_counter() - start) * 1000,
        len(text or ""),
    )
    return text


# =============================================================================
# LANGCHAIN LITELLM CACHING FOR TRANSACTIONAL CHATBOT
# =============================================================================

_cached_llm = None
_cache_initialized = False
_local_cache_initialized = False
_local_cache_db = None


def initialize_llm_cache():
    """Initialize SQLite-based cache for LLM responses in the Transactional Chatbot."""
    global _cache_initialized

    if _cache_initialized:
        return True

    if not LANGCHAIN_AVAILABLE or set_llm_cache is None or SQLiteCache is None:
        logger.warning(
            "LangChain LiteLLM caching unavailable. Install langchain, langchain-litellm, and langchain-community to enable caching."
        )
        return False

    try:
        cache_db_path = ".langchain_cache.db"
        set_llm_cache(SQLiteCache(database_path=cache_db_path))
        _cache_initialized = True
        logger.info("LLM caching initialized | database=%s", cache_db_path)
        return True
    except Exception as e:
        logger.exception("Failed to initialize LLM cache: %s", e)
        return False


def initialize_local_prompt_cache():
    """Initialize a local SQLite prompt cache for fallback caching."""
    global _local_cache_initialized, _local_cache_db

    if _local_cache_initialized and _local_cache_db is not None:
        return True

    try:
        cache_db_path = ".litellm_prompt_cache.db"
        _local_cache_db = sqlite3.connect(cache_db_path, check_same_thread=False)
        cursor = _local_cache_db.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_cache (
                prompt_hash TEXT PRIMARY KEY,
                prompt TEXT,
                model TEXT,
                temperature REAL,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _local_cache_db.commit()
        _local_cache_initialized = True
        logger.info("Local prompt cache initialized | database=%s", cache_db_path)
        return True
    except Exception as e:
        logger.exception("Failed to initialize local prompt cache: %s", e)
        _local_cache_initialized = False
        _local_cache_db = None
        return False


def _get_cached_prompt(prompt: str, model: str, temperature: float, ttl_seconds: int = 86400):
    if not initialize_local_prompt_cache() or _local_cache_db is None:
        return None

    try:
        prompt_hash = hashlib.sha256(f"{model}|{temperature}|{prompt}".encode("utf-8")).hexdigest()
        cursor = _local_cache_db.cursor()
        cursor.execute(
            "SELECT response, created_at FROM prompt_cache WHERE prompt_hash = ?",
            (prompt_hash,)
        )
        row = cursor.fetchone()
        if row:
            response_text, created_at_str = row
            try:
                # created_at is in YYYY-MM-DD HH:MM:SS format
                created_time = datetime.datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                now_utc = datetime.datetime.utcnow()
                age_seconds = (now_utc - created_time).total_seconds()
                if age_seconds < ttl_seconds:
                    logger.info("Local prompt cache hit | model=%s prompt_hash=%s | age_sec=%.1f", model, prompt_hash, age_seconds)
                    return response_text
                else:
                    logger.info("Local prompt cache entry expired | model=%s prompt_hash=%s | age_sec=%.1f", model, prompt_hash, age_seconds)
                    cursor.execute("DELETE FROM prompt_cache WHERE prompt_hash = ?", (prompt_hash,))
                    _local_cache_db.commit()
            except Exception as ex:
                logger.warning("Cache TTL parsing failed, returning cached value: %s", ex)
                return response_text
    except Exception as e:
        logger.exception("Failed to read from local prompt cache: %s", e)
    return None


def _store_cached_prompt(prompt: str, model: str, temperature: float, response: str):
    if not initialize_local_prompt_cache() or _local_cache_db is None:
        return

    try:
        prompt_hash = hashlib.sha256(f"{model}|{temperature}|{prompt}".encode("utf-8")).hexdigest()
        cursor = _local_cache_db.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO prompt_cache (prompt_hash, prompt, model, temperature, response) VALUES (?, ?, ?, ?, ?)",
            (prompt_hash, prompt, model, temperature, response)
        )
        _local_cache_db.commit()
        logger.info("Stored prompt cache entry | model=%s prompt_hash=%s", model, prompt_hash)
    except Exception as e:
        logger.exception("Failed to write to local prompt cache: %s", e)


def count_tokens(text: str, model: str = "gpt-4-0613") -> int:
    """Count tokens for arbitrary text using tiktoken, with a safe word-count fallback."""
    if text is None:
        return 0
    text = str(text)
    if tiktoken is None:
        return len(text.split())
    try:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning("Token counting failed, falling back to word count: %s", e)
        return len(text.split())


def count_token_usage(prompt: str, response: str, model: str = "gpt-4-0613") -> dict:
    """Return input/output/total token counts for a prompt/response pair."""
    prompt_tokens = count_tokens(prompt, model=model)
    response_tokens = count_tokens(response, model=model)
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
    }


def get_cached_litellm_instance(model: str = "anthropic.claude-3-haiku-20240307-v1:0", temperature: float = 0.0):
    """Get or create a ChatLiteLLM instance with caching enabled."""
    global _cached_llm

    if not LANGCHAIN_AVAILABLE or ChatLiteLLM is None:
        logger.debug("LangChain LiteLLM not available for caching")
        return None

    try:
        if _cached_llm is None:
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY")
            if not api_key:
                logger.warning("No API key found for LiteLLM caching; set OPENAI_API_KEY or LITELLM_API_KEY")
                return None

            _cached_llm = ChatLiteLLM(model=model, temperature=temperature, api_key=api_key)
            logger.info("ChatLiteLLM instance created with caching | model=%s", model)

        return _cached_llm
    except Exception as e:
        logger.exception("Error creating ChatLiteLLM instance: %s", e)
        return None


def call_llm_with_cache(prompt: str, model: str = "anthropic.claude-3-haiku-20240307-v1:0", temperature: float = 0.0) -> str:
    """Call LLM with caching.

    Attempts LangChain LiteLLM cache first, then falls back to a local SQLite prompt cache.
    Identical prompts return cached responses, reducing repeated API calls.
    """
    # Keep a local cache even when LangChain LiteLLM packages are unavailable.
    cached_prompt_response = _get_cached_prompt(prompt, model, temperature)
    if cached_prompt_response is not None:
        return cached_prompt_response

    if initialize_llm_cache():
        try:
            llm = get_cached_litellm_instance(model=model, temperature=temperature)
            if llm is not None:
                logger.info("Invoking cached LiteLLM | model=%s | prompt_chars=%d", model, len(prompt))
                response = llm.invoke(prompt)
                if hasattr(response, "content"):
                    result = response.content
                else:
                    result = str(response)
                result_text = (result or "").strip()
                _store_cached_prompt(prompt, model, temperature, result_text)
                logger.info("Cached LLM response received | response_chars=%d", len(result_text))
                return result_text
            else:
                logger.debug("Cached LiteLLM instance unavailable, falling back to standard call")
        except Exception as e:
            logger.exception("Error in cached LiteLLM call, falling back to standard call: %s", e)

    # Fallback standard call with local cache persistence
    response_text = call_llm(prompt, model=model, temperature=temperature)
    _store_cached_prompt(prompt, model, temperature, response_text)
    return response_text


# Initialize cache on module load (best-effort)
if LANGCHAIN_AVAILABLE:
    try:
        initialize_llm_cache()
    except Exception:
        pass

def call_llm_haiku(
    prompt: str,
    model: str="anthropic.claude-3-haiku-20240307-v1:0",
    api_key: str = None,
    timeout: int = 60,
    temperature: float = 0,
) -> str:
    """
    Call an LLM to convert natural language to SQL.
    Preference order:
      1. AWS Bedrock when `BEDROCK_HAIKU_MODEL_ID` is configured
      2. OpenAI ChatCompletion when OpenAI API key is available
    """

    # Use Bedrock if configured via env var
    bedrock_model = os.getenv('BEDROCK_HAIKU_MODEL_ID')
    if bedrock_model:
        # Avoid Opus on-demand ValidationException
        #if bedrock_model == 'anthropic.claude-opus-4-8':
        #    bedrock_model = 'anthropic.claude-3-haiku-20240307-v1:0'
        logger.info("Using AWS Bedrock backend | model=%s prompt_chars=%s", bedrock_model, len(prompt))
        if boto3 is None:
            raise RuntimeError('boto3 package not installed')
        try:
            start = time.perf_counter()
            bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION') or None,
            )

            messages = [{
                "role": "user",
                "content": [{"text": prompt}]
            }]

            response = bedrock_client.converse(
                modelId=bedrock_model,
                messages=messages,
                inferenceConfig={
                    'temperature': temperature,
                },
            )

            # Extract response text robustly
            output_text = ""
            out = response.get('output') or {}
            msg = out.get('message') or {}
            content = msg.get('content') or []
            if content and isinstance(content, list):
                first = content[0]
                if isinstance(first, dict):
                    output_text = first.get('text') or first.get('body') or ""

            logger.info(
                "Bedrock response received | elapsed_ms=%.2f response_chars=%s",
                (time.perf_counter() - start) * 1000,
                len(output_text or ""),
            )

            return (output_text or "").strip()
        except Exception as e:
            logger.exception("Bedrock call failed: %s", e)
            raise RuntimeError(f'Bedrock call failed: {e}')

    # Fallback to OpenAI
    if OpenAI is None:
        raise RuntimeError('openai package not installed and Bedrock not configured')
    resolved_key = api_key or os.getenv('OPENAI_API_KEY')
    if not resolved_key:
        raise RuntimeError('OPENAI_API_KEY not set and no api_key provided')

    logger.info("Using OpenAI backend | model=%s prompt_chars=%s", model, len(prompt))
    start = time.perf_counter()
    openai_client = OpenAI(api_key=resolved_key)
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=timeout,
    )
    text = resp.choices[0].message.content
    logger.info(
        "OpenAI response received | elapsed_ms=%.2f response_chars=%s",
        (time.perf_counter() - start) * 1000,
        len(text or ""),
    )
    return text


def call_llm_haiku2(
    prompt: str,
    model: str="anthropic.claude-3-haiku-20240307-v1:0",
    api_key: str = None,
    timeout: int = 60,
    temperature: float = 0.7,
) -> str:
    """
    Generate dynamic textual and visual suggestions based on conversation history
    and the columns/data of the latest result DataFrame 
    """

    # Use Bedrock if configured via env var
    bedrock_model = os.getenv('BEDROCK_HAIKU_MODEL_ID')
    if bedrock_model:
        # Avoid Opus on-demand ValidationException
        #if bedrock_model == 'anthropic.claude-opus-4-8':
        #    bedrock_model = 'anthropic.claude-3-haiku-20240307-v1:0'
        logger.info("Using AWS Bedrock backend | model=%s prompt_chars=%s", bedrock_model, len(prompt))
        if boto3 is None:
            raise RuntimeError('boto3 package not installed')
        try:
            start = time.perf_counter()
            bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION') or None,
            )

            messages = [{
                "role": "user",
                "content": [{"text": prompt}]
            }]

            response = bedrock_client.converse(
                modelId=bedrock_model,
                messages=messages,
                inferenceConfig={
                    'temperature': temperature,
                },
            )

            # Extract response text robustly
            output_text = ""
            out = response.get('output') or {}
            msg = out.get('message') or {}
            content = msg.get('content') or []
            if content and isinstance(content, list):
                first = content[0]
                if isinstance(first, dict):
                    output_text = first.get('text') or first.get('body') or ""

            logger.info(
                "Bedrock response received | elapsed_ms=%.2f response_chars=%s",
                (time.perf_counter() - start) * 1000,
                len(output_text or ""),
            )

            return (output_text or "").strip()
        except Exception as e:
            logger.exception("Bedrock call failed: %s", e)
            raise RuntimeError(f'Bedrock call failed: {e}')

    # Fallback to OpenAI

    logger.info("Using Anthropic backend | model=%s prompt_chars=%s", model, len(prompt))
    start = time.perf_counter()
    openai_client = OpenAI(api_key=resolved_key)
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=timeout,
    )
    text = resp.choices[0].message.content
    logger.info(
        "OpenAI response received | elapsed_ms=%.2f response_chars=%s",
        (time.perf_counter() - start) * 1000,
        len(text or ""),
    )
    return text