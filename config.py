import os
import logging
import time
try:
    import boto3
except ImportError:
    boto3 = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
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
    model: str = 'gpt-4',
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

def call_llm_with_tokens(
    prompt: str,
    model: str = 'gpt-4',
    api_key: str = None,
    timeout: int = 60,
    temperature: float = 0.0,
) -> dict:
    """
    Call an LLM and return both response text and token counts.
    Returns: {
        'text': str,
        'input_tokens': int,
        'output_tokens': int,
        'total_tokens': int
    }
    """
    
    # Use Bedrock if configured via env var
    bedrock_model = os.getenv('BEDROCK_MODEL_ID')
    if bedrock_model:
        logger.info("Using AWS Bedrock backend (with tokens) | model=%s", bedrock_model)
        if boto3 is None:
            raise RuntimeError('boto3 package not installed')
        try:
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

            # Extract response text
            output_text = ""
            out = response.get('output') or {}
            msg = out.get('message') or {}
            content = msg.get('content') or []
            if content and isinstance(content, list):
                first = content[0]
                if isinstance(first, dict):
                    output_text = first.get('text') or first.get('body') or ""

            # Extract token usage from Bedrock response
            usage = response.get('usage', {})
            input_tokens = usage.get('inputTokens', 0)
            output_tokens = usage.get('outputTokens', 0)
            total_tokens = input_tokens + output_tokens

            logger.info("Bedrock response with tokens | input=%d output=%d total=%d", input_tokens, output_tokens, total_tokens)

            return {
                'text': (output_text or "").strip(),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens
            }
        except Exception as e:
            logger.exception("Bedrock call failed: %s", e)
            raise RuntimeError(f'Bedrock call failed: {e}')

    # Fallback to OpenAI
    if OpenAI is None:
        raise RuntimeError('openai package not installed and Bedrock not configured')
    resolved_key = api_key or os.getenv('OPENAI_API_KEY')
    if not resolved_key:
        raise RuntimeError('OPENAI_API_KEY not set and no api_key provided')

    logger.info("Using OpenAI backend (with tokens) | model=%s", model)
    openai_client = OpenAI(api_key=resolved_key)
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=timeout,
    )
    
    text = resp.choices[0].message.content
    input_tokens = resp.usage.prompt_tokens if resp.usage else 0
    output_tokens = resp.usage.completion_tokens if resp.usage else 0
    total_tokens = resp.usage.total_tokens if resp.usage else 0

    logger.info("OpenAI response with tokens | input=%d output=%d total=%d", input_tokens, output_tokens, total_tokens)

    return {
        'text': text,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens
    }

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


def call_llm_haiku_with_tokens(
    prompt: str,
    model: str = 'gpt-4',
    api_key: str = None,
    timeout: int = 60,
    temperature: float = 0.1,
) -> dict:
    """
    Call an LLM Haiku and return both response text and token counts.
    Returns: {
        'text': str,
        'input_tokens': int,
        'output_tokens': int,
        'total_tokens': int
    }
    """

    # Use Bedrock if configured via env var
    bedrock_model = os.getenv('BEDROCK_HAIKU_MODEL_ID')
    if bedrock_model:
        logger.info("Using AWS Bedrock Haiku backend (with tokens) | model=%s", bedrock_model)
        if boto3 is None:
            raise RuntimeError('boto3 package not installed')
        try:
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

            # Extract response text
            output_text = ""
            out = response.get('output') or {}
            msg = out.get('message') or {}
            content = msg.get('content') or []
            if content and isinstance(content, list):
                first = content[0]
                if isinstance(first, dict):
                    output_text = first.get('text') or first.get('body') or ""

            # Extract token usage from Bedrock response
            usage = response.get('usage', {})
            input_tokens = usage.get('inputTokens', 0)
            output_tokens = usage.get('outputTokens', 0)
            total_tokens = input_tokens + output_tokens

            logger.info("Bedrock Haiku response with tokens | input=%d output=%d total=%d", input_tokens, output_tokens, total_tokens)

            return {
                'text': (output_text or "").strip(),
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens
            }
        except Exception as e:
            logger.exception("Bedrock call failed: %s", e)
            raise RuntimeError(f'Bedrock call failed: {e}')

    # Fallback to OpenAI
    if OpenAI is None:
        raise RuntimeError('openai package not installed and Bedrock not configured')
    resolved_key = api_key or os.getenv('OPENAI_API_KEY')
    if not resolved_key:
        raise RuntimeError('OPENAI_API_KEY not set and no api_key provided')

    logger.info("Using OpenAI backend Haiku (with tokens) | model=%s", model)
    openai_client = OpenAI(api_key=resolved_key)
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        timeout=timeout,
    )
    
    text = resp.choices[0].message.content
    input_tokens = resp.usage.prompt_tokens if resp.usage else 0
    output_tokens = resp.usage.completion_tokens if resp.usage else 0
    total_tokens = resp.usage.total_tokens if resp.usage else 0

    logger.info("OpenAI Haiku response with tokens | input=%d output=%d total=%d", input_tokens, output_tokens, total_tokens)

    return {
        'text': text,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens
    }