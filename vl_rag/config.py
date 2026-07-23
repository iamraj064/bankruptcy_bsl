import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from . import logger_config
logger = logger_config.setup_logger("config")

# Model & API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

def call_llm(prompt: str, system_prompt: str = None, temperature: float = TEMPERATURE) -> str:
    """
    Unified LLM Invocation wrapper reading environment config.
    Priority Order:
      1. OpenAI API (if OPENAI_API_KEY set)
      2. AWS Bedrock Converse API (if AWS credentials set)
      3. Native structured fallback
    """
    # 1. OpenAI Integration
    openai_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            logger.info("Calling OpenAI LLM (%s)...", OPENAI_MODEL)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("OpenAI LLM call failed: %s. Trying fallbacks...", e)

    # 2. AWS Bedrock Integration
    aws_key = os.getenv("AWS_ACCESS_KEY_ID") or AWS_ACCESS_KEY_ID
    if aws_key:
        try:
            import boto3
            bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=aws_key,
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", AWS_SECRET_ACCESS_KEY),
                region_name=os.getenv("AWS_REGION", AWS_REGION)
            )
            messages = [{"role": "user", "content": [{"text": prompt}]}]
            logger.info("Calling AWS Bedrock LLM (%s)...", BEDROCK_MODEL_ID)
            
            converse_kwargs = {
                "modelId": BEDROCK_MODEL_ID,
                "messages": messages,
                "inferenceConfig": {"temperature": temperature}
            }
            if system_prompt:
                converse_kwargs["system"] = [{"text": system_prompt}]

            response = bedrock_client.converse(**converse_kwargs)
            output = response.get("output", {}).get("message", {}).get("content", [])
            if output and isinstance(output, list):
                return output[0].get("text", "")
        except Exception as e:
            logger.warning("AWS Bedrock LLM call failed: %s.", e)

    return ""

_llm_cache = {}

def call_llm_with_cache(prompt: str, system_prompt: str = None, temperature: float = TEMPERATURE, *args, **kwargs) -> str:
    cache_key = (prompt, system_prompt, temperature)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]
    res = call_llm(prompt, system_prompt, temperature)
    if res:
        _llm_cache[cache_key] = res
    return res