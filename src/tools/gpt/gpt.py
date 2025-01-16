import os
import time
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from dataclasses import dataclass
import backoff

from src.tools.gpt.gemini_client import GeminiClient
from src.tools.gpt.logger import init_logger, SUCCESS_ICON, ERROR_ICON
from src.tools.gpt.ollama_client import OllamaClient

ai_client = None

# 设置日志记录
logger = init_logger('api_calls')

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
env_path = os.path.join(project_root, '.env')

# 加载环境变量
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)
    logger.info(f"{SUCCESS_ICON} 已加载环境变量: {env_path}")
else:
    logger.warning(f"{ERROR_ICON} 未找到环境变量文件: {env_path}")

# 验证环境变量
ai_platform = os.getenv("AI_PLATFORM")
if ai_platform == 'ollama':
    ollama_address = os.getenv("OLLAMA_ADDRESS")
    ollama_model = os.getenv("OLLAMA_MODEL")
    if not ollama_address:
        logger.error(f"{ERROR_ICON} 未找到 OLLAMA_ADDRESS 环境变量")
        raise ValueError("OLLAMA_ADDRESS not found in environment variables")
    if not ollama_model:
        logger.error(f"{ERROR_ICON} 未找到 OLLAMA_MODEL 环境变量")
        raise ValueError("OLLAMA_MODEL not found in environment variables")
    ai_client = OllamaClient(ollama_address, ollama_model, logger)
elif ai_platform == 'gemini':
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")
    if not api_key:
        logger.error(f"{ERROR_ICON} 未找到 GEMINI_API_KEY 环境变量")
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    if not model:
        logger.error(f"{ERROR_ICON} 未找到 GEMINI_MODEL 环境变量")
        raise ValueError("GEMINI_MODEL not found in environment variables")
    ai_client = GeminiClient(api_key, model, logger)


# class GptClient:
#     def get_chat_completion(self, messages, max_retries=3, initial_retry_delay=1) -> str:
#         pass



