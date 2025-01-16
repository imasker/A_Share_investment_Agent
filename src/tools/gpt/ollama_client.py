import os
import time
import google.generativeai as genai
from ollama import Client
import backoff

from src.tools.gpt.logger import WAIT_ICON, SUCCESS_ICON, ERROR_ICON


class OllamaClient:
    def __init__(self, address, model, logger):
        self.client = Client(
            host=address,
            headers={}
        )
        self.model = model
        self.logger = logger

    def generate_content_with_retry(self, contents):
        """带重试机制的内容生成函数"""
        try:
            self.logger.info(f"{WAIT_ICON} 正在调用 Ollama API...")
            self.logger.info(f"请求内容: {contents[:500]}..." if len(
                str(contents)) > 500 else f"请求内容: {contents}")

            response = self.client.chat(model=self.model, messages=contents)

            self.logger.info(f"{SUCCESS_ICON} API 调用成功")
            self.logger.info(f"响应内容: {response.message.content}..." if len(
                str(response.message.content)) > 500 else f"响应内容: {response.message.content}")
            return response
        except Exception as e:
            self.logger.error(f"{ERROR_ICON} API 调用失败: {str(e)}")
            self.logger.error(f"错误详情: {str(e)}")
            raise e

    def get_chat_completion(self, messages, max_retries=3, initial_retry_delay=1):
        """获取聊天完成结果，包含重试逻辑"""
        try:
            self.logger.info(f"{WAIT_ICON} 使用模型: {self.model}")
            self.logger.debug(f"消息内容: {messages}")

            for attempt in range(max_retries):
                try:
                    # 调用 API
                    response = self.generate_content_with_retry(
                        contents=messages
                    )

                    if response is None:
                        self.logger.warning(
                            f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries}: API 返回空值")
                        if attempt < max_retries - 1:
                            retry_delay = initial_retry_delay * (2 ** attempt)
                            self.logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                            time.sleep(retry_delay)
                            continue
                        return None

                    self.logger.debug(f"API 原始响应: {response.message.content}")
                    self.logger.info(f"{SUCCESS_ICON} 成功获取响应")
                    return response.message.content

                except Exception as e:
                    self.logger.error(
                        f"{ERROR_ICON} 尝试 {attempt + 1}/{max_retries} 失败: {str(e)}")
                    if attempt < max_retries - 1:
                        retry_delay = initial_retry_delay * (2 ** attempt)
                        self.logger.info(f"{WAIT_ICON} 等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                    else:
                        self.logger.error(f"{ERROR_ICON} 最终错误: {str(e)}")
                        return None

        except Exception as e:
            self.logger.error(f"{ERROR_ICON} get_chat_completion 发生错误: {str(e)}")
            return None
