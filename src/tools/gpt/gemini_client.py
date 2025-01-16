import os
import time
import google.generativeai as genai
import backoff

from src.tools.gpt.logger import WAIT_ICON, SUCCESS_ICON, ERROR_ICON


class GeminiClient:
    def __init__(self, api_key, model, logger):
        genai.configure(api_key=api_key)
        self.model = model
        self.logger = logger

    @backoff.on_exception(
        backoff.expo,
        (Exception),
        max_tries=5,
        max_time=300,
        giveup=lambda e: "AFC is enabled" not in str(e)
    )
    def generate_content_with_retry(self, contents, config=None):
        """带重试机制的内容生成函数"""
        try:
            self.logger.info(f"{WAIT_ICON} 正在调用 Gemini API...")
            self.logger.info(f"请求内容: {contents[:500]}..." if len(
                str(contents)) > 500 else f"请求内容: {contents}")
            self.logger.info(f"请求配置: {config}")

            client = genai.GenerativeModel(self.model, **config)
            response = client.generate_content(
                contents=contents
            )

            self.logger.info(f"{SUCCESS_ICON} API 调用成功")
            self.logger.info(f"响应内容: {response.text[:500]}..." if len(
                str(response.text)) > 500 else f"响应内容: {response.text}")
            return response
        except Exception as e:
            if "AFC is enabled" in str(e):
                self.logger.warning(f"{ERROR_ICON} 触发 API 限制，等待重试... 错误: {str(e)}")
                time.sleep(5)
                raise e
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
                    # 转换消息格式
                    prompt = ""
                    system_instruction = None

                    for message in messages:
                        role = message["role"]
                        content = message["content"]
                        if role == "system":
                            system_instruction = content
                        elif role == "user":
                            prompt += f"User: {content}\n"
                        elif role == "assistant":
                            prompt += f"Assistant: {content}\n"

                    # 准备配置
                    config = {}
                    if system_instruction:
                        config['system_instruction'] = system_instruction

                    # 调用 API
                    response = self.generate_content_with_retry(
                        contents=prompt.strip(),
                        config=config
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

                    self.logger.debug(f"API 原始响应: {response.text}")
                    self.logger.info(f"{SUCCESS_ICON} 成功获取响应")
                    return response.text

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
