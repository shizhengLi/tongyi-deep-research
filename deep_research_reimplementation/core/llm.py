"""
通义深度研究复现 - 大语言模型接口
支持多种LLM后端
"""

import json
import requests
from typing import Dict, List, Any, Optional
from .base_agent import BaseLLM, Message
import logging

logger = logging.getLogger(__name__)

class MockLLM(BaseLLM):
    """模拟LLM，用于测试和演示"""

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.responses = [
            "这是一个模拟的LLM响应。在实际应用中，这里会调用真实的LLM API。",
            "基于您的查询，我建议使用搜索工具来获取相关信息。",
            "让我分析一下当前情况并制定下一步计划。"
        ]

    def chat(self, messages: List[Message], **kwargs) -> str:
        """聊天接口"""
        # 简单的响应逻辑
        last_message = messages[-1].content if messages else ""
        return f"关于'{last_message}'的响应：{self.responses[len(messages) % len(self.responses)]}"

    def generate_with_tools(self, messages: List[Message], tools: List[Any], **kwargs) -> Dict[str, Any]:
        """带工具生成的接口"""
        # 模拟工具调用响应
        return {
            "content": """Think: 我需要分析用户的查询并选择合适的工具。
Report: 当前研究状态显示需要搜索更多信息来回答用户的问题。
Action: 使用工具 web_search 查询相关信息""",
            "function_call": None
        }

class OpenAILLM(BaseLLM):
    """OpenAI LLM接口"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: List[Message], **kwargs) -> str:
        """聊天接口"""
        try:
            # 格式化消息
            formatted_messages = []
            for msg in messages:
                if isinstance(msg.content, str):
                    formatted_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
                elif isinstance(msg.content, list):
                    # 处理多模态内容
                    content_parts = []
                    for part in msg.content:
                        if isinstance(part, dict):
                            if "text" in part:
                                content_parts.append({"type": "text", "text": part["text"]})
                            elif "image_url" in part:
                                content_parts.append({"type": "image_url", "image_url": part["image_url"]})
                    formatted_messages.append({
                        "role": msg.role,
                        "content": content_parts
                    })

            payload = {
                "model": self.model,
                "messages": formatted_messages,
                **kwargs
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            return f"API调用失败: {str(e)}"

    def generate_with_tools(self, messages: List[Message], tools: List[Any], **kwargs) -> Dict[str, Any]:
        """带工具生成的接口"""
        try:
            # 格式化工具
            formatted_tools = []
            for tool in tools:
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.get_schema()
                    }
                })

            # 格式化消息
            formatted_messages = []
            for msg in messages:
                if isinstance(msg.content, str):
                    formatted_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })

            payload = {
                "model": self.model,
                "messages": formatted_messages,
                "tools": formatted_tools,
                "tool_choice": "auto",
                **kwargs
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            message = result["choices"][0]["message"]

            return {
                "content": message.get("content", ""),
                "function_call": message.get("tool_calls")
            }

        except Exception as e:
            logger.error(f"OpenAI工具调用失败: {e}")
            return {
                "content": f"工具调用失败: {str(e)}",
                "function_call": None
            }

class QwenLLM(BaseLLM):
    """通义千问LLM接口"""

    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages: List[Message], **kwargs) -> str:
        """聊天接口"""
        try:
            # 格式化消息
            input_messages = []
            for msg in messages:
                if msg.role == "system":
                    continue  # Qwen API的系统消息处理方式不同
                role = "user" if msg.role == "user" else "assistant"
                input_messages.append({
                    "role": role,
                    "content": msg.content if isinstance(msg.content, str) else str(msg.content)
                })

            payload = {
                "model": self.model,
                "input": {
                    "messages": input_messages
                },
                "parameters": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 2000)
                }
            }

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result["output"]["text"]

        except Exception as e:
            logger.error(f"Qwen API调用失败: {e}")
            return f"API调用失败: {str(e)}"

    def generate_with_tools(self, messages: List[Message], tools: List[Any], **kwargs) -> Dict[str, Any]:
        """带工具生成的接口"""
        try:
            # Qwen的工具调用方式可能需要特殊处理
            # 这里简化处理，直接调用chat接口
            content = self.chat(messages, **kwargs)

            # 简单的工具调用解析
            if "使用工具" in content or "use_tool" in content:
                # 这里可以添加更复杂的工具调用解析逻辑
                return {
                    "content": content,
                    "function_call": "simulated_tool_call"
                }

            return {
                "content": content,
                "function_call": None
            }

        except Exception as e:
            logger.error(f"Qwen工具调用失败: {e}")
            return {
                "content": f"工具调用失败: {str(e)}",
                "function_call": None
            }

class GeminiLLM(BaseLLM):
    """Google Gemini LLM接口"""

    def __init__(self, api_key: str, model: str = "gemini-pro"):
        self.api_key = api_key
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.headers = {
            "Content-Type": "application/json"
        }

    def chat(self, messages: List[Message], **kwargs) -> str:
        """聊天接口"""
        try:
            # 构建对话历史
            contents = []
            for msg in messages:
                if msg.role == "system":
                    # 系统消息作为第一条用户消息的一部分
                    continue

                role = "user" if msg.role == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content if isinstance(msg.content, str) else str(msg.content)}]
                })

            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "maxOutputTokens": kwargs.get("max_tokens", 2000)
                }
            }

            params = {"key": self.api_key}
            response = requests.post(
                self.base_url,
                headers=self.headers,
                params=params,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]

        except Exception as e:
            logger.error(f"Gemini API调用失败: {e}")
            return f"API调用失败: {str(e)}"

    def generate_with_tools(self, messages: List[Message], tools: List[Any], **kwargs) -> Dict[str, Any]:
        """带工具生成的接口"""
        # Gemini的工具调用功能需要特殊处理
        # 这里简化处理，直接调用chat接口
        content = self.chat(messages, **kwargs)

        return {
            "content": content,
            "function_call": None
        }

def create_llm(provider: str, **kwargs) -> BaseLLM:
    """创建LLM实例的工厂函数"""
    provider = provider.lower()

    if provider == "mock":
        return MockLLM(kwargs.get("model_name", "mock-model"))
    elif provider == "openai":
        return OpenAILLM(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url", "https://api.openai.com/v1"),
            model=kwargs.get("model", "gpt-3.5-turbo")
        )
    elif provider == "qwen":
        return QwenLLM(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "qwen-turbo")
        )
    elif provider == "gemini":
        return GeminiLLM(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "gemini-pro")
        )
    else:
        raise ValueError(f"不支持的LLM提供商: {provider}")