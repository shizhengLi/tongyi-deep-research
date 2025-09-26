"""
通义深度研究复现 - 工具实现
实现了搜索、访问、代码解释等核心工具
"""

import json
import requests
import re
from typing import Dict, Any, List, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
import logging
from .base_agent import BaseTool, ToolResult, ToolType

logger = logging.getLogger(__name__)

class SearchTool(BaseTool):
    """搜索工具"""

    def __init__(self, api_key: str = None, search_engine: str = "google"):
        super().__init__(
            name="web_search",
            description="使用搜索引擎查找相关信息"
        )
        self.api_key = api_key
        self.search_engine = search_engine
        self.tool_type = ToolType.SEARCH

    def execute(self, query: str, num_results: int = 5, **kwargs) -> ToolResult:
        """执行搜索"""
        try:
            # 模拟搜索结果（实际应用中需要调用真实搜索API）
            mock_results = self._mock_search_results(query, num_results)

            return ToolResult(
                tool_name=self.name,
                arguments={"query": query, "num_results": num_results},
                result=mock_results,
                success=True
            )
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return ToolResult(
                tool_name=self.name,
                arguments={"query": query, "num_results": num_results},
                result="",
                success=False,
                error=str(e)
            )

    def _mock_search_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """模拟搜索结果（用于演示）"""
        # 这里使用模拟数据，实际应用中应该调用真实搜索API
        return [
            {
                "title": f"关于'{query}'的搜索结果 {i+1}",
                "url": f"https://example.com/search_result_{i+1}",
                "snippet": f"这是关于'{query}'的第{i+1}个搜索结果摘要...",
                "content": f"详细内容：这是关于'{query}'的详细信息，包含了相关的背景知识和最新发展。"
            }
            for i in range(min(num_results, 5))
        ]

    def get_schema(self) -> Dict[str, Any]:
        """获取工具schema"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询字符串"
                },
                "num_results": {
                    "type": "integer",
                    "description": "返回结果数量",
                    "default": 5
                }
            },
            "required": ["query"]
        }

class VisitTool(BaseTool):
    """网页访问工具"""

    def __init__(self, max_content_length: int = 5000):
        super().__init__(
            name="web_visit",
            description="访问指定URL并提取网页内容"
        )
        self.max_content_length = max_content_length
        self.tool_type = ToolType.VISIT
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def execute(self, url: str, **kwargs) -> ToolResult:
        """执行网页访问"""
        try:
            # 验证URL
            if not self._is_valid_url(url):
                return ToolResult(
                    tool_name=self.name,
                    arguments={"url": url},
                    result="",
                    success=False,
                    error=f"无效的URL: {url}"
                )

            # 获取网页内容
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            # 解析网页内容
            content = self._extract_content(response.text, url)

            return ToolResult(
                tool_name=self.name,
                arguments={"url": url},
                result=content,
                success=True
            )

        except Exception as e:
            logger.error(f"访问网页失败: {e}")
            return ToolResult(
                tool_name=self.name,
                arguments={"url": url},
                result="",
                success=False,
                error=str(e)
            )

    def _is_valid_url(self, url: str) -> bool:
        """验证URL是否有效"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """提取网页内容"""
        soup = BeautifulSoup(html, 'html.parser')

        # 移除不需要的元素
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        # 提取标题
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "无标题"

        # 提取正文
        content = soup.get_text()
        content = re.sub(r'\s+', ' ', content).strip()

        # 限制内容长度
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."

        return {
            "url": url,
            "title": title_text,
            "content": content,
            "links": self._extract_links(soup, url),
            "images": self._extract_images(soup, url)
        }

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """提取链接"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href:
                absolute_url = urljoin(base_url, href)
                text = link.get_text().strip()
                if text and absolute_url:
                    links.append({
                        "text": text[:50],  # 限制链接文本长度
                        "url": absolute_url
                    })

        # 限制链接数量
        return links[:10]

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """提取图片"""
        images = []
        for img in soup.find_all('img', src=True):
            src = img.get('src')
            if src:
                absolute_url = urljoin(base_url, src)
                alt = img.get('alt', '')
                images.append({
                    "alt": alt[:50],
                    "url": absolute_url
                })

        # 限制图片数量
        return images[:5]

    def get_schema(self) -> Dict[str, Any]:
        """获取工具schema"""
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "要访问的网页URL"
                }
            },
            "required": ["url"]
        }

class CodeInterpreterTool(BaseTool):
    """代码解释器工具"""

    def __init__(self, timeout: int = 30):
        super().__init__(
            name="code_interpreter",
            description="执行Python代码进行数据分析和计算"
        )
        self.timeout = timeout
        self.tool_type = ToolType.CODE_INTERPRETER

    def execute(self, code: str, **kwargs) -> ToolResult:
        """执行代码"""
        try:
            # 安全检查（这里只做简单检查，实际应用中需要更完善的安全措施）
            if not self._is_safe_code(code):
                return ToolResult(
                    tool_name=self.name,
                    arguments={"code": code},
                    result="",
                    success=False,
                    error="代码包含不安全的操作"
                )

            # 执行代码（这里使用模拟执行，实际应用中需要沙箱环境）
            result = self._execute_code_safely(code)

            return ToolResult(
                tool_name=self.name,
                arguments={"code": code},
                result=result,
                success=True
            )

        except Exception as e:
            logger.error(f"代码执行失败: {e}")
            return ToolResult(
                tool_name=self.name,
                arguments={"code": code},
                result="",
                success=False,
                error=str(e)
            )

    def _is_safe_code(self, code: str) -> bool:
        """简单的代码安全检查"""
        forbidden_keywords = [
            'import os', 'import subprocess', 'import sys',
            'exec(', 'eval(', '__import__',
            'open(', 'file(', 'input(',
            'rm ', 'del ', 'rmdir'
        ]

        code_lower = code.lower()
        for keyword in forbidden_keywords:
            if keyword in code_lower:
                return False

        return True

    def _execute_code_safely(self, code: str) -> str:
        """安全执行代码（模拟）"""
        # 这里只是一个模拟实现
        # 实际应用中需要使用真正的沙箱环境

        # 简单的数学表达式计算
        if re.match(r'^[\d\+\-\*\/\(\)\s\.]+$', code):
            try:
                result = eval(code)
                return f"计算结果: {result}"
            except:
                return "表达式计算错误"

        # 如果不是简单的数学表达式，返回模拟结果
        return f"代码执行结果:\n{code}\n\n这是一个模拟的执行结果。在实际应用中，这里会返回真实的代码执行输出。"

    def get_schema(self) -> Dict[str, Any]:
        """获取工具schema"""
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的Python代码"
                }
            },
            "required": ["code"]
        }

class CalculatorTool(BaseTool):
    """计算器工具"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="执行数学计算"
        )
        self.tool_type = ToolType.CALCULATOR

    def execute(self, expression: str, **kwargs) -> ToolResult:
        """执行计算"""
        try:
            # 安全的表达式计算
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return ToolResult(
                    tool_name=self.name,
                    arguments={"expression": expression},
                    result="",
                    success=False,
                    error="表达式包含不允许的字符"
                )

            result = eval(expression)
            return ToolResult(
                tool_name=self.name,
                arguments={"expression": expression},
                result=str(result),
                success=True
            )

        except Exception as e:
            logger.error(f"计算失败: {e}")
            return ToolResult(
                tool_name=self.name,
                arguments={"expression": expression},
                result="",
                success=False,
                error=str(e)
            )

    def get_schema(self) -> Dict[str, Any]:
        """获取工具schema"""
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如: 2 + 3 * 4"
                }
            },
            "required": ["expression"]
        }

class ImageSearchTool(BaseTool):
    """图像搜索工具"""

    def __init__(self, api_key: str = None):
        super().__init__(
            name="image_search",
            description="搜索相关图像"
        )
        self.api_key = api_key
        self.tool_type = ToolType.IMAGE_SEARCH

    def execute(self, query: str, num_results: int = 5, **kwargs) -> ToolResult:
        """执行图像搜索"""
        try:
            # 模拟图像搜索结果
            mock_results = self._mock_image_search_results(query, num_results)

            return ToolResult(
                tool_name=self.name,
                arguments={"query": query, "num_results": num_results},
                result=mock_results,
                success=True
            )

        except Exception as e:
            logger.error(f"图像搜索失败: {e}")
            return ToolResult(
                tool_name=self.name,
                arguments={"query": query, "num_results": num_results},
                result="",
                success=False,
                error=str(e)
            )

    def _mock_image_search_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """模拟图像搜索结果"""
        return [
            {
                "title": f"关于'{query}'的图像 {i+1}",
                "url": f"https://example.com/image_{i+1}.jpg",
                "thumbnail": f"https://example.com/thumbnail_{i+1}.jpg",
                "description": f"这是关于'{query}'的第{i+1}个图像结果"
            }
            for i in range(min(num_results, 5))
        ]

    def get_schema(self) -> Dict[str, Any]:
        """获取工具schema"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "图像搜索查询"
                },
                "num_results": {
                    "type": "integer",
                    "description": "返回结果数量",
                    "default": 5
                }
            },
            "required": ["query"]
        }

def create_default_tools() -> List[BaseTool]:
    """创建默认工具集"""
    return [
        SearchTool(),
        VisitTool(),
        CodeInterpreterTool(),
        CalculatorTool(),
        ImageSearchTool()
    ]