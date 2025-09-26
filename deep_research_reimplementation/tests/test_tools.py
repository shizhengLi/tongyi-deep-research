"""
通义深度研究复现 - 工具测试
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.tools import (
    SearchTool, VisitTool, CodeInterpreterTool,
    CalculatorTool, ImageSearchTool, create_default_tools
)

class TestSearchTool(unittest.TestCase):
    """测试搜索工具"""

    def setUp(self):
        self.tool = SearchTool()

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.tool.name, "web_search")
        self.assertEqual(self.tool.description, "使用搜索引擎查找相关信息")

    def test_execute_success(self):
        """测试成功执行搜索"""
        result = self.tool.execute(query="测试查询", num_results=3)
        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "web_search")
        self.assertEqual(result.arguments["query"], "测试查询")
        self.assertEqual(result.arguments["num_results"], 3)

    def test_execute_with_default_params(self):
        """测试使用默认参数执行"""
        result = self.tool.execute(query="测试查询")
        self.assertTrue(result.success)
        self.assertEqual(result.arguments["num_results"], 5)

    def test_get_schema(self):
        """测试获取schema"""
        schema = self.tool.get_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("query", schema["properties"])
        self.assertIn("num_results", schema["properties"])
        self.assertIn("query", schema["required"])

    def test_mock_search_results(self):
        """测试模拟搜索结果"""
        results = self.tool._mock_search_results("测试", 3)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn("title", result)
            self.assertIn("url", result)
            self.assertIn("snippet", result)
            self.assertIn("content", result)

class TestVisitTool(unittest.TestCase):
    """测试网页访问工具"""

    def setUp(self):
        self.tool = VisitTool()

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.tool.name, "web_visit")
        self.assertEqual(self.tool.description, "访问指定URL并提取网页内容")

    @patch('requests.Session.get')
    def test_execute_success(self, mock_get):
        """测试成功访问网页"""
        # 模拟HTTP响应
        mock_response = Mock()
        mock_response.text = """
        <html>
            <head><title>测试页面</title></head>
            <body>
                <h1>测试标题</h1>
                <p>测试内容</p>
                <a href="/link1">链接1</a>
                <img src="/image1.jpg" alt="图片1">
            </body>
        </html>
        """
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = self.tool.execute(url="https://example.com")
        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "web_visit")
        self.assertIn("title", result.result)
        self.assertEqual(result.result["title"], "测试页面")

    def test_execute_invalid_url(self):
        """测试访问无效URL"""
        result = self.tool.execute(url="invalid_url")
        self.assertFalse(result.success)
        self.assertIn("无效的URL", result.error)

    def test_is_valid_url(self):
        """测试URL验证"""
        self.assertTrue(self.tool._is_valid_url("https://example.com"))
        self.assertTrue(self.tool._is_valid_url("http://test.com/path"))
        self.assertFalse(self.tool._is_valid_url("invalid_url"))
        self.assertFalse(self.tool._is_valid_url(""))

    def test_extract_content(self):
        """测试内容提取"""
        html = """
        <html>
            <head><title>测试页面</title></head>
            <body>
                <script>console.log('应该被移除');</script>
                <style>.test{color:red;}</style>
                <h1>主标题</h1>
                <p>这是一个测试段落，包含一些   空格和    换行符。</p>
                <nav>导航栏内容</nav>
                <a href="https://example.com/link1">链接文本1</a>
                <a href="https://example.com/link2">链接文本2</a>
                <img src="https://example.com/img1.jpg" alt="图片1">
                <img src="https://example.com/img2.jpg" alt="图片2">
            </body>
        </html>
        """

        content = self.tool._extract_content(html, "https://example.com")

        self.assertIn("url", content)
        self.assertIn("title", content)
        self.assertIn("content", content)
        self.assertIn("links", content)
        self.assertIn("images", content)

        self.assertEqual(content["title"], "测试页面")
        self.assertIn("主标题", content["content"])
        self.assertNotIn("console.log", content["content"])  # script被移除
        self.assertNotIn("导航栏内容", content["content"])  # nav被移除

        # 测试链接提取
        self.assertEqual(len(content["links"]), 2)
        self.assertEqual(content["links"][0]["text"], "链接文本1")

        # 测试图片提取
        self.assertEqual(len(content["images"]), 2)
        self.assertEqual(content["images"][0]["alt"], "图片1")

    def test_content_length_limit(self):
        """测试内容长度限制"""
        # 创建长内容
        long_content = "x" * 10000
        html = f"<html><body><p>{long_content}</p></body></html>"

        content = self.tool._extract_content(html, "https://example.com")
        self.assertLessEqual(len(content["content"]), self.tool.max_content_length + 3)  # +3 for "..."

    def test_get_schema(self):
        """测试获取schema"""
        schema = self.tool.get_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("url", schema["properties"])
        self.assertIn("url", schema["required"])

class TestCodeInterpreterTool(unittest.TestCase):
    """测试代码解释器工具"""

    def setUp(self):
        self.tool = CodeInterpreterTool()

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.tool.name, "code_interpreter")
        self.assertEqual(self.tool.description, "执行Python代码进行数据分析和计算")

    def test_execute_safe_math_expression(self):
        """测试执行安全的数学表达式"""
        result = self.tool.execute(code="2 + 3 * 4")
        self.assertTrue(result.success)
        self.assertIn("计算结果: 14", result.result)

    def test_execute_unsafe_code(self):
        """测试执行不安全的代码"""
        result = self.tool.execute(code="import os; os.system('ls')")
        self.assertFalse(result.success)
        self.assertIn("不安全的操作", result.error)

    def test_is_safe_code(self):
        """测试代码安全检查"""
        self.assertTrue(self.tool._is_safe_code("2 + 2"))
        self.assertTrue(self.tool._is_safe_code("print('hello')"))
        self.assertFalse(self.tool._is_safe_code("import os"))
        self.assertFalse(self.tool._is_safe_code("exec('dangerous')"))
        self.assertFalse(self.tool._is_safe_code("open('file.txt')"))

    def test_get_schema(self):
        """测试获取schema"""
        schema = self.tool.get_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("code", schema["properties"])
        self.assertIn("code", schema["required"])

class TestCalculatorTool(unittest.TestCase):
    """测试计算器工具"""

    def setUp(self):
        self.tool = CalculatorTool()

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.tool.name, "calculator")
        self.assertEqual(self.tool.description, "执行数学计算")

    def test_execute_simple_calculation(self):
        """测试简单计算"""
        result = self.tool.execute(expression="2 + 3 * 4")
        self.assertTrue(result.success)
        self.assertEqual(result.result, "14")

    def test_execute_complex_calculation(self):
        """测试复杂计算"""
        result = self.tool.execute(expression="(10 + 5) * 2 / 3")
        self.assertTrue(result.success)
        self.assertEqual(result.result, "10.0")

    def test_execute_invalid_expression(self):
        """测试无效表达式"""
        result = self.tool.execute(expression="2 + * 3")
        self.assertFalse(result.success)
        # 错误消息可能包含"syntax error"或类似信息
        self.assertIn("syntax", result.error.lower())

    def test_execute_unsafe_expression(self):
        """测试不安全的表达式"""
        result = self.tool.execute(expression="2 + eval('dangerous')")
        self.assertFalse(result.success)
        self.assertIn("不允许的字符", result.error)

    def test_get_schema(self):
        """测试获取schema"""
        schema = self.tool.get_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("expression", schema["properties"])
        self.assertIn("expression", schema["required"])

class TestImageSearchTool(unittest.TestCase):
    """测试图像搜索工具"""

    def setUp(self):
        self.tool = ImageSearchTool()

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.tool.name, "image_search")
        self.assertEqual(self.tool.description, "搜索相关图像")

    def test_execute_success(self):
        """测试成功执行图像搜索"""
        result = self.tool.execute(query="测试查询", num_results=3)
        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "image_search")
        self.assertEqual(result.arguments["query"], "测试查询")

    def test_execute_with_default_params(self):
        """测试使用默认参数执行"""
        result = self.tool.execute(query="测试查询")
        self.assertTrue(result.success)
        self.assertEqual(result.arguments["num_results"], 5)

    def test_mock_image_search_results(self):
        """测试模拟图像搜索结果"""
        results = self.tool._mock_image_search_results("测试", 3)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn("title", result)
            self.assertIn("url", result)
            self.assertIn("thumbnail", result)
            self.assertIn("description", result)

    def test_get_schema(self):
        """测试获取schema"""
        schema = self.tool.get_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("query", schema["properties"])
        self.assertIn("num_results", schema["properties"])
        self.assertIn("query", schema["required"])

class TestCreateDefaultTools(unittest.TestCase):
    """测试创建默认工具集"""

    def test_create_default_tools(self):
        """测试创建默认工具集"""
        tools = create_default_tools()
        self.assertEqual(len(tools), 5)

        tool_names = [tool.name for tool in tools]
        self.assertIn("web_search", tool_names)
        self.assertIn("web_visit", tool_names)
        self.assertIn("code_interpreter", tool_names)
        self.assertIn("calculator", tool_names)
        self.assertIn("image_search", tool_names)

if __name__ == '__main__':
    unittest.main()