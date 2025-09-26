"""
通义深度研究复现 - 基础Agent测试
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.base_agent import (
    BaseTool, BaseLLM, BaseDeepResearchAgent,
    ResearchState, Message, ToolResult, ToolType
)

class MockTool(BaseTool):
    """模拟工具用于测试"""

    def __init__(self, name="test_tool", description="测试工具"):
        super().__init__(name, description)
        self.tool_type = ToolType.SEARCH

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            arguments=kwargs,
            result="测试结果",
            success=True
        )

    def get_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "test_param": {"type": "string"}
            }
        }

class MockLLM(BaseLLM):
    """模拟LLM用于测试"""

    def chat(self, messages, **kwargs) -> str:
        return "模拟LLM响应"

    def generate_with_tools(self, messages, tools, **kwargs) -> dict:
        return {
            "content": "Think: 测试思考\nReport: 测试报告\nAction: 使用工具 test_tool",
            "function_call": None
        }

class TestResearchState(unittest.TestCase):
    """测试研究状态管理"""

    def setUp(self):
        self.state = ResearchState()

    def test_initial_state(self):
        """测试初始状态"""
        self.assertEqual(len(self.state.messages), 0)
        self.assertEqual(len(self.state.tool_results), 0)
        self.assertEqual(self.state.research_report, "")
        self.assertEqual(self.state.current_round, 0)

    def test_add_message(self):
        """测试添加消息"""
        message = Message(role="user", content="测试消息")
        self.state.add_message(message)
        self.assertEqual(len(self.state.messages), 1)
        self.assertEqual(self.state.messages[0].content, "测试消息")

    def test_add_tool_result(self):
        """测试添加工具结果"""
        result = ToolResult(
            tool_name="test_tool",
            arguments={},
            result="测试结果",
            success=True
        )
        self.state.add_tool_result(result)
        self.assertEqual(len(self.state.tool_results), 1)
        self.assertEqual(self.state.tool_results[0].result, "测试结果")

    def test_update_report(self):
        """测试更新报告"""
        self.state.update_report("测试报告")
        self.assertEqual(self.state.research_report, "测试报告")

    def test_should_continue(self):
        """测试继续条件"""
        self.assertTrue(self.state.should_continue())

        # 模拟达到最大轮次
        self.state.current_round = self.state.max_rounds
        self.assertFalse(self.state.should_continue())

    def test_get_context_summary(self):
        """测试获取上下文摘要"""
        self.state.update_report("测试报告内容")
        summary = self.state.get_context_summary()
        self.assertIn("测试报告内容", summary)

class TestMessage(unittest.TestCase):
    """测试消息类"""

    def test_message_creation(self):
        """测试消息创建"""
        message = Message(role="user", content="测试内容")
        self.assertEqual(message.role, "user")
        self.assertEqual(message.content, "测试内容")
        self.assertIsInstance(message.timestamp, datetime)

    def test_message_with_list_content(self):
        """测试列表形式的消息内容"""
        content = [{"type": "text", "text": "测试内容"}]
        message = Message(role="user", content=content)
        self.assertEqual(message.content, content)

class TestToolResult(unittest.TestCase):
    """测试工具结果类"""

    def test_tool_result_creation(self):
        """测试工具结果创建"""
        result = ToolResult(
            tool_name="test_tool",
            arguments={"param": "value"},
            result="测试结果",
            success=True
        )
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.arguments, {"param": "value"})
        self.assertEqual(result.result, "测试结果")
        self.assertTrue(result.success)
        self.assertIsInstance(result.timestamp, datetime)

    def test_tool_result_with_error(self):
        """测试带错误的工具结果"""
        result = ToolResult(
            tool_name="test_tool",
            arguments={},
            result="",
            success=False,
            error="测试错误"
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "测试错误")

class TestMockTool(unittest.TestCase):
    """测试模拟工具"""

    def setUp(self):
        self.tool = MockTool()

    def test_tool_execution(self):
        """测试工具执行"""
        result = self.tool.execute(test_param="测试值")
        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "test_tool")
        self.assertEqual(result.arguments["test_param"], "测试值")

    def test_tool_schema(self):
        """测试工具schema"""
        schema = self.tool.get_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)

class TestMockLLM(unittest.TestCase):
    """测试模拟LLM"""

    def setUp(self):
        self.llm = MockLLM()

    def test_chat(self):
        """测试聊天接口"""
        messages = [Message(role="user", content="测试消息")]
        response = self.llm.chat(messages)
        self.assertEqual(response, "模拟LLM响应")

    def test_generate_with_tools(self):
        """测试带工具生成"""
        messages = [Message(role="user", content="测试消息")]
        tools = [MockTool()]
        response = self.llm.generate_with_tools(messages, tools)
        self.assertIn("content", response)
        self.assertIn("function_call", response)

class TestBaseDeepResearchAgent(unittest.TestCase):
    """测试基础深度研究Agent"""

    def setUp(self):
        self.llm = MockLLM()
        self.tools = [MockTool()]

        # 创建一个具体的Agent实现来测试基础功能
        class TestAgent(BaseDeepResearchAgent):
            def research(self, query, **kwargs):
                # 简单的research方法实现
                return {"query": query, "result": "测试结果"}

        self.agent = TestAgent(
            llm=self.llm,
            tools=self.tools,
            system_message="测试系统消息",
            max_rounds=10
        )

    def test_reset(self):
        """测试重置"""
        self.agent.state.current_round = 5
        self.agent.reset()
        self.assertEqual(self.agent.state.current_round, 0)

    def test_format_tools_for_llm(self):
        """测试格式化工具信息"""
        tools_info = self.agent.format_tools_for_llm()
        self.assertEqual(len(tools_info), 1)
        self.assertEqual(tools_info[0]["name"], "test_tool")

    def test_create_system_prompt(self):
        """测试创建系统提示"""
        prompt = self.agent.create_system_prompt("测试任务")
        self.assertIn("测试系统消息", prompt)
        self.assertIn("测试任务", prompt)

    def test_parse_agent_response(self):
        """测试解析Agent响应"""
        response = """Think: 这是思考过程
Report: 这是报告内容
Action: 这是动作内容"""

        parsed = self.agent.parse_agent_response(response)
        self.assertEqual(parsed["think"], "这是思考过程")
        self.assertEqual(parsed["report"], "这是报告内容")
        self.assertEqual(parsed["action"], "这是动作内容")

    def test_execute_action_success(self):
        """测试成功执行动作"""
        action = "使用工具 test_tool"
        result = self.agent.execute_action(action)
        self.assertTrue(result.success)

    def test_execute_action_unknown_tool(self):
        """测试执行未知工具"""
        action = "使用工具 unknown_tool"
        result = self.agent.execute_action(action)
        self.assertFalse(result.success)
        self.assertIn("不存在", result.error)

    def test_execute_action_final_answer(self):
        """测试最终答案动作"""
        action = "这是最终答案"
        result = self.agent.execute_action(action)
        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "final_answer")

    def test_get_research_summary(self):
        """测试获取研究摘要"""
        self.agent.state.messages.append(Message(role="user", content="测试查询"))
        self.agent.state.tool_results.append(ToolResult(
            tool_name="test_tool",
            arguments={},
            result="测试结果",
            success=True
        ))

        summary = self.agent.get_research_summary()
        self.assertIn("测试查询", summary["query"])
        self.assertEqual(summary["total_tools_used"], 1)
        self.assertEqual(summary["success_rate"], 1.0)

if __name__ == '__main__':
    unittest.main()