"""
通义深度研究复现 - WebDancer Agent测试
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.web_dancer_agent import WebDancerAgent
from core.base_agent import Message, ToolResult
from core.tools import create_default_tools

class TestWebDancerAgent(unittest.TestCase):
    """测试WebDancer Agent"""

    def setUp(self):
        """设置测试环境"""
        self.agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=3  # 减少测试轮次
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.agent.llm)
        self.assertEqual(len(self.agent.tools), 5)  # 默认5个工具
        self.assertEqual(self.agent.max_rounds, 3)
        self.assertIn("WebDancer", self.agent.system_message)

    def test_reset(self):
        """测试重置"""
        # 添加一些状态
        self.agent.state.current_round = 2
        self.agent.state.research_report = "测试报告"

        # 重置
        self.agent.reset()

        # 验证重置
        self.assertEqual(self.agent.state.current_round, 0)
        self.assertEqual(self.agent.state.research_report, "")

    def test_research_initialization(self):
        """测试研究任务初始化"""
        query = "什么是人工智能？"

        # 开始研究
        self.agent.state.add_message(Message(role="user", content=query))
        self.agent.state.add_message(Message(role="system", content=self.agent.create_system_prompt()))

        # 验证初始化
        self.assertEqual(len(self.agent.state.messages), 2)
        self.assertEqual(self.agent.state.messages[0].role, "system")
        self.assertEqual(self.agent.state.messages[1].role, "user")
        self.assertEqual(self.agent.state.messages[1].content, query)

    def test_prepare_context(self):
        """测试准备上下文"""
        # 添加消息
        self.agent.state.add_message(Message(role="system", content="系统消息"))
        self.agent.state.add_message(Message(role="user", content="用户查询"))
        self.agent.state.update_report("测试报告")

        # 准备上下文
        context = self.agent._prepare_context()

        # 验证上下文包含必要信息
        self.assertGreaterEqual(len(context), 2)
        roles = [msg.role for msg in context]
        self.assertIn("system", roles)
        self.assertIn("user", roles)

        # 验证包含研究状态摘要
        context_content = " ".join([str(msg.content) for msg in context])
        self.assertIn("测试报告", context_content)

    def test_generate_agent_response(self):
        """测试生成Agent响应"""
        # 准备上下文
        context = [
            Message(role="system", content="系统消息"),
            Message(role="user", content="测试查询")
        ]

        # 生成响应
        response = self.agent._generate_agent_response(context)

        # 验证响应
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_parse_agent_response(self):
        """测试解析Agent响应"""
        response = """Think: 我需要分析用户的查询并制定研究策略
Report: 当前研究显示需要进一步了解人工智能的定义和应用
Action: 使用工具 web_search 搜索'人工智能定义'"""

        parsed = self.agent.parse_agent_response(response)

        self.assertEqual(parsed["think"], "我需要分析用户的查询并制定研究策略")
        self.assertEqual(parsed["report"], "当前研究显示需要进一步了解人工智能的定义和应用")
        self.assertEqual(parsed["action"], "使用工具 web_search 搜索'人工智能定义'")

    def test_parse_agent_response_multiline(self):
        """测试解析多行Agent响应"""
        response = """Think: 这是一个复杂的问题，
需要分步骤分析。
Report: 当前研究发现：
1. 人工智能是计算机科学的一个分支
2. 它涉及机器学习和深度学习
Action: 继续深入研究"""

        parsed = self.agent.parse_agent_response(response)

        self.assertIn("分步骤分析", parsed["think"])
        self.assertIn("机器学习和深度学习", parsed["report"])
        self.assertEqual(parsed["action"], "继续深入研究")

    def test_execute_action_tool_call(self):
        """测试执行工具调用动作"""
        action = "使用工具 web_search 查询'人工智能'"
        result = self.agent.execute_action(action)

        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "web_search")

    def test_execute_action_final_answer(self):
        """测试执行最终答案动作"""
        action = "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
        result = self.agent.execute_action(action)

        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "final_answer")
        self.assertEqual(result.result, action)

    def test_execute_action_unknown_tool(self):
        """测试执行未知工具"""
        action = "使用工具 unknown_tool"
        result = self.agent.execute_action(action)

        self.assertFalse(result.success)
        self.assertIn("不存在", result.error)

    def test_should_stop_research_finish_keywords(self):
        """测试停止研究条件 - 完成关键词"""
        response = {
            "think": "研究已完成",
            "report": "详细的研究报告",
            "action": "完成研究"
        }

        self.assertTrue(self.agent._should_stop_research(response))

    def test_should_stop_research_english_keywords(self):
        """测试停止研究条件 - 英文关键词"""
        response = {
            "think": "Research complete",
            "report": "Detailed research report",
            "action": "finish the research"
        }

        self.assertTrue(self.agent._should_stop_research(response))

    def test_should_stop_research_comprehensive_report(self):
        """测试停止研究条件 - 完整报告"""
        # 设置长报告
        self.agent.state.update_report("x" * 600)  # 超过500字符

        response = {
            "think": "继续分析",
            "report": "x" * 600,
            "action": "最终答案：这是最终的研究结论。"
        }

        self.assertTrue(self.agent._should_stop_research(response))

    def test_should_stop_research_max_rounds(self):
        """测试停止研究条件 - 最大轮次"""
        self.agent.state.current_round = self.agent.max_rounds

        response = {
            "think": "继续研究",
            "report": "简短报告",
            "action": "继续搜索"
        }

        self.assertTrue(self.agent._should_stop_research(response))

    def test_should_continue_research(self):
        """测试继续研究条件"""
        response = {
            "think": "需要更多信息",
            "report": "简短报告",
            "action": "使用工具 web_search 搜索更多信息"
        }

        self.assertFalse(self.agent._should_stop_research(response))

    def test_generate_final_report_with_existing_report(self):
        """测试生成最终报告 - 有现有报告"""
        self.agent.state.update_report("这是研究过程中积累的报告内容。")
        self.agent.state.current_round = 2
        self.agent.state.tool_results.append(ToolResult(
            tool_name="web_search",
            arguments={},
            result="搜索结果",
            success=True
        ))

        # 生成最终报告
        final_report = self.agent._generate_final_report()

        # 验证报告内容
        self.assertIsInstance(final_report, str)
        self.assertGreater(len(final_report), 0)

    def test_generate_final_report_without_existing_report(self):
        """测试生成最终报告 - 无现有报告"""
        self.agent.state.current_round = 1
        self.agent.state.tool_results.append(ToolResult(
            tool_name="web_search",
            arguments={},
            result="搜索结果",
            success=False
        ))

        # 生成最终报告
        final_report = self.agent._generate_final_report()

        # 验证报告内容
        self.assertIsInstance(final_report, str)
        self.assertIn("无法提供完整的答案", final_report)

    def test_get_research_details(self):
        """测试获取研究详情"""
        # 添加一些测试数据
        self.agent.state.add_message(Message(role="user", content="测试查询"))
        self.agent.state.add_tool_result(ToolResult(
            tool_name="web_search",
            arguments={},
            result="搜索结果",
            success=True
        ))
        self.agent.state.add_thought("测试思考")
        self.agent.state.add_action({"round": 1, "action": "测试动作"})

        # 获取详情
        details = self.agent.get_research_details()

        # 验证详情
        self.assertIn("messages", details)
        self.assertIn("tool_results", details)
        self.assertIn("thought_history", details)
        self.assertIn("action_history", details)
        self.assertIn("research_report", details)
        self.assertIn("summary", details)

        self.assertEqual(len(details["messages"]), 1)
        self.assertEqual(len(details["tool_results"]), 1)
        self.assertEqual(len(details["thought_history"]), 1)
        self.assertEqual(len(details["action_history"]), 1)

    @patch.object(WebDancerAgent, '_should_stop_research')
    @patch.object(WebDancerAgent, '_generate_agent_response')
    def test_research_full_cycle(self, mock_generate_response, mock_should_stop):
        """测试完整的研究周期"""
        # 设置mock
        mock_generate_response.return_value = """Think: 分析查询
Report: 初步研究结果
Action: 使用工具 web_search"""
        mock_should_stop.return_value = True  # 第一轮就停止

        # 执行研究
        query = "测试查询"
        result = self.agent.research(query)

        # 验证结果
        self.assertIn("query", result)
        self.assertIn("final_report", result)
        self.assertIn("research_summary", result)
        self.assertIn("success", result)

        self.assertEqual(result["query"], query)
        self.assertTrue(result["success"])

        # 验证状态更新
        self.assertEqual(self.agent.state.current_round, 1)
        self.assertEqual(len(self.agent.state.messages), 3)  # system, user, function
        self.assertEqual(len(self.agent.state.tool_results), 1)

    def test_create_system_prompt(self):
        """测试创建系统提示"""
        task_description = "研究人工智能的发展历史"
        prompt = self.agent.create_system_prompt(task_description)

        self.assertIn("WebDancer", prompt)
        self.assertIn("Think-Report-Action的循环", prompt)
        self.assertIn(task_description, prompt)

    def test_format_tools_for_llm(self):
        """测试格式化工具信息"""
        tools_info = self.agent.format_tools_for_llm()

        self.assertIsInstance(tools_info, list)
        self.assertEqual(len(tools_info), 5)

        for tool_info in tools_info:
            self.assertIn("name", tool_info)
            self.assertIn("description", tool_info)
            self.assertIn("parameters", tool_info)

if __name__ == '__main__':
    unittest.main()