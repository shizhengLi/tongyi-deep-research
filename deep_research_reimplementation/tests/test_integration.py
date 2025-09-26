"""
通义深度研究复现 - 集成测试
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.web_dancer_agent import WebDancerAgent
from core.tools import SearchTool, VisitTool, create_default_tools
from core.base_agent import Message

class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_full_research_cycle(self):
        """测试完整的研究周期"""
        # 创建Agent
        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=2  # 限制轮次以便测试
        )

        # 执行研究任务
        query = "什么是机器学习？"
        result = agent.research(query)

        # 验证结果结构
        self.assertIn("query", result)
        self.assertIn("final_report", result)
        self.assertIn("research_summary", result)
        self.assertIn("success", result)

        self.assertEqual(result["query"], query)
        self.assertIsInstance(result["final_report"], str)
        self.assertIsInstance(result["research_summary"], dict)
        self.assertTrue(result["success"])

        # 验证研究摘要
        summary = result["research_summary"]
        self.assertIn("total_rounds", summary)
        self.assertIn("total_tools_used", summary)
        self.assertIn("research_report", summary)
        self.assertIn("success_rate", summary)

        self.assertGreaterEqual(summary["total_rounds"], 1)
        self.assertGreaterEqual(summary["total_tools_used"], 0)
        self.assertGreaterEqual(summary["success_rate"], 0.0)
        self.assertLessEqual(summary["success_rate"], 1.0)

    def test_research_with_tool_usage(self):
        """测试使用工具的研究"""
        # 创建Agent
        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1  # 限制轮次
        )

        # 执行研究任务
        query = "查找关于量子计算的最新研究"
        result = agent.research(query)

        # 验证工具被使用
        summary = result["research_summary"]
        self.assertGreater(summary["total_tools_used"], 0)

    def test_research_state_management(self):
        """测试研究状态管理"""
        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1
        )

        # 执行第一个研究任务
        query1 = "什么是深度学习？"
        result1 = agent.research(query1)

        # 验证状态
        self.assertEqual(agent.state.current_round, 1)
        self.assertGreater(len(agent.state.messages), 0)

        # 执行第二个研究任务（应该重置状态）
        query2 = "什么是强化学习？"
        result2 = agent.research(query2)

        # 验证状态重置
        self.assertEqual(agent.state.current_round, 1)
        self.assertEqual(result2["query"], query2)

    def test_multiple_queries_comparison(self):
        """测试多个查询的比较"""
        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1
        )

        queries = [
            "什么是人工智能？",
            "什么是机器学习？",
            "什么是深度学习？"
        ]

        results = []
        for query in queries:
            result = agent.research(query)
            results.append(result)

        # 验证所有查询都成功执行
        for i, result in enumerate(results):
            self.assertTrue(result["success"])
            self.assertEqual(result["query"], queries[i])
            self.assertIsInstance(result["final_report"], str)

    def test_error_handling(self):
        """测试错误处理"""
        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1
        )

        # 执行可能出错的研究任务
        query = "这是一个可能导致错误的查询"
        result = agent.research(query)

        # 验证错误处理
        self.assertIn("success", result)
        # 即使有错误，也应该返回结构化的结果

    def test_research_details(self):
        """测试研究详情获取"""
        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1
        )

        # 执行研究任务
        query = "测试查询"
        result = agent.research(query)

        # 获取详细的研究信息
        details = agent.get_research_details()

        # 验证详情结构
        self.assertIn("messages", details)
        self.assertIn("tool_results", details)
        self.assertIn("thought_history", details)
        self.assertIn("action_history", details)
        self.assertIn("research_report", details)
        self.assertIn("summary", details)

        # 验证详情内容
        self.assertIsInstance(details["messages"], list)
        self.assertIsInstance(details["tool_results"], list)
        self.assertIsInstance(details["thought_history"], list)
        self.assertIsInstance(details["action_history"], list)
        self.assertIsInstance(details["research_report"], str)
        self.assertIsInstance(details["summary"], dict)

    def test_custom_tools_integration(self):
        """测试自定义工具集成"""
        # 创建自定义工具
        class CustomTestTool:
            def __init__(self):
                self.name = "custom_tool"
                self.description = "自定义测试工具"

            def execute(self, **kwargs):
                return "自定义工具执行结果"

            def get_schema(self):
                return {"type": "object", "properties": {}}

        # 创建带有自定义工具的Agent
        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1
        )
        agent.tools["custom_tool"] = CustomTestTool()

        # 执行研究任务
        query = "使用自定义工具"
        result = agent.research(query)

        # 验证结果
        self.assertTrue(result["success"])
        self.assertIn("final_report", result)

    def test_performance_basic(self):
        """测试基本性能"""
        import time

        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1
        )

        # 测试响应时间
        start_time = time.time()
        result = agent.research("性能测试查询")
        end_time = time.time()

        # 验证响应时间在合理范围内（小于10秒）
        response_time = end_time - start_time
        self.assertLess(response_time, 10.0)

        # 验证结果正确性
        self.assertTrue(result["success"])

    def test_memory_usage_basic(self):
        """测试基本内存使用"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1
        )

        # 执行研究任务
        result = agent.research("内存测试查询")

        # 检查内存增长
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增长应该在合理范围内（小于100MB）
        self.assertLess(memory_increase, 100 * 1024 * 1024)

        # 验证结果正确性
        self.assertTrue(result["success"])

    def test_concurrent_research(self):
        """测试并发研究（简化版）"""
        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1
        )

        # 顺序执行多个研究任务（模拟并发）
        queries = [
            "查询1",
            "查询2",
            "查询3"
        ]

        results = []
        for query in queries:
            result = agent.research(query)
            results.append(result)

        # 验证所有结果
        self.assertEqual(len(results), 3)
        for i, result in enumerate(results):
            self.assertTrue(result["success"])
            self.assertEqual(result["query"], queries[i])

class TestScenarios(unittest.TestCase):
    """测试具体场景"""

    def test_academic_research_scenario(self):
        """测试学术研究场景"""
        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1
        )

        query = "请研究量子计算在密码学中的应用前景"
        result = agent.research(query)

        self.assertTrue(result["success"])
        self.assertIn("final_report", result)

        # 验证研究摘要包含有意义的信息
        summary = result["research_summary"]
        self.assertGreaterEqual(summary["total_rounds"], 1)

    def test_business_analysis_scenario(self):
        """测试商业分析场景"""
        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1
        )

        query = "分析人工智能在金融科技领域的市场机会"
        result = agent.research(query)

        self.assertTrue(result["success"])
        self.assertIn("final_report", result)

    def test_technical_explanation_scenario(self):
        """测试技术解释场景"""
        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1
        )

        query = "解释Transformer架构的工作原理"
        result = agent.research(query)

        self.assertTrue(result["success"])
        self.assertIn("final_report", result)

    def test_current_events_scenario(self):
        """测试时事查询场景"""
        agent = WebDancerAgent(
            llm_provider="mock",
            max_rounds=1
        )

        query = "最近在人工智能领域有哪些重要突破？"
        result = agent.research(query)

        self.assertTrue(result["success"])
        self.assertIn("final_report", result)

if __name__ == '__main__':
    # 运行集成测试
    unittest.main(verbosity=2)