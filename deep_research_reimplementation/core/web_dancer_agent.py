"""
通义深度研究复现 - WebDancer Agent
实现自主信息检索Agency
"""

import json
from typing import Dict, List, Any, Optional
from .base_agent import BaseDeepResearchAgent, Message, ResearchState
from .tools import create_default_tools
from .llm import create_llm
import logging

logger = logging.getLogger(__name__)

class WebDancerAgent(BaseDeepResearchAgent):
    """WebDancer Agent - 自主信息检索Agency"""

    def __init__(self,
                 llm_provider: str = "mock",
                 llm_config: Dict[str, Any] = None,
                 tools: List = None,
                 max_rounds: int = 20,
                 system_message: str = None):
        """
        初始化WebDancer Agent

        Args:
            llm_provider: LLM提供商 ("mock", "openai", "qwen", "gemini")
            llm_config: LLM配置参数
            tools: 工具列表
            max_rounds: 最大轮次
            system_message: 系统消息
        """
        # 创建LLM
        if llm_config is None:
            llm_config = {}
        self.llm = create_llm(llm_provider, **llm_config)

        # 创建工具
        if tools is None:
            tools = create_default_tools()

        # 默认系统消息
        if system_message is None:
            system_message = """你是WebDancer，一个专业的深度研究助手。你的任务是通过自主的信息检索和分析来解决复杂的问题。

你的工作流程是：
1. Think: 深入分析当前情况，制定精准的研究策略
2. Report: 系统化地整合和更新研究发现
3. Action: 执行搜索、访问或其他必要操作

在每个回合中，你需要：
- 评估当前信息是否充分
- 确定下一步最有效的行动
- 避免重复搜索相同内容
- 逐步构建完整的答案

记住，深度研究需要耐心和系统性，不要急于给出不完整的答案。"""

        super().__init__(
            llm=self.llm,
            tools=tools,
            system_message=system_message,
            max_rounds=max_rounds
        )

    def research(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        执行研究任务

        Args:
            query: 研究查询
            **kwargs: 其他参数

        Returns:
            研究结果字典
        """
        self.reset()

        # 添加用户查询
        user_message = Message(role="user", content=query)
        self.state.add_message(user_message)

        logger.info(f"开始研究任务: {query}")

        # 创建任务描述
        task_description = f"""
研究任务: {query}

要求:
1. 全面分析问题的各个方面
2. 从多个来源收集信息
3. 验证信息的准确性
4. 提供详细的分析和结论
5. 确保信息是最新的

请按照Think-Report-Action的循环进行深入研究。
"""

        # 添加系统消息
        system_message = Message(
            role="system",
            content=self.create_system_prompt(task_description)
        )
        self.state.add_message(system_message)

        # 开始研究循环
        while self.state.should_continue():
            self.state.increment_round()
            logger.info(f"研究轮次 {self.state.current_round}")

            # 准备上下文
            context = self._prepare_context()

            # 生成Agent响应
            response = self._generate_agent_response(context)

            # 解析响应
            parsed_response = self.parse_agent_response(response)

            # 处理Think部分
            if parsed_response['think']:
                self.state.add_thought(parsed_response['think'])
                logger.info(f"Agent思考: {parsed_response['think'][:100]}...")

            # 处理Report部分
            if parsed_response['report']:
                self.state.update_report(parsed_response['report'])
                logger.info(f"研究报告更新")

            # 处理Action部分
            if parsed_response['action']:
                self.state.add_action({
                    "round": self.state.current_round,
                    "action": parsed_response['action']
                })

                # 执行动作
                tool_result = self.execute_action(parsed_response['action'])
                self.state.add_tool_result(tool_result)

                # 添加工具结果到对话历史
                if tool_result.success:
                    result_message = Message(
                        role="function",
                        content=f"工具 {tool_result.tool_name} 执行结果: {tool_result.result}"
                    )
                else:
                    result_message = Message(
                        role="function",
                        content=f"工具 {tool_result.tool_name} 执行失败: {tool_result.error}"
                    )

                self.state.add_message(result_message)

                logger.info(f"执行动作: {parsed_response['action'][:50]}...")

            # 检查是否应该结束
            if self._should_stop_research(parsed_response):
                logger.info("研究完成，准备结束")
                break

        # 生成最终报告
        final_report = self._generate_final_report()

        return {
            "query": query,
            "final_report": final_report,
            "research_summary": self.get_research_summary(),
            "success": True
        }

    def _prepare_context(self) -> List[Message]:
        """准备上下文"""
        context = []

        # 添加系统消息
        if self.state.messages and self.state.messages[0].role == "system":
            context.append(self.state.messages[0])

        # 添加用户消息
        if self.state.messages and len(self.state.messages) > 1:
            context.extend([msg for msg in self.state.messages[1:] if msg.role in ["user", "assistant", "function"]])

        # 添加当前研究状态摘要
        if self.state.research_report or self.state.tool_results:
            context_summary = Message(
                role="system",
                content=f"当前研究状态:\n{self.state.get_context_summary()}"
            )
            context.append(context_summary)

        return context

    def _generate_agent_response(self, context: List[Message]) -> str:
        """生成Agent响应"""
        try:
            # 使用带工具的生成
            response = self.llm.generate_with_tools(
                messages=context,
                tools=list(self.tools.values()),
                temperature=0.7,
                max_tokens=2000
            )

            return response.get("content", "")

        except Exception as e:
            logger.error(f"生成Agent响应失败: {e}")
            return f"生成响应失败: {str(e)}"

    def _should_stop_research(self, parsed_response: Dict[str, str]) -> bool:
        """判断是否应该停止研究"""
        action = parsed_response['action'].lower()

        # 如果动作包含"完成"、"finish"、"end"等关键词
        stop_keywords = ["完成", "finish", "end", "最终答案", "final answer"]
        for keyword in stop_keywords:
            if keyword in action:
                return True

        # 如果报告看起来很完整且动作是给出答案
        if len(self.state.research_report) > 500 and ("答案" in action or "answer" in action):
            return True

        # 如果达到最大轮次
        if self.state.current_round >= self.max_rounds:
            return True

        return False

    def _generate_final_report(self) -> str:
        """生成最终报告"""
        # 如果已经有研究报告，则基于其生成最终报告
        if self.state.research_report:
            final_prompt = f"""
基于以下研究报告，请生成一个结构化的最终答案：

研究报告：
{self.state.research_report}

研究过程摘要：
- 总共进行了 {self.state.current_round} 轮研究
- 使用了 {len(self.state.tool_results)} 次工具调用
- 成功率: {sum(1 for r in self.state.tool_results if r.success) / len(self.state.tool_results) * 100:.1f}%

请提供一个完整、准确、结构化的最终答案。
"""

            try:
                final_response = self.llm.chat([
                    Message(role="system", content="你是一个专业的报告撰写助手。"),
                    Message(role="user", content=final_prompt)
                ])
                return final_response
            except Exception as e:
                logger.error(f"生成最终报告失败: {e}")
                return self.state.research_report
        else:
            # 如果没有研究报告，生成一个简单的总结
            return f"""
研究任务完成。
- 研究轮次: {self.state.current_round}
- 工具调用次数: {len(self.state.tool_results)}
- 成功的调用: {sum(1 for r in self.state.tool_results if r.success)}

由于研究过程中遇到问题，无法提供完整的答案。请重新尝试或检查相关配置。
"""

    def get_research_details(self) -> Dict[str, Any]:
        """获取详细的研究信息"""
        return {
            "messages": [msg.__dict__ for msg in self.state.messages],
            "tool_results": [result.__dict__ for result in self.state.tool_results],
            "thought_history": self.state.thought_history,
            "action_history": self.state.action_history,
            "research_report": self.state.research_report,
            "summary": self.get_research_summary()
        }

# 示例使用
if __name__ == "__main__":
    # 创建WebDancer Agent
    agent = WebDancerAgent(
        llm_provider="mock",
        max_rounds=5
    )

    # 执行研究任务
    query = "什么是量子计算？它有哪些应用前景？"
    result = agent.research(query)

    print("=== 最终报告 ===")
    print(result["final_report"])

    print("\n=== 研究摘要 ===")
    print(json.dumps(result["research_summary"], indent=2, ensure_ascii=False))