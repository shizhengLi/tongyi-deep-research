"""
通义深度研究复现 - 基础Agent框架
基于WebDancer和WebResearcher的架构设计
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Iterator, Union
from dataclasses import dataclass
from enum import Enum
import json
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolType(Enum):
    """工具类型枚举"""
    SEARCH = "search"
    VISIT = "visit"
    CODE_INTERPRETER = "code_interpreter"
    IMAGE_SEARCH = "image_search"
    CALCULATOR = "calculator"

@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Message:
    """消息格式"""
    role: str  # "user", "assistant", "system", "function"
    content: Union[str, List[Dict[str, Any]]]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class BaseTool(ABC):
    """基础工具抽象类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tool_type = None

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        pass

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """获取工具schema"""
        pass

class BaseLLM(ABC):
    """基础大语言模型抽象类"""

    @abstractmethod
    def chat(self, messages: List[Message], **kwargs) -> str:
        """聊天接口"""
        pass

    @abstractmethod
    def generate_with_tools(self, messages: List[Message], tools: List[BaseTool], **kwargs) -> Dict[str, Any]:
        """带工具生成的接口"""
        pass

class ResearchState:
    """研究状态管理"""

    def __init__(self):
        self.messages: List[Message] = []
        self.tool_results: List[ToolResult] = []
        self.research_report: str = ""
        self.thought_history: List[str] = []
        self.action_history: List[Dict[str, Any]] = []
        self.current_round: int = 0
        self.max_rounds: int = 20

    def add_message(self, message: Message):
        """添加消息"""
        self.messages.append(message)

    def add_tool_result(self, result: ToolResult):
        """添加工具结果"""
        self.tool_results.append(result)

    def update_report(self, report: str):
        """更新研究报告"""
        self.research_report = report

    def add_thought(self, thought: str):
        """添加思考过程"""
        self.thought_history.append(thought)

    def add_action(self, action: Dict[str, Any]):
        """添加动作记录"""
        self.action_history.append(action)

    def increment_round(self):
        """增加轮次"""
        self.current_round += 1

    def should_continue(self) -> bool:
        """判断是否应该继续"""
        return self.current_round < self.max_rounds

    def get_context_summary(self, max_tokens: int = 4000) -> str:
        """获取上下文摘要"""
        # 这里实现上下文压缩逻辑
        context_parts = []

        # 添加当前研究报告
        if self.research_report:
            context_parts.append(f"当前研究报告:\n{self.research_report}")

        # 添加最近的工具结果
        recent_results = self.tool_results[-5:] if len(self.tool_results) > 5 else self.tool_results
        if recent_results:
            context_parts.append("\n最近的工具执行结果:")
            for result in recent_results:
                context_parts.append(f"- {result.tool_name}: {str(result.result)[:200]}...")

        # 添加思考历史
        if self.thought_history:
            context_parts.append(f"\n最近的思考过程:\n{self.thought_history[-1]}")

        summary = "\n".join(context_parts)

        # 简单的长度控制
        if len(summary) > max_tokens:
            summary = summary[:max_tokens] + "..."

        return summary

class BaseDeepResearchAgent(ABC):
    """基础深度研究Agent"""

    def __init__(self,
                 llm: BaseLLM,
                 tools: List[BaseTool],
                 system_message: str = "你是一个专业的深度研究助手。",
                 max_rounds: int = 20):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.system_message = system_message
        self.max_rounds = max_rounds
        self.state = ResearchState()
        self.state.max_rounds = max_rounds

    def reset(self):
        """重置Agent状态"""
        self.state = ResearchState()
        self.state.max_rounds = self.max_rounds

    def format_tools_for_llm(self) -> List[Dict[str, Any]]:
        """格式化工具信息供LLM使用"""
        tools_info = []
        for tool in self.tools.values():
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.get_schema()
            })
        return tools_info

    def create_system_prompt(self, task_description: str = "") -> str:
        """创建系统提示"""
        tools_desc = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools.values()])

        prompt = f"""{self.system_message}

可用工具:
{tools_desc}

你的任务是通过思考(Think)、报告(Report)和行动(Action)的循环来完成深度研究任务。

在每个回合中，你应该：
1. Think: 分析当前情况，制定下一步计划
2. Report: 更新你的研究报告，整合新的发现
3. Action: 选择合适的工具或给出最终答案

输出格式：
```
Think: [你的思考过程]
Report: [更新的研究报告]
Action: [工具调用或最终答案]
```

{task_description}
"""
        return prompt

    def parse_agent_response(self, response: str) -> Dict[str, str]:
        """解析Agent响应"""
        result = {"think": "", "report": "", "action": ""}

        lines = response.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('Think:'):
                current_section = 'think'
                result['think'] = line[6:].strip()
            elif line.startswith('Report:'):
                current_section = 'report'
                result['report'] = line[8:].strip()
            elif line.startswith('Action:'):
                current_section = 'action'
                result['action'] = line[7:].strip()
            elif current_section and line:
                if current_section == 'think':
                    result['think'] += ' ' + line
                elif current_section == 'report':
                    result['report'] += ' ' + line
                elif current_section == 'action':
                    result['action'] += ' ' + line

        return result

    def execute_action(self, action: str) -> ToolResult:
        """执行动作"""
        try:
            # 简单的动作解析
            if action.startswith('使用工具 ') or action.startswith('use_tool '):
                tool_name = action.split(' ', 1)[1]
                if tool_name in self.tools:
                    # 这里需要更复杂的参数解析
                    result = self.tools[tool_name].execute()
                    return result
                else:
                    return ToolResult(
                        tool_name="unknown",
                        arguments={},
                        result=f"工具 {tool_name} 不存在",
                        success=False,
                        error=f"工具 {tool_name} 不存在"
                    )
            else:
                # 返回最终答案
                return ToolResult(
                    tool_name="final_answer",
                    arguments={},
                    result=action,
                    success=True
                )
        except Exception as e:
            logger.error(f"执行动作时出错: {e}")
            return ToolResult(
                tool_name="error",
                arguments={},
                result=str(e),
                success=False,
                error=str(e)
            )

    @abstractmethod
    def research(self, query: str, **kwargs) -> Dict[str, Any]:
        """执行研究任务"""
        pass

    def get_research_summary(self) -> Dict[str, Any]:
        """获取研究摘要"""
        return {
            "query": self.state.messages[0].content if self.state.messages else "",
            "total_rounds": self.state.current_round,
            "total_tools_used": len(self.state.tool_results),
            "research_report": self.state.research_report,
            "thought_history": self.state.thought_history,
            "action_history": self.state.action_history,
            "success_rate": sum(1 for r in self.state.tool_results if r.success) / len(self.state.tool_results) if self.state.tool_results else 0
        }