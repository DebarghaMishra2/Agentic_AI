import json
import os
from typing import Any, Callable, Dict

from groq import Groq


class GroqProcessAgent:
    """Tool-calling agent for the P2P Process Intelligence project."""

    def __init__(
        self,
        get_kpis: Callable,
        get_bottlenecks: Callable,
        get_exceptions: Callable,
        get_pr_risk: Callable,
        search_process_knowledge: Callable,
        model: str = "openai/gpt-oss-120b",
    ):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not found."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

        self.tool_registry = {
            "get_kpis": get_kpis,
            "get_bottlenecks": get_bottlenecks,
            "get_exceptions": get_exceptions,
            "get_pr_risk": get_pr_risk,
            "search_process_knowledge": search_process_knowledge,
        }

        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "get_kpis",
                    "description": (
                        "Return overall Purchase-to-Pay KPIs, including "
                        "PR volume, cycle time, rejection rate, and "
                        "three-way-match metrics."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_bottlenecks",
                    "description": (
                        "Return the slowest P2P activity transitions "
                        "with occurrence and duration metrics."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "top_n": {
                                "type": "integer",
                                "description": (
                                    "Number of bottlenecks to return."
                                ),
                                "default": 5,
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_exceptions",
                    "description": (
                        "Return important P2P process exceptions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_pr_risk",
                    "description": (
                        "Return predictive slow-PR risk information "
                        "for a specific Purchase Requisition."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pr_id": {
                                "type": "string",
                                "description": (
                                    "Purchase Requisition ID, "
                                    "e.g. PR1."
                                ),
                            }
                        },
                        "required": ["pr_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_process_knowledge",
                    "description": (
                        "Search the process-intelligence RAG "
                        "knowledge base for relevant evidence."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": (
                                    "Analytical question to search."
                                ),
                            }
                        },
                        "required": ["question"],
                    },
                },
            },
        ]

    def _execute_tool(
        self,
        tool_call
    ) -> Dict[str, Any]:

        name = tool_call.function.name

        try:
            arguments = json.loads(
                tool_call.function.arguments or "{}"
            )
        except json.JSONDecodeError as exc:
            return {
                "error": f"Invalid tool arguments: {exc}"
            }

        function = self.tool_registry.get(name)

        if function is None:
            return {
                "error": f"Unknown tool: {name}"
            }

        try:
            return {
                "result": function(**arguments)
            }

        except Exception as exc:
            return {
                "error": str(exc)
            }

    def run(
        self,
        question: str,
        max_iterations: int = 5
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an Agentic Process Intelligence Analyst. "
                    "Use the available tools to analyze the "
                    "Purchase-to-Pay process. "
                    "Use get_kpis for overall performance, "
                    "get_bottlenecks for delays, "
                    "get_exceptions for exceptions, "
                    "get_pr_risk for a specific PR, "
                    "and search_process_knowledge for broader evidence. "
                    "Never invent metrics. "
                    "Treat tool output as the source of truth. "
                    "Distinguish facts from interpretation and do not "
                    "claim causality unless the data supports it. "
                    "Give concise business-oriented answers."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        for _ in range(max_iterations):

            response = (
                self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_definitions,
                    tool_choice="auto",
                    temperature=0,
                )
            )

            message = response.choices[0].message

            if not message.tool_calls:
                return (
                    message.content
                    or "The model returned no text."
                )

            messages.append(message)

            for tool_call in message.tool_calls:

                result = self._execute_tool(
                    tool_call
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": json.dumps(
                            result,
                            default=str
                        ),
                    }
                )

        return (
            "The agent reached its maximum "
            "reasoning iterations."
        )