"""
Interactive Chat CLI for the AI Order Agent.

Features:
- Session Memory (MemorySaver) -- Agent nho toan bo cuoc tro chuyen
- Human-in-the-Loop -- Yeu cau xac nhan truoc khi luu don hang
- Rich Terminal UI -- Giao dien dep, hien thi tool calls real-time
- Fuzzy Search -- Tim kiem san pham thong minh (chong loi chinh ta)

Usage:
    python src/chat.py
    python src/chat.py --provider google
    python src/chat.py --provider ollama --model llama3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Fix import paths
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

from src.agent.graph import build_tools, build_system_prompt
from src.core.llm import build_chat_model
from src.utils.data_store import OrderDataStore

console = Console()

BANNER = r"""
[bold blue]
  +------------------------------------------------------+
  |   AI Order Agent -- Cua hang Dien tu                 |
  |                                                      |
  |   Features:                                          |
  |   - Session Memory (nho ca cuoc tro chuyen)          |
  |   - Human-in-the-Loop (xac nhan truoc khi luu)       |
  |   - Fuzzy Search (chong loi chinh ta)                |
  |   - Rich Terminal UI                                 |
  |                                                      |
  |   Go 'quit' de thoat | 'reset' de xoa tri nho       |
  +------------------------------------------------------+
[/bold blue]
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive chat with AI Order Agent")
    parser.add_argument("--provider", default="google", help="LLM provider (google, openai, ollama)")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature (0.3 for natural chat)")
    return parser.parse_args()


def build_chat_agent(provider: str, model_name: str | None, temperature: float):
    """Build a LangGraph agent with Memory + Human-in-the-Loop interrupt."""
    memory = MemorySaver()
    store = OrderDataStore(ROOT_DIR / "data", ROOT_DIR / "artifacts" / "orders")
    tools = build_tools(store)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=temperature)

    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=build_system_prompt(),
        checkpointer=memory,
        # Human-in-the-Loop: Interrupt BEFORE tools execute
        interrupt_before=["tools"],
    )
    return agent


def display_tool_call(tool_name: str, tool_args: dict):
    """Pretty-print a tool call with arguments."""
    table = Table(title=f"Tool: [bold]{tool_name}[/bold]", show_header=True, border_style="dim")
    table.add_column("Argument", style="cyan", width=20)
    table.add_column("Value", style="white")
    for key, value in tool_args.items():
        val_str = str(value)
        if len(val_str) > 80:
            val_str = val_str[:77] + "..."
        table.add_row(key, val_str)
    console.print(table)


def display_tool_result(tool_name: str, output: str):
    """Show tool result in a compact panel."""
    display_output = output if len(output) < 500 else output[:497] + "..."
    console.print(Panel(display_output, title=f"Ket qua: [bold]{tool_name}[/bold]", border_style="green", expand=False))


def has_save_order_call(messages) -> bool:
    """Check if the last AI message contains a save_order tool call."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "save_order":
                    return True
        break
    return False


def process_stream_event(event):
    """Process a single stream event and display it."""
    last_msg = event["messages"][-1]

    if isinstance(last_msg, AIMessage) and last_msg.content:
        if not getattr(last_msg, "tool_calls", None):
            console.print(Panel(
                Markdown(str(last_msg.content)),
                title="[bold cyan]AI Assistant[/bold cyan]",
                border_style="cyan",
            ))

    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        for tc in last_msg.tool_calls:
            display_tool_call(tc["name"], tc.get("args", {}))

    if isinstance(last_msg, ToolMessage):
        display_tool_result(
            str(getattr(last_msg, "name", "tool")),
            str(last_msg.content),
        )


def resume_agent(agent, config):
    """Resume a paused agent and display results."""
    for event in agent.stream(None, config=config, stream_mode="values"):
        process_stream_event(event)


def main():
    args = parse_args()
    console.print(BANNER)

    agent = build_chat_agent(args.provider, args.model, args.temperature)
    config = {"configurable": {"thread_id": "interactive_session_1"}}
    turn_count = 0

    while True:
        try:
            user_input = Prompt.ask("\n[bold yellow]Khach hang[/bold yellow]")
            stripped = user_input.strip().lower()

            if stripped in ("quit", "exit", "q"):
                console.print("[dim]Tam biet![/dim]")
                break

            if stripped == "reset":
                agent = build_chat_agent(args.provider, args.model, args.temperature)
                config = {"configurable": {"thread_id": f"interactive_session_{turn_count + 1}"}}
                console.print("[bold green]Da xoa tri nho. Bat dau phien moi![/bold green]")
                continue

            if not stripped:
                continue

            turn_count += 1
            console.print("[dim]Dang suy nghi...[/dim]")

            # Stream events from the agent
            for event in agent.stream(
                {"messages": [("user", user_input)]},
                config=config,
                stream_mode="values",
            ):
                process_stream_event(event)

            # Human-in-the-Loop: Check if save_order is pending
            state = agent.get_state(config)
            if state.next:
                pending_messages = state.values.get("messages", [])
                if has_save_order_call(pending_messages):
                    console.print()
                    console.print(Panel(
                        "[bold yellow]Agent muon luu don hang.\n"
                        "Ban co dong y chot don khong?[/bold yellow]",
                        border_style="yellow",
                    ))
                    confirmed = Confirm.ask("[bold]Xac nhan chot don?[/bold]", default=True)

                    if confirmed:
                        console.print("[dim]Da xac nhan. Dang luu don hang...[/dim]")
                        resume_agent(agent, config)
                    else:
                        console.print("[bold red]Da huy. Don hang KHONG duoc luu.[/bold red]")
                        for event in agent.stream(
                            {"messages": [("user", "Huy don hang. Khong luu.")]},
                            config=config,
                            stream_mode="values",
                        ):
                            process_stream_event(event)
                else:
                    resume_agent(agent, config)

        except KeyboardInterrupt:
            console.print("\n[dim]Tam biet![/dim]")
            break
        except Exception as e:
            console.print(f"[bold red]Loi: {e}[/bold red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    main()
