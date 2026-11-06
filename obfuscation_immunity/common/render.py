"""One pair as an HTML page for wandb: a timeline.

A title and a quiet line of counts, then every event in the order it
happened, each a dot on one rail: the system prompt (folded), the puzzle,
the honest answer (reasoning folded), the request for a sneaky one, the
sneaky answer, then what comes after (the grade, the judge's verdict, the
reward). The dot's colour says what the event is: grey for the
environment, blue for the policy, violet for the judge, and green, amber
or red for a verdict.
"""

import html
import re
from collections.abc import Sequence

TURN = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL)
PIECE = re.compile(
    r"(<think>.*?(?:</think>|$)"
    r"|<tool_call>.*?(?:</tool_call>|$)"
    r"|<tool_response>.*?(?:</tool_response>|$))",
    re.DOTALL,
)
COMMAND = re.compile(r"<parameter=command>\n?(.*?)\n?</parameter>", re.DOTALL)
STYLE = """
:root { --env: #9a9aa0; --agent: #2f5bd3; --method: #7b3fbf;
  --good: #1f7a3a; --warn: #b07800; --bad: #b3261e; --ink: #1d1d1f;
  --muted: #8a8a8e; --line: #e4e4e8; }
body { margin: 0; background: #fff; color: var(--ink);
  font: 14px/1.6 -apple-system, system-ui, "Segoe UI", sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 22px 16px 48px; }
h1 { text-align: center; font-size: 19px; font-weight: 650; margin: 0; }
.subtitle { text-align: center; color: var(--muted); font-style: italic;
  margin: 2px 0 22px; }
.timeline { position: relative; padding-left: 30px; }
.timeline::before { content: ""; position: absolute; left: 10px; top: 6px;
  bottom: 6px; width: 2px; background: var(--line); }
.event { position: relative; margin: 0 0 16px; --dot: var(--env); }
.event::before { content: ""; position: absolute; left: -26px; top: 5px;
  width: 10px; height: 10px; border-radius: 50%; background: var(--dot);
  box-shadow: 0 0 0 3px #fff; }
.event .who { font-size: 12px; font-weight: 600; color: var(--dot);
  margin-bottom: 2px; }
.event.assistant { --dot: var(--agent); }
.event.method { --dot: var(--method); }
.event.good { --dot: var(--good); }
.event.warn { --dot: var(--warn); }
.event.bad { --dot: var(--bad); }
.event.good .prose, .event.warn .prose, .event.bad .prose {
  font-weight: 600; color: var(--dot); }
.event.task .prose { background: #f6f6f8; border-radius: 8px;
  padding: 8px 12px; }
.prose { white-space: pre-wrap; }
summary { cursor: pointer; color: var(--muted); font-style: italic; }
details .prose { color: #5a5a60; border-left: 2px solid var(--line);
  padding-left: 10px; margin: 4px 0; }
pre { font: 12.5px/1.5 ui-monospace, Menlo, monospace; white-space: pre-wrap;
  margin: 4px 0; padding: 8px 12px; border-radius: 8px; }
pre.command { background: #eef2fc; color: #1e3a8a; }
pre.output { background: #fafafa; color: #555;
  border: 1px solid var(--line); }
"""
WHO = {
    "system": "System",
    "task": "Puzzle",
    "assistant": "Policy",
    "tool": "Tool output",
    "user": "Request",
}


def folded(label: str, inner: str, fold: bool) -> str:
    """`inner` under a summary line that opens it, or as is."""
    if not fold:
        return inner
    return f"<details><summary>{label}</summary>{inner}</details>"


def prose(text: str) -> str:
    """Plain text, escaped, line breaks kept; empty text stays empty."""
    text = text.strip()
    return f'<div class="prose">{html.escape(text)}</div>' if text else ""


def piece(text: str, fold: bool) -> str:
    """One stretch of text: reasoning (folded if `fold`), a command or its
    output in monospace, anything else as prose.
    """
    if text.startswith("<think>"):
        body = text.removeprefix("<think>").removesuffix("</think>")
        return folded("thinking", prose(body), fold) if body.strip() else ""
    if text.startswith("<tool_call>"):
        found = COMMAND.search(text)
        command = (found[1] if found else text).strip()
        return f'<pre class="command">$ {html.escape(command)}</pre>'
    if text.startswith("<tool_response>"):
        body = text.removeprefix("<tool_response>")
        body = body.removesuffix("</tool_response>").strip()
        return f'<pre class="output">{html.escape(body)}</pre>'
    return prose(text)


def pieces(text: str, fold: bool) -> str:
    """Every stretch of `text`, rendered."""
    return "".join(piece(p, fold) for p in PIECE.split(text) if p)


def kind(role: str, content: str, first_user: bool) -> str:
    """What a turn is: the first user turn is the task, a user turn that
    holds tool responses is tool output.
    """
    if role == "user" and content.lstrip().startswith("<tool_response>"):
        return "tool"
    if role == "user" and first_user:
        return "task"
    return role


def event(kind: str, who: str, inner: str) -> str:
    """One dot on the rail: `kind` sets its colour, `who` names it."""
    title = f'<div class="who">{html.escape(who)}</div>'
    return f'<div class="event {kind}">{title}{inner}</div>'


def chat(text: str) -> list[str]:
    """A policy's run as events, the system prompt and reasoning folded."""
    out, seen_user = [], False
    for role, content in TURN.findall(text):
        name = kind(role, content, not seen_user)
        seen_user |= role == "user"
        inner = pieces(content, fold=True)
        if name == "system":
            inner = folded("the rules the policy was given", inner, True)
        out.append(event(name, WHO.get(name, name), inner))
    return out


def page(
    transcript: str,
    title: str,
    subtitle: str,
    after: Sequence[tuple[str, str, str, str]],
) -> str:
    """The episode's timeline: title, subtitle, the agent's run, then the
    `after` events as (kind, who, text, fold): `kind` is method, good,
    warn, bad or empty; the text's reasoning is shown open, and the whole
    text is folded under `fold` when that is not empty.
    """
    events = chat(transcript)
    for k, who, text, fold in after:
        inner = folded(fold, pieces(text, False), bool(fold))
        events.append(event(k, who, inner))
    head = (
        f"<h1>{html.escape(title)}</h1>"
        f'<p class="subtitle">{html.escape(subtitle)}</p>'
    )
    style = f'<meta charset="utf-8"><style>{STYLE}</style>'
    timeline = f'<div class="timeline">{"".join(events)}</div>'
    return f"{style}<main>{head}{timeline}</main>"
