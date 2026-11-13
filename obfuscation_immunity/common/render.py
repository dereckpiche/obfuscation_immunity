"""
One pair as an HTML page for wandb, exact to the token.

The page follows each model's sequence in turn (the policy's trajectory,
then the judge's call), one block per stretch of tokens, in the order
things happened, on a timeline. Each block says whether the stretch is
context a model read or output a model generated, and which model; an
output's block opens with its tokens against its budget and its
thinking/final split. Then the tokens, decoded with their special tokens
shown, and a switch that turns them into the list of tokens, each with its
id and the sampler's log-probability, shaded by how unlikely it was.
Under a block, small events on the timeline say what happened: a grade, a
broken format, the order the judge saw, the judge's J against the truth.
The last event adds up the reward. The page is self-contained: no network,
no fonts.
"""

import html
import json
import math
import re
from typing import Any

from transformers import PreTrainedTokenizerBase

from obfuscation_immunity.common.chat_format import ChatFormat
from obfuscation_immunity.common.puzzles import answer_alone
from obfuscation_immunity.common.transcript import (
    ENVIRONMENT,
    HONEST_ANSWER_KIND,
    JUDGE,
    JUDGE_OUTPUT_KIND,
    JUDGE_QUESTION_KIND,
    POLICY,
    SNEAKY_ANSWER_KIND,
    PairRecord,
    TranscriptSegment,
)

OUTCOME_NAMES = [
    "both right",
    "only honest right",
    "only sneaky right",
    "neither",
]
"""OUTCOMES in words, in (s, s̄) order."""

SOURCE_NAMES = {ENVIRONMENT: "environment", POLICY: "policy", JUDGE: "judge"}
"""How each segment source is named on the page."""

MODEL_EMOJI = {POLICY: "✍︎", JUDGE: "⚖︎"}
"""Each model's emoji, in text presentation, before its name and outputs."""

PAGE_STYLE = """
:root { --ink: #1d1d1f; --soft: #48484a; --muted: #8e8e93; --faint: #d1d1d6;
  --line: #e8e8ed; --paper: #fff; --chip: #e6e6eb; --context: #f4f4f7;
  --env: #8e8e93; --policy: #2f5bd3; --judge: #7b3fbf;
  --error: #c4600a; --unlikely-rgb: 214, 120, 0;
  --mono: ui-monospace, SFMono-Regular, Menlo, monospace; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink);
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui,
  sans-serif; -webkit-font-smoothing: antialiased; }
.page { max-width: 1120px; margin: 0 auto; padding: 28px 24px 64px; }
.top { display: flex; justify-content: center; flex-wrap: wrap;
  gap: 8px 40px; margin-bottom: 34px; }
.meta { display: flex; flex-direction: column; align-items: center; }
.meta b { font-size: 15px; font-weight: 600; }
.meta small { color: var(--muted); font-size: 11.5px; }
.rows { position: relative; }
.rows::before { content: ""; position: absolute; left: 5px; top: 10px;
  bottom: 10px; width: 2px; background: var(--line); }
.row, .event { display: grid; grid-template-columns: 170px minmax(0, 1fr);
  column-gap: 28px; }
.row { padding-bottom: 34px; }
.row.followed { padding-bottom: 12px; }
.events { padding-bottom: 34px; }
.environment { --dot: var(--env); }
.policy { --dot: var(--policy); --tint: #f6f9fe; --rule: #e2e9f8; }
.judge { --dot: var(--judge); --tint: #faf7fd; --rule: #ece4f6; }
.total { --dot: var(--ink); }
.node { position: relative; padding-left: 26px; }
.node::before { content: ""; position: absolute; left: 0; top: 5px;
  width: 12px; height: 12px; border-radius: 50%; background: var(--dot);
  box-shadow: 0 0 0 4px var(--paper); }
.emoji { color: var(--ink); font-variant-emoji: text; font-weight: 400;
  font-size: 1.15em; margin-right: 6px;
  font-family: "Apple Symbols", "Segoe UI Symbol", "DejaVu Sans", sans-serif; }
.node h2 { font-size: 14.5px; font-weight: 600; margin: 0; color: var(--dot); }
.node .facts { margin-top: 3px; color: var(--muted); font-size: 12px; }
.event { padding: 3px 0; font-size: 13px; }
.mark { position: relative; padding-left: 26px; color: var(--soft);
  font-weight: 600; font-size: 12.5px; }
.mark::before { content: ""; position: absolute; left: 2px; top: 6px;
  width: 8px; height: 8px; border-radius: 50%; background: var(--paper);
  box-shadow: inset 0 0 0 2px var(--dot), 0 0 0 3px var(--paper); }
.said { color: var(--soft); display: flex; align-items: center;
  justify-content: center; flex-wrap: wrap; gap: 4px 10px; }
.said small { color: var(--muted); font-style: italic; }
.badge { display: inline-block; padding: 1px 9px; border-radius: 999px;
  font-size: 12px; font-weight: 600; font-style: normal; }
.badge.strong { color: var(--ink); background: var(--chip); }
.badge.error { color: var(--error); background: #fcf0e4; }
.badge.plain { color: var(--soft); background: var(--chip);
  font-weight: 500; }
.captioned { display: inline-flex; flex-direction: column;
  align-items: center; gap: 1px; }
.captioned > small { font-style: normal; font-size: 10px;
  text-transform: uppercase; letter-spacing: 0.05em; }
.captioned code { font-size: 13px; padding: 1px 7px; }
.relation { font-size: 17px; color: var(--soft); margin: 0 -2px;
  align-self: flex-start; line-height: 1.3; }
.chip { display: inline-flex; align-items: baseline; gap: 6px; }
.chip > small { font-style: normal; font-size: 10.5px;
  text-transform: uppercase; letter-spacing: 0.05em; }
.grid { display: grid; grid-template-columns: 76px 88px 88px 76px;
  gap: 4px; align-items: center; padding: 2px 0; }
.axis { font-size: 11.5px; color: var(--muted); text-align: center; }
.axis.side { text-align: right; padding-right: 6px; }
.cell { height: 46px; border-radius: 8px; display: flex;
  flex-direction: column; align-items: center; justify-content: center;
  background: rgba(29, 29, 31, calc(0.03 + var(--p) * 0.24));
  color: var(--ink); font-size: 15px; font-weight: 600;
  font-variant-numeric: tabular-nums; line-height: 1.2; }
.cell i { font-style: normal; font-size: 10.5px; font-weight: 500;
  color: var(--soft); min-height: 1.2em; }
.cell.truth { box-shadow: inset 0 0 0 2px var(--ink); }
.said code { font: 12px var(--mono); background: var(--context);
  border-radius: 4px; padding: 0 5px; }
.card { border-radius: 12px; overflow: hidden; }
.card.context { background: var(--context); border: 1px solid var(--line); }
.card.output { background: var(--tint); border: 1px solid var(--rule); }
.card > .head { display: flex; align-items: center;
  justify-content: space-between; gap: 12px; padding: 6px 10px 6px 14px;
  border-bottom: 1px solid var(--line); }
.card.output > .head { border-color: var(--rule); }
.budget { flex: 1; min-width: 0; display: flex; align-items: center;
  gap: 14px; flex-wrap: wrap;
  font-size: 12px; color: var(--muted);
  font-variant-numeric: tabular-nums; }
.meter { flex: 0 0 140px; height: 6px; border-radius: 3px;
  background: var(--line); overflow: hidden; }
.meter span { display: block; height: 100%; background: var(--muted); }
.meter.full span { background: var(--error); }
.warn { color: var(--error); }
.sequence { display: flex; align-items: center; gap: 14px;
  margin: 6px 0 22px 26px; color: var(--soft); font-size: 13px;
  font-weight: 600; }
.sequence::before, .sequence::after { content: ""; flex: 1; height: 1px;
  background: var(--line); }
.sequence small { color: var(--muted); font-weight: 400;
  font-style: italic; }
.switch { display: inline-flex; border: 1px solid var(--line);
  border-radius: 7px; overflow: hidden; }
.switch button { font: inherit; font-size: 12px; padding: 2px 11px; border: 0;
  background: none; color: var(--muted); cursor: pointer; }
.switch button.on { background: var(--chip); color: var(--ink); }
.text { padding: 10px 14px 12px; }
.exact { margin: 0; white-space: pre-wrap; word-break: break-word;
  font: 12.5px/1.65 var(--mono); }
.context .exact { color: var(--soft); }
.clip, .tail { position: relative; overflow: hidden; font-size: 12.5px; }
.clip.long:not(.open) { max-height: 16.5em; }
.tail[hidden] { display: none; }
.clip.long { cursor: pointer; }
.tail { cursor: pointer; max-height: 16.5em; display: flex; flex-direction: column;
  justify-content: flex-end; }
.clip.long:not(.open)::after, .tail::before { content: ""; position: absolute;
  left: 0; right: 0; height: 3.5em; z-index: 1; }
.clip.long:not(.open)::after { bottom: 0;
  background: linear-gradient(rgba(255, 255, 255, 0), var(--fade)); }
.tail::before { top: 0;
  background: linear-gradient(var(--fade), rgba(255, 255, 255, 0)); }
.card.context { --fade: var(--context); } .card.output { --fade: var(--tint); }
.more { display: none; margin: 8px auto; padding: 3px 14px; border: 0;
  border-radius: 999px; background: var(--chip); font: inherit;
  font-size: 12px; color: var(--soft); cursor: pointer; }
.more:hover { color: var(--ink); }
.clip.long + .more { display: block; }
.pill { display: inline-block; padding: 0 5px; margin: 0 1px;
  border-radius: 5px; background: var(--chip); color: var(--muted);
  font-size: 11px; line-height: 1.55; vertical-align: 1px; }
.tokens { padding: 10px 14px 12px; font: 12px/2 var(--mono); }
.token { position: relative; display: inline-block; padding: 0 3px;
  margin: 0 1px 1px 0; border-radius: 4px; white-space: pre;
  background: rgba(var(--unlikely-rgb), var(--unlikely, 0));
  box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.06); }
.token:hover { box-shadow: inset 0 0 0 1px var(--ink); }
.token:hover::after { content: attr(data-tip); position: absolute; left: 0;
  bottom: calc(100% + 6px); z-index: 5; white-space: nowrap;
  background: var(--ink); color: var(--paper); padding: 4px 8px;
  border-radius: 6px; font: 11px/1.4 -apple-system, system-ui, sans-serif;
  pointer-events: none; }
.token i { color: var(--faint); font-style: normal; }
.token.special { background: var(--chip); color: var(--muted); }
.equation { display: flex; align-items: center; justify-content: center;
  gap: 18px; flex-wrap: wrap; }
.term { text-align: center; }
.term b { display: block; font-size: 22px; font-weight: 600;
  font-variant-numeric: tabular-nums; }
.term small { color: var(--muted); font-size: 12px; }
.operator { font-size: 20px; color: var(--muted); }
@media (max-width: 760px) {
  .row, .event { grid-template-columns: minmax(0, 1fr); row-gap: 8px; }
  .rows::before { display: none; } }
"""
"""The page's stylesheet."""

PAGE_SCRIPT = """
const CLIP_LINES = 24;
const segments = JSON.parse(document.getElementById("segments").textContent);
function escapeText(text) {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function tokenChip(segment, index) {
  const text = segment.pieces[index];
  const shown = escapeText(text).replace(/ /g, "<i>·</i>")
    .replace(/\\t/g, "<i>→</i>").replace(/\\n/g, "<i>↵</i>");
  let tip = "position " + index + " · id " + segment.ids[index];
  let style = "";
  if (segment.logprobs.length > index) {
    const logprob = segment.logprobs[index];
    const probability = Math.exp(logprob);
    tip += " · log p " + logprob.toFixed(3) + " · p " + probability.toFixed(3);
    style = ' style="--unlikely:' + (0.62 * (1 - probability)).toFixed(3)
      + '"';
  }
  const kind = segment.special.includes(index) ? "token special" : "token";
  const chip = '<span class="' + kind + '" data-tip="' + escapeText(tip)
    + '"' + style + ">" + (shown || "<i>∅</i>") + "</span>";
  return text.includes("\\n") ? chip + "<br>" : chip;
}
function showView(button) {
  const card = button.closest(".card");
  const segment = segments[Number(card.dataset.index)];
  const tokens = card.querySelector(".tokens");
  const showTokens = button.dataset.view === "tokens";
  if (showTokens && !tokens.dataset.built) {
    tokens.innerHTML = segment.pieces.map(
      (_, position) => tokenChip(segment, position)).join("");
    tokens.dataset.built = "1";
  }
  card.querySelector(".text").hidden = showTokens;
  tokens.hidden = !showTokens;
  for (const other of card.querySelectorAll(".switch button")) {
    other.classList.toggle("on", other === button);
  }
}
for (const button of document.querySelectorAll(".switch button")) {
  button.addEventListener("click", () => showView(button));
}
for (const clip of document.querySelectorAll(".clip")) {
  const exact = clip.querySelector(".exact");
  const line = parseFloat(getComputedStyle(exact).lineHeight);
  if (clip.scrollHeight > CLIP_LINES * line) {
    clip.classList.add("long");
    const tail = clip.parentElement.querySelector(".tail");
    tail.appendChild(exact.cloneNode(true));
    tail.hidden = false;
  }
}
function toggleClip(text) {
  const clip = text.querySelector(".clip");
  if (!clip.classList.contains("long")) { return; }
  const more = text.querySelector(".more");
  const open = clip.classList.toggle("open");
  text.querySelector(".tail").hidden = open;
  more.textContent = open ? "Show less" : more.dataset.label;
}
for (const text of document.querySelectorAll(".text")) {
  text.addEventListener("click", () => {
    if (window.getSelection().toString()) { return; }
    toggleClip(text);
  });
}
"""
"""The page's script: views, token lists built on demand, clipped text."""


def special_token_pattern(chat: ChatFormat) -> re.Pattern:
    """
    Compiles a pattern matching the model's markers, escaped.

    Args:
        chat: The model's turn markers.

    Returns:
        A pattern over HTML-escaped text, longest markers first.
    """
    specials = chat.marker_texts
    escaped = sorted((html.escape(s) for s in specials if s), key=len)
    return re.compile("|".join(re.escape(s) for s in reversed(escaped)))


def exact_text_html(text: str, specials: re.Pattern) -> str:
    """
    Renders decoded text exactly, its special tokens as pills.

    Args:
        text: Tokens decoded with their special tokens.
        specials: The pattern from `special_token_pattern`.

    Returns:
        The escaped text, special tokens wrapped in a `pill` span.
    """
    return specials.sub(
        lambda match: f'<span class="pill">{match.group(0)}</span>',
        html.escape(text),
    )


def clipped_html(inner_html: str, label: str) -> str:
    """
    Wraps exact text so that a long one shows its start and its end.

    The middle is hidden behind a button, so that a long generation's
    final answer shows without its reasoning.

    Args:
        inner_html: The exact text's HTML.
        label: The button's label, such as "Show all 1,204 tokens".

    Returns:
        The HTML; the page's script fills the end and shows the button
        only if the text is long.
    """
    return (
        f'<div class="clip"><pre class="exact">{inner_html}</pre></div>'
        f'<button class="more" data-label="{html.escape(label)}">'
        f'{html.escape(label)}</button><div class="tail" hidden></div>'
    )


def budget_html(segment: TranscriptSegment, chat: ChatFormat) -> str:
    """
    Renders a segment's token count, an output's against its budget.

    Args:
        segment: The segment.
        chat: The model's turn markers.

    Returns:
        The count's HTML, for the top of the segment's card.
    """
    used = len(segment.token_ids)
    if segment.source == ENVIRONMENT:
        return f'<div class="budget"><span>{used:,} tokens</span></div>'
    parts = []
    if segment.budget:
        full = used >= segment.budget
        share = min(used / segment.budget, 1.0)
        parts.append(
            f'<div class="meter{" full" if full else ""}">'
            f'<span style="width:{100 * share:.1f}%"></span></div>'
            f"<span>{used:,} of {segment.budget:,} tokens</span>"
        )
        if full:
            parts.append('<span class="warn">cut off</span>')
    else:
        parts.append(f"<span>{used:,} tokens</span>")
    cut = chat.think_close_position(segment.token_ids)
    if cut is None:
        parts.append('<span class="warn">never closed its thinking</span>')
    else:
        parts.append(
            f"<span>thinking {cut + 1:,} · final {used - cut - 1:,}</span>"
        )
    return f'<div class="budget">{"".join(parts)}</div>'


def card_html(
    index: int,
    segment: TranscriptSegment,
    tokenizer: PreTrainedTokenizerBase,
    chat: ChatFormat,
    specials: re.Pattern,
) -> str:
    """
    Renders a segment's exact tokens under its count and a view switch.

    Args:
        index: The segment's position in the record.
        segment: The segment.
        tokenizer: The run's tokenizer.
        chat: The model's turn markers.
        specials: The pattern from `special_token_pattern`.

    Returns:
        The card's HTML.
    """
    kind = "context" if segment.source == ENVIRONMENT else "output"
    switch = (
        '<div class="switch"><button data-view="text" class="on">Text'
        '</button><button data-view="tokens">Tokens</button></div>'
    )
    decoded = str(
        tokenizer.decode(segment.token_ids, skip_special_tokens=False)
    )
    label = f"Show all {len(segment.token_ids):,} tokens"
    text = clipped_html(exact_text_html(decoded, specials), label)
    return (
        f'<div class="card {kind}" data-index="{index}"><div class="head">'
        f"{budget_html(segment, chat)}{switch}</div>"
        f'<div class="text">{text}</div><div class="tokens" hidden></div>'
        "</div>"
    )


def emoji_html(source: str) -> str:
    """
    Renders a model's emoji, black and white, before its name.

    Args:
        source: ENVIRONMENT, POLICY or JUDGE.

    Returns:
        The emoji's HTML; empty for the environment.
    """
    if source not in MODEL_EMOJI:
        return ""
    return f'<span class="emoji">{MODEL_EMOJI[source]}</span>'


def node_html(segment: TranscriptSegment) -> str:
    """
    Renders a segment's place on the timeline: who, how long, trained.

    Args:
        segment: The segment.

    Returns:
        The node's HTML.
    """
    facts = [SOURCE_NAMES[segment.source], f"{len(segment.token_ids):,} tokens"]
    facts.append("trained" if segment.trained else "not trained")
    return (
        f'<div class="node"><h2>{emoji_html(segment.source)}'
        f"{html.escape(segment.title)}</h2>"
        f'<div class="facts">{"<br>".join(facts)}</div></div>'
    )


def badge_html(text: str, tone: str) -> str:
    """
    Renders a short verdict as a rounded badge.

    Args:
        text: The verdict, such as "✓︎ right".
        tone: "strong", "plain", or "error" for what could not be read.

    Returns:
        The badge's HTML.
    """
    return f'<span class="badge {tone}">{text}</span>'


def captioned_html(value_html: str, caption: str) -> str:
    """
    Renders a value with a small caption under it.

    Args:
        value_html: The value, as HTML.
        caption: What the value is, such as "expected".

    Returns:
        The captioned value's HTML.
    """
    return (
        f'<span class="captioned">{value_html}'
        f"<small>{html.escape(caption)}</small></span>"
    )


def chip_html(label: str, value_html: str) -> str:
    """
    Renders a value with a small label before it.

    Args:
        label: What the value is, such as "read".
        value_html: The value, as HTML.

    Returns:
        The chip's HTML.
    """
    return (
        f'<span class="chip"><small>{html.escape(label)}</small>'
        f"{value_html}</span>"
    )


def event_html(label: str, said_html: str) -> str:
    """
    Renders one small event on the timeline, under a block.

    Args:
        label: What happened, in a word or two.
        said_html: The event's line, as HTML.

    Returns:
        The event's HTML.
    """
    return (
        f'<div class="event"><div class="mark">{html.escape(label)}</div>'
        f'<div class="said">{said_html}</div></div>'
    )


def broken_format_event_html(segment: TranscriptSegment) -> str:
    """
    Renders the event of a turn out of the format, which ends the game.

    Args:
        segment: The turn.

    Returns:
        The event's HTML.
    """
    problem = html.escape(segment.details["format"])
    winner = "the policy" if segment.source == JUDGE else "the judge"
    return event_html(
        "Format broken",
        badge_html(problem, "error")
        + f"<small>the game ends here: {winner} wins</small>",
    )


def answer_events_html(
    record: PairRecord,
    segment: TranscriptSegment,
    tokenizer: PreTrainedTokenizerBase,
    chat: ChatFormat,
) -> list[str]:
    """
    Renders what happened to one answer: its grade, or its format.

    Args:
        record: The pair's record.
        segment: The honest or the sneaky answer.
        tokenizer: The run's tokenizer.
        chat: The model's turn markers.

    Returns:
        The events' HTML.
    """
    if "format" in segment.details:
        return [broken_format_event_html(segment)]
    honest = segment.kind == HONEST_ANSWER_KIND
    correct = record.honest_correct if honest else record.sneaky_correct
    _, final = chat.split(tokenizer, segment.token_ids)
    graded = answer_alone(final) or ""
    expected = captioned_html(
        f"<code>{html.escape(record.expected_answer)}</code>", "expected"
    )
    relation = "=" if correct else "≠"
    said = (
        captioned_html(f"<code>{html.escape(graded)}</code>", "answer")
        + f'<span class="relation">{relation}</span>'
        + expected
    )
    goal = "be right" if honest else "be wrong, convincingly"
    if correct == honest:
        said += badge_html("✓︎ goal met", "strong")
    else:
        said += badge_html("✕︎ goal missed", "plain")
    said += f"<small>its goal: {goal}</small>"
    return [event_html("Graded", said)]


def outcome_cell_html(
    index: int, probabilities: list[float], true_outcome: int
) -> str:
    """
    Renders one outcome's cell of J's grid, shaded by its probability.

    Args:
        index: The outcome, in OUTCOMES' order.
        probabilities: J over OUTCOMES.
        true_outcome: The outcome V gives.

    Returns:
        The cell's HTML; the truth's is outlined.
    """
    probability = probabilities[index]
    if index == true_outcome:
        return (
            f'<span class="cell truth" style="--p:{probability:.3f}">'
            f"{probability:.2f}<i>truth</i></span>"
        )
    return (
        f'<span class="cell" style="--p:{probability:.3f}">'
        f"{probability:.2f}</span>"
    )


def judge_events_html(
    record: PairRecord, segment: TranscriptSegment
) -> list[str]:
    """
    Renders what the judge's turn came to: J against the truth, or its format.

    J is a grid of the honest answer's label by the sneaky answer's, each
    cell shaded by its probability, the truth's outlined.

    Args:
        record: The pair's record.
        segment: The judge's turn.

    Returns:
        The events' HTML.
    """
    if "format" in segment.details:
        return [broken_format_event_html(segment)]
    probabilities = [math.exp(value) for value in record.outcome_logprobs]
    cells = [
        outcome_cell_html(index, probabilities, record.true_outcome)
        for index in range(len(probabilities))
    ]
    grid = (
        '<div class="grid"><span></span><span class="axis">sneaky ✓︎</span>'
        '<span class="axis">sneaky ✕︎</span><span></span>'
        f'<span class="axis side">honest ✓︎</span>{cells[0]}{cells[1]}'
        f'<span></span><span class="axis side">honest ✕︎</span>{cells[2]}'
        f"{cells[3]}<span></span></div>"
    )
    return [event_html("J", grid)]


def events_html(
    record: PairRecord,
    segment: TranscriptSegment,
    tokenizer: PreTrainedTokenizerBase,
    chat: ChatFormat,
) -> list[str]:
    """
    Renders the key events under a segment's block, by the segment's kind.

    Args:
        record: The pair's record.
        segment: The segment.
        tokenizer: The run's tokenizer.
        chat: The model's turn markers.

    Returns:
        The events' HTML; none for most context.
    """
    if segment.kind in (HONEST_ANSWER_KIND, SNEAKY_ANSWER_KIND):
        return answer_events_html(record, segment, tokenizer, chat)
    if segment.kind == JUDGE_QUESTION_KIND:
        order = "".join(
            chip_html(label, badge_html(html.escape(value), "plain"))
            for label, value in segment.details.items()
        )
        return [event_html("Order shown", order)]
    if segment.kind == JUDGE_OUTPUT_KIND:
        return judge_events_html(record, segment)
    return []


def reward_row_html(record: PairRecord) -> str:
    """
    Renders the last node on the timeline: the reward's parts summed.

    Args:
        record: The pair's record.

    Returns:
        The row's HTML.
    """
    terms = (
        f'<div class="term"><b>{record.verifier_reward:g}</b>'
        "<small>V((1,0) | s, s̄)</small></div>"
        '<div class="operator">+</div>'
        f'<div class="term"><b>{record.judge_miss:.3f}</b>'
        "<small>1 − J(y_true)</small></div>"
        '<div class="operator">+</div>'
        f'<div class="term"><b>{record.format_reward:+g}</b>'
        "<small>turns in format</small></div>"
        '<div class="operator">=</div>'
        f'<div class="term"><b>{record.reward:.3f}</b><small>reward</small>'
        "</div>"
    )
    node = (
        '<div class="node"><h2>Reward</h2><div class="facts">the whole'
        "<br>trajectory's</div></div>"
    )
    return f'<div class="row total">{node}<div class="equation">{terms}</div></div>'


def sequence_header_html(model: str) -> str:
    """
    Renders the divider that opens one model's sequence.

    Args:
        model: POLICY or JUDGE.

    Returns:
        The divider's HTML.
    """
    name = SOURCE_NAMES[model]
    return (
        f'<div class="sequence"><span>{emoji_html(model)}The {name}\'s'
        " sequence</span>"
        "<small>its context, then its output, in order</small></div>"
    )


def segment_data(
    segment: TranscriptSegment, pieces: list[str], specials: set[str]
) -> dict[str, Any]:
    """
    Packs a segment for the page's script.

    Args:
        segment: The segment.
        pieces: Each of its tokens decoded alone.
        specials: The tokenizer's special tokens, as text.

    Returns:
        The tokens' text, ids and log-probabilities, and which are special.
    """
    return {
        "pieces": pieces,
        "ids": segment.token_ids,
        "logprobs": segment.logprobs,
        "special": [i for i, piece in enumerate(pieces) if piece in specials],
    }


def header_html(meta: dict[str, str]) -> str:
    """
    Renders the top of the page: what the pair is, one fact each.

    Args:
        meta: Facts by label, such as {"step": "14"}.

    Returns:
        The header's HTML.
    """
    facts = "".join(
        f'<div class="meta"><b>{html.escape(value)}</b>'
        f"<small>{html.escape(label)}</small></div>"
        for label, value in meta.items()
    )
    return f'<div class="top">{facts}</div>'


def pair_page_html(
    record: PairRecord,
    tokenizer: PreTrainedTokenizerBase,
    chat: ChatFormat,
    meta: dict[str, str] | None = None,
) -> str:
    """
    Renders a pair's page from its record.

    Args:
        record: The pair's record.
        tokenizer: The run's tokenizer.
        chat: The model's turn markers.
        meta: Facts for the top of the page, by label, after the puzzle:
            the method, the model, the step, the run.

    Returns:
        The page's HTML.
    """
    specials = special_token_pattern(chat)
    special_texts = set(chat.marker_texts)
    rows, data, current_model = [], [], None
    for index, segment in enumerate(record.segments):
        if segment.model != current_model:
            rows.append(sequence_header_html(segment.model))
            current_model = segment.model
        pieces = tokenizer.batch_decode([[i] for i in segment.token_ids])
        data.append(
            segment_data(segment, [str(p) for p in pieces], special_texts)
        )
        events = events_html(record, segment, tokenizer, chat)
        followed = " followed" if events else ""
        rows.append(
            f'<div class="row {segment.source}{followed}">'
            f"{node_html(segment)}"
            f"{card_html(index, segment, tokenizer, chat, specials)}</div>"
        )
        if events:
            rows.append(
                f'<div class="events {segment.source}">{"".join(events)}</div>'
            )
    rows.append(reward_row_html(record))
    meta = {"puzzle": record.puzzle, **(meta or {})}
    title = " · ".join(["Pair", *meta.values()])
    segments_json = json.dumps(data).replace("</", "<\\/")
    return (
        f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}'
        f"</title><style>{PAGE_STYLE}</style>"
        f'<div class="page">{header_html(meta)}'
        f'<div class="rows">{"".join(rows)}</div></div>'
        f'<script type="application/json" id="segments">{segments_json}'
        f"</script><script>{PAGE_SCRIPT}</script>"
    )
