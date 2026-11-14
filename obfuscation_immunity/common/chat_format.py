"""
The model's own turn markers, and the one shape every turn must take.

A model's turn is its thinking, the tag that closes it, the separator its
chat template puts after that tag, its reply, and the token that ends its
turn: nothing before, between or after. The markers differ by model, so
each model family names them (`chat:` in configs/family/); Qwen3.5's are
</think>, "\\n\\n" and <|im_end|>. The end of a turn is the chat template's,
which is not always the tokenizer's end-of-text token.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

OUT_OF_BUDGET = "out of budget"
"""What is wrong with a turn that never ended."""

UNCLOSED_THINKING = "never closed its thinking"
"""What is wrong with a turn that ended without closing its thinking."""

STRAY_MARKER = "a stray marker"
"""What is wrong with a turn that wrote a marker anywhere else."""

REPLY_OUT_OF_FORMAT = "reply not in the format"
"""What is wrong with a turn whose reply is not exactly the format."""


@dataclass(frozen=True)
class ChatFormat:
    """
    A model's turn markers, as token ids, and its separator.

    Attributes:
        think_close_id: The token that closes the thinking.
        end_of_turn_id: The token that ends a turn.
        separator: What the chat template writes between the closing tag
            and the reply.
        marker_ids: Every special or added token of the tokenizer: the
            chat template's markers, none of which may appear in a turn
            but the closing tag, once, and the end of the turn, last.
        marker_texts: The same markers, as text, for the pair's page.
    """

    think_close_id: int
    end_of_turn_id: int
    separator: str
    marker_ids: frozenset[int]
    marker_texts: frozenset[str]

    @classmethod
    def build(cls, tokenizer: Any, chat: Mapping[str, Any]) -> "ChatFormat":
        """
        Reads a family's `chat:` markers from the run's tokenizer.

        Args:
            tokenizer: The run's tokenizer.
            chat: The family's `think_close`, `separator` and
                `end_of_turn`.

        Returns:
            The format.

        Raises:
            ValueError: If a marker is not a single token of the tokenizer.
        """
        added = tokenizer.added_tokens_decoder
        texts = {token.content for token in added.values()}
        texts |= set(tokenizer.all_special_tokens)
        ids = set(added) | set(tokenizer.all_special_ids)

        def marker_id(key: str, text: str) -> int:
            """
            Looks up one marker's token id.

            Args:
                key: The `chat:` entry it comes from.
                text: The marker.

            Returns:
                Its token id.

            Raises:
                ValueError: If it is not a marker token of the tokenizer.
            """
            token_id = tokenizer.convert_tokens_to_ids(text)
            if token_id not in ids:
                raise ValueError(f"{key} {text!r} is not a marker token")
            return token_id

        return cls(
            think_close_id=marker_id("think_close", chat["think_close"]),
            end_of_turn_id=marker_id("end_of_turn", chat["end_of_turn"]),
            separator=chat["separator"],
            marker_ids=frozenset(ids),
            marker_texts=frozenset(texts),
        )

    def ended(self, token_ids: list[int]) -> bool:
        """
        Tells whether a turn ended itself, rather than ran out of budget.

        Args:
            token_ids: The tokens the model generated in the turn.

        Returns:
            Whether its last token ends the turn.
        """
        return bool(token_ids) and token_ids[-1] == self.end_of_turn_id

    def think_close_position(self, token_ids: list[int]) -> int | None:
        """
        Finds where a turn closes its thinking.

        Args:
            token_ids: The tokens the model generated in the turn.

        Returns:
            The position of the first closing tag; None if there is none.
        """
        if self.think_close_id in token_ids:
            return token_ids.index(self.think_close_id)
        return None

    def without_separator(self, after: str) -> str:
        """
        Removes the separator the model wrote before its reply.

        Args:
            after: What the turn wrote after its closing tag.

        Returns:
            The text after the separator, if it starts with it.
        """
        if after.startswith(self.separator):
            return after[len(self.separator) :]
        return after

    def split(self, tokenizer: Any, token_ids: list[int]) -> tuple[str, str]:
        """
        Splits a turn at its closing tag into its thinking and its reply.

        Args:
            tokenizer: The run's tokenizer.
            token_ids: The tokens the model generated in the turn.

        Returns:
            The thinking, stripped, and the reply, exactly: after the
            separator, up to the end of the turn. With no closing tag,
            everything is thinking and the reply is empty.
        """
        cut = self.think_close_position(token_ids)
        if cut is None:
            cut = len(token_ids)
        body = token_ids[cut + 1 :]
        if self.ended(body):
            body = body[:-1]
        thinking = tokenizer.decode(token_ids[:cut], skip_special_tokens=True)
        thinking = str(thinking).strip()
        return thinking, self.without_separator(str(tokenizer.decode(body)))

    def problem(
        self,
        tokenizer: Any,
        token_ids: list[int],
        reply_in_format: Callable[[str], bool],
    ) -> str:
        """
        Says what is wrong with a turn, if anything.

        The turn must be: thinking (no marker; empty is how chat templates
        render a turn with no reasoning), the closing tag, the separator, a
        reply with no other space around it that `reply_in_format`
        accepts, and the end of the turn.

        Args:
            tokenizer: The run's tokenizer.
            token_ids: The tokens the model generated in the turn.
            reply_in_format: Tells whether a reply is in its exact format.

        Returns:
            One of the module's problems; empty if the turn is clean.
        """
        if not self.ended(token_ids):
            return OUT_OF_BUDGET
        body = token_ids[:-1]
        cut = self.think_close_position(body)
        if cut is None:
            return UNCLOSED_THINKING
        if any(i in self.marker_ids for i in body[:cut] + body[cut + 1 :]):
            return STRAY_MARKER
        after = str(tokenizer.decode(body[cut + 1 :]))
        if not after.startswith(self.separator):
            return REPLY_OUT_OF_FORMAT
        reply = self.without_separator(after)
        if not reply or reply != reply.strip():
            return REPLY_OUT_OF_FORMAT
        if not reply_in_format(reply):
            return REPLY_OUT_OF_FORMAT
        return ""
