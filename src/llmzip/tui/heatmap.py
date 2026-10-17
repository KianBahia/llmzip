"""Token heatmap widget: the file's text, soft-wrapped, one background colour per token.

Uses Textual's line API so thousands of tokens stay cheap to render and a click maps
exactly to a token: the layout records, for every VISUAL line, the
(col_start, col_end, token_index) span of each piece drawn on it.

Display-only choices (the analysed text is untouched):
  * tabs are drawn as 4 spaces and "\r" is not drawn;
  * a token that would otherwise occupy no cells (a lone "\n", "\r", or the first part
    of a character split across tokens) is drawn as one space so it can be seen and
    selected;
  * a token that would cross the right edge moves to the next line; a token wider
    than the view is split across lines but stays one selectable token.
"""

from dataclasses import dataclass

from rich.cells import cell_len
from rich.segment import Segment
from rich.style import Style
from textual.binding import Binding
from textual.geometry import Size
from textual.message import Message
from textual.reactive import reactive
from textual.scroll_view import ScrollView
from textual.strip import Strip

from llmzip.inspect_cmd import bucket


@dataclass
class Span:
    start: int  # first cell column
    end: int  # one past the last cell column
    token: int
    text: str


def _display(text: str) -> str:
    return text.replace("\r", "").replace("\t", "    ")


def layout(texts: list[str], width: int) -> list[list[Span]]:
    """Lay tokens out into visual lines of at most `width` cells."""
    width = max(width, 1)
    lines: list[list[Span]] = [[]]
    col = 0

    def newline():
        nonlocal col
        lines.append([])
        col = 0

    def place(piece: str, idx: int):
        nonlocal col
        w = cell_len(piece)
        if col + w > width and col > 0:  # move the whole piece to the next line
            newline()
        while cell_len(piece) > width - col:  # wider than the view: split by character
            head, used = "", 0
            for ch in piece:
                if used + cell_len(ch) > width - col:
                    break
                head += ch
                used += cell_len(ch)
            if not head:  # a single wide char that cannot fit on an empty line
                head, used = piece[0], cell_len(piece[0])
            lines[-1].append(Span(col, col + used, idx, head))
            piece = piece[len(head):]
            newline()
            if not piece:
                return
        if piece:
            lines[-1].append(Span(col, col + cell_len(piece), idx, piece))
            col += cell_len(piece)

    for idx, raw in enumerate(texts):
        parts = _display(raw).split("\n")
        drawn = False
        for k, part in enumerate(parts):
            if k > 0:
                if not drawn:  # a lone newline: give it one visible cell first
                    place(" ", idx)
                    drawn = True
                newline()
            if part:
                place(part, idx)
                drawn = True
        if not drawn:
            place(" ", idx)
    return lines


class HeatmapView(ScrollView, can_focus=True):
    """Scrollable, soft-wrapped token heatmap with a token cursor."""

    # A stable gutter keeps the content width fixed when the scrollbar appears, so the
    # wrap width used by layout() always equals the drawn width.
    DEFAULT_CSS = """
    HeatmapView { scrollbar-gutter: stable; overflow-x: hidden; }
    """

    BINDINGS = [
        Binding("left", "move(-1)", "Prev token", show=False),
        Binding("right", "move(1)", "Next token", show=False),
        Binding("up", "line(-1)", "Line up", show=False),
        Binding("down", "line(1)", "Line down", show=False),
    ]

    cursor: reactive[int] = reactive(0)

    class TokenSelected(Message):
        def __init__(self, index: int) -> None:
            self.index = index
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._texts: list[str] = []
        self._bits: list[float] = []
        self._lines: list[list[Span]] = [[]]
        self._first: list[tuple[int, int]] = []  # token -> (visual line, start col) of its first piece
        self._width = 0

    # --- data and layout ----------------------------------------------------------

    def set_tokens(self, texts: list[str], bits: list[float]) -> None:
        self._texts, self._bits = texts, bits
        self._width = 0
        self._relayout()
        self.cursor = 0
        self.scroll_home(animate=False)
        if texts:
            self.post_message(self.TokenSelected(0))

    def clear(self) -> None:
        self.set_tokens([], [])

    def _relayout(self) -> None:
        width = self.scrollable_content_region.width or self.size.width or 80
        if width == self._width and self._texts:
            return
        self._width = width
        self._lines = layout(self._texts, width) if self._texts else [[]]
        self._first = [(-1, 0)] * len(self._texts)
        for y, line in enumerate(self._lines):
            for span in line:
                if self._first[span.token][0] < 0:
                    self._first[span.token] = (y, span.start)
        self.virtual_size = Size(width, len(self._lines))
        self.refresh()

    def on_resize(self) -> None:
        self._relayout()

    @property
    def lines(self) -> list[list[Span]]:
        return self._lines

    def position_of(self, token: int) -> tuple[int, int]:
        """(visual line, column) where `token` starts."""
        return self._first[token]

    def token_at(self, x: int, y: int) -> int | None:
        """Token under content cell (x, y), where y is a visual line index."""
        if not 0 <= y < len(self._lines):
            return None
        for span in self._lines[y]:
            if span.start <= x < span.end:
                return span.token
        return None

    # --- rendering ----------------------------------------------------------------

    def render_line(self, y: int) -> Strip:
        line_no = y + self.scroll_offset.y
        width = self.scrollable_content_region.width
        if not 0 <= line_no < len(self._lines):
            return Strip.blank(width)
        segments = []
        for span in self._lines[line_no]:
            bg, fg, _ = bucket(self._bits[span.token])
            style = Style(color=fg, bgcolor=bg)
            if span.token == self.cursor:
                style += Style(reverse=True, bold=True, underline=True)
            segments.append(Segment(span.text, style))
        return Strip(segments).extend_cell_length(width)

    # --- cursor -------------------------------------------------------------------

    def select(self, index: int) -> None:
        if not self._texts:
            return
        self.cursor = max(0, min(index, len(self._texts) - 1))

    def watch_cursor(self, old: int, new: int) -> None:
        if not self._texts:
            return
        y, _ = self._first[new]
        if y >= 0:
            top = self.scroll_offset.y
            height = self.scrollable_content_region.height or 1
            if not top <= y < top + height:
                self.scroll_to(y=max(0, y - height // 2), animate=False)
        self.refresh()
        self.post_message(self.TokenSelected(new))

    def action_move(self, delta: int) -> None:
        self.select(self.cursor + delta)

    def action_line(self, delta: int) -> None:
        if not self._texts:
            return
        y, x = self._first[self.cursor]
        target = y + delta
        while 0 <= target < len(self._lines) and not self._lines[target]:
            target += delta  # skip empty visual lines
        if not 0 <= target < len(self._lines):
            return
        spans = self._lines[target]
        hit = next((s for s in spans if s.start <= x < s.end), None)
        if hit is None:
            hit = max((s for s in spans if s.start <= x), key=lambda s: s.start, default=spans[0])
        self.select(hit.token)

    def on_click(self, event) -> None:
        offset = event.get_content_offset(self)
        if offset is None:
            return
        token = self.token_at(offset.x, offset.y + self.scroll_offset.y)
        if token is not None:
            self.focus()
            self.select(token)
