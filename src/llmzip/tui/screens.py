"""Modal screens: overwrite confirmation and the key-binding help."""

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

KEYS = [
    ("1 / 2 / 3 / 4", "Overview / Heatmap / Stats / Compare tab"),
    ("tab / shift+tab", "move focus (file tree, buttons, heatmap, tables)"),
    ("ctrl+o", "compress the selected file"),
    ("ctrl+d", "decompress the selected .lz file"),
    ("v", "verify (round trip in memory, nothing written)"),
    ("a", "analyze: build the heatmap and stats"),
    ("b", "run the benchmark comparison"),
    ("c", "cancel the running job (nothing is written)"),
    ("t", "show / hide the file tree"),
    ("← → ↑ ↓ / click", "heatmap: move the token cursor"),
    ("enter", "stats table: jump to that token in the heatmap"),
    ("?", "this help"),
    ("q", "quit"),
]


class ConfirmScreen(ModalScreen[bool]):
    """Yes/No question; dismisses with True only on an explicit yes."""

    BINDINGS = [Binding("y", "answer(True)", "Yes"), Binding("n", "answer(False)", "No"),
                Binding("escape", "answer(False)", "No")]
    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    ConfirmScreen > Vertical { width: 64; height: auto; padding: 1 2; border: thick $warning; background: $surface; }
    ConfirmScreen Horizontal { height: auto; align: center middle; margin-top: 1; }
    ConfirmScreen Button { margin: 0 1; }
    """

    def __init__(self, question: str) -> None:
        super().__init__()
        self.question = question

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(Text(self.question))
            with Horizontal():
                yield Button("Overwrite (y)", id="yes", variant="warning")
                yield Button("Cancel (n)", id="no", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#no", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_answer(self, yes: bool) -> None:
        self.dismiss(yes)


class HelpScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape,question_mark,q", "dismiss_help", "Close")]
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen > Vertical { width: 76; height: auto; padding: 1 2; border: thick $primary; background: $surface; }
    """

    def compose(self) -> ComposeResult:
        text = Text("llmzip terminal app: keys\n\n", style="bold")
        for key, what in KEYS:
            text.append(f"{key:>18}  ", style="bold")
            text.append(f"{what}\n")
        text.append("\nPress escape or ? to close.", style="dim")
        with Vertical():
            yield Static(text)

    def action_dismiss_help(self) -> None:
        self.dismiss(None)
