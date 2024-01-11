import random

from typing import Iterable

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes

from utils import handle_user_message

class Theme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.slate,
        secondary_hue: colors.Color | str = colors.green,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Cousine"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Cousine"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

chat_interface = gr.ChatInterface(
    fn=handle_user_message,
    css='footer, .wrapper label {visibility: hidden};',
    theme=Theme(),
    # submit_btn=None,
    # stop_btn=None,
    # retry_btn=None,
    # undo_btn=None,
    clear_btn=None,
    analytics_enabled=False
)
