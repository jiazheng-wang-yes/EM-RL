"""Times New Roman for paper figures.

Set PAPER_FONT_DIR to a directory with the regular, bold, italic and bold italic
Times New Roman TTF files when the font is not installed on the system.
"""
import os
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager


def configure_fonts():
    if directory := os.environ.get('PAPER_FONT_DIR'):
        for path in sorted(Path(directory).iterdir()):
            if path.suffix.lower() != '.ttf':
                continue
            font_manager.fontManager.addfont(path)
    for style, weight in [('normal', 'normal'), ('normal', 'bold'),
                          ('italic', 'normal'), ('italic', 'bold')]:
        font_manager.findfont(font_manager.FontProperties(
            family='Times New Roman', style=style, weight=weight),
            fallback_to_default=False)
    mpl.rcParams.update({
        'font.family': 'Times New Roman',
        # STIX provides Times-compatible math with calligraphic sets and transpose.
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
