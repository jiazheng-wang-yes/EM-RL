"""Clarify Figure 1's removal result while preserving its vector artwork."""
from pathlib import Path

import pymupdf
from matplotlib.font_manager import findfont

root = Path(__file__).resolve().parents[2]
figures = root / 'paper_draft/arr_2026/figures'
doc = pymupdf.open(figures / 'archive/overview_v2_before_removal_label.pdf')
page = doc[0]
span = next(
    span
    for block in page.get_text('dict')['blocks']
    for line in block.get('lines', [])
    for span in line['spans']
    if span['text'] == 'found in six settings'
)
label = 'removal in six settings'
font_path = findfont('DejaVu Sans')
font = pymupdf.Font(fontfile=font_path)
width = font.text_length(label, fontsize=span['size'])
box = pymupdf.Rect(span['bbox'])
x = (box.x0 + box.x1 - width) / 2
page.add_redact_annot(box, fill=False)
page.apply_redactions(images=0, graphics=0)
page.insert_font(fontname='RemovalLabel', fontfile=font_path)
color = tuple(((span['color'] >> shift) & 255) / 255 for shift in (16, 8, 0))
page.insert_text((x, span['origin'][1]), label, fontname='RemovalLabel',
                 fontsize=span['size'], color=color)
doc.save(figures / 'overview_v2.pdf', garbage=4, deflate=True)
