"""Check the ARR manuscript and render specified pages from its PDF.

Run from the repository root. JSON goes to eval_runs; review images go to figures.
The diagnostic PDF omits review rulers and page numbers for ACL pubcheck only.
The submission PDF is never edited by this script.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = {
    'acl.sty': '19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de',
    'acl_natbib.bst': '6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, default=ROOT / 'paper_draft/arr_2026/main.pdf')
    parser.add_argument('--diagnostic-pdf', type=Path)
    parser.add_argument('--render', type=int, nargs='*', default=[])
    args = parser.parse_args()
    paper = ROOT / 'paper_draft/arr_2026'
    out = ROOT / 'eval_runs/persona_control_stage9/arr_submission_20261011'
    images = ROOT / 'figures/persona_control/arr_submission_20261011'
    out.mkdir(parents=True, exist_ok=True)
    src = (paper / 'main.tex').read_text()
    active = re.sub(r'(?m)(?<!\x5c)%.*$', '', src)
    abstract = active.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
    abstract = re.sub(r'\x5ctextbf\{([^}]+)\}', r'\1', abstract).replace(r'\%', '%').strip()
    abstract = re.sub(r'\x5chref\{[^}]+\}\{([^}]+)\}', r'\1', abstract)
    cited = sorted({k.strip() for item in re.findall(r'\x5ccite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}', active)
                    for k in item.split(',')})
    bibliography = (paper / 'references.bib').read_text()
    keys = set(re.findall(r'@\w+\{([^,]+),', bibliography))
    doc = pymupdf.open(args.pdf)
    report = dict(pdf=str(args.pdf), pdf_sha256=hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
                  pages=len(doc), abstract_words=len(abstract.split()), abstract_plain_text=abstract,
                  review_mode=r'\usepackage[review]{acl}' in active,
                  official_style={name: hashlib.sha256((paper / name).read_bytes()).hexdigest() == digest
                                  for name, digest in OFFICIAL.items()},
                  official_style_commit='d5adc823ff0f80f98c80405ca0ab66c68e684409',
                  cited_references=len(cited), missing_citations=sorted(set(cited) - keys),
                  metadata=doc.metadata, headings={}, links=[], fonts=[], margin_spans=[],
                  invisible_text_spans=[], review_ruler_spans=0)
    headings = ['Conclusion', 'Limitations', 'Ethical considerations', 'References',
                'Artifact documentation and annotation instructions']
    fonts = {}
    links = set()
    sizes = []
    for i, page in enumerate(doc):
        sizes.append([round(page.rect.width, 2), round(page.rect.height, 2)])
        spans = [span for block in page.get_text('dict')['blocks']
                 for line in block.get('lines', []) for span in line['spans']]
        for h in headings:
            start = h.split(' and ')[0]
            if any(start in span['text'] and span['size'] > 11.5
                   and ('Medi' in span['font'] or 'Bold' in span['font']) for span in spans):
                report['headings'].setdefault(h, []).append(i + 1)
        for link in page.get_links():
            if link.get('uri'):
                links.add(link['uri'])
        for font in page.get_fonts():
            fonts[font[0]] = font[3]
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    text = span['text'].strip()
                    ruler = text.isdigit() and span['color'] != 0 and (x1 < 60 or x0 > 550)
                    page_number = text == str(i + 1) and y0 > 790
                    if ruler:
                        report['review_ruler_spans'] += 1
                    if ruler or page_number:
                        if args.diagnostic_pdf:
                            page.add_redact_annot(span['bbox'], fill=(1, 1, 1))
                        continue
                    # Two-point tolerance includes glyph bearings and microtype protrusion.
                    if text and (x0 < 68.87 or x1 > 526.41 or y0 < 68.87 or y1 > 777.3):
                        report['margin_spans'].append(dict(page=i + 1, text=text, bbox=span['bbox']))
        for trace in page.get_texttrace():
            if trace['type'] == 3 or trace.get('opacity', 1) == 0:
                report['invisible_text_spans'].append(dict(page=i + 1, type=trace['type']))
        if i + 1 in args.render:
            images.mkdir(parents=True, exist_ok=True)
            page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False, annots=False).save(images / f'page_{i+1:02d}.png')
        if args.diagnostic_pdf:
            page.apply_redactions()
    report['page_sizes_points'] = sizes
    report['all_pages_A4'] = all(abs(w - 595.28) < .1 and abs(h - 841.89) < .1 for w, h in sizes)
    report['links'] = sorted(links)
    report['fonts'] = [dict(name=name, embedded=bool(doc.extract_font(xref)[3]))
                       for xref, name in fonts.items()]
    report['all_fonts_embedded'] = all(f['embedded'] for f in report['fonts'])
    report['body_ends_within_8_pages'] = report['headings'].get('Limitations', [999])[0] <= 8
    if args.diagnostic_pdf:
        args.diagnostic_pdf.parent.mkdir(parents=True, exist_ok=True)
        doc.save(args.diagnostic_pdf)
    (out / 'paper_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ['pages', 'abstract_words', 'official_style',
                                           'all_pages_A4', 'all_fonts_embedded',
                                           'body_ends_within_8_pages', 'cited_references',
                                           'missing_citations', 'headings']}, indent=2))
    print('Margin spans:', len(report['margin_spans']), '; invisible spans:', len(report['invisible_text_spans']))


if __name__ == '__main__':
    main()
