"""Check that the text of a manuscript md survives in a rendered two-column PDF.

Why: the LibreOffice PDF of 논문초안_v12 lost one digit at a justified line break ("36.9 mm" was printed as "6.9 mm"),
and a reviewer read the wrong number. This script compares every sentence of the md with the PDF text, read column
by column and across the full width, ignoring whitespace. A sentence that is only interrupted by a column or page break is not reported.

Run: python3 scripts/check_pdf_text.py manuscript/X.md manuscript/X.pdf [--review]
Exit status 1 when text is lost or changed. --review skips the author block and the acknowledgement, which the blind copy omits.
"""
import html
import re
import subprocess
import sys


def lines_of(words):
    """Group words into text lines. Full-size words define the lines; raised or small words (superscripts) join the nearest line."""
    big = sorted((w for w in words if w[3] - w[1] > 11), key=lambda w: (w[1] + w[3]) / 2); centres = []
    for w in big:
        c = (w[1] + w[3]) / 2
        if centres and c - centres[-1][0] < 4:
            centres[-1][1].append(w)
        else:
            centres.append([c, [w]])
    for w in words:
        if w[3] - w[1] > 11:
            continue
        c = (w[1] + w[3]) / 2
        if centres:
            near = min(centres, key=lambda line: abs(line[0] - c))
            if abs(near[0] - c) < 8:
                near[1].append(w); continue
        centres.append([c, [w]])
    return [''.join(w[4] for w in sorted(line[1], key=lambda w: w[0])) for line in sorted(centres, key=lambda line: line[0])]


def pdf_text(pdf):
    """Two readings of every page: column by column (body) and across the full width (title block, abstract, reference tables).

    The word boxes of `pdftotext -bbox` are ordered here, because the reading order of plain pdftotext breaks on justified
    lines with wide word gaps.
    """
    xml = subprocess.run(['pdftotext', '-bbox', pdf, '-'], capture_output=True, text=True, check=True).stdout
    columns, full = [], []
    for page in re.finditer(r'(?s)<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>', xml):
        half = float(page.group(1)) / 2
        words = [(float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), html.unescape(m.group(5)))
                 for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', page.group(3))]
        columns += lines_of([w for w in words if w[0] < half]) + lines_of([w for w in words if w[0] >= half])
        full += lines_of(words)
    return ''.join(columns) + ''.join(full)


def norm(text):
    text = re.sub(r'\*\*|\^|_\{|\}|\$\$', '', text)                # md markup of the builders
    text = text.replace('−', '-').replace('–', '-')
    return re.sub(r'[\s\-]+', '', text)                            # a hyphen at a line end is dropped by pdftotext


def sentences(md, review):
    body = md
    if review:
        body = re.sub(r'(?s)\A(# [^\n]*\n\n[^\n]*\n).*?(## ABSTRACT)', r'\1\n\2', body)      # author block
        body = re.sub(r'(?s)## 후 기.*?(## References)', r'\1', body)
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '|', '$$')):
            continue
        if line.startswith('!['):
            line = line[2:line.index('](')]
        for sent in re.split(r'(?<=[.다])\s+', line):
            if len(norm(sent)) >= 8:
                yield sent


def main():
    md_path, pdf_path = sys.argv[1], sys.argv[2]; review = '--review' in sys.argv
    flat = norm(pdf_text(pdf_path)); lost = []
    for sent in sentences(open(md_path, encoding='utf-8').read(), review):
        s = norm(sent)
        if s in flat:
            continue
        a = len(s)
        while a > 0 and s[:a] not in flat:
            a -= 1
        b = len(s)
        while b > 0 and s[-b:] not in flat:
            b -= 1
        if a + b >= len(s):
            continue                                               # interrupted by a column or page break only
        lost.append((sent, s[max(0, a - 14):a], s[a:len(s) - b], s[len(s) - b:len(s) - b + 14]))
    for sent, before, middle, after in lost:
        print(f'LOST OR CHANGED: "{middle}" between "{before}" and "{after}"\n   in: {sent[:90]}')
    print(f'{pdf_path}: {"text intact" if not lost else str(len(lost)) + " sentence(s) differ"}')
    return 1 if lost else 0


if __name__ == '__main__':
    sys.exit(main())
