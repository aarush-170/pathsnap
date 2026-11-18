"""Shared ReportLab styles and helpers for the PDF report builders."""
from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, Table, TableStyle

INK, MUTED, ACCENT, LIGHT = colors.HexColor("#1f2933"), colors.HexColor("#52606d"), \
    colors.HexColor("#2463eb"), colors.HexColor("#eef2f7")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=22, leading=27, textColor=INK, spaceAfter=4)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=11, textColor=MUTED, alignment=TA_CENTER, spaceAfter=14)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=15, leading=19, textColor=ACCENT, spaceBefore=8, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=11.5, leading=15, textColor=INK, spaceBefore=6, spaceAfter=3)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=10, leading=14, textColor=INK, spaceAfter=5)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.8, leading=11.5, spaceAfter=0)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
CAP = ParagraphStyle("CAP", parent=BODY, fontSize=8.5, leading=11, textColor=MUTED, alignment=TA_CENTER, spaceAfter=8)
BUL = ParagraphStyle("BUL", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)

PAGE_W = A4[0] - 4 * cm


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(rows, widths, header=True, zebra=True):
    data = [[P(c, CELLB if (header and i == 0) else CELL) if isinstance(c, str) else c for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd2d9")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe4f0")))
    if zebra:
        for r in range(1 if header else 0, len(rows)):
            if r % 2 == 0:
                st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f7f9fb")))
    t.setStyle(TableStyle(st))
    return t
