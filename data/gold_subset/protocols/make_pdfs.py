#!/usr/bin/env python3
"""
Generate print-ready HI-CSG-R handwriting collection PDFs from protocol .md files.

Updated layout behavior:
  - one WRITE_START/WRITE_END block per page;
  - fixed block geometry on every page;
  - exactly 18 ruled handwriting lines on every page;
  - no bottom QC checkboxes;
  - no footer with source file / page counter;
  - concise exact-copying instructions for writers;
  - compact multi-column source text fitting, so long blocks do not shrink the writing area;
  - Latin letters/English words are bold in mixed Russian/English source blocks;
  - S02 pages show a 3-14 day interval reminder in the header;
  - handwriting lines include pale line numbers.

Example:
  python make_printable_pdfs_fixed_18lines.py \
      --input /mnt/data/hi_csg_r_protocols_exact_no_semifree_reduced_en \
      --output /mnt/data/hi_csg_r_printable_pdfs_fixed_18 \
      --combined

Dependencies:
  pip install reportlab pypdf
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

try:
    from reportlab.graphics import renderPDF
    from reportlab.graphics.barcode import qr
    from reportlab.graphics.shapes import Drawing
    QR_AVAILABLE = True
except Exception:
    QR_AVAILABLE = False

try:
    from pypdf import PdfReader, PdfWriter
    PYPDF_AVAILABLE = True
except Exception:
    try:
        from PyPDF2 import PdfReader, PdfWriter  # type: ignore
        PYPDF_AVAILABLE = True
    except Exception:
        PYPDF_AVAILABLE = False


@dataclasses.dataclass
class ProtocolBlock:
    participant_id: str
    version: str
    block_id: str
    session: str
    page: str
    lang: str
    script: str
    task_type: str
    condition: str
    transcription_expected: str
    title: str
    text: str
    source_file: str
    ordinal: int


@dataclasses.dataclass
class Protocol:
    participant_id: str
    version: str
    general_instruction: str
    source_path: Path
    blocks: List[ProtocolBlock]


@dataclasses.dataclass
class LayoutConfig:
    page_size: Tuple[float, float]
    margin_x: float = 12 * mm
    margin_y: float = 10 * mm
    header_h: float = 31 * mm
    instruction_h: float = 15 * mm
    source_h: float = 60 * mm
    gap: float = 2.0 * mm
    writing_line_count: int = 18
    font_regular: str = "HIRegular"
    font_bold: str = "HIBold"
    font_mono: str = "HIMono"
    with_qr: bool = False
    draw_crop_marks: bool = False


FIELD_RE = re.compile(r"^\[([A-Z0-9_]+):\s*(.*?)\]\s*$")
BLOCK_START_RE = re.compile(r"^\[BLOCK_ID:\s*(.*?)\]\s*$")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s\-()]{0,3}){9,}\d(?!\d)")
CYRILLIC_LETTER_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
LATIN_RUN_RE = re.compile(r"[A-Za-z]+")

DEFAULT_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]
DEFAULT_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
]
DEFAULT_MONO_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
]


def first_existing(paths: Sequence[str]) -> Optional[str]:
    for p in paths:
        if p and Path(p).exists():
            return p
    return None


def register_fonts(font_path: Optional[str] = None, bold_path: Optional[str] = None, mono_path: Optional[str] = None) -> Tuple[str, str, str]:
    """Register Unicode fonts. Falls back to built-in PDF fonts if no TTF is found."""
    regular = font_path or first_existing(DEFAULT_FONT_CANDIDATES)
    bold = bold_path or first_existing(DEFAULT_BOLD_CANDIDATES)
    mono = mono_path or first_existing(DEFAULT_MONO_CANDIDATES)

    if regular:
        pdfmetrics.registerFont(TTFont("HIRegular", regular))
        regular_name = "HIRegular"
    else:
        regular_name = "Helvetica"

    if bold:
        pdfmetrics.registerFont(TTFont("HIBold", bold))
        bold_name = "HIBold"
    else:
        bold_name = "Helvetica-Bold"

    if mono:
        pdfmetrics.registerFont(TTFont("HIMono", mono))
        mono_name = "HIMono"
    else:
        mono_name = "Courier"

    return regular_name, bold_name, mono_name


def normalize_text(s: str) -> str:
    return s.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def infer_participant_from_filename(name: str) -> str:
    m = re.search(r"P\d{3,}", name)
    return m.group(0) if m else Path(name).stem


def parse_protocol(path: Path) -> Protocol:
    raw = normalize_text(path.read_text(encoding="utf-8"))
    lines = raw.splitlines()
    global_meta: Dict[str, str] = {}
    blocks: List[ProtocolBlock] = []
    i = 0
    ordinal = 0

    while i < len(lines):
        line = lines[i].strip()
        field = FIELD_RE.match(line)
        if field and field.group(1) != "BLOCK_ID" and not blocks:
            global_meta[field.group(1)] = field.group(2)
            i += 1
            continue

        block_start = BLOCK_START_RE.match(line)
        if not block_start:
            i += 1
            continue

        meta: Dict[str, str] = {"BLOCK_ID": block_start.group(1)}
        i += 1
        while i < len(lines) and lines[i].strip() != "WRITE_START":
            f = FIELD_RE.match(lines[i].strip())
            if f:
                meta[f.group(1)] = f.group(2)
            i += 1

        if i >= len(lines) or lines[i].strip() != "WRITE_START":
            raise ValueError(f"{path.name}: block {meta.get('BLOCK_ID')} has no WRITE_START")
        i += 1

        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip() != "WRITE_END":
            text_lines.append(lines[i].rstrip())
            i += 1
        if i >= len(lines):
            raise ValueError(f"{path.name}: block {meta.get('BLOCK_ID')} has no WRITE_END")
        i += 1

        ordinal += 1
        participant_id = global_meta.get("PARTICIPANT_ID", infer_participant_from_filename(path.name))
        version = global_meta.get("VERSION", "unknown")
        blocks.append(
            ProtocolBlock(
                participant_id=participant_id,
                version=version,
                block_id=meta.get("BLOCK_ID", ""),
                session=meta.get("SESSION", ""),
                page=meta.get("PAGE", ""),
                lang=meta.get("LANG", ""),
                script=meta.get("SCRIPT", ""),
                task_type=meta.get("TASK_TYPE", ""),
                condition=meta.get("CONDITION", ""),
                transcription_expected=meta.get("TRANSCRIPTION_EXPECTED", ""),
                title=meta.get("TITLE", ""),
                text="\n".join(text_lines).strip(),
                source_file=path.name,
                ordinal=ordinal,
            )
        )

    if not blocks:
        raise ValueError(f"{path.name}: no protocol blocks found")

    return Protocol(
        participant_id=global_meta.get("PARTICIPANT_ID", infer_participant_from_filename(path.name)),
        version=global_meta.get("VERSION", "unknown"),
        general_instruction=global_meta.get("GENERAL_INSTRUCTION", ""),
        source_path=path,
        blocks=blocks,
    )


def collect_protocol_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".md":
            raise ValueError("Input file must be .md")
        return [input_path]
    files = sorted(input_path.glob("P*_protocol.md"))
    if not files:
        files = sorted(input_path.glob("*.md"))
    return [p for p in files if p.is_file()]


def text_width(text: str, font: str, size: float) -> float:
    return pdfmetrics.stringWidth(text, font, size)


def wrap_one_paragraph(paragraph: str, font: str, size: float, max_width: float) -> List[str]:
    if paragraph == "":
        return [""]
    words = paragraph.split(" ")
    lines: List[str] = []
    cur = ""
    for word in words:
        if word == "":
            continue
        candidate = word if not cur else cur + " " + word
        if text_width(candidate, font, size) <= max_width:
            cur = candidate
            continue
        if cur:
            lines.append(cur)
        if text_width(word, font, size) <= max_width:
            cur = word
            continue
        buf = ""
        for ch in word:
            candidate_char = buf + ch
            if text_width(candidate_char, font, size) <= max_width:
                buf = candidate_char
            else:
                if buf:
                    lines.append(buf)
                buf = ch
        cur = buf
    if cur:
        lines.append(cur)
    return lines


def wrap_text_preserve_newlines(text: str, font: str, size: float, max_width: float) -> List[str]:
    paragraphs = text.splitlines() or [""]
    out: List[str] = []
    for idx, paragraph in enumerate(paragraphs):
        out.extend(wrap_one_paragraph(paragraph.strip(), font, size, max_width))
        if idx != len(paragraphs) - 1 and paragraph.strip() == "":
            out.append("")
    return out


def draw_round_rect(c: canvas.Canvas, x: float, y: float, w: float, h: float, stroke=colors.black, fill=None, radius: float = 4) -> None:
    c.saveState()
    c.setStrokeColor(stroke)
    if fill is not None:
        c.setFillColor(fill)
        c.roundRect(x, y, w, h, radius, stroke=1, fill=1)
    else:
        c.roundRect(x, y, w, h, radius, stroke=1, fill=0)
    c.restoreState()


def draw_text_lines(c: canvas.Canvas, lines: Sequence[str], x: float, y_top: float, font: str, size: float, line_h: float, fill=colors.black) -> float:
    c.saveState()
    c.setFont(font, size)
    c.setFillColor(fill)
    y = y_top
    for line in lines:
        c.drawString(x, y, line)
        y -= line_h
    c.restoreState()
    return y


def is_mixed_ru_en_text(text: str) -> bool:
    """True when a source block contains both Cyrillic and Latin letters."""
    return bool(CYRILLIC_LETTER_RE.search(text) and LATIN_LETTER_RE.search(text))


def should_bold_latin_in_block(block: ProtocolBlock) -> bool:
    """Bold Latin runs only in mixed Russian/English source blocks."""
    return block.lang == "mixed" or is_mixed_ru_en_text(block.text)


def local_block_label(block: ProtocolBlock) -> str:
    """Use the B-number from BLOCK_ID, not the global block ordinal."""
    match = re.search(r"_B(\d{2,3})(?:_|$)", block.block_id)
    if match:
        return f"B{match.group(1)}"
    return f"B{block.ordinal:02d}"


def styled_text_width(text: str, base_font: str, latin_bold_font: str, size: float, bold_latin: bool) -> float:
    if not bold_latin or not text:
        return text_width(text, base_font, size)
    width = 0.0
    pos = 0
    for match in LATIN_RUN_RE.finditer(text):
        if match.start() > pos:
            width += text_width(text[pos:match.start()], base_font, size)
        width += text_width(match.group(0), latin_bold_font, size)
        pos = match.end()
    if pos < len(text):
        width += text_width(text[pos:], base_font, size)
    return width


def draw_styled_string(c: canvas.Canvas, text: str, x: float, y: float, base_font: str, latin_bold_font: str, size: float, bold_latin: bool) -> None:
    if not bold_latin:
        c.setFont(base_font, size)
        c.drawString(x, y, text)
        return
    cur_x = x
    pos = 0
    for match in LATIN_RUN_RE.finditer(text):
        if match.start() > pos:
            chunk = text[pos:match.start()]
            c.setFont(base_font, size)
            c.drawString(cur_x, y, chunk)
            cur_x += text_width(chunk, base_font, size)
        chunk = match.group(0)
        c.setFont(latin_bold_font, size)
        c.drawString(cur_x, y, chunk)
        cur_x += text_width(chunk, latin_bold_font, size)
        pos = match.end()
    if pos < len(text):
        chunk = text[pos:]
        c.setFont(base_font, size)
        c.drawString(cur_x, y, chunk)


def draw_text_lines_styled(
    c: canvas.Canvas,
    lines: Sequence[str],
    x: float,
    y_top: float,
    base_font: str,
    latin_bold_font: str,
    size: float,
    line_h: float,
    bold_latin: bool,
    fill=colors.black,
) -> float:
    c.saveState()
    c.setFillColor(fill)
    y = y_top
    for line in lines:
        draw_styled_string(c, line, x, y, base_font, latin_bold_font, size, bold_latin)
        y -= line_h
    c.restoreState()
    return y


def wrap_one_paragraph_styled(paragraph: str, base_font: str, latin_bold_font: str, size: float, max_width: float, bold_latin: bool) -> List[str]:
    if paragraph == "":
        return [""]
    words = paragraph.split(" ")
    lines: List[str] = []
    cur = ""
    for word in words:
        if word == "":
            continue
        candidate = word if not cur else cur + " " + word
        if styled_text_width(candidate, base_font, latin_bold_font, size, bold_latin) <= max_width:
            cur = candidate
            continue
        if cur:
            lines.append(cur)
        if styled_text_width(word, base_font, latin_bold_font, size, bold_latin) <= max_width:
            cur = word
            continue
        buf = ""
        for ch in word:
            candidate_char = buf + ch
            if styled_text_width(candidate_char, base_font, latin_bold_font, size, bold_latin) <= max_width:
                buf = candidate_char
            else:
                if buf:
                    lines.append(buf)
                buf = ch
        cur = buf
    if cur:
        lines.append(cur)
    return lines


def wrap_text_preserve_newlines_styled(text: str, base_font: str, latin_bold_font: str, size: float, max_width: float, bold_latin: bool) -> List[str]:
    paragraphs = text.splitlines() or [""]
    out: List[str] = []
    for idx, paragraph in enumerate(paragraphs):
        out.extend(wrap_one_paragraph_styled(paragraph.strip(), base_font, latin_bold_font, size, max_width, bold_latin))
        if idx != len(paragraphs) - 1 and paragraph.strip() == "":
            out.append("")
    return out


def draw_label(c: canvas.Canvas, x: float, y: float, label: str, value: str, font: str, bold: str, size: float = 7.2) -> None:
    c.saveState()
    c.setFont(bold, size)
    c.drawString(x, y, f"{label}:")
    offset = text_width(f"{label}: ", bold, size)
    c.setFont(font, size)
    c.drawString(x + offset, y, value)
    c.restoreState()


def draw_qr_code(c: canvas.Canvas, payload: str, x: float, y: float, size: float) -> None:
    if not QR_AVAILABLE:
        return
    qrw = qr.QrCodeWidget(payload)
    bounds = qrw.getBounds()
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    drawing = Drawing(size, size, transform=[size / width, 0, 0, size / height, 0, 0])
    drawing.add(qrw)
    renderPDF.draw(drawing, c, x, y)


def tempo_instruction(condition: str) -> str:
    cond = condition.lower().strip()
    if cond == "slow":
        return "медленно и разборчиво, но без искусственного рисования букв"
    if cond == "fast":
        return "быстрее обычного, с естественным почерком, но читаемо"
    if cond == "slow_fast":
        return "сначала медленно, затем быстрее обычного, оба варианта читаемо"
    return "обычный естественный почерк, без намеренного украшения или упрощения"


def instruction_for(block: ProtocolBlock) -> str:
    parts = [
        "Перепишите текст из блока ниже дословно: пунктуацию, регистр, символы и цифры.",
        f"Темп: {tempo_instruction(block.condition)}.",
        "При ошибке аккуратно зачеркните фрагмент и продолжайте.",
    ]
    if should_bold_latin_in_block(block):
        parts.append("В смешанных RU/EN блоках латиница в образце выделена полужирным.")
    return " ".join(parts)


def draw_header(c: canvas.Canvas, block: ProtocolBlock, cfg: LayoutConfig) -> None:
    page_w, page_h = cfg.page_size
    x0 = cfg.margin_x
    y_top = page_h - cfg.margin_y
    content_w = page_w - 2 * cfg.margin_x
    header_y = y_top - cfg.header_h

    draw_round_rect(c, x0, header_y, content_w, cfg.header_h, stroke=colors.black, fill=colors.white, radius=5)

    c.setFont(cfg.font_bold, 10.8)
    c.drawString(x0 + 4 * mm, y_top - 6.8 * mm, "HI-CSG-R writing form")
    c.setFont(cfg.font_bold, 14.6)
    compact = f"{block.participant_id}  {block.session} / P{block.page} / {local_block_label(block)}"
    right_reserve = 33 * mm if cfg.with_qr else 4 * mm
    c.drawRightString(x0 + content_w - right_reserve, y_top - 6.8 * mm, compact)
    if block.session.upper() == "S02":
        c.setFont(cfg.font_bold, 7.8)
        c.setFillColor(colors.HexColor("#555555"))
        c.drawRightString(
            x0 + content_w - right_reserve,
            y_top - 11.3 * mm,
            "S02: проходить через 3–14 дней после S01",
        )
        c.setFillColor(colors.black)

    if cfg.with_qr:
        qr_payload = json.dumps(
            {
                "participant_id": block.participant_id,
                "session": block.session,
                "page": block.page,
                "block_id": block.block_id,
                "task_type": block.task_type,
                "condition": block.condition,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        draw_qr_code(c, qr_payload, x0 + content_w - 27 * mm, header_y + 3.5 * mm, 22 * mm)

    y1 = y_top - 15.2 * mm
    col1 = x0 + 4 * mm
    col2 = x0 + 75 * mm
    col3 = x0 + 126 * mm
    draw_label(c, col1, y1, "BLOCK_ID", block.block_id, cfg.font_regular, cfg.font_bold, 7.0)
    draw_label(c, col1, y1 - 5 * mm, "TITLE", block.title, cfg.font_regular, cfg.font_bold, 7.0)
    draw_label(c, col1, y1 - 10 * mm, "VERSION", block.version, cfg.font_regular, cfg.font_bold, 7.0)
    draw_label(c, col2, y1 - 10 * mm, "TASK", block.task_type, cfg.font_regular, cfg.font_bold, 7.0)
    draw_label(c, col3, y1, "LANG/SCRIPT", f"{block.lang}/{block.script}", cfg.font_regular, cfg.font_bold, 7.0)
    draw_label(c, col3, y1 - 5 * mm, "CONDITION", block.condition, cfg.font_regular, cfg.font_bold, 7.0)
    draw_label(c, col3, y1 - 10 * mm, "EXPECTED", block.transcription_expected, cfg.font_regular, cfg.font_bold, 7.0)


def draw_instruction_box(c: canvas.Canvas, block: ProtocolBlock, cfg: LayoutConfig, x: float, y_top: float, w: float) -> float:
    h = cfg.instruction_h
    y = y_top - h
    draw_round_rect(c, x, y, w, h, stroke=colors.black, fill=colors.white, radius=4)
    c.setFont(cfg.font_bold, 8.0)
    c.drawString(x + 3 * mm, y_top - 4.7 * mm, "Указание")
    lines = wrap_text_preserve_newlines(instruction_for(block), cfg.font_regular, 6.8, w - 6 * mm)
    draw_text_lines(c, lines[:3], x + 3 * mm, y_top - 8.6 * mm, cfg.font_regular, 6.8, 3.1 * mm)
    return y - cfg.gap


def source_font_for(block: ProtocolBlock, cfg: LayoutConfig) -> str:
    if block.lang in {"mixed", "en"} or "numeric" in block.task_type.lower() or any(ch.isdigit() for ch in block.text):
        return cfg.font_mono
    return cfg.font_regular


def choose_source_layout(block: ProtocolBlock, cfg: LayoutConfig, inner_w: float, inner_h: float) -> Tuple[float, float, int, List[str], bool]:
    """Fit source text into a fixed box. Returns font size, line height, column count, lines, truncated."""
    font = source_font_for(block, cfg)
    bold_latin = should_bold_latin_in_block(block)
    task = block.task_type.lower()
    preferred_cols = [1, 2, 3]
    if "target" in task or "numeric" in task or block.lang == "mixed" or len(block.text) > 700:
        preferred_cols = [2, 3, 1]

    gutter = 3 * mm
    for size_i in range(84, 37, -2):  # 8.4pt down to 3.8pt
        size = size_i / 10
        line_h = size * 1.16
        for cols in preferred_cols:
            col_w = (inner_w - gutter * (cols - 1)) / cols
            if col_w <= 12 * mm:
                continue
            lines = wrap_text_preserve_newlines_styled(block.text, font, cfg.font_bold, size, col_w, bold_latin)
            capacity_per_col = max(1, int(inner_h // line_h))
            if len(lines) <= capacity_per_col * cols:
                return size, line_h, cols, lines, False

    size = 3.8
    line_h = size * 1.12
    cols = 3
    col_w = (inner_w - gutter * (cols - 1)) / cols
    lines = wrap_text_preserve_newlines_styled(block.text, font, cfg.font_bold, size, col_w, bold_latin)
    capacity = max(1, int(inner_h // line_h)) * cols
    if len(lines) > capacity:
        lines = lines[: max(1, capacity - 1)] + ["[... clipped; increase --source-height-mm ...]"]
        return size, line_h, cols, lines, True
    return size, line_h, cols, lines, False


def draw_source_box(c: canvas.Canvas, block: ProtocolBlock, cfg: LayoutConfig, x: float, y_top: float, w: float) -> Tuple[float, bool]:
    h = cfg.source_h
    y = y_top - h
    draw_round_rect(c, x, y, w, h, stroke=colors.black, fill=colors.white, radius=4)
    c.setFont(cfg.font_bold, 8.0)
    c.drawString(x + 3 * mm, y_top - 4.8 * mm, "Блок для переписывания")

    inner_x = x + 3 * mm
    inner_w = w - 6 * mm
    inner_top = y_top - 9.5 * mm
    inner_h = h - 12.5 * mm
    size, line_h, cols, lines, truncated = choose_source_layout(block, cfg, inner_w, inner_h)
    font = source_font_for(block, cfg)
    bold_latin = should_bold_latin_in_block(block)
    gutter = 3 * mm
    col_w = (inner_w - gutter * (cols - 1)) / cols
    capacity_per_col = max(1, int(inner_h // line_h))

    for col in range(cols):
        start = col * capacity_per_col
        end = start + capacity_per_col
        col_lines = lines[start:end]
        if not col_lines:
            continue
        col_x = inner_x + col * (col_w + gutter)
        if bold_latin:
            draw_text_lines_styled(c, col_lines, col_x, inner_top, font, cfg.font_bold, size, line_h, True)
        else:
            draw_text_lines(c, col_lines, col_x, inner_top, font, size, line_h)

    return y - cfg.gap, truncated


def draw_writing_area(c: canvas.Canvas, cfg: LayoutConfig, x: float, y_top: float, w: float, y_bottom: float) -> List[float]:
    h = y_top - y_bottom
    draw_round_rect(c, x, y_bottom, w, h, stroke=colors.black, fill=None, radius=4)
    c.setFont(cfg.font_bold, 8.2)
    c.drawString(x + 3 * mm, y_top - 5.0 * mm, "Поле для письма")

    start_y = y_top - 12.5 * mm
    end_y = y_bottom + 8.0 * mm
    if cfg.writing_line_count == 1:
        y_positions = [(start_y + end_y) / 2]
    else:
        line_gap = (start_y - end_y) / (cfg.writing_line_count - 1)
        y_positions = [start_y - i * line_gap for i in range(cfg.writing_line_count)]

    c.saveState()
    line_color = colors.HexColor("#D7D7D7")
    c.setStrokeColor(line_color)
    c.setFillColor(line_color)
    c.setLineWidth(0.35)
    c.setFont(cfg.font_regular, 6.2)
    for idx, y in enumerate(y_positions, start=1):
        c.drawRightString(x + 4.2 * mm, y - 1.0 * mm, f"{idx:02d}")
        c.line(x + 5 * mm, y, x + w - 5 * mm, y)
    c.restoreState()
    return y_positions


def draw_crop_marks(c: canvas.Canvas, cfg: LayoutConfig) -> None:
    if not cfg.draw_crop_marks:
        return
    page_w, page_h = cfg.page_size
    l = 5 * mm
    off = 4 * mm
    c.saveState()
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.4)
    for x in [off, page_w - off]:
        c.line(x, off, x, off + l)
        c.line(x, page_h - off, x, page_h - off - l)
    for y in [off, page_h - off]:
        c.line(off, y, off + l, y)
        c.line(page_w - off, y, page_w - off - l, y)
    c.restoreState()


def render_block_page(c: canvas.Canvas, block: ProtocolBlock, cfg: LayoutConfig) -> Dict[str, object]:
    page_w, page_h = cfg.page_size
    x = cfg.margin_x
    content_w = page_w - 2 * cfg.margin_x
    top = page_h - cfg.margin_y

    draw_header(c, block, cfg)
    y = top - cfg.header_h - cfg.gap
    y = draw_instruction_box(c, block, cfg, x, y, content_w)
    y, truncated = draw_source_box(c, block, cfg, x, y, content_w)

    writing_bottom = cfg.margin_y
    y_positions = draw_writing_area(c, cfg, x, y, content_w, writing_bottom)
    draw_crop_marks(c, cfg)
    c.showPage()

    return {
        "block_id": block.block_id,
        "writing_line_count": len(y_positions),
        "source_truncated": truncated,
    }


def generate_pdf(protocol: Protocol, output_pdf: Path, cfg: LayoutConfig) -> List[Dict[str, object]]:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_pdf), pagesize=cfg.page_size, pageCompression=1)
    c.setTitle(f"HI-CSG-R printable sheets - {protocol.participant_id}")
    c.setAuthor("HI-CSG-R protocol generator")
    c.setSubject("Handwriting dataset collection sheets")
    page_reports: List[Dict[str, object]] = []
    for block in protocol.blocks:
        page_reports.append(render_block_page(c, block, cfg))
    c.save()
    return page_reports


def merge_pdfs(pdf_paths: Sequence[Path], output_pdf: Path) -> None:
    if not PYPDF_AVAILABLE:
        print("WARNING: pypdf/PyPDF2 is not available; combined PDF was not created.", file=sys.stderr)
        return
    writer = PdfWriter()
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            writer.add_page(page)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as f:
        writer.write(f)


def inspect_pdf_page_count(path: Path) -> int:
    if PYPDF_AVAILABLE:
        return len(PdfReader(str(path)).pages)
    return -1


def validate_protocol(protocol: Protocol) -> List[str]:
    issues: List[str] = []
    ids = [b.block_id for b in protocol.blocks]
    if len(ids) != len(set(ids)):
        issues.append(f"{protocol.source_path.name}: duplicate BLOCK_ID")
    if not protocol.participant_id:
        issues.append(f"{protocol.source_path.name}: missing PARTICIPANT_ID")
    for b in protocol.blocks:
        if not b.text.strip():
            issues.append(f"{b.block_id}: empty text")
        if EMAIL_RE.search(b.text):
            issues.append(f"{b.block_id}: possible email detected")
        if b.lang != "mixed" and "numeric" not in b.task_type.lower() and PHONE_RE.search(b.text):
            issues.append(f"{b.block_id}: possible phone-like pattern detected")
        for attr in ["session", "page", "lang", "script", "task_type", "condition", "transcription_expected", "title"]:
            if not getattr(b, attr):
                issues.append(f"{b.block_id}: missing {attr}")
        if "manual" in b.transcription_expected.lower():
            issues.append(f"{b.block_id}: manual transcription block remains; expected exact-copy protocol")
    return issues


def page_size_from_arg(name: str) -> Tuple[float, float]:
    if name.upper() == "A4":
        return A4
    if name.upper() == "LETTER":
        return LETTER
    raise ValueError("--paper must be A4 or LETTER")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate print-ready PDFs from HI-CSG-R Markdown protocols.")
    parser.add_argument("--input", "-i", required=True, help="Protocol .md file or directory with .md protocols")
    parser.add_argument("--output", "-o", required=True, help="Output directory for generated PDFs")
    parser.add_argument("--paper", default="A4", choices=["A4", "LETTER"], help="Paper size, default: A4")
    parser.add_argument("--combined", action="store_true", help="Also create a merged all-participants PDF")
    parser.add_argument("--qr", action="store_true", help="Add QR code with block metadata to every page header")
    parser.add_argument("--crop-marks", action="store_true", help="Draw small crop marks")
    parser.add_argument("--font", default=None, help="Path to regular TTF font with Cyrillic/Latin support")
    parser.add_argument("--bold-font", default=None, help="Path to bold TTF font")
    parser.add_argument("--mono-font", default=None, help="Path to monospaced TTF font")
    parser.add_argument("--source-height-mm", type=float, default=60.0, help="Fixed source text box height in mm, default: 60")
    parser.add_argument("--lines", type=int, default=18, help="Number of ruled handwriting lines on each page, default: 18")
    parser.add_argument("--manifest", default="pdf_manifest.json", help="Manifest filename inside output directory")
    args = parser.parse_args(argv)

    if args.lines <= 0:
        raise SystemExit("--lines must be positive")
    if args.source_height_mm < 35:
        raise SystemExit("--source-height-mm should be at least 35 mm")

    regular, bold, mono = register_fonts(args.font, args.bold_font, args.mono_font)
    cfg = LayoutConfig(
        page_size=page_size_from_arg(args.paper),
        source_h=args.source_height_mm * mm,
        writing_line_count=args.lines,
        font_regular=regular,
        font_bold=bold,
        font_mono=mono,
        with_qr=bool(args.qr),
        draw_crop_marks=args.crop_marks,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)
    files = collect_protocol_files(input_path)
    if not files:
        raise SystemExit(f"No .md protocol files found in {input_path}")

    protocols = [parse_protocol(p) for p in files]
    all_issues: List[str] = []
    for protocol in protocols:
        all_issues.extend(validate_protocol(protocol))
    if all_issues:
        print("WARNING: protocol validation issues:", file=sys.stderr)
        for issue in all_issues:
            print(f"  - {issue}", file=sys.stderr)

    pdf_paths: List[Path] = []
    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "input": str(input_path),
        "output": str(output_dir),
        "paper": args.paper,
        "fixed_layout": {
            "writing_lines_per_page": cfg.writing_line_count,
            "source_height_mm": args.source_height_mm,
            "header_height_mm": cfg.header_h / mm,
            "instruction_height_mm": cfg.instruction_h / mm,
        },
        "removed_elements": ["bottom scan checkboxes", "bottom source filename footer", "bottom page counter"],
        "fonts": {"regular": regular, "bold": bold, "mono": mono},
        "protocol_count": len(protocols),
        "issues": all_issues,
        "pdfs": [],
    }

    for protocol in protocols:
        out_pdf = output_dir / f"{protocol.participant_id}_printable_sheets.pdf"
        page_reports = generate_pdf(protocol, out_pdf, cfg)
        pages = inspect_pdf_page_count(out_pdf)
        pdf_paths.append(out_pdf)
        line_counts = sorted(set(int(r["writing_line_count"]) for r in page_reports))
        truncated_blocks = [str(r["block_id"]) for r in page_reports if r["source_truncated"]]
        manifest["pdfs"].append(
            {
                "participant_id": protocol.participant_id,
                "source_md": str(protocol.source_path),
                "pdf": str(out_pdf),
                "blocks": len(protocol.blocks),
                "pages": pages,
                "writing_line_counts_detected": line_counts,
                "source_truncated_blocks": truncated_blocks,
            }
        )
        print(f"Generated {out_pdf} ({len(protocol.blocks)} sheets, lines/page={line_counts})")
        if truncated_blocks:
            print(f"WARNING: source text clipped in {len(truncated_blocks)} blocks for {protocol.participant_id}", file=sys.stderr)

    if args.combined:
        combined_path = output_dir / "ALL_participants_printable_sheets.pdf"
        merge_pdfs(pdf_paths, combined_path)
        if combined_path.exists():
            manifest["combined_pdf"] = str(combined_path)
            manifest["combined_pages"] = inspect_pdf_page_count(combined_path)
            print(f"Generated {combined_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote manifest {manifest_path}")

    # Fail only if geometry requirement is not met; protocol text warnings remain warnings.
    bad_line_counts = []
    clipped = []
    for pdf_info in manifest["pdfs"]:
        counts = pdf_info["writing_line_counts_detected"]
        if counts != [args.lines]:
            bad_line_counts.append({"participant_id": pdf_info["participant_id"], "counts": counts})
        if pdf_info["source_truncated_blocks"]:
            clipped.append({"participant_id": pdf_info["participant_id"], "blocks": pdf_info["source_truncated_blocks"]})
    if bad_line_counts:
        print("ERROR: not all pages have the requested number of writing lines.", file=sys.stderr)
        return 2
    if clipped:
        print("ERROR: some source text was clipped. Increase --source-height-mm or reduce header/instruction sizes.", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())