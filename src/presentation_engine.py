"""Gelişmiş Otomatik PowerPoint Sunum Üretim Motoru

Ders notundan profesyonel .pptx dosyası üretir:
- Koyu lacivert / vurgu mavi renk paleti
- Tablo (Table) slayt desteği
- İki sütun (Two-Column) düzeni
- Renkli Callout (vurgu) kutuları
- Her slayt için detaylı Konuşmacı Notları
- Kapak + Gündem + İçerik + Teşekkür slaytları
"""

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PPTX_DIR = BASE_DIR / "static" / "presentations"
PPTX_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Renk Paleti
# ─────────────────────────────────────────────
def _rgb(r, g, b):
    from pptx.dml.color import RGBColor
    return RGBColor(r, g, b)

PALET = {
    "lacivert":    (0x1A, 0x23, 0x56),
    "beyaz":       (0xFF, 0xFF, 0xFF),
    "acik_mavi":   (0xE8, 0xF0, 0xFE),
    "vurgu":       (0x42, 0x85, 0xF4),
    "vurgu2":      (0x34, 0xA8, 0x53),   # yeşil callout
    "vurgu3":      (0xFB, 0xBC, 0x04),   # sarı callout
    "vurgu4":      (0xEA, 0x43, 0x35),   # kırmızı callout
    "metin_koyu":  (0x20, 0x20, 0x2E),
    "metin_acik":  (0xB0, 0xC4, 0xDE),
    "tablo_bas":   (0x1A, 0x23, 0x56),
    "tablo_tek":   (0xF0, 0xF4, 0xFF),
    "tablo_cift":  (0xE8, 0xF0, 0xFE),
    "cerceve":     (0x42, 0x85, 0xF4),
}


# ─────────────────────────────────────────────
# Yardımcı: Birim Dönüşüm
# ─────────────────────────────────────────────
def _inch(n):
    from pptx.util import Inches
    return Inches(n)

def _pt(n):
    from pptx.util import Pt
    return Pt(n)

def _emu(n):
    from pptx.util import Emu
    return Emu(n)


# ─────────────────────────────────────────────
# Yardımcı: Arka Plan
# ─────────────────────────────────────────────
def _arka_plan(slide, prs, renk_tuple):
    from pptx.dml.color import RGBColor
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        0, 0, prs.slide_width, prs.slide_height,
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(*renk_tuple)
    shape.line.fill.background()
    sp = slide.shapes._spTree
    sp.remove(shape._element)
    sp.insert(2, shape._element)


def _dikdortgen(slide, renk_tuple, left, top, width, height, cerceve=False):
    from pptx.dml.color import RGBColor
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(*renk_tuple)
    if cerceve:
        shape.line.color.rgb = RGBColor(*PALET["cerceve"])
        shape.line.width = _pt(1.5)
    else:
        shape.line.fill.background()
    return shape


# ─────────────────────────────────────────────
# Yardımcı: Metin Kutusu
# ─────────────────────────────────────────────
def _metin_kutusu(slide, metin, left, top, width, height,
                  font_size=18, renk=None, kalin=False,
                  hizalama=None, word_wrap=True):
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    if renk is None:
        renk = PALET["metin_koyu"]
    if hizalama is None:
        hizalama = PP_ALIGN.LEFT

    txb = slide.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame
    tf.word_wrap = word_wrap
    p = tf.paragraphs[0]
    p.alignment = hizalama
    run = p.add_run()
    run.text = metin
    run.font.size = _pt(font_size)
    run.font.color.rgb = RGBColor(*renk)
    run.font.bold = kalin
    return txb


# ─────────────────────────────────────────────
# Yardımcı: Başlık Şeridi
# ─────────────────────────────────────────────
def _baslik_seridi(slide, prs, baslik, slayt_no="", arka_renk=None):
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    if arka_renk is None:
        arka_renk = PALET["lacivert"]

    serit = slide.shapes.add_shape(
        1, 0, 0, prs.slide_width, _inch(1.25),
    )
    serit.fill.solid()
    serit.fill.fore_color.rgb = RGBColor(*arka_renk)
    serit.line.fill.background()
    sp = slide.shapes._spTree
    sp.remove(serit._element)
    sp.insert(2, serit._element)

    # Sol vurgu çizgisi
    cubuk = slide.shapes.add_shape(
        1, 0, _inch(1.25), _inch(0.12), _inch(6.25),
    )
    cubuk.fill.solid()
    cubuk.fill.fore_color.rgb = RGBColor(*PALET["vurgu"])
    cubuk.line.fill.background()

    # Başlık metni
    _metin_kutusu(
        slide, baslik,
        _inch(0.25), _inch(0.08),
        _inch(11.5), _inch(1.05),
        font_size=26, renk=PALET["beyaz"], kalin=True,
    )
    if slayt_no:
        _metin_kutusu(
            slide, slayt_no,
            _inch(11.8), _inch(0.08),
            _inch(1.3), _inch(1.05),
            font_size=14, renk=PALET["metin_acik"],
            hizalama=__import__("pptx.enum.text", fromlist=["PP_ALIGN"]).PP_ALIGN.RIGHT,
        )


# ─────────────────────────────────────────────
# Yardımcı: Konuşmacı Notu
# ─────────────────────────────────────────────
def _konusmaci_notu(slide, metin: str) -> None:
    try:
        ns = slide.notes_slide
        tf = ns.notes_text_frame
        tf.text = metin
    except Exception:
        pass


# ─────────────────────────────────────────────
# Yardımcı: İçerik Listesi (bullet)
# ─────────────────────────────────────────────
def _icerik_listesi(slide, maddeler: List[str],
                    left, top, width, height,
                    font_size=17, renk=None):
    from pptx.dml.color import RGBColor
    if renk is None:
        renk = PALET["metin_koyu"]

    txb = slide.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame
    tf.word_wrap = True

    for i, madde in enumerate(maddeler):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_before = _pt(5)
        p.space_after = _pt(3)
        run = p.add_run()
        run.text = f"▪  {madde}"
        run.font.size = _pt(font_size)
        run.font.color.rgb = RGBColor(*renk)


# ─────────────────────────────────────────────
# Yardımcı: Callout Kutusu
# ─────────────────────────────────────────────
def _callout_kutusu(slide, metin: str, left, top, width, height,
                    renk_tuple=None, font_size=15, baslik: str = ""):
    from pptx.dml.color import RGBColor
    if renk_tuple is None:
        renk_tuple = PALET["vurgu2"]

    # Çerçeve
    cerceve = slide.shapes.add_shape(1, left, top, width, height)
    cerceve.fill.solid()
    cerceve.fill.fore_color.rgb = RGBColor(*renk_tuple)
    cerceve.fill.fore_color.rgb  # trigger
    # Şeffaf iç dolgu için: hafif tonu al
    r, g, b = renk_tuple
    cerceve.fill.fore_color.rgb = RGBColor(
        min(255, r + 180), min(255, g + 180), min(255, b + 180)
    )
    cerceve.line.color.rgb = RGBColor(*renk_tuple)
    cerceve.line.width = _pt(2)

    # Sol renkli şerit
    serit = slide.shapes.add_shape(1, left, top, _inch(0.1), height)
    serit.fill.solid()
    serit.fill.fore_color.rgb = RGBColor(*renk_tuple)
    serit.line.fill.background()

    # Metin
    ic_metin = f"{baslik}\n{metin}" if baslik else metin
    _metin_kutusu(
        slide, ic_metin,
        left + _inch(0.15), top + _inch(0.05),
        width - _inch(0.2), height - _inch(0.1),
        font_size=font_size, renk=PALET["metin_koyu"], kalin=bool(baslik),
    )


# ─────────────────────────────────────────────
# Yardımcı: Tablo Slaytı
# ─────────────────────────────────────────────
def _tablo_slayt_ekle(prs, baslik: str, satirlar: List[List[str]],
                       sutun_basliklari: List[str], slayt_no: str = "") -> Any:
    """Tablo verisi içeren slayt oluşturur ve döndürür."""
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt
    from pptx.enum.text import PP_ALIGN

    layout_bos = prs.slide_layouts[6]
    slide = prs.slides.add_slide(layout_bos)
    _arka_plan(slide, prs, PALET["acik_mavi"])
    _baslik_seridi(slide, prs, baslik, slayt_no)

    rows = len(satirlar) + 1  # +1 başlık satırı
    cols = len(sutun_basliklari)
    if rows < 2 or cols < 1:
        _konusmaci_notu(slide, f"{baslik} — Tablo verisi eksik.")
        return slide

    tbl_left = _inch(0.5)
    tbl_top = _inch(1.45)
    tbl_width = _inch(12.3)
    tbl_height = _inch(5.7)

    table = slide.shapes.add_table(rows, cols, tbl_left, tbl_top, tbl_width, tbl_height).table

    # Sütun genişlikleri eşit
    col_w = tbl_width // cols
    for ci in range(cols):
        table.columns[ci].width = col_w

    # Başlık satırı
    for ci, baslik_metin in enumerate(sutun_basliklari):
        hucre = table.cell(0, ci)
        hucre.text = baslik_metin
        hucre.fill.solid()
        hucre.fill.fore_color.rgb = RGBColor(*PALET["tablo_bas"])
        p = hucre.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.runs[0] if p.runs else p.add_run()
        run.font.bold = True
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(*PALET["beyaz"])

    # Veri satırları
    for ri, satir in enumerate(satirlar):
        arka = PALET["tablo_tek"] if ri % 2 == 0 else PALET["tablo_cift"]
        for ci, deger in enumerate(satir[:cols]):
            hucre = table.cell(ri + 1, ci)
            hucre.text = str(deger)
            hucre.fill.solid()
            hucre.fill.fore_color.rgb = RGBColor(*arka)
            p = hucre.text_frame.paragraphs[0]
            if p.runs:
                p.runs[0].font.size = Pt(12)
                p.runs[0].font.color.rgb = RGBColor(*PALET["metin_koyu"])

    not_metni = (
        f"{baslik} tablosu — {len(satirlar)} satır, {cols} sütun.\n"
        + "  |  ".join(sutun_basliklari)
        + f"\n\nSatır sayısı: {len(satirlar)}"
    )
    _konusmaci_notu(slide, not_metni)
    return slide


# ─────────────────────────────────────────────
# Yardımcı: İki Sütun Slaytı
# ─────────────────────────────────────────────
def _iki_sutun_slayt_ekle(prs, baslik: str,
                            sol_maddeler: List[str],
                            sag_maddeler: List[str],
                            sol_baslik: str = "Sol Sütun",
                            sag_baslik: str = "Sağ Sütun",
                            slayt_no: str = "") -> Any:
    """İkili sütun düzeninde slayt oluşturur."""
    from pptx.dml.color import RGBColor
    layout_bos = prs.slide_layouts[6]
    slide = prs.slides.add_slide(layout_bos)
    _arka_plan(slide, prs, PALET["acik_mavi"])
    _baslik_seridi(slide, prs, baslik, slayt_no)

    sutun_gen = _inch(5.9)
    icerik_top = _inch(1.5)
    icerik_yuk = _inch(5.7)

    # Sol sütun başlığı + içerik
    _dikdortgen(slide, PALET["vurgu"],
                _inch(0.3), icerik_top,
                sutun_gen, _inch(0.45))
    _metin_kutusu(slide, sol_baslik,
                  _inch(0.4), icerik_top + _inch(0.03),
                  sutun_gen - _inch(0.1), _inch(0.4),
                  font_size=14, renk=PALET["beyaz"], kalin=True)
    _icerik_listesi(slide, sol_maddeler,
                    _inch(0.3), icerik_top + _inch(0.5),
                    sutun_gen, icerik_yuk - _inch(0.5))

    # Sağ sütun başlığı + içerik
    _dikdortgen(slide, PALET["vurgu2"],
                _inch(6.5), icerik_top,
                sutun_gen, _inch(0.45))
    _metin_kutusu(slide, sag_baslik,
                  _inch(6.6), icerik_top + _inch(0.03),
                  sutun_gen - _inch(0.1), _inch(0.4),
                  font_size=14, renk=PALET["beyaz"], kalin=True)
    _icerik_listesi(slide, sag_maddeler,
                    _inch(6.5), icerik_top + _inch(0.5),
                    sutun_gen, icerik_yuk - _inch(0.5))

    # Dikey ayraç
    ayrac = slide.shapes.add_shape(
        1, _inch(6.3), icerik_top, _inch(0.04), icerik_yuk
    )
    ayrac.fill.solid()
    from pptx.dml.color import RGBColor
    ayrac.fill.fore_color.rgb = RGBColor(*PALET["vurgu"])
    ayrac.line.fill.background()

    _konusmaci_notu(slide, (
        f"{baslik}\n\n"
        f"SOL — {sol_baslik}:\n" + "\n".join(f"  • {m}" for m in sol_maddeler) + "\n\n"
        f"SAĞ — {sag_baslik}:\n" + "\n".join(f"  • {m}" for m in sag_maddeler)
    ))
    return slide


# ─────────────────────────────────────────────
# Metin → Slayt Yapısına Dönüştürme
# ─────────────────────────────────────────────
def _metni_slaytlara_donustur(text: str, max_slides: int = 8) -> List[Dict[str, Any]]:
    """Metni başlık + madde listesi slayt yapısına dönüştürür."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    slides: List[Dict[str, Any]] = []
    current_title = ""
    current_bullets: List[str] = []

    def _kaydet(title: str, bullets: List[str]) -> None:
        if not title and not bullets:
            return
        slides.append({
            "title": title or "Ders Notu",
            "bullets": bullets[:7],
            "notes": f"{title}\n\n" + "\n".join(f"• {b}" for b in bullets),
        })

    for line in lines:
        is_heading = (
            line.startswith("#")
            or re.match(r"^\*{0,2}[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]{2,50}:?\*{0,2}$", line)
            or (len(line) < 60 and line == line.upper() and len(line) > 3)
        )
        if is_heading:
            _kaydet(current_title, current_bullets)
            current_title = re.sub(r"^#+\s*|\*+", "", line).strip().rstrip(":")
            current_bullets = []
        else:
            clean = re.sub(r"^[\-\*•►▪▶→]\s*", "", line).strip()
            if clean:
                current_bullets.append(clean)

    _kaydet(current_title, current_bullets)

    if not slides:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
        chunk = max(1, len(sentences) // max_slides)
        for i in range(0, len(sentences), chunk):
            group = sentences[i: i + chunk]
            slides.append({
                "title": f"Konu {len(slides) + 1}",
                "bullets": group[:7],
                "notes": " ".join(group),
            })

    return slides[:max_slides]


# ─────────────────────────────────────────────
# Ana Fonksiyon
# ─────────────────────────────────────────────
def generate_presentation(
    text: str,
    title: str = "Ders Notu Sunumu",
    subtitle: str = "Otomatik Oluşturuldu",
    max_slides: int = 8,
    filename: Optional[str] = None,
    tablo_verisi: Optional[List[Dict[str, Any]]] = None,
    iki_sutun_verisi: Optional[List[Dict[str, Any]]] = None,
    callout_listesi: Optional[List[Dict[str, str]]] = None,
) -> Path:
    """Verilen ders notu / özet metninden gelişmiş .pptx dosyası üretir.

    Args:
        text: Slaytlara dönüştürülecek metin.
        title: Kapak slaytı başlığı.
        subtitle: Kapak alt başlığı.
        max_slides: Maksimum içerik slaytı sayısı.
        filename: Çıktı dosya adı (opsiyonel).
        tablo_verisi: [{"baslik": "...", "sutunlar": [...], "satirlar": [[...]]}]
        iki_sutun_verisi: [{"baslik":"...", "sol":["..."], "sag":["..."], ...}]
        callout_listesi: [{"metin":"...", "tip":"bilgi"|"uyari"|"ipucu"|"hata"}]

    Returns:
        Oluşturulan .pptx dosyasının Path nesnesi.
    """
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt, Emu
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
    except ImportError as exc:
        raise ImportError(
            "python-pptx yüklü değil. 'pip install python-pptx' komutunu çalıştırın."
        ) from exc

    if not text.strip():
        raise ValueError("Sunum oluşturmak için metin boş olamaz.")

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)

    layout_bos = prs.slide_layouts[6]

    # ── 1. KAPAK SLAYTI ───────────────────────────────────────
    slide_kapak = prs.slides.add_slide(layout_bos)
    _arka_plan(slide_kapak, prs, PALET["lacivert"])

    # Gradient etkisi için üst şerit
    serit_ust = slide_kapak.shapes.add_shape(
        1, 0, 0, prs.slide_width, _inch(0.08)
    )
    serit_ust.fill.solid()
    serit_ust.fill.fore_color.rgb = RGBColor(*PALET["vurgu"])
    serit_ust.line.fill.background()

    # Alt yatay çizgi
    _dikdortgen(slide_kapak, PALET["vurgu"],
                _inch(0.8), _inch(2.9), _inch(11.6), _inch(0.06))

    # Başlık
    _metin_kutusu(
        slide_kapak, title,
        _inch(0.8), _inch(1.0), _inch(11.5), _inch(1.7),
        font_size=42, renk=PALET["beyaz"], kalin=True,
        hizalama=PP_ALIGN.LEFT,
    )
    # Alt başlık
    _metin_kutusu(
        slide_kapak, subtitle,
        _inch(0.8), _inch(3.1), _inch(11.5), _inch(0.9),
        font_size=20, renk=PALET["metin_acik"],
        hizalama=PP_ALIGN.LEFT,
    )
    # Logo benzeri daire
    daire = slide_kapak.shapes.add_shape(
        9,  # Oval
        _inch(11.1), _inch(4.8), _inch(1.5), _inch(1.5),
    )
    daire.fill.solid()
    daire.fill.fore_color.rgb = RGBColor(*PALET["vurgu"])
    daire.line.fill.background()

    _konusmaci_notu(slide_kapak, (
        f"SUNUM BAŞLIĞI: {title}\n"
        f"ALT BAŞLIK: {subtitle}\n\n"
        "Bu sunum Ders Notu Özetleyici sistemi tarafından otomatik oluşturulmuştur.\n"
        "Lütfen slaytları sunuma başlamadan önce gözden geçiriniz."
    ))

    # ── 2. GÜNDEM SLAYTI ─────────────────────────────────────
    slide_yapilari = _metni_slaytlara_donustur(text, max_slides=max_slides)

    slide_gundem = prs.slides.add_slide(layout_bos)
    _arka_plan(slide_gundem, prs, PALET["acik_mavi"])
    _baslik_seridi(slide_gundem, prs, "📋  Gündem", "00")

    gundem_maddeler = [s["title"] for s in slide_yapilari]
    if tablo_verisi:
        for tv in tablo_verisi:
            gundem_maddeler.append(f"📊 Tablo: {tv.get('baslik', 'Tablo')}")
    if iki_sutun_verisi:
        for isv in iki_sutun_verisi:
            gundem_maddeler.append(f"⚖️ Karşılaştırma: {isv.get('baslik', '')}")

    _icerik_listesi(
        slide_gundem, gundem_maddeler[:10],
        _inch(1.0), _inch(1.5), _inch(11.0), _inch(5.7),
        font_size=18,
    )
    _konusmaci_notu(slide_gundem, "GÜNDEM\n\nBu slayt sunumun genel yapısını özetlemektedir.\n"
                    + "\n".join(f"  {i+1}. {m}" for i, m in enumerate(gundem_maddeler)))

    # ── 3. İÇERİK SLAYTLARI ──────────────────────────────────
    toplam = len(slide_yapilari)
    for sno, slayt_verisi in enumerate(slide_yapilari, 1):
        slide = prs.slides.add_slide(layout_bos)
        _arka_plan(slide, prs, PALET["acik_mavi"])
        _baslik_seridi(slide, prs, slayt_verisi["title"], f"{sno:02d}/{toplam:02d}")

        bullets = slayt_verisi["bullets"]

        # Eğer callout_listesi bu slayta denk geliyorsa callout kutusu ekle
        slayt_callout = None
        if callout_listesi and sno <= len(callout_listesi):
            slayt_callout = callout_listesi[sno - 1]

        if slayt_callout:
            # Maddeleri daraltılmış alanda göster
            _icerik_listesi(
                slide, bullets,
                _inch(0.3), _inch(1.5), _inch(8.0), _inch(5.6),
            )
            tip = slayt_callout.get("tip", "bilgi").lower()
            renk_map = {
                "bilgi": PALET["vurgu"],
                "ipucu": PALET["vurgu2"],
                "uyari": PALET["vurgu3"],
                "hata":  PALET["vurgu4"],
            }
            callout_renk = renk_map.get(tip, PALET["vurgu"])
            baslik_map = {
                "bilgi": "💡 Bilgi",
                "ipucu": "✅ İpucu",
                "uyari": "⚠️ Uyarı",
                "hata":  "❌ Dikkat",
            }
            _callout_kutusu(
                slide,
                metin=slayt_callout.get("metin", ""),
                left=_inch(8.5), top=_inch(1.6),
                width=_inch(4.6), height=_inch(2.5),
                renk_tuple=callout_renk,
                baslik=baslik_map.get(tip, "📌 Not"),
            )
        else:
            _icerik_listesi(
                slide, bullets,
                _inch(0.3), _inch(1.45), _inch(12.8), _inch(5.8),
            )

        _konusmaci_notu(slide, slayt_verisi.get("notes", slayt_verisi["title"]))

    # ── 4. TABLO SLAYTLARI ───────────────────────────────────
    if tablo_verisi:
        for ti, tv in enumerate(tablo_verisi, 1):
            _tablo_slayt_ekle(
                prs,
                baslik=tv.get("baslik", f"Tablo {ti}"),
                satirlar=tv.get("satirlar", []),
                sutun_basliklari=tv.get("sutunlar", []),
                slayt_no=f"T{ti}",
            )

    # ── 5. İKİ SÜTUN SLAYTLARI ───────────────────────────────
    if iki_sutun_verisi:
        for ki, isv in enumerate(iki_sutun_verisi, 1):
            _iki_sutun_slayt_ekle(
                prs,
                baslik=isv.get("baslik", f"Karşılaştırma {ki}"),
                sol_maddeler=isv.get("sol", []),
                sag_maddeler=isv.get("sag", []),
                sol_baslik=isv.get("sol_baslik", "Sol"),
                sag_baslik=isv.get("sag_baslik", "Sağ"),
                slayt_no=f"K{ki}",
            )

    # ── 6. TEŞEKKÜR SLAYTI ────────────────────────────────────
    slide_son = prs.slides.add_slide(layout_bos)
    _arka_plan(slide_son, prs, PALET["lacivert"])
    _dikdortgen(slide_son, PALET["vurgu"],
                _inch(2.5), _inch(3.45), _inch(8.3), _inch(0.07))
    _metin_kutusu(
        slide_son, "Teşekkürler",
        _inch(1.5), _inch(2.1), _inch(10.0), _inch(1.6),
        font_size=52, renk=PALET["beyaz"], kalin=True,
        hizalama=PP_ALIGN.CENTER,
    )
    _metin_kutusu(
        slide_son,
        "Bu sunum Ders Notu Özetleyici tarafından otomatik olarak oluşturulmuştur.",
        _inch(1.5), _inch(3.7), _inch(10.0), _inch(0.8),
        font_size=15, renk=PALET["metin_acik"],
        hizalama=PP_ALIGN.CENTER,
    )
    _konusmaci_notu(slide_son,
                    f"Sunum sonu. Toplam slayt: {len(prs.slides)}\nBaşlık: {title}")

    # ── KAYDET ────────────────────────────────────────────────
    if not filename:
        filename = f"sunum_{uuid.uuid4().hex[:8]}.pptx"
    elif not filename.endswith(".pptx"):
        filename += ".pptx"

    output_path = PPTX_DIR / filename
    prs.save(str(output_path))
    logger.info(
        "PowerPoint oluşturuldu: %s | %d slayt | %.1f KB",
        output_path.name, len(prs.slides), output_path.stat().st_size / 1024,
    )
    return output_path
