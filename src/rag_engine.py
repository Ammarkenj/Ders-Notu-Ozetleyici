import io
import logging
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import chromadb
from chromadb.config import Settings
import pypdf

logger = logging.getLogger(__name__)

# ChromaDB veritabanı dizini
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "processed" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "lecture_notes"

_CHROMA_CLIENT = None
_COLLECTION = None


def get_chroma_collection():
    """ChromaDB istemcisini ve ders notları koleksiyonunu döndürür.

    Koleksiyon silinmişse veya referans eskiyse otomatik olarak
    get_or_create_collection ile yeniden oluşturur ve döndürür.
    """
    global _CHROMA_CLIENT, _COLLECTION
    try:
        if _CHROMA_CLIENT is None:
            _CHROMA_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # Eğer önceden alınmış bir koleksiyon varsa, hâlâ geçerli olduğunu doğrula
        if _COLLECTION is not None:
            try:
                # count() çağrısı koleksiyonun ChromaDB'de mevcut olup olmadığını test eder
                _COLLECTION.count()
                return _COLLECTION
            except Exception as check_err:
                logger.warning("Mevcut koleksiyon referansı geçersiz (%s), yeniden oluşturuluyor.", check_err)
                _COLLECTION = None

        _COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Ders notu parçaları ve gömmeleri"},
        )
        return _COLLECTION
    except Exception as exc:
        logger.warning("get_chroma_collection hatası, istemci sıfırlanıyor: %s", exc)
        try:
            _CHROMA_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))
            _COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Ders notu parçaları ve gömmeleri"},
            )
            return _COLLECTION
        except Exception as final_exc:
            logger.error("ChromaDB koleksiyonu oluşturulamadı: %s", final_exc)
            raise


def split_text_into_chunks(
    text: str, chunk_size: int = 400, overlap: int = 50
) -> List[str]:
    """Metni anlam bütünlüğünü koruyacak şekilde örtüşmeli parçalara (chunk) böler."""
    cleaned = text.strip()
    if not cleaned:
        return []

    # Paragrafları veya cümleleri tespit et
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]

    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}".strip() if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Paragraf tek başına chunk_size'dan büyükse cümle bazlı böl
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?]) +", para)
                sub_chunk = ""
                for sent in sentences:
                    if len(sub_chunk) + len(sent) <= chunk_size:
                        sub_chunk = f"{sub_chunk} {sent}".strip() if sub_chunk else sent
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                        sub_chunk = sent
                if sub_chunk:
                    current_chunk = sub_chunk
                else:
                    current_chunk = ""
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def add_text_to_rag(
    text: str,
    source: str = "manuel_not",
    metadata: Optional[Dict[str, Any]] = None,
    chunk_size: int = 400,
    overlap: int = 50,
) -> int:
    """Metni parçalayarak ChromaDB koleksiyonuna ekler.

    Returns:
        Eklenen parça sayısı (int).
    """
    collection = get_chroma_collection()
    chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return 0

    base_meta = metadata or {}
    base_meta["source"] = source

    u_id_prefix = uuid.uuid4().hex[:10]
    ids = [f"doc_{u_id_prefix}_{i}" for i in range(len(chunks))]
    metadatas = [{**base_meta, "chunk_index": i} for i in range(len(chunks))]

    try:
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )
    except Exception as add_err:
        logger.warning("ChromaDB add hatası (%s), koleksiyon yeniden oluşturulup tekrar deneniyor...", add_err)
        global _COLLECTION
        _COLLECTION = None
        collection = get_chroma_collection()
        collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )

    logger.info("ChromaDB'ye %d parça başarıyla eklendi (Kaynak: %s)", len(chunks), source)
    return len(chunks)


def add_pdf_to_rag(
    pdf_input: Union[str, Path, bytes],
    source_name: str = "pdf_belgesi",
) -> int:
    """PDF belgesinden metin ayıklar ve ChromaDB'ye ekler.

    Normal metin tabanlı PDF'lerden doğrudan metin çıkarır.
    Sayfada metin katmanı bulunamazsa ve görsel varsa (taranmış ders notları),
    OCR motoru ile görsellerden metin çıkararak indekslemeyi tamamlar.
    """
    try:
        if isinstance(pdf_input, (str, Path)):
            reader = pypdf.PdfReader(str(pdf_input), strict=False)
        elif isinstance(pdf_input, bytes):
            reader = pypdf.PdfReader(io.BytesIO(pdf_input), strict=False)
        else:
            raise TypeError("Desteklenmeyen PDF girdi türü.")
    except Exception as init_err:
        logger.warning("PDF okuyucu başlatılamadı (%s): %s", source_name, init_err)
        return 0

    # Şifreli PDF koruması (standart boş şifreli PDF'leri açar)
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception as e:
            logger.warning("PDF şifresi çözülemedi: %s", e)

    total_chunks = 0
    try:
        pages_list = list(reader.pages)
    except Exception as pages_err:
        logger.warning("PDF sayfaları listelenemedi (%s): %s", source_name, pages_err)
        return 0

    for page_idx, page in enumerate(pages_list):
        page_text = ""
        try:
            page_text = page.extract_text() or ""
        except Exception as extract_err:
            logger.warning("Sayfa %d metin çıkarma hatası: %s", page_idx + 1, extract_err)

        # Eğer sayfadan doğrudan metin çıkarılamadıysa ve görsel varsa (taranmış notlar), OCR yap
        if not page_text.strip() and hasattr(page, "images"):
            try:
                from src.ocr_engine import extract_text_from_image
                img_texts = []
                for img in page.images:
                    try:
                        extracted = extract_text_from_image(img.data)
                        if extracted.strip():
                            img_texts.append(extracted.strip())
                    except Exception as ocr_err:
                        logger.warning("Sayfa %d görsel OCR hatası: %s", page_idx + 1, ocr_err)
                if img_texts:
                    page_text = "\n".join(img_texts)
            except Exception as ocr_module_err:
                logger.warning("OCR desteği çalıştırılamadı: %s", ocr_module_err)

        if page_text.strip():
            count = add_text_to_rag(
                text=page_text,
                source=source_name,
                metadata={"page": page_idx + 1, "type": "pdf"},
            )
            total_chunks += count

    return total_chunks


def add_docx_to_rag(
    docx_input: Union[str, Path, bytes],
    source_name: str = "word_belgesi",
) -> int:
    """Word (.docx) belgesinden paragraf metinlerini ayıklar ve ChromaDB'ye ekler."""
    try:
        from docx import Document  # python-docx
    except ImportError as exc:
        raise ImportError("python-docx yüklü değil. 'pip install python-docx' komutunu çalıştırın.") from exc

    if isinstance(docx_input, (str, Path)):
        doc = Document(str(docx_input))
    elif isinstance(docx_input, bytes):
        doc = Document(io.BytesIO(docx_input))
    else:
        raise TypeError("Desteklenmeyen DOCX girdi türü.")

    total_chunks = 0
    current_block: List[str] = []
    block_char_count = 0
    BLOCK_LIMIT = 1200  # Paragrafları mantıklı bloklara böl

    def _flush_block(block: List[str]) -> int:
        combined = "\n".join(block).strip()
        if combined:
            return add_text_to_rag(text=combined, source=source_name, metadata={"type": "docx"})
        return 0

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if block_char_count + len(text) > BLOCK_LIMIT:
            total_chunks += _flush_block(current_block)
            current_block = [text]
            block_char_count = len(text)
        else:
            current_block.append(text)
            block_char_count += len(text)

    total_chunks += _flush_block(current_block)
    logger.info("DOCX -> ChromaDB: %d parça eklendi (Kaynak: %s)", total_chunks, source_name)
    return total_chunks


def add_excel_to_rag(
    excel_input: Union[str, Path, bytes],
    source_name: str = "excel_belgesi",
    dosya_uzantisi: str = ".xlsx",
) -> int:
    """Excel (.xlsx) veya CSV (.csv) tablosunu okuyup ChromaDB'ye ekler.

    Her satırı bir metin belgesi olarak biçimlendirip ChromaDB koleksiyonuna kaydeder.
    Ayrıca sütun başlıklarından bağlamsal açıklama oluşturur.

    Args:
        excel_input: Dosya yolu (str/Path) veya ham bayt (bytes).
        source_name:  ChromaDB'deki kaynak etiketi.
        dosya_uzantisi: '.xlsx' veya '.csv' — bytes girildiğinde hangi format olduğunu belirtir.

    Returns:
        ChromaDB'ye eklenen parça sayısı.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas yüklü değil. 'pip install pandas openpyxl' çalıştırın.") from exc

    # Dosyayı oku
    try:
        if isinstance(excel_input, (str, Path)):
            uzanti = Path(excel_input).suffix.lower()
            if uzanti == ".csv":
                df = pd.read_csv(str(excel_input), encoding="utf-8", on_bad_lines="skip")
            else:
                df = pd.read_excel(str(excel_input))
        elif isinstance(excel_input, bytes):
            import io as _io
            if dosya_uzantisi.lower() == ".csv":
                df = pd.read_csv(_io.BytesIO(excel_input), encoding="utf-8", on_bad_lines="skip")
            else:
                df = pd.read_excel(_io.BytesIO(excel_input))
        else:
            raise TypeError("Desteklenmeyen Excel girdi türü.")
    except Exception as e:
        raise ValueError(f"Excel/CSV okunurken hata: {e}") from e

    if df.empty:
        logger.warning("Excel/CSV dosyası boş: %s", source_name)
        return 0

    # Sütun başlıklarından bağlamsal açıklama
    sutun_basliklari = list(df.columns.astype(str))
    baslik_cumle = "Bu tablo şu sütunları içermektedir: " + ", ".join(sutun_basliklari) + "."
    total_chunks = add_text_to_rag(
        baslik_cumle,
        source=source_name,
        metadata={"type": "excel_header", "columns": ", ".join(sutun_basliklari)},
    )

    # Satırları metin olarak ekle (max 500 satır)
    df = df.fillna("").head(500)
    satirlar_metni: List[str] = []
    for _, row in df.iterrows():
        parcalar = [f"{col}: {str(val).strip()}" for col, val in row.items() if str(val).strip()]
        if parcalar:
            satirlar_metni.append(" | ".join(parcalar))

    # 10 satırı bir metin bloğuna birleştir
    BLOK = 10
    for i in range(0, len(satirlar_metni), BLOK):
        blok_metin = "\n".join(satirlar_metni[i: i + BLOK])
        total_chunks += add_text_to_rag(
            blok_metin,
            source=source_name,
            metadata={"type": "excel_data", "rows": f"{i+1}-{min(i+BLOK, len(satirlar_metni))}"},
        )

    logger.info("Excel/CSV -> ChromaDB: %d parça eklendi (Kaynak: %s)", total_chunks, source_name)
    return total_chunks


def query_rag_notes(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """Kullanıcı sorusuyla en ilgili ders notu parçalarını getirir."""
    global _COLLECTION
    try:
        collection = get_chroma_collection()
        total_docs = collection.count()
    except Exception as e:
        logger.warning("query_rag_notes koleksiyon erişim hatası, yenileniyor: %s", e)
        _COLLECTION = None
        collection = get_chroma_collection()
        total_docs = collection.count()

    if total_docs == 0:
        return []

    limit = min(n_results, total_docs)
    try:
        results = collection.query(
            query_texts=[query],
            n_results=limit,
        )
    except Exception as q_err:
        logger.warning("collection.query hatası, yenileniyor: %s", q_err)
        _COLLECTION = None
        collection = get_chroma_collection()
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
        )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if "distances" in results else []

    formatted_results = []
    for idx, doc in enumerate(documents):
        formatted_results.append({
            "text": doc,
            "metadata": metadatas[idx] if idx < len(metadatas) else {},
            "distance": distances[idx] if idx < len(distances) else None,
        })
    return formatted_results


DURAK_KELIMELER = {
    "ve", "veya", "ile", "bir", "bu", "da", "de", "ki", "ne", "mi", "mu", "mü",
    "için", "olan", "gibi", "ise", "ama", "fakat", "ancak", "çok", "en", "her",
    "olarak", "aynı", "daha", "kadar", "var", "yok", "şu", "o", "biz", "siz",
    "onlar", "tarafından", "şekilde", "üzerine", "göre", "nedeniyle", "bunu",
    "buna", "bunun", "yani", "hem", "ya", "the", "of", "in", "is", "and", "to", "a",
    "bölüm", "ünite", "sayfa", "kaynak", "ödev", "hafta", "ders"
}

FIIL_EKLERI = ("dır", "dir", "dur", "dür", "tir", "tır", "tur", "tür", "sıdır", "sidir", "sudur", "südür")
GECERSIZ_KELIMELER = {
    "sağlar", "yapar", "eder", "olur", "gelir", "gerekir", "bulunur", "verir", "alır",
    "sağlamak", "yapmak", "etmek", "çalışır", "sağlayan", "yapan", "eden", "olan",
    "içeren", "tanımlanan", "katmandır", "modelidir", "yöntemidir", "bağlanmasıdır",
    "protokolüdür", "sürecidir", "yapısıdır"
}


def _clean_flowchart_label(text: str, max_words: int = 4) -> str:
    """Mermaid flowchart düğüm metnini temizler; parantez, köşeli ayraç ve çift tırnakları ayıklar."""
    cleaned = re.sub(r'["`#*•\-–,/\\;()\[\]{}]', " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words])
    return cleaned.strip() or "Kavram"


def _is_valid_topic(term: str) -> bool:
    """Terimin anlamlı bir ana konu başlığı olup olmadığını doğrular."""
    if not term or len(term) < 3:
        return False
    t_lower = term.lower().strip()
    words = t_lower.split()
    if not words:
        return False

    # Başta veya sonda olmaması gereken kelimeler
    bad_starters = {"bu", "şu", "o", "bunlar", "şunlar", "tablo", "şekil", "örnek", "resim", "sayfa", "bölüm", "ünite", "kaynak", "ödev", "hafta", "not", "ders"}
    if words[0] in bad_starters:
        return False

    bad_words = {"verirken", "sıralandığı", "olarak", "gibi", "ile", "için", "olan", "veya", "yaparken", "ederek", "olduğu", "görüldüğü", "bulunan", "içeren"}
    if any(w in bad_words for w in words):
        return False

    if t_lower in DURAK_KELIMELER or t_lower in GECERSIZ_KELIMELER:
        return False

    bad_suffixes = ("diği", "dığı", "duğu", "düğü", "ken", "erek", "arak", "mesi", "ması", "mek", "mak", "miş", "mış", "muş", "müş", "ecek", "acak", "iyor", "ıyor", "uyor", "üyor", "malı", "meli", "dır", "dir", "dur", "dür", "tir", "tır", "tur", "tür", "sıdır", "sidir")
    if any(t_lower.endswith(s) for s in bad_suffixes) and len(t_lower) <= 12:
        return False

    return True


def _extract_document_concepts(combined_text: str):
    """Belgeden tanımlanmış kavramları ve açıklamalarını ayıklar."""
    concepts = []
    seen = set()

    # 1. "Kavram: Tanım" veya "Kavram - Tanım" kalıpları (Parantez içi yabancı dil eklerini temizler)
    defs = re.findall(r"^([A-ZÇĞİÖŞÜ0-9][\w\sçğıöşü\'\(\)\-]{2,50})\s*[:\-–]\s*(.+)$", combined_text, re.MULTILINE)
    for term, definition in defs:
        t_clean = re.sub(r"\(.*?\)", "", term).strip()
        t_clean = re.sub(r"^[0-9]+[\.\-]\s*", "", t_clean).strip()
        d_clean = definition.strip().rstrip(".- ") + "."
        if _is_valid_topic(t_clean) and t_clean.lower() not in seen and len(d_clean) > 15:
            seen.add(t_clean.lower())
            concepts.append((t_clean, d_clean))

    # 2. "X ... olarak tanımlanır / denir / adlandırılır" cümleleri
    cumleler = re.split(r"(?<=[.!?])\s+", combined_text)
    for c in cumleler:
        c = c.strip()
        if len(c) < 25 or len(c) > 300:
            continue
        m = re.match(
            r"^([A-ZÇĞİÖŞÜ0-9][\w\sçğıöşü\'\(\)\-]{2,40})\s*[,;]?\s*(.*(?:olarak tanımlanır|olarak adlandırılır|denir|ifade edilir|anlamına gelir|sürecidir|yapısıdır|modelidir).*)",
            c,
            re.IGNORECASE,
        )
        if m:
            raw_t = m.group(1).strip()
            t_clean = re.sub(r"\(.*?\)", "", raw_t).strip()
            t_clean = re.sub(r"^[0-9]+[\.\-]\s*", "", t_clean).strip()
            d_clean = c.rstrip(".- ") + "."
            if _is_valid_topic(t_clean) and t_clean.lower() not in seen:
                seen.add(t_clean.lower())
                concepts.append((t_clean, d_clean))

    return concepts


def _generate_mindmap_rag_answer(retrieved_items: List[Dict[str, Any]], collection) -> str:
    """Kavram Akış Şeması (Mermaid flowchart LR):
    Merkezde belgenin adı, etrafında renkli kutularla ana ve alt konuları bağlar.
    """
    texts = [item["text"] for item in retrieved_items if item.get("text")]
    try:
        extra_data = collection.get(limit=25)
        docs = extra_data.get("documents", []) or []
        for d in docs:
            if d and d not in texts:
                texts.append(d)
    except Exception:
        pass

    combined_text = "\n".join(texts)

    # 1. Merkez Düğüm: Yüklenen belgenin kaynak adını bul
    center_name = "Ders Notları"
    for item in retrieved_items:
        src = item.get("metadata", {}).get("source", "")
        if src and not src.lower().startswith("kaynak"):
            clean_src = re.sub(r"\.[a-zA-Z0-9]+$", "", src)
            clean_src = re.sub(r"^[0-9]+[\.\-_]\s*", "", clean_src)
            clean_src = clean_src.replace("_", " ").replace("-", " ").strip()
            if clean_src and len(clean_src) > 2 and clean_src.lower() not in ["ders notu", "pdf belgesi"]:
                center_name = clean_src
                break

    center_title = _clean_flowchart_label(center_name, max_words=4)
    center_node_label = f"📘 {center_title}"

    # 2. Tanımlı kavramları ve ana başlıkları ayıkla
    extracted_concepts = _extract_document_concepts(combined_text)

    ana_dallar = []
    seen_dallar = set()
    # Merkez başlığı dallara tekrar eklememek için işaretle
    seen_dallar.add(center_title.lower())
    for word in center_title.lower().split():
        if len(word) > 4:
            seen_dallar.add(word)

    # Öncelik 1: Belgedeki gerçek kavram ve terim başlıkları
    for term, definition in extracted_concepts:
        clean_t = _clean_flowchart_label(term, max_words=3)
        if (
            _is_valid_topic(clean_t)
            and clean_t.lower() not in seen_dallar
            and clean_t.lower() != center_title.lower()
            and len(ana_dallar) < 5
        ):
            seen_dallar.add(clean_t.lower())
            ana_dallar.append((clean_t, definition))

    # Öncelik 2: Büyük harfli teknik terim öbekleri (örn: İkili Sayı Sistemi, 2'ye Tümleme)
    if len(ana_dallar) < 4:
        phrases = re.findall(r"\b([A-ZÇĞİÖŞÜ0-9][\wçğıöşü\']+(?:\s+[A-ZÇĞİÖŞÜ0-9][\wçğıöşü\']+){1,2})\b", combined_text)
        for ph in phrases:
            clean_ph = _clean_flowchart_label(ph, max_words=3)
            if _is_valid_topic(clean_ph) and clean_ph.lower() not in seen_dallar and len(ana_dallar) < 5:
                seen_dallar.add(clean_ph.lower())
                ana_dallar.append((clean_ph, ""))

    # Öncelik 3: Frekansı yüksek anlamlı anahtar kelimeler
    if len(ana_dallar) < 4:
        kelimeler = re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]{4,}", combined_text)
        sayac = Counter(k.lower() for k in kelimeler if _is_valid_topic(k))
        for kw, _ in sayac.most_common(15):
            clean_kw = _clean_flowchart_label(kw.capitalize(), max_words=2)
            if _is_valid_topic(clean_kw) and clean_kw.lower() not in seen_dallar and len(ana_dallar) < 5:
                seen_dallar.add(clean_kw.lower())
                ana_dallar.append((clean_kw, ""))

    # Yedek güvenli başlıklar
    yedekler = ["Temel Kavramlar", "Mimari ve Yapı", "Temel İlkeler", "Uygulama Alanları", "Analiz ve Yöntemler"]
    for yd in yedekler:
        if len(ana_dallar) < 4 and yd.lower() not in seen_dallar:
            seen_dallar.add(yd.lower())
            ana_dallar.append((yd, ""))

    # 3. Mermaid flowchart LR diyagramı inşa et
    flowchart_lines = [
        "flowchart LR",
        f'    ROOT(["{center_node_label}"])\n'
    ]

    main_node_ids = []
    sub_node_ids = []
    cumleler = re.split(r"(?<=[.!?])\s+", combined_text)

    for i, (dal_label, context_desc) in enumerate(ana_dallar[:5], 1):
        m_id = f"M{i}"
        main_node_ids.append(m_id)
        flowchart_lines.append(f'    {m_id}(["{dal_label}"])')
        flowchart_lines.append(f"    ROOT --> {m_id}")

        alt_kavramlar = []
        seen_alt = set()

        if context_desc:
            desc_words = [
                w for w in re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü0-9][\wçğıöşü\']{2,}", context_desc)
                if _is_valid_topic(w) and w.lower() not in dal_label.lower() and w.lower() not in center_name.lower()
            ]
            for dw in desc_words:
                clean_dw = _clean_flowchart_label(dw.capitalize(), max_words=2)
                if clean_dw.lower() not in seen_alt and clean_dw.lower() not in dal_label.lower() and clean_dw.lower() not in center_name.lower():
                    seen_alt.add(clean_dw.lower())
                    alt_kavramlar.append(clean_dw)
                    if len(alt_kavramlar) >= 3:
                        break

        if len(alt_kavramlar) < 2:
            key_words = set(re.findall(r"\w+", dal_label.lower()))
            for c in cumleler:
                if any(kw in c.lower() for kw in key_words):
                    c_terms = [
                        w for w in re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü0-9][\wçğıöşü\']{3,}", c)
                        if _is_valid_topic(w) and w.lower() not in dal_label.lower() and w.lower() not in center_name.lower()
                    ]
                    for ct in c_terms:
                        clean_ct = _clean_flowchart_label(ct.capitalize(), max_words=2)
                        if clean_ct.lower() not in seen_alt and clean_ct.lower() not in dal_label.lower() and clean_ct.lower() not in center_name.lower():
                            seen_alt.add(clean_ct.lower())
                            alt_kavramlar.append(clean_ct)
                            if len(alt_kavramlar) >= 3:
                                break
                if len(alt_kavramlar) >= 3:
                    break

        if len(alt_kavramlar) < 2:
            default_subs = ["Temel Özellikler", "Çalışma Prensibi", "Uygulama Alanı"]
            for ds in default_subs:
                if len(alt_kavramlar) < 3 and ds.lower() not in seen_alt:
                    alt_kavramlar.append(ds)

        for j, ak in enumerate(alt_kavramlar[:3], 1):
            s_id = f"S{i}_{j}"
            sub_node_ids.append(s_id)
            flowchart_lines.append(f'    {s_id}["{ak}"]')
            flowchart_lines.append(f"    {m_id} --> {s_id}")

        flowchart_lines.append("")

    # 4. Stil Tanımları (Modern classDef)
    flowchart_lines.append("    %% Stil Tanımları")
    flowchart_lines.append("    classDef rootStyle fill:#4338ca,stroke:#6366f1,stroke-width:3px,color:#ffffff,font-weight:bold,font-size:13px;")
    flowchart_lines.append("    classDef mainStyle fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#e0e7ff,font-weight:600,font-size:12px;")
    flowchart_lines.append("    classDef subStyle fill:#0f172a,stroke:#10b981,stroke-width:1.5px,color:#f1f5f9,font-size:11px;")
    flowchart_lines.append("    class ROOT rootStyle;")
    if main_node_ids:
        flowchart_lines.append(f"    class {','.join(main_node_ids)} mainStyle;")
    if sub_node_ids:
        flowchart_lines.append(f"    class {','.join(sub_node_ids)} subStyle;")

    return "\n".join(flowchart_lines)


def _generate_glossary_rag_answer(retrieved_items: List[Dict[str, Any]]) -> str:
    """Kavram Sözlüğü Modu: Metinlerdeki önemli teknik terimleri ve tanımları ayıklayıp Markdown formatında döndürür."""
    combined_text = "\n".join(item["text"] for item in retrieved_items if item.get("text"))
    extracted_concepts = _extract_document_concepts(combined_text)

    sozluk_maddeleri = []
    seen = set()

    for term, definition in extracted_concepts:
        if term.lower() not in seen and len(sozluk_maddeleri) < 10:
            seen.add(term.lower())
            sozluk_maddeleri.append(f"- **{term}:** {definition}")

    # Yeterli terim çıkmadıysa anahtar kelimeleri açıklayıcı cümlelerle eşleştir
    if len(sozluk_maddeleri) < 5:
        cumleler = re.split(r"(?<=[.!?])\s+", combined_text)
        kelimeler = re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]{4,}", combined_text)
        kelime_sayac = Counter(k.lower() for k in kelimeler if _is_valid_topic(k))
        for kw, _ in kelime_sayac.most_common(15):
            if kw in seen:
                continue
            for c in cumleler:
                c_clean = c.strip()
                if kw in c_clean.lower() and len(c_clean) > 30:
                    seen.add(kw)
                    sozluk_maddeleri.append(f"- **{kw.capitalize()}:** {c_clean.rstrip('.- ') + '.'}")
                    break
            if len(sozluk_maddeleri) >= 8:
                break

    if not sozluk_maddeleri:
        sozluk_maddeleri = [
            "- **Temel Kavram:** Ders notlarında incelenen ana prensipler ve teorik çerçeve.",
            "- **Uygulama Alanı:** Konunun pratikteki işleyişi ve sağladığı çözümler."
        ]

    return "### 📖 Ders Notları Kavram Sözlüğü (Glossary)\n\n" + "\n".join(sozluk_maddeleri)


def _generate_exam_predictions_rag_answer(retrieved_items: List[Dict[str, Any]], collection) -> str:
    """Sınav Tahmini Modu: Notlardaki konulardan çıkması muhtemel 5 adet klasik/kavramsal soru ve cevap anahtarı üretir."""
    combined_text = "\n".join(item["text"] for item in retrieved_items if item.get("text"))
    extracted_concepts = _extract_document_concepts(combined_text)

    onemli_konular = [term for term, _ in extracted_concepts]

    # Yeterli değilse predict_exam_questions veya kelime sıklığından al
    if len(onemli_konular) < 5:
        try:
            exam_report = predict_exam_questions(top_n=5)
            for k in exam_report.get("en_onemli_konular", []):
                w = k["konu_anahtar_kelimesi"]
                if _is_valid_topic(w) and w.lower() not in [t.lower() for t in onemli_konular]:
                    onemli_konular.append(w.capitalize())
        except Exception:
            pass

    if len(onemli_konular) < 5:
        kelimeler = re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]{4,}", combined_text)
        kelime_sayac = Counter(k.lower() for k in kelimeler if _is_valid_topic(k))
        for kw, _ in kelime_sayac.most_common(10):
            if kw.lower() not in [t.lower() for t in onemli_konular] and len(onemli_konular) < 5:
                onemli_konular.append(kw.capitalize())

    default_konular = ["Temel Kavramlar", "Yöntemler", "İlkeler", "Sistem Yapısı", "Analiz"]
    for dk in default_konular:
        if len(onemli_konular) < 5 and dk.lower() not in [k.lower() for k in onemli_konular]:
            onemli_konular.append(dk)

    cumleler = re.split(r"(?<=[.!?])\s+", combined_text)

    soru_kaliplari = [
        "'{konu}' kavramını açıklayınız ve ders notlarındaki temel ilkeleriyle birlikte değerlendiriniz.",
        "'{konu}' konusunun temel özellikleri ve çalışma mantığı nedir? Maddeler halinde açıklayınız.",
        "'{konu}' bağlamında karşılaşılan başlıca problemler ve çözüm yaklaşımları nelerdir?",
        "'{konu}' ile ilgili yöntemlerin sağladığı temel avantajlar ve uygulama alanları nelerdir?",
        "'{konu}' konusunda dikkat edilmesi gereken teorik kurallar ve kritik noktalar nelerdir?",
    ]

    soru_bloklari = []
    for i, konu in enumerate(onemli_konular[:5]):
        konu_baslik = konu.capitalize()
        soru_metni = soru_kaliplari[i % len(soru_kaliplari)].format(konu=konu_baslik)

        # Doğrudan tanımlı kavramı kontrol et
        cevap_cumlesi = ""
        for term, definition in extracted_concepts:
            if term.lower() == konu.lower():
                cevap_cumlesi = definition
                break

        if not cevap_cumlesi:
            for c in cumleler:
                c_clean = c.strip()
                if konu.lower() in c_clean.lower() and len(c_clean) > 25:
                    cevap_cumlesi = c_clean.rstrip(".- ") + "."
                    break

        if not cevap_cumlesi:
            cevap_cumlesi = f"{konu_baslik} konusu ders notlarında sıklıkla vurgulanmış olup sınavda ana hatları ve mekanizması sorulabilir."

        soru_bloklari.append(f"### Soru {i + 1}: {soru_metni}\n**Cevap/Önemli Nokta:** {cevap_cumlesi}")

    return "### 🎯 Olası Sınav Sorusu Tahminleri (Vize / Final)\n\n" + "\n\n".join(soru_bloklari)


def _generate_summary_rag_answer(retrieved_items: List[Dict[str, Any]], query: str, is_summary: bool) -> str:
    """Genel Özet ve Normal Sorular: Kaynakça metni yapıştırmadan temiz Markdown maddeleri döndürür."""
    combined_text = "\n".join(item["text"] for item in retrieved_items if item.get("text"))
    
    query_words = set(re.findall(r"\w+", query.lower()))
    cumleler = re.split(r"(?<=[.!?])\s+", combined_text)
    
    ranked_sentences = []
    seen = set()

    for sent in cumleler:
        s = sent.strip()
        s_clean = s.lstrip("-*• 0123456789.)")
        if len(s_clean) < 25 or s_clean.lower() in seen:
            continue
        seen.add(s_clean.lower())

        s_words = set(re.findall(r"\w+", s.lower()))
        overlap = len(query_words.intersection(s_words))
        ranked_sentences.append((overlap, s_clean))

    ranked_sentences.sort(key=lambda x: x[0], reverse=True)
    secilen_maddeler = [item[1] for item in ranked_sentences[:7]]

    if not secilen_maddeler:
        secilen_maddeler = [s.strip() for s in cumleler if len(s.strip()) > 30][:5]

    if is_summary:
        header = "### 📌 Ders Notları Genel Özeti\n\n"
        return header + "\n".join(f"- {m}" for m in secilen_maddeler)
    else:
        return "\n".join(f"- {m}" for m in secilen_maddeler)


def ask_rag_notes(
    query: str,
    n_results: int = 3,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Akıllı Not RAG: Kullanıcı sorusuna veya Studio aracına göre dinamik ve akıllı yanıt üretir."""
    collection = get_chroma_collection()
    total_docs = collection.count()

    if total_docs == 0:
        return {
            "query": query,
            "answer": "Sistemde henüz kayıtlı ders notu bulunmuyor veya bu soruyla ilgili içerik tespit edilemedi. Lütfen önce ders notu veya PDF yükleyin.",
            "sources": [],
            "found_relevant": False,
        }

    q_lower = query.lower()
    is_mindmap = (
        "mindmap" in q_lower
        or "zihin haritası" in q_lower
        or "zihin haritasi" in q_lower
        or "hiyerarşik" in q_lower
        or "akış şeması" in q_lower
        or "akis semasi" in q_lower
        or "flowchart" in q_lower
        or "graph" in q_lower
        or "kavram şeması" in q_lower
        or "kavram semasi" in q_lower
        or "kavram haritası" in q_lower
        or "kavram haritasi" in q_lower
    )
    is_glossary = "sözlük" in q_lower or "sozluk" in q_lower or "terim" in q_lower or "glossary" in q_lower
    is_exam = "sınav" in q_lower or "sinav" in q_lower or "vize" in q_lower or "final" in q_lower or "tahmin" in q_lower
    is_summary = "özet" in q_lower or "ozet" in q_lower or "maddeler halinde" in q_lower

    limit_count = max(n_results, 8) if (is_mindmap or is_glossary or is_exam or is_summary) else n_results
    retrieved_items = query_rag_notes(query=query, n_results=limit_count)

    # Eğer sorgu semantik olarak az sonuç döndürdüyse doğrudan koleksiyondaki belgelerle takviye et
    if len(retrieved_items) < 3 and total_docs > 0:
        coll_data = collection.get(limit=min(12, total_docs))
        docs = coll_data.get("documents", []) or []
        metas = coll_data.get("metadatas", []) or []
        existing_texts = {item["text"] for item in retrieved_items}
        for idx, doc in enumerate(docs):
            if doc and doc.strip() and doc not in existing_texts:
                retrieved_items.append({
                    "text": doc,
                    "metadata": metas[idx] if idx < len(metas) else {},
                    "distance": None,
                })

    # Kaynak özetini derle (UI tarafındaki rozetler için)
    sources_summary = []
    seen_sources = set()
    for item in retrieved_items:
        src = item["metadata"].get("source", "Ders Notu")
        page = item["metadata"].get("page")
        key = (src, page)
        if key not in seen_sources:
            seen_sources.add(key)
            sources_summary.append({
                "source_id": len(sources_summary) + 1,
                "source": src,
                "page": page,
                "snippet": item["text"][:300] + ("..." if len(item["text"]) > 300 else ""),
            })

    # Dinamik Akıllı Mod Seçimi
    if is_mindmap:
        answer_text = _generate_mindmap_rag_answer(retrieved_items, collection)
    elif is_glossary:
        answer_text = _generate_glossary_rag_answer(retrieved_items)
    elif is_exam:
        answer_text = _generate_exam_predictions_rag_answer(retrieved_items, collection)
    else:
        answer_text = _generate_summary_rag_answer(retrieved_items, query, is_summary)

    return {
        "query": query,
        "answer": answer_text,
        "sources": sources_summary,
        "found_relevant": True,
        "total_sources_found": len(retrieved_items),
    }


def clear_rag_notes() -> bool:
    """Tüm ChromaDB ders notlarını sıfırlar/temizler ve hemen yeni boş koleksiyon başlatır."""
    global _CHROMA_CLIENT, _COLLECTION
    try:
        if _CHROMA_CLIENT is None:
            _CHROMA_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))

        try:
            _CHROMA_CLIENT.delete_collection(name=COLLECTION_NAME)
        except Exception as del_err:
            logger.info("ChromaDB koleksiyon silme notu (zaten silinmiş olabilir): %s", del_err)

        # Koleksiyon silindikten hemen sonra yeni boş koleksiyon oluştur
        _COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Ders notu parçaları ve gömmeleri"},
        )
        logger.info("ChromaDB hafızası temizlendi ve yeni boş koleksiyon oluşturuldu.")
        return True
    except Exception as e:
        logger.error("ChromaDB temizlenirken hata: %s", e)
        # Hata olsa bile client ve koleksiyonu sıfırdan oluşturmayı garanti et
        try:
            _CHROMA_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))
            _COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Ders notu parçaları ve gömmeleri"},
            )
            return True
        except Exception as retry_err:
            logger.error("ChromaDB yeniden oluşturma hatası: %s", retry_err)
            return False


def get_rag_stats() -> Dict[str, Any]:
    """ChromaDB içerisindeki not sayısı ve durumunu döndürür."""
    global _COLLECTION
    try:
        collection = get_chroma_collection()
        total_chunks = collection.count()
    except Exception as e:
        logger.warning("get_rag_stats koleksiyon hatası, yenileniyor: %s", e)
        _COLLECTION = None
        collection = get_chroma_collection()
        total_chunks = collection.count()

    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": total_chunks,
        "storage_path": str(CHROMA_DIR),
    }


def predict_exam_questions(
    past_questions_text: str = "",
    top_n: int = 5,
) -> Dict[str, Any]:
    """Ders notları ve çıkmış sorulardan sınav soru tahmini yapar.

    ChromaDB'deki tüm ders notu parçalarını ve isteğe bağlı olarak verilen
    geçmiş sınav sorularını analiz ederek:
    - Hoca'nın en sık odaklandığı konuları tespit eder
    - Sınavda çıkma ihtimali yüksek konuları listeler
    - Tahmin edilen yeni soru önerileri üretir
    - Her konu için olasılık skoru hesaplar

    Args:
        past_questions_text: Geçmiş sınav soruları (düz metin). Boş bırakılabilir.
        top_n: Kaç tahmin sorusu üretileceği.

    Returns:
        JSON serileştirilmiş tahmin raporu.
    """
    collection = get_chroma_collection()
    total_chunks = collection.count()

    if total_chunks == 0:
        return {
            "durum": "hata",
            "mesaj": "Sistemde kayıtlı ders notu bulunamadı. Önce not yükleyin.",
            "tahminler": [],
        }

    # Tüm parçaları çek (max 50)
    limit = min(50, total_chunks)
    data = collection.get(limit=limit)
    tum_metinler: List[str] = data.get("documents", []) or []
    tum_metinler = [m for m in tum_metinler if m and m.strip()]

    # Konu / anahtar kelime frekans analizi
    DURAK_KELIMELER = {
        "ve", "veya", "ile", "bir", "bu", "da", "de", "ki", "ne", "mi",
        "için", "olan", "gibi", "ise", "ama", "çok", "en", "her", "olan",
        "olarak", "aynı", "olan", "daha", "kadar", "olan", "var", "ise",
        "the", "of", "in", "is", "and", "to", "a",
    }

    kelime_sayaci: Counter = Counter()
    cumle_sayaci: Counter = Counter()

    for metin in tum_metinler:
        kelimeler = re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]{4,}", metin.lower())
        anlamli = [k for k in kelimeler if k not in DURAK_KELIMELER]
        kelime_sayaci.update(anlamli)

        cumleler = re.split(r"(?<=[.!?])\s+", metin)
        for cumle in cumleler:
            cumle = cumle.strip()
            if len(cumle) > 20:
                cumle_sayaci[cumle] += 1

    # Geçmiş sorulardan konu tespiti
    gecmis_konu_agirligi: Counter = Counter()
    if past_questions_text.strip():
        gecmis_kelimeler = re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]{4,}", past_questions_text.lower())
        gecmis_anlamli = [k for k in gecmis_kelimeler if k not in DURAK_KELIMELER]
        gecmis_konu_agirligi.update(gecmis_anlamli)

    # En sık geçen konular (ders notlarından)
    en_siklar = kelime_sayaci.most_common(30)

    # Olasılık skoru = (ders notundaki frekans * 0.6) + (geçmiş sorulardaki frekans * 0.4)
    skor_tablosu: List[Dict[str, Any]] = []
    for kelime, not_frekans in en_siklar:
        gecmis_frekans = gecmis_konu_agirligi.get(kelime, 0)
        olasilik_skoru = round((not_frekans * 0.6 + gecmis_frekans * 0.4) / max(total_chunks, 1), 4)
        skor_tablosu.append({
            "konu_anahtar_kelimesi": kelime,
            "ders_notunda_gecme_sayisi": not_frekans,
            "gecmis_sorularda_gecme": gecmis_frekans,
            "tahmin_skoru": olasilik_skoru,
        })

    skor_tablosu.sort(key=lambda x: x["tahmin_skoru"], reverse=True)
    en_onemli_konular = skor_tablosu[:top_n]

    # Tahmin edilen sorular üret
    tahmin_sorular: List[Dict[str, str]] = []
    soru_sablonlari = [
        "'{konu}' kavramını açıklayınız ve önemini belirtiniz.",
        "'{konu}' ile ilgili temel özellikler nelerdir? Örnekler vererek anlatınız.",
        "'{konu}' bağlamında karşılaşılan problemler nasıl çözülür?",
        "'{konu}' konusunda doğru/yanlış sorular: Bu ifadenin doğruluğunu tartışınız.",
        "'{konu}' ile ilgili tarihsel gelişim veya farklı yaklaşımları karşılaştırınız.",
    ]

    for i, konu_bilgisi in enumerate(en_onemli_konular):
        konu = konu_bilgisi["konu_anahtar_kelimesi"]
        sablon = soru_sablonlari[i % len(soru_sablonlari)]
        skor = konu_bilgisi["tahmin_skoru"]
        tahmin_sorular.append({
            "soru_no": i + 1,
            "tahmin_soru": sablon.format(konu=konu.capitalize()),
            "konu": konu.capitalize(),
            "cikmasi_muhtemel_neden": (
                f"'{konu}' terimi ders notlarında {konu_bilgisi['ders_notunda_gecme_sayisi']} kez geçmekte"
                + (f" ve geçmiş sınavlarda {konu_bilgisi['gecmis_sorularda_gecme']} kez sorulmuştur." if konu_bilgisi['gecmis_sorularda_gecme'] > 0 else ".")
            ),
            "tahmin_skoru": skor,
            "onem_seviyesi": "Yüksek" if skor > 0.3 else ("Orta" if skor > 0.1 else "Düşük"),
        })

    return {
        "durum": "basarili",
        "analiz_edilen_parca_sayisi": len(tum_metinler),
        "gecmis_soru_analizi_yapildi": bool(past_questions_text.strip()),
        "en_onemli_konular": en_onemli_konular,
        "tahmin_sorular": tahmin_sorular,
        "not": (
            "Bu tahminler ders notlarındaki konu yoğunluğuna ve geçmiş sınav sorusu analizine dayalıdır. "
            "Kesin sonuç vermez; rehber niteliğindedir."
        ),
    }
