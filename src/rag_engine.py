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

from src.gemini_service import GeminiUnavailableError, generate_with_distillation  # noqa: F401
from src.quiz_generator import _extract_json_block

logger = logging.getLogger(__name__)

# ChromaDB veritabanı dizini
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "processed" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "lecture_notes"

# Gemini'ye gönderilecek maksimum bağlam uzunluğu (karakter)
MAX_CONTEXT_CHARS = 12000

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


def _clean_mermaid_flowchart(raw_code: Optional[str]) -> Optional[str]:
    """Gemini çıktısını geçerli ve hatasız bir Mermaid.js flowchart LR şemasına dönüştürür."""
    if not raw_code or not raw_code.strip():
        return None
    code = raw_code.strip()
    # Markdown kod bloklarını ayıkla
    if "```" in code:
        code = re.sub(r"```(?:mermaid|flowchart|graph)?\s*", "", code, flags=re.IGNORECASE)
        code = code.replace("```", "").strip()

    # flowchart LR direktifini kontrol et ve garantile
    first_line_match = re.search(r"^\s*(flowchart|graph)\s+([A-Za-z]+)", code, re.IGNORECASE)
    if first_line_match:
        code = re.sub(r"^\s*(flowchart|graph)\s+([A-Za-z]+)", "flowchart LR", code, count=1, flags=re.IGNORECASE)
    elif not code.strip().lower().startswith("flowchart lr"):
        code = "flowchart LR\n" + code

    # En azından birkaç bağlantı ok işareti (-->) içermelidir
    if code.count("-->") < 1:
        return None

    # Parantez/köşeli ayraç temizliği: düğüm metinleri içindeki parantez/tırnak hatalarını temizle
    def _sanitize_bracket_label(match):
        pre = match.group(1)
        inner = match.group(2)
        post = match.group(3)
        cleaned_inner = re.sub(r'["`()\[\]{}]', ' ', inner)
        cleaned_inner = re.sub(r'\s+', ' ', cleaned_inner).strip()
        return f'{pre}{cleaned_inner}{post}'

    code = re.sub(r'(\w+\(\[")([^"]*)("\]\))', _sanitize_bracket_label, code)
    code = re.sub(r'(\w+\[")([^"]*)("\])', _sanitize_bracket_label, code)

    # classDef stillerini kontrol et; eksikse modern stilleri ekle
    if "classDef" not in code:
        code += "\n\n    %% Stil Tanımları\n"
        code += "    classDef rootStyle fill:#4338ca,stroke:#6366f1,stroke-width:3px,color:#ffffff,font-weight:bold,font-size:13px;\n"
        code += "    classDef mainStyle fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#e0e7ff,font-weight:600,font-size:12px;\n"
        code += "    classDef subStyle fill:#0f172a,stroke:#10b981,stroke-width:1.5px,color:#f1f5f9,font-size:11px;\n"
        code += "    class ROOT rootStyle;\n"

    return code.strip()


def _build_context(retrieved_items: List[Dict[str, Any]], limit: int = MAX_CONTEXT_CHARS) -> str:
    """ChromaDB'den bulunan ham bağlam parçalarını Gemini'ye iletilecek tek metne birleştirir (kurye modu)."""
    parts: List[str] = []
    total = 0
    for item in retrieved_items:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        if total + len(text) > limit:
            break
        parts.append(text)
        total += len(text)
    return "\n\n".join(parts)


def _post_mindmap(raw: str) -> Optional[str]:
    return _clean_mermaid_flowchart(raw)


def _post_glossary(raw: str) -> Optional[str]:
    if raw and "- **" in raw:
        if not raw.startswith("### 📖"):
            raw = "### 📖 Ders Notları Kavram Sözlüğü (Glossary)\n\n" + raw.strip()
        return raw.strip()
    return None


def _post_exam(raw: str) -> Optional[str]:
    if raw and ("### Soru" in raw or "### 🎯" in raw):
        if not raw.startswith("### 🎯"):
            raw = "### 🎯 Olası Sınav Sorusu Tahminleri (Vize / Final)\n\n" + raw.strip()
        return raw.strip()
    return None


def _post_summary(raw: str) -> Optional[str]:
    if raw and len(raw.strip()) > 20:
        if not raw.startswith("### 📌"):
            raw = "### 📌 Ders Notları Genel Özeti\n\n" + raw.strip()
        return raw.strip()
    return None


def _post_qa(raw: str) -> Optional[str]:
    return raw.strip() if raw and len(raw.strip()) > 20 else None


def _build_prompt(task_type: str, query: str, context: str) -> str:
    """Görev türüne göre Gemini prompt'unu oluşturur."""
    ctx_block = f"DERS NOTU BAĞLAMI:\n\"\"\"\n{context}\n\"\"\""

    if task_type == "mindmap":
        return (
            "Aşağıdaki ders notundaki ana ve alt kavramları Mermaid.js flowchart LR (soldan sağa kavram akış şeması) olarak görselleştir.\n\n"
            "KESİN VE TAVİZSİZ KURALLAR:\n"
            "1. Yanıtın SADECE ve SADECE geçerli Mermaid.js kodu olmalıdır. Asla markdown bloğu (```), açıklama veya yorum yazma.\n"
            "2. Doğrudan 'flowchart LR' ile başla.\n"
            "3. Merkez düğüm ders notunun ana konusu olmalı: ROOT([\"📘 Ana Konu\"])\n"
            "4. En az 3-4 adet ana kavram düğümü ekle: ROOT --> M1([\"Ana Kavram 1\"])\n"
            "5. Her ana kavrama en az 2 adet alt kavram bağla: M1 --> S1_1[\"Alt Kavram 1\"]\n"
            "6. Düğüm metinleri içinde ASLA parantez (), köşeli ayraç [], süslü ayraç {}, çift tırnak \" veya tek tırnak ' KULLANMA.\n"
            "7. Şemanın sonuna mutlaka modern classDef stillerini ekle:\n"
            "    classDef rootStyle fill:#4338ca,stroke:#6366f1,stroke-width:3px,color:#ffffff,font-weight:bold,font-size:13px;\n"
            "    classDef mainStyle fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#e0e7ff,font-weight:600,font-size:12px;\n"
            "    classDef subStyle fill:#0f172a,stroke:#10b981,stroke-width:1.5px,color:#f1f5f9,font-size:11px;\n"
            "    class ROOT rootStyle;\n\n"
            f"{ctx_block}"
        )
    if task_type == "glossary":
        return (
            "Aşağıdaki ders notu metninde geçen tüm önemli teknik terimleri, kavramları ve tanımlarını akademik bir titizlikle çıkar.\n"
            "Format tam olarak şöyle olmalıdır:\n"
            "### 📖 Ders Notları Kavram Sözlüğü (Glossary)\n\n"
            "- **Terim 1:** Detaylı akademik açıklama ve tanım.\n"
            "- **Terim 2:** Detaylı akademik açıklama ve tanım.\n\n"
            f"{ctx_block}"
        )
    if task_type == "exam":
        return (
            "Aşağıdaki ders notu bağlamından üniversite sınavlarında (Vize / Final) çıkması en muhtemel 5 adet klasik veya analitik sınav sorusu "
            "ve bunların detaylı cevap anahtarını/önemli noktalarını hazırla.\n\n"
            "Format tam olarak şöyle olmalıdır:\n"
            "### 🎯 Olası Sınav Sorusu Tahminleri (Vize / Final)\n\n"
            "### Soru 1: Soru metni?\n"
            "**Cevap/Önemli Nokta:** Detaylı cevap açıklaması ve kritik noktalar.\n\n"
            "### Soru 2: Soru metni?\n"
            "**Cevap/Önemli Nokta:** Detaylı cevap açıklaması ve kritik noktalar.\n\n"
            f"{ctx_block}"
        )
    if task_type == "summary":
        return (
            "Aşağıdaki ders notu bağlamını en yüksek akademik standartta özetle. "
            "Önemli kavramları, ilişkileri ve sonuçları maddeler halinde Markdown ile sun.\n\n"
            "Format:\n"
            "### 📌 Ders Notları Genel Özeti\n\n"
            "- **Kavram/Nokta:** Açıklama.\n\n"
            f"{ctx_block}"
        )
    return (
        "Aşağıdaki ders notu bağlamını kullanarak kullanıcının sorusuna akademik, doğrudan, net ve kapsamlı bir yanıt ver. "
        "Yalnızca bağlamdaki bilgilere dayan; bağlamda yoksa bunu belirt.\n"
        "Maddeli açıklamalar veya net paragraflar kullan.\n\n"
        f"KULLANICI SORUSU: {query}\n\n"
        f"{ctx_block}"
    )


_TASK_CONFIG = {
    # task_type: (postprocess, temperature, max_output_tokens)
    "mindmap": (_post_mindmap, 0.1, 3000),
    "glossary": (_post_glossary, 0.2, 3000),
    "exam": (_post_exam, 0.2, 3500),
    "summary": (_post_summary, 0.2, 3000),
    "qa": (_post_qa, 0.2, 2500),
}


def ask_rag_notes(
    query: str,
    n_results: int = 3,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Akıllı Not RAG (Kurye Modu).

    Lokal CPU + ChromaDB yalnızca bağlam bulur; metin üretiminin tamamı Gemini'dedir
    (gemini-2.5-flash). Gemini yanıt veremezse GeminiUnavailableError fırlatılır.
    Başarılı her yanıt ai_cache.json'a ve data/fine_tuning_dataset.jsonl'e yazılır.
    """
    collection = get_chroma_collection()
    total_docs = collection.count()

    if total_docs == 0:
        return {
            "query": query,
            "answer": "Sistemde henüz kayıtlı ders notu bulunmuyor veya bu soruyla ilgili içerik tespit edilemedi. Lütfen önce ders notu veya PDF yükleyin.",
            "sources": [],
            "found_relevant": False,
            "total_sources_found": 0,
        }

    q_lower = query.lower()
    is_mindmap = any(k in q_lower for k in (
        "mindmap", "zihin haritası", "zihin haritasi", "hiyerarşik", "akış şeması", "akis semasi",
        "flowchart", "graph", "kavram şeması", "kavram semasi", "kavram haritası", "kavram haritasi",
    ))
    is_glossary = "sözlük" in q_lower or "sozluk" in q_lower or "terim" in q_lower or "glossary" in q_lower
    is_exam = any(k in q_lower for k in ("sınav", "sinav", "vize", "final", "tahmin"))
    is_summary = "özet" in q_lower or "ozet" in q_lower or "maddeler halinde" in q_lower

    if is_mindmap:
        task_type = "mindmap"
    elif is_glossary:
        task_type = "glossary"
    elif is_exam:
        task_type = "exam"
    elif is_summary:
        task_type = "summary"
    else:
        task_type = "qa"

    # --- Lokal CPU işi: vektör arama ile bağlam bulma ---
    limit_count = max(n_results, 8) if task_type != "qa" else n_results
    retrieved_items = query_rag_notes(query=query, n_results=limit_count)

    # Sorgu semantik olarak az sonuç döndürdüyse koleksiyondan takviye et
    if len(retrieved_items) < 3:
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

    context = _build_context(retrieved_items)
    postprocess, temperature, max_out = _TASK_CONFIG[task_type]

    # --- Tüm zeka işi Gemini'de: cache -> Gemini -> distillation kaydı ---
    answer_text = generate_with_distillation(
        task_type=task_type,
        instruction=query,
        context=context,
        prompt=_build_prompt(task_type, query, context),
        temperature=temperature,
        max_output_tokens=max(max_out, max_tokens),
        postprocess=postprocess,
    )

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
    """Vize/Final sınav sorusu tahmini.

    Lokal CPU yalnızca ChromaDB'den ham bağlamı toplar ve kelime frekansı çıkarır;
    soruların üretimi Gemini (gemini-2.5-flash) tarafından yapılır.
    Gemini kullanılamazsa GeminiUnavailableError fırlatılır.
    """
    collection = get_chroma_collection()
    total_chunks = collection.count()

    if total_chunks == 0:
        return {
            "durum": "hata",
            "mesaj": "Sistemde kayıtlı ders notu bulunamadı. Önce not yükleyin.",
            "tahminler": [],
        }

    data = collection.get(limit=min(50, total_chunks))
    tum_metinler: List[str] = [m for m in (data.get("documents", []) or []) if m and m.strip()]

    # Lokal frekans analizi (üretim değil; yalnızca konu ağırlığı için)
    durak = {
        "ve", "veya", "ile", "bir", "bu", "da", "de", "ki", "ne", "mi", "için", "olan", "gibi",
        "ise", "ama", "çok", "en", "her", "olarak", "aynı", "daha", "kadar", "var",
        "the", "of", "in", "is", "and", "to", "a",
    }
    kelime_sayaci: Counter = Counter()
    for metin in tum_metinler:
        kelime_sayaci.update(k for k in re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]{4,}", metin.lower()) if k not in durak)

    gecmis_agirlik: Counter = Counter()
    if past_questions_text.strip():
        gecmis_agirlik.update(
            k for k in re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]{4,}", past_questions_text.lower()) if k not in durak
        )

    skor_tablosu: List[Dict[str, Any]] = []
    for kelime, frekans in kelime_sayaci.most_common(30):
        gecmis = gecmis_agirlik.get(kelime, 0)
        skor_tablosu.append({
            "konu_anahtar_kelimesi": kelime,
            "ders_notunda_gecme_sayisi": frekans,
            "gecmis_sorularda_gecme": gecmis,
            "tahmin_skoru": round((frekans * 0.6 + gecmis * 0.4) / max(total_chunks, 1), 4),
        })
    skor_tablosu.sort(key=lambda x: x["tahmin_skoru"], reverse=True)
    en_onemli_konular = skor_tablosu[:top_n]

    # --- Gemini: soru üretimi ---
    context = _build_context([{"text": m} for m in tum_metinler])
    konular = ", ".join(k["konu_anahtar_kelimesi"] for k in en_onemli_konular)
    prompt = (
        f"Aşağıdaki ders notlarına dayanarak üniversite Vize/Final sınavında çıkması en muhtemel {top_n} soruyu tahmin et.\n"
        + (f"Geçmiş sınav soruları (üslup/odak için referans):\n{past_questions_text.strip()}\n\n" if past_questions_text.strip() else "")
        + f"Notlarda öne çıkan anahtar kelimeler: {konular}\n\n"
        "SADECE geçerli bir JSON objesi döndür (markdown bloğu yok):\n"
        "{\"tahmin_sorular\": [{\"soru_no\": 1, \"tahmin_soru\": \"...\", \"konu\": \"...\", "
        "\"cikmasi_muhtemel_neden\": \"...\", \"tahmin_skoru\": 0.0-1.0 arası sayı, "
        "\"onem_seviyesi\": \"Yüksek|Orta|Düşük\"}]}\n\n"
        f"DERS NOTU BAĞLAMI:\n\"\"\"\n{context}\n\"\"\""
    )

    def _post(raw: str):
        parsed = _extract_json_block(raw)
        sorular = parsed.get("tahmin_sorular") if parsed else None
        if not isinstance(sorular, list) or not sorular:
            return None
        for i, s in enumerate(sorular, 1):
            if not isinstance(s, dict) or not s.get("tahmin_soru"):
                return None
            s["soru_no"] = i
        return {"tahmin_sorular": sorular[:top_n]}

    uretilen = generate_with_distillation(
        task_type="exam_json",
        instruction=f"Ders notlarından en olası {top_n} sınav sorusunu tahmin et. Geçmiş sorular: {past_questions_text.strip()}",
        context=context,
        prompt=prompt,
        temperature=0.3,
        max_output_tokens=3500,
        postprocess=_post,
    )

    return {
        "durum": "basarili",
        "analiz_edilen_parca_sayisi": len(tum_metinler),
        "gecmis_soru_analizi_yapildi": bool(past_questions_text.strip()),
        "en_onemli_konular": en_onemli_konular,
        "tahmin_sorular": uretilen["tahmin_sorular"],
        "not": (
            "Sorular Gemini (gemini-2.5-flash) tarafından ders notlarınıza ve konu yoğunluğuna göre üretilmiştir. "
            "Kesin sonuç vermez; rehber niteliğindedir."
        ),
    }
