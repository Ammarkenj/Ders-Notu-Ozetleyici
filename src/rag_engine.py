import io
import logging
import re
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
    """ChromaDB istemcisini ve ders notları koleksiyonunu döndürür."""
    global _CHROMA_CLIENT, _COLLECTION
    if _CHROMA_CLIENT is None or _COLLECTION is None:
        _CHROMA_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Ders notu parçaları ve gömmeleri"},
        )
    return _COLLECTION


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

    existing_count = collection.count()
    ids = [f"doc_{existing_count + i}_{abs(hash(c)) % 100000}" for i, c in enumerate(chunks)]
    metadatas = [{**base_meta, "chunk_index": i} for i in range(len(chunks))]

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
    """PDF belgesinden metin ayıklar ve ChromaDB'ye ekler."""
    if isinstance(pdf_input, (str, Path)):
        reader = pypdf.PdfReader(str(pdf_input))
    elif isinstance(pdf_input, bytes):
        reader = pypdf.PdfReader(io.BytesIO(pdf_input))
    else:
        raise TypeError("Desteklenmeyen PDF girdi türü.")

    total_chunks = 0
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            count = add_text_to_rag(
                text=page_text,
                source=source_name,
                metadata={"page": page_idx + 1},
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

    collection = get_chroma_collection()
    total_docs = collection.count()
    if total_docs == 0:
        return []

    limit = min(n_results, total_docs)
    results = collection.query(
        query_texts=[query],
        n_results=limit,
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


def ask_rag_notes(query: str, n_results: int = 3) -> Dict[str, Any]:
    """NotebookLM tarzı: Ders notları üzerinden kullanıcı sorusunu yanıtlar.

    Alakalı parçaları ChromaDB'den çeker ve bu bağlam üzerinden doğrudan
    kaynaklı ve sentezlenmiş yanıt üretir.
    """
    retrieved_items = query_rag_notes(query=query, n_results=n_results)

    if not retrieved_items:
        return {
            "query": query,
            "answer": "Sistemde henüz kayıtlı ders notu bulunmuyor veya bu soruyla ilgili içerik tespit edilemedi. Lütfen önce ders notu veya PDF yükleyin.",
            "sources": [],
            "found_relevant": False,
        }

    # Bağlamı birleştir
    context_blocks = []
    sources_summary = []
    for i, item in enumerate(retrieved_items, 1):
        src = item["metadata"].get("source", "Bilinmeyen Kaynak")
        page = item["metadata"].get("page")
        page_info = f" (Sayfa: {page})" if page else ""
        header = f"[Kaynak {i}: {src}{page_info}]"
        context_blocks.append(f"{header}\n{item['text']}")
        sources_summary.append({
            "source_id": i,
            "source": src,
            "page": page,
            "snippet": item["text"][:150] + ("..." if len(item["text"]) > 150 else ""),
        })

    combined_context = "\n\n".join(context_blocks)

    # Bağlamdan yanıt sentezi (Cümle alaka analizi ve özet bağlam)
    query_words = set(re.findall(r"\w+", query.lower()))
    relevant_sentences = []

    for item in retrieved_items:
        sentences = re.split(r"(?<=[.!?])\s+", item["text"])
        for sent in sentences:
            sent_words = set(re.findall(r"\w+", sent.lower()))
            overlap = query_words.intersection(sent_words)
            if overlap:
                relevant_sentences.append((len(overlap), sent.strip()))

    relevant_sentences.sort(key=lambda x: x[0], reverse=True)
    top_excerpts = [s[1] for s in relevant_sentences[:4]]

    if top_excerpts:
        answer_text = (
            f"Ders notlarına göre:\n\n"
            + "\n".join(f"• {exc}" for exc in top_excerpts)
            + f"\n\nDetaylı Bağlam:\n{combined_context[:500]}..."
        )
    else:
        answer_text = f"Ders notlarında bulunan ilgili içerik:\n\n{combined_context}"

    return {
        "query": query,
        "answer": answer_text,
        "sources": sources_summary,
        "found_relevant": True,
        "total_sources_found": len(retrieved_items),
    }


def clear_rag_notes() -> bool:
    """Tüm ChromaDB ders notlarını sıfırlar/temizler."""
    global _CHROMA_CLIENT, _COLLECTION
    try:
        if _CHROMA_CLIENT is None:
            _CHROMA_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _CHROMA_CLIENT.delete_collection(name=COLLECTION_NAME)
        _COLLECTION = _CHROMA_CLIENT.get_or_create_collection(name=COLLECTION_NAME)
        return True
    except Exception as e:
        logger.error("ChromaDB temizlenirken hata: %s", e)
        return False


def get_rag_stats() -> Dict[str, Any]:
    """ChromaDB içerisindeki not sayısı ve durumunu döndürür."""
    collection = get_chroma_collection()
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": collection.count(),
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
