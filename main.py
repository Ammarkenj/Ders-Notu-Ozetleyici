import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from src.ocr_engine import extract_text_from_image
from src.rag_engine import (
    add_docx_to_rag,
    add_excel_to_rag,
    add_pdf_to_rag,
    add_text_to_rag,
    ask_rag_notes,
    clear_rag_notes,
    get_chroma_collection,
    get_rag_stats,
    predict_exam_questions,
    query_rag_notes,
)
from src.summarizer import (
    generate_glossary,
    generate_mindmap,
    generate_quiz_and_flashcards,
    summarize_text,
)
from src.tts_engine import generate_audio_summary, get_turkish_voices
from src.presentation_engine import generate_presentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ders_notu_ozetleyici")

# Dizin tanımları
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = STATIC_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
PRESENTATION_DIR = STATIC_DIR / "presentations"
PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Ders Notu Özetleyici")

# CORS ayarları (Eğer projende varsa app = FastAPI()'nin hemen altında olmalı)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. APP OLUŞTURULDUKTAN SONRA EKLENMESİ GEREKEN KISIM: Arayüzü bağla
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Statik dosyaları (özellikle oluşturulan ses dosyalarını) dışa aç
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==========================================
# Pydantic Veri Modelleri
# ==========================================

class TextSummarizeRequest(BaseModel):
    text: str = Field(..., description="Özetlenecek metin", min_length=1)
    max_length: Optional[int] = Field(150, description="Maksimum token uzunluğu")
    min_length: Optional[int] = Field(30, description="Minimum token uzunluğu")
    model_name: Optional[str] = Field(None, description="Özel Hugging Face model adı")


class SummarizeResponse(BaseModel):
    status: str
    original_text: str
    summary: str
    original_length: int
    summary_length: int


class TextExtractionResponse(BaseModel):
    status: str
    extracted_text: str
    character_count: int


class IndexNotesRequest(BaseModel):
    text: str = Field(..., description="ChromaDB'ye eklenecek ders notu metni", min_length=5)
    source_name: Optional[str] = Field("manuel_not", description="Notun başlığı veya kaynak adı")


class AskNotesRequest(BaseModel):
    question: str = Field(..., description="Ders notlarına sorulacak soru", min_length=2)
    n_results: Optional[int] = Field(3, description="Taranacak en alakalı parça sayısı", ge=1, le=10)


class AskNotesResponse(BaseModel):
    status: str
    query: str
    answer: str
    found_relevant: bool
    sources: List[Dict[str, Any]]
    total_sources_found: int


class GenerateQuizRequest(BaseModel):
    text: Optional[str] = Field(
        None,
        description="Quiz üretilecek ders notu metni (boş bırakılırsa RAG veritabanındaki notlar kullanılır)",
    )
    num_questions: Optional[int] = Field(5, description="Üretilecek soru ve kart adedi", ge=1, le=20)


class QuizItem(BaseModel):
    id: int
    question: str
    options: List[str]
    correct_answer: str
    explanation: str


class FlashcardItem(BaseModel):
    id: int
    front: str
    back: str
    topic: Optional[str] = None


class GenerateQuizResponse(BaseModel):
    status: str
    quiz: List[QuizItem]
    flashcards: List[FlashcardItem]
    total_questions: int
    total_flashcards: int


class AudioSummaryRequest(BaseModel):
    text: str = Field(..., description="Seslendirilecek özet metni", min_length=2)
    voice: Optional[str] = Field(
        "tr-TR-EmelNeural",
        description="Türkçe ses modeli: 'tr-TR-EmelNeural' (Varsayılan) veya 'tr-TR-AhmetNeural'",
    )
    rate: Optional[str] = Field("+0%", description="Konuşma hızı (örn: '+0%', '+10%', '-10%')")
    pitch: Optional[str] = Field("+0Hz", description="Ses tonu (örn: '+0Hz', '+5Hz')")


class AudioSummaryResponse(BaseModel):
    status: str
    message: str
    filename: str
    audio_url: str
    file_path: str


# ==========================================
# Genel & Durum Endpoint'leri
# ==========================================

@app.get("/", tags=["Genel"])
def root():
    return {
        "status": "online",
        "app_name": "Ders Notu Özetleyici & NotebookLM RAG API",
        "version": "2.1.0",
        "docs_url": "/docs",
    }


@app.get("/rag-stats", tags=["RAG (Notlara Soru Sorma)"])
def rag_stats_endpoint():
    """ChromaDB'deki kayıtlı not parçaları sayısını ve durumunu döndürür."""
    return get_rag_stats()


@app.delete("/rag-notes", tags=["RAG (Notlara Soru Sorma)"])
def clear_notes_endpoint():
    """ChromaDB'de saklanan tüm ders notlarını temizler."""
    success = clear_rag_notes()
    return {"status": "success" if success else "error", "message": "Ders notları temizlendi."}


# ==========================================
# RAG: Not İndeksleme & Soru Sorma Endpoint'leri
# ==========================================

@app.post(
    "/index-notes",
    tags=["RAG (Notlara Soru Sorma)"],
    summary="Metin Notlarını ChromaDB'ye Kaydet",
)
def index_notes_endpoint(request: IndexNotesRequest):
    """Metin formatındaki ders notlarını parçalayarak ChromaDB'ye ekler."""
    try:
        chunk_count = add_text_to_rag(
            text=request.text,
            source=request.source_name or "manuel_not",
        )
        return {
            "status": "success",
            "message": "Ders notu başarıyla indekslendi.",
            "chunks_added": chunk_count,
        }
    except Exception as exc:
        logger.error("Not indeksleme hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Not indekslenirken hata oluştu: {str(exc)}",
        )


@app.post(
    "/index-pdf",
    tags=["RAG (Notlara Soru Sorma)"],
    summary="PDF Belgesini ChromaDB'ye Kaydet",
)
async def index_pdf_endpoint(
    file: UploadFile = File(..., description="Ders notu PDF belgesi"),
):
    """Yüklenen PDF dosyasının sayfalarını ayıklar ve ChromaDB'ye kaydeder."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Yüklenen PDF dosyası boş.",
            )

        chunks_added = add_pdf_to_rag(content, source_name=file.filename or "uploaded_pdf")
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_added": chunks_added,
            "message": f"{file.filename} başarıyla indekslendi ({chunks_added} parça).",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("PDF indeksleme hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"PDF işlenirken hata oluştu: {str(exc)}",
        )


@app.post(
    "/ask-notes",
    response_model=AskNotesResponse,
    status_code=status.HTTP_200_OK,
    tags=["RAG (Notlara Soru Sorma)"],
    summary="Ders Notlarına Soru Sor (NotebookLM RAG)",
)
def ask_notes_endpoint(request: AskNotesRequest):
    """Ders notları içerisinden en alakalı kısımları bularak soruyu yanıtlar."""
    try:
        result = ask_rag_notes(query=request.question, n_results=request.n_results or 3)
        return AskNotesResponse(
            status="success",
            query=result["query"],
            answer=result["answer"],
            found_relevant=result["found_relevant"],
            sources=result["sources"],
            total_sources_found=result.get("total_sources_found", len(result["sources"])),
        )
    except Exception as exc:
        logger.error("Soru yanıtlama hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Soru yanıtlanırken hata oluştu: {str(exc)}",
        )


# ==========================================
# Quiz & Soru Kartı Üretimi Endpoint'i
# ==========================================

@app.post(
    "/generate-quiz",
    response_model=GenerateQuizResponse,
    status_code=status.HTTP_200_OK,
    tags=["Quiz & Kartlar"],
    summary="Ders Notundan Otomatik Quiz ve Flashcard Üret",
)
def generate_quiz_endpoint(request: GenerateQuizRequest):
    """Verilen veya ChromaDB'de kayıtlı ders notlarından otomatik test soruları ve çalışma kartları üretir."""
    try:
        target_text = ""
        if request.text and request.text.strip():
            target_text = request.text.strip()
        else:
            collection = get_chroma_collection()
            docs_data = collection.get(limit=10)
            if docs_data and docs_data.get("documents"):
                target_text = "\n\n".join(docs_data["documents"])

        if not target_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Quiz üretmek için metin gönderilmedi ve sistemde kayıtlı ders notu bulunamadı.",
            )

        data = generate_quiz_and_flashcards(
            text=target_text,
            num_questions=request.num_questions or 5,
        )

        return GenerateQuizResponse(
            status="success",
            quiz=data["quiz"],
            flashcards=data["flashcards"],
            total_questions=data["total_questions"],
            total_flashcards=data["total_flashcards"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Quiz üretim hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quiz üretilirken hata oluştu: {str(exc)}",
        )


# ==========================================
# Edge-TTS Seslendirme (Podcast / Sesli Özet) Endpoint'leri
# ==========================================

@app.post(
    "/audio-summary",
    response_model=AudioSummaryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Seslendirme (TTS)"],
    summary="Özet Metnini MP3 Ses Dosyasına Dönüştür",
)
async def audio_summary_endpoint(request: AudioSummaryRequest):
    """Verilen özet metnini Edge-TTS ile MP3 formatında seslendirir ve statik bağlantısını döndürür."""
    try:
        saved_file = await generate_audio_summary(
            text=request.text,
            voice=request.voice or "tr-TR-AhmetNeural",
            rate=request.rate or "+0%",
            pitch=request.pitch or "+0Hz",
        )

        audio_url = f"/static/audio/{saved_file.name}"

        return AudioSummaryResponse(
            status="success",
            message="Özet seslendirmesi başarıyla tamamlandı.",
            filename=saved_file.name,
            audio_url=audio_url,
            file_path=str(saved_file),
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Seslendirme hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ses dosyası oluşturulurken hata oluştu: {str(exc)}",
        )


@app.get(
    "/audio-voices",
    tags=["Seslendirme (TTS)"],
    summary="Kullanılabilir Türkçe Ses Modelleri",
)
async def list_voices_endpoint():
    """Kullanılabilir Türkçe ses modellerini listeler."""
    voices = await get_turkish_voices()
    return {"status": "success", "voices": voices}


# ==========================================
# Özetleme & OCR Endpoint'leri
# ==========================================

@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Özetleme"],
    summary="Metin Özetleme",
)
def summarize_endpoint(request: TextSummarizeRequest):
    """Doğrudan verilen metin içeriğini Hugging Face transformers modeli ile özetler."""
    try:
        raw_text = request.text.strip()
        if not raw_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Özetlenecek metin boş olamaz.",
            )

        summary = summarize_text(
            text=raw_text,
            max_length=request.max_length or 150,
            min_length=request.min_length or 30,
            model_name=request.model_name,
        )

        return SummarizeResponse(
            status="success",
            original_text=raw_text,
            summary=summary,
            original_length=len(raw_text),
            summary_length=len(summary),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Özetleme hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Özetleme işlemi sırasında bir hata oluştu: {str(exc)}",
        )


@app.post(
    "/summarize-image",
    response_model=SummarizeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Görsel & Özetleme"],
    summary="Görselden Metin Çıkar ve Özetle",
)
async def summarize_image_endpoint(
    file: UploadFile = File(..., description="Ders notu görseli (PNG, JPG, JPEG vb.)"),
    max_length: int = Form(150, description="Maksimum token uzunluğu"),
    min_length: int = Form(30, description="Minimum token uzunluğu"),
    auto_index_rag: bool = Form(True, description="Çıkarılan metin otomatik olarak RAG veritabanına eklensin mi?"),
):
    """Yüklenen ders notu görselinden metin çıkarır (OCR), özetler ve istenirse RAG veritabanına ekler."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Yüklenen görsel dosyası boş.",
            )

        extracted_text = extract_text_from_image(content)
        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Görselden okunabilir bir metin çıkarılamadı.",
            )

        if auto_index_rag:
            try:
                add_text_to_rag(extracted_text, source=file.filename or "gorsel_notu")
            except Exception as rag_err:
                logger.warning("Görsel metni RAG veritabanına eklenemedi: %s", rag_err)

        summary = summarize_text(
            text=extracted_text,
            max_length=max_length,
            min_length=min_length,
        )

        return SummarizeResponse(
            status="success",
            original_text=extracted_text,
            summary=summary,
            original_length=len(extracted_text),
            summary_length=len(summary),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Görsel özetleme hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Görsel özetleme sırasında bir hata oluştu: {str(exc)}",
        )


@app.post(
    "/extract-text",
    response_model=TextExtractionResponse,
    status_code=status.HTTP_200_OK,
    tags=["OCR"],
    summary="Yalnızca Görselden Metin Çıkarma",
)
async def extract_text_endpoint(
    file: UploadFile = File(..., description="Ders notu görseli"),
):
    """Yüklenen görselden yalnızca OCR ile metin çıkarır."""
    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Yüklenen dosya boş.",
            )

        extracted_text = extract_text_from_image(content)
        return TextExtractionResponse(
            status="success",
            extracted_text=extracted_text,
            character_count=len(extracted_text),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("OCR hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"OCR işlemi sırasında hata oluştu: {str(exc)}",
        )


# ==========================================
# Word (.docx) İndeksleme Endpoint'i
# ==========================================

@app.post(
    "/index-docx",
    tags=["RAG (Notlara Soru Sorma)"],
    summary="Word (.docx) Belgesini ChromaDB'ye Kaydet",
)
async def index_docx_endpoint(
    file: UploadFile = File(..., description="Ders notu Word belgesi (.docx)"),
):
    """Yüklenen Word (.docx) dosyasının paragraflarını ayıklar ve ChromaDB'ye kaydeder."""
    try:
        if not file.filename or not file.filename.lower().endswith(".docx"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Lütfen geçerli bir .docx dosyası yükleyin.",
            )
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Yüklenen Word dosyası boş.",
            )
        chunks_added = add_docx_to_rag(content, source_name=file.filename or "word_belgesi")
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_added": chunks_added,
            "message": f"{file.filename} başarıyla indekslendi ({chunks_added} parça).",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("DOCX indeksleme hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Word dosyası işlenirken hata oluştu: {str(exc)}",
        )


# ==========================================
# Sınav Sorusu Tahmin Motoru Endpoint'i
# ==========================================

class ExamPredictionRequest(BaseModel):
    past_questions: Optional[str] = Field(
        "",
        description="Geçmiş sınav soruları (düz metin). Boş bırakılabilir.",
    )
    top_n: Optional[int] = Field(5, description="Tahmin edilecek konu/soru sayısı", ge=1, le=20)


@app.post(
    "/predict-exam-questions",
    tags=["Sınav Tahmin Motoru"],
    summary="Ders Notlarından Olası Sınav Sorusu Tahmini",
)
def predict_exam_questions_endpoint(request: ExamPredictionRequest):
    """Yüklü ders notlarını ve (opsiyonel) geçmiş sınav sorularını analiz ederek çıkması muhtemel konuları ve soruları tahmin eder."""
    try:
        result = predict_exam_questions(
            past_questions_text=request.past_questions or "",
            top_n=request.top_n or 5,
        )
        return result
    except Exception as exc:
        logger.error("Sınav tahmini hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sınav sorusu tahmini yapılırken hata oluştu: {str(exc)}",
        )


# ==========================================
# Otomatik Sunum Üretimi Endpoint'i
# ==========================================

class GeneratePresentationRequest(BaseModel):
    text: str = Field(..., description="Sunuma dönüştürülecek ders notu / özet metni", min_length=10)
    title: Optional[str] = Field("Ders Notu Sunumu", description="Kapak başlığı")
    subtitle: Optional[str] = Field("Otomatik Oluşturuldu", description="Kapak alt başlığı")
    max_slides: Optional[int] = Field(8, description="Maksimum slayt sayısı (kapak hariç)", ge=1, le=20)
    filename: Optional[str] = Field(None, description="Kaydedilecek dosya adı (opsiyonel)")


@app.post(
    "/generate-presentation",
    tags=["Sunum Üretimi"],
    summary="Ders Notundan Otomatik PowerPoint (.pptx) Sunumu Oluştur",
)
def generate_presentation_endpoint(request: GeneratePresentationRequest):
    """Verilen metin içeriğinden profesyonel bir PowerPoint sunumu oluşturur ve indirme bağlantısını döndürür."""
    try:
        output_path = generate_presentation(
            text=request.text,
            title=request.title or "Ders Notu Sunumu",
            subtitle=request.subtitle or "Otomatik Oluşturuldu",
            max_slides=request.max_slides or 8,
            filename=request.filename,
        )
        pptx_url = f"/static/presentations/{output_path.name}"
        return {
            "status": "success",
            "message": "PowerPoint sunumu başarıyla oluşturuldu.",
            "filename": output_path.name,
            "download_url": pptx_url,
            "file_path": str(output_path),
            "slide_info": "Kapak + içerik slaytları + teşekkür slaytı",
        }
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err),
        )
    except Exception as exc:
        logger.error("Sunum üretim hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sunum oluşturulurken hata oluştu: {str(exc)}",
        )


# ==========================================
# Excel (.xlsx / .csv) İndeksleme Endpoint'i
# ==========================================

@app.post(
    "/index-excel",
    tags=["RAG (Notlara Soru Sorma)"],
    summary="Excel (.xlsx / .csv) Tablosunu ChromaDB'ye Kaydet",
)
async def index_excel_endpoint(
    file: UploadFile = File(..., description="Ders verisi Excel (.xlsx) veya CSV (.csv) dosyası"),
):
    """Yüklenen Excel veya CSV tablosunu satır satır okuyup ChromaDB'ye kaydeder."""
    try:
        uzanti = Path(file.filename or "").suffix.lower()
        if uzanti not in (".xlsx", ".csv", ".xls"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Lütfen geçerli bir .xlsx, .xls veya .csv dosyası yükleyin.",
            )
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Yüklenen dosya boş.",
            )
        chunks_added = add_excel_to_rag(
            content,
            source_name=file.filename or "excel_belgesi",
            dosya_uzantisi=uzanti,
        )
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_added": chunks_added,
            "message": f"{file.filename} başarıyla indekslendi ({chunks_added} parça).",
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Excel indeksleme hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Excel dosyası işlenirken hata oluştu: {str(exc)}",
        )


# ==========================================
# Kavram Sözlüğü (Glossary) Endpoint'i
# ==========================================

class GlossaryRequest(BaseModel):
    text: str = Field(..., description="Sözlük üretilecek ders notu metni", min_length=10)


@app.post(
    "/generate-glossary",
    tags=["Analiz & Sözlük"],
    summary="Ders Notundan Kavram Sözlüğü, Formül ve Anahtar Kelime Listesi Üret",
)
def generate_glossary_endpoint(request: GlossaryRequest):
    """Ders notundaki terimleri, tanımları, formülleri ve anahtar kelimeleri çıkararak JSON sözlük döndürür."""
    try:
        sonuc = generate_glossary(request.text)
        return {"status": "success", **sonuc}
    except Exception as exc:
        logger.error("Sözlük üretim hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kavram sözlüğü oluşturulurken hata: {str(exc)}",
        )


# ==========================================
# Zihin Haritası (MindMap) Endpoint'i
# ==========================================

class MindmapRequest(BaseModel):
    text: str = Field(..., description="Zihin haritası üretilecek ders notu metni", min_length=10)
    max_nodes: Optional[int] = Field(30, description="Maksimum düğüm sayısı", ge=5, le=80)


@app.post(
    "/generate-mindmap",
    tags=["Analiz & Sözlük"],
    summary="Ders Notundan D3.js/vis.js Uyumlu Zihin Haritası Verisi Üret",
)
def generate_mindmap_endpoint(request: MindmapRequest):
    """Ders notundaki kavram ilişkilerini analiz ederek düğüm + kenar (node-edge) yapısında zihin haritası döndürür."""
    try:
        sonuc = generate_mindmap(request.text, max_nodes=request.max_nodes or 30)
        return {"status": "success", **sonuc}
    except Exception as exc:
        logger.error("Zihin haritası hatası: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Zihin haritası oluşturulurken hata: {str(exc)}",
        )
