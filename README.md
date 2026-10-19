# Ders Notu Özetleyici & Akıllı Not RAG Asistanı

Bu proje, ders notları görsellerinden OCR ile metin çıkaran, Hugging Face ile Türkçe özetleyen ve ChromaDB tabanlı RAG mimarisiyle ders notlarına soru sorma (Akıllı Not benzeri), otomatik quiz ve çalışma kartı (flashcard) üretme yeteneklerine sahip bir yapay zeka asistanıdır.

## Özellikler

- **Görsel Ön İşleme & OCR (`src/ocr_engine.py`):** OpenCV (grayscale, thresholding) ve EasyOCR/Tesseract ile görselden metin çıkarma.
- **Türkçe Özetleme (`src/summarizer.py`):** Hugging Face transformers pipeline ile Türkçe ders notu özetleme.
- **RAG Motoru (`src/rag_engine.py`):** ChromaDB vektör veritabanı ile ders notlarını kalıcı olarak indeksleme ve akıllı soru yanıtlama.
- **Quiz & Flashcard Üretimi (`src/summarizer.py`):** Ders notlarındaki kavram ve tanımlardan otomatik test soruları (A, B, C, D) ve soru kartları üretimi.
- **FastAPI Backend (`main.py`):** Modern, dokümante edilmiş ve asenkron REST API.

## Proje Yapısı

```
Ders-Notu-Ozetleyici/
├── data/
│   ├── raw/
│   └── processed/
│       └── chroma_db/      # Vektör veritabanı (RAG)
├── models/
├── src/
│   ├── __init__.py
│   ├── ocr_engine.py       # Görsel işleme & OCR
│   ├── rag_engine.py       # ChromaDB RAG & Soru yanıtlama
│   └── summarizer.py       # Özetleme, Quiz & Flashcard üretimi
├── static/
│   └── audio/          # Üretilen MP3 ses dosyaları
├── tests/
│   ├── __init__.py
│   └── test_tum_sistem.py # 6 aşamalı kapsamlı sistem testi
├── .gitignore
├── main.py                 # FastAPI sunucusu & endpoint'ler
├── README.md
└── requirements.txt
```

## API Endpoint'leri

| Metot | Endpoint | Açıklama |
|---|---|---|
| `GET` | `/` | API durum ve servis bilgisi |
| `POST` | `/summarize` | Doğrudan metin özetleme |
| `POST` | `/summarize-image` | Görselden metin çıkarma ve özetleme |
| `POST` | `/extract-text` | Görselden yalnızca OCR ile metin çıkarma |
| `POST` | `/index-notes` | Metin notlarını ChromaDB'ye kaydetme |
| `POST` | `/index-pdf` | PDF ders notunu ChromaDB'ye kaydetme |
| `POST` | `/ask-notes` | **Ders notlarına soru sorma (Akıllı Not RAG)** |
| `POST` | `/generate-quiz` | **Ders notundan Quiz ve Soru Kartı üretme** |
| `POST` | `/audio-summary` | **Özet metnini MP3 ses dosyasına dönüştürme (Edge-TTS)** |
| `GET` | `/audio-voices` | Kullanılabilir Türkçe ses modellerini listeleme |
| `GET` | `/rag-stats` | Kayıtlı ders notu istatistikleri |
| `DELETE` | `/rag-notes` | İndekslenen notları temizleme |

## Kurulum ve Çalıştırma

1. Sanal ortamı aktif edin:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Sunucuyu başlatın:
   ```bash
   uvicorn main:app --reload
   ```

3. Swagger UI dokümantasyonuna erişin:
   - Tarayıcınızda: `http://127.0.0.1:8000/docs`

4. Sistem testlerini çalıştırın:
   ```bash
   pytest tests/test_tum_sistem.py -v
   ```
