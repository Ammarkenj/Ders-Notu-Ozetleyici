"""Ders Notu Özetleyici & RAG Asistanı - Kapsamlı Sistem Test Paketi

Bu test dosyası, projenin 6 temel çekirdek bileşenini sırasıyla ve bağımsız
olarak doğrular:
1. Görsel Ön İşleme ve OCR Metin Çıkarma (OpenCV & EasyOCR)
2. Türkçe NLP Metin Özetleme (Hugging Face / Kural Tabanlı)
3. ChromaDB Vektör Veritabanı ve Soru Yanıtlama (Akıllı Not RAG)
4. Otomatik Quiz (Çoktan Seçmeli Test) ve Çalışma Kartı (Flashcard) Üretimi
5. Edge-TTS ile MP3 Sesli Özet Oluşturma ve Statik Dosya Kontrolü
6. FastAPI REST API Sunucusu ve Tüm Uç Noktaların HTTP 200 Doğrulaması
"""

import asyncio
import sys
from pathlib import Path

# Proje kök dizinini sys.path'e ekle (tests/ klasöründen importları garantiye al)
KOK_DIZIN = Path(__file__).resolve().parent.parent
if str(KOK_DIZIN) not in sys.path:
    sys.path.insert(0, str(KOK_DIZIN))

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app
from src.ocr_engine import extract_text_from_image, preprocess_image
from src.rag_engine import (
    add_docx_to_rag,
    add_excel_to_rag,
    add_pdf_to_rag,
    add_text_to_rag,
    ask_rag_notes,
    clear_rag_notes,
    get_rag_stats,
    predict_exam_questions,
    query_rag_notes,
)
from src.summarizer import (
    generate_quiz_and_flashcards, summarize_text,
    generate_glossary, generate_mindmap,
)
from src.tts_engine import generate_audio_summary
from src.presentation_engine import generate_presentation


@pytest.fixture(scope="session", autouse=True)
def cleanup_chromadb_session():
    """Tüm testler bittiğinde test verilerini ChromaDB'den otomatik olarak temizler."""
    yield
    clear_rag_notes()
    RenkliKonsol.bilgi("Test oturumu tamamlandı: ChromaDB test verileri temizlendi.")


class RenkliKonsol:
    """Konsola renkli ve simgeli bildirimler yazdırmak için yardımcı sınıf."""

    YESIL = "\033[92m"
    MAVI = "\033[94m"
    SARI = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    KALIN = "\033[1m"

    @classmethod
    def baslik(cls, metin: str) -> None:
        print(f"\n{cls.CYAN}{cls.KALIN}{'=' * 70}{cls.RESET}")
        print(f"{cls.CYAN}{cls.KALIN}  {metin}{cls.RESET}")
        print(f"{cls.CYAN}{cls.KALIN}{'=' * 70}{cls.RESET}")

    @classmethod
    def basarili(cls, metin: str) -> None:
        print(f"  {cls.YESIL}[BAŞARILI]{cls.RESET} {metin}")

    @classmethod
    def bilgi(cls, metin: str) -> None:
        print(f"  {cls.MAVI}[BİLGİ]{cls.RESET}    {metin}")

    @classmethod
    def detay(cls, etiket: str, deger: str) -> None:
        print(f"     * {cls.SARI}{etiket}:{cls.RESET} {deger}")


def test_01_gorsel_ocr_metin_cikarma():
    """1. Aşama: OpenCV ön işleme ve OCR metin ayıklama testi."""
    RenkliKonsol.baslik("AŞAMA 1: Görsel Ön İşleme ve OCR Metin Çıkarma Testi")

    # 1. Sentetik net bir ders notu görseli oluştur (Beyaz zemin üzerine siyah metin)
    gorsel_yukseklik = 120
    gorsel_genislik = 500
    ornek_gorsel = np.ones((gorsel_yukseklik, gorsel_genislik, 3), dtype=np.uint8) * 255

    test_metni = "FIZIK DERS NOTU"
    cv2.putText(
        ornek_gorsel,
        test_metni,
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    RenkliKonsol.bilgi(f"Sentetik test görseli oluşturuldu (Boyut: {gorsel_genislik}x{gorsel_yukseklik} piksel).")

    # 2. OpenCV Ön İşleme Kontrolü (Grayscale & Thresholding)
    on_islenmis_gorsel = preprocess_image(ornek_gorsel, apply_threshold=True, apply_blur=True)
    assert on_islenmis_gorsel is not None, "Ön işlenmiş görsel boş olamaz."
    assert on_islenmis_gorsel.shape == (gorsel_yukseklik, gorsel_genislik), "Ön işleme görsel boyutunu korumalıdır."
    assert len(on_islenmis_gorsel.shape) == 2, "Ön işleme sonucu tek kanallı gri seviye (grayscale) olmalıdır."
    RenkliKonsol.basarili("OpenCV Gri tonlama ve Thresholding ön işlemesi başarıyla doğrulandı.")

    # 3. OCR Metin Çıkarma Kontrolü
    ayiklanan_metin = extract_text_from_image(ornek_gorsel, languages=["tr", "en"], use_preprocessing=True)
    assert isinstance(ayiklanan_metin, str), "OCR çıktısı metin (str) türünde olmalıdır."
    assert len(ayiklanan_metin.strip()) > 0, "OCR görselden en az bir karakter çıkarabilmelidir."

    temiz_metin = ayiklanan_metin.upper()
    assert "FIZIK" in temiz_metin or "NOTU" in temiz_metin or "DERS" in temiz_metin, (
        f"Görseldeki beklenen anahtar kelimeler bulunamadı. Okunan metin: '{ayiklanan_metin}'"
    )

    RenkliKonsol.basarili("OCR motoru metni hatasız ayıkladı.")
    RenkliKonsol.detay("Ayıklanan Metin", ayiklanan_metin.strip())


def test_02_turkce_metin_ozetleme():
    """2. Aşama: Türkçe NLP metin özetleme modeli testi."""
    RenkliKonsol.baslik("AŞAMA 2: Türkçe Metin Özetleme Modeli Testi")

    ornek_ders_notu = (
        "Hücre tüm canlıların en temel yapısal ve işlevsel birimidir. "
        "Hücre teorisine göre bütün canlılar bir veya birden fazla hücreden meydana gelir. "
        "Hücre zarı seçici geçirgen bir yapıya sahip olup hücreye giren ve çıkan maddeleri kontrol eder. "
        "Çekirdek hücrenin genetik bilgi deposudur ve DNA moleküllerini barındırır. "
        "Sitoplazma ise organellerin bulunduğu akışkan sıvıdır. "
        "Mitokondri hücresel solunum yoluyla enerji üretiminden sorumludur."
    )
    RenkliKonsol.bilgi(f"Orijinal ders notu hazırlandı ({len(ornek_ders_notu)} karakter, {len(ornek_ders_notu.split())} kelime).")

    ozet_sonucu = summarize_text(
        text=ornek_ders_notu,
        max_length=100,
        min_length=20,
    )

    assert isinstance(ozet_sonucu, str), "Özet metni str formatında olmalıdır."
    assert len(ozet_sonucu.strip()) > 0, "Üretilen özet boş olamaz."
    assert len(ozet_sonucu) <= len(ornek_ders_notu), "Özet metni orijinal metinden daha uzun olamaz."

    RenkliKonsol.basarili("Türkçe metin özetleme işlemi başarıyla tamamlandı.")
    RenkliKonsol.detay("Orijinal Karakter", str(len(ornek_ders_notu)))
    RenkliKonsol.detay("Özet Karakter", str(len(ozet_sonucu)))
    RenkliKonsol.detay("Özet İçeriği", ozet_sonucu)


def test_03_chromadb_vektor_veritabani_rag():
    """3. Aşama: ChromaDB vektör araması ve notlara soru sorma (RAG) testi."""
    RenkliKonsol.baslik("AŞAMA 3: ChromaDB Vektör Veritabanı ve RAG Soru Yanıtlama Testi")

    # Önceki test kalıntılarını temizle
    clear_rag_notes()

    ders_notu_parcasi = (
        "Newton'un Birinci Hareket Yasası: Bir cisim üzerine dengelenmemiş net bir dış kuvvet etki etmedikçe, "
        "cisim duruyorsa durmaya, hareket ediyorsa sabit hızla doğrusal hareketine devam eder. "
        "Bu ilke eylemsizlik yasası olarak adlandırılır. "
        "Newton'un İkinci Hareket Yasası ise F = m * a bağıntısıyla ifade edilir. "
        "Net kuvvet, kütle ile ivmenin çarpımına eşittir."
    )

    # 1. Notu ChromaDB koleksiyonuna indeksle
    eklenen_parca_sayisi = add_text_to_rag(
        text=ders_notu_parcasi,
        source="Fizik_Ders_Notu_Hafta_3",
    )
    assert eklenen_parca_sayisi > 0, "ChromaDB'ye en az bir parça indekslenmiş olmalıdır."
    RenkliKonsol.basarili(f"ChromaDB koleksiyonuna {eklenen_parca_sayisi} adet vektör parçası eklendi.")

    # 2. Vektör Benzerlik Araması Testi
    arama_sorgusu = "Eylemsizlik yasası neyi ifade eder?"
    benzer_parcalar = query_rag_notes(query=arama_sorgusu, n_results=2)
    assert len(benzer_parcalar) > 0, "Soru ile ilgili en az bir parça getirilmelidir."
    assert "eylemsizlik" in benzer_parcalar[0]["text"].lower(), "Getirilen parçada anahtar kavram yer almalıdır."
    RenkliKonsol.basarili("Vektör benzerlik araması en alakalı bağlamı başarıyla çekti.")

    # 3. Akıllı Not Tarzı RAG Yanıt Üretimi Testi
    rag_sonucu = ask_rag_notes(query="Newton'un birinci yasası nedir?", n_results=2)
    assert isinstance(rag_sonucu, dict), "RAG yanıtı sözlük formatında dönmelidir."
    assert rag_sonucu["found_relevant"] is True, "İlgili bağlam tespit edilmiş olmalıdır."
    assert len(rag_sonucu["answer"].strip()) > 0, "RAG yanıtı boş olamaz."
    assert len(rag_sonucu["sources"]) > 0, "En az bir kaynak referansı içermelidir."

    RenkliKonsol.basarili("RAG motoru bağlam destekli yanıtı ve kaynakları üretti.")
    RenkliKonsol.detay("Soru", rag_sonucu["query"])
    RenkliKonsol.detay("Kaynak Sayısı", str(len(rag_sonucu["sources"])))
    RenkliKonsol.detay("Üretilen Yanıt", rag_sonucu["answer"][:180] + "...")


def test_04_otomatik_quiz_ve_soru_karti_uretimi():
    """4. Aşama: Otomatik test soruları (Quiz) ve çalışma kartı (Flashcard) üretimi testi."""
    RenkliKonsol.baslik("AŞAMA 4: Otomatik Quiz ve Flashcard (Soru Kartı) Üretim Testi")

    biyoloji_notu = (
        "Fotosentez: Kloroplast organelinde ışık enerjisinin kimyasal enerjiye dönüştürülerek besin üretilmesidir.\n"
        "Klorofil: Fotosentez sırasında ışık enerjisini soğuran yeşil renkli pigmenttir.\n"
        "Solunum: Hücrelerin besin moleküllerini parçalayarak ATP enerjisi elde etmesi sürecidir.\n"
        "ATP: Hücresel metabolik faaliyetlerde doğrudan kullanılan temel enerji molekülüdür."
    )

    istenen_soru_adedi = 3
    uretim_sonucu = generate_quiz_and_flashcards(
        text=biyoloji_notu,
        num_questions=istenen_soru_adedi,
    )

    assert isinstance(uretim_sonucu, dict), "Üretim sonucu sözlük türünde olmalıdır."
    quiz_listesi = uretim_sonucu.get("quiz", [])
    kart_listesi = uretim_sonucu.get("flashcards", [])

    assert len(quiz_listesi) >= 2, "En az 2 adet çoktan seçmeli soru üretilmelidir."
    assert len(kart_listesi) >= 2, "En az 2 adet soru kartı üretilmelidir."

    # Çoktan seçmeli soru yapısını doğrula
    ilk_soru = quiz_listesi[0]
    assert "question" in ilk_soru, "Soruda 'question' alanı bulunmalıdır."
    assert "options" in ilk_soru and len(ilk_soru["options"]) == 4, "Sorunun 4 seçeneği (A, B, C, D) olmalıdır."
    assert "correct_answer" in ilk_soru and ilk_soru["correct_answer"] in ["A", "B", "C", "D"], (
        "Doğru cevap A, B, C veya D olmalıdır."
    )
    assert "explanation" in ilk_soru, "Soru açıklaması bulunmalıdır."
    RenkliKonsol.basarili(f"{len(quiz_listesi)} adet 4 seçenekli Quiz sorusu başarıyla oluşturuldu.")
    RenkliKonsol.detay("Örnek Soru", ilk_soru["question"])
    RenkliKonsol.detay("Doğru Şık", ilk_soru["correct_answer"])

    # Çalışma kartı yapısını doğrula
    ilk_kart = kart_listesi[0]
    assert "front" in ilk_kart, "Kartın ön yüzü ('front') bulunmalıdır."
    assert "back" in ilk_kart, "Kartın arka yüzü ('back') bulunmalıdır."
    RenkliKonsol.basarili(f"{len(kart_listesi)} adet çalışma kartı (Flashcard) başarıyla oluşturuldu.")
    RenkliKonsol.detay("Kart Ön Yüz", ilk_kart["front"])
    RenkliKonsol.detay("Kart Arka Yüz", ilk_kart["back"])


def test_05_sesli_ozet_mp3_uretimi():
    """5. Aşama: Edge-TTS ile MP3 sesli özet üretimi ve static/audio/ dizin kontrolü."""
    RenkliKonsol.baslik("AŞAMA 5: Edge-TTS ile MP3 Sesli Özet Üretim Testi")

    seslendirilecek_metin = "Bu test aşamasında Edge-TTS kullanılarak ders notu özeti MP3 formatında seslendirilmektedir."

    # Asenkron fonksiyonu çalıştır
    uretilen_ses_yolu = asyncio.run(
        generate_audio_summary(
            text=seslendirilecek_metin,
            voice="tr-TR-EmelNeural",
            rate="+0%",
        )
    )

    dosya_nesnesi = Path(uretilen_ses_yolu)
    assert dosya_nesnesi.exists(), f"Üretilen ses dosyası diskte bulunamadı: {dosya_nesnesi}"
    assert dosya_nesnesi.suffix.lower() == ".mp3", "Üretilen dosya uzantısı '.mp3' olmalıdır."
    assert dosya_nesnesi.stat().st_size > 0, "Oluşturulan MP3 dosyasının boyutu 0'dan büyük olmalıdır."

    # Hedef klasörün 'static/audio' olduğunu doğrula
    assert "static" in dosya_nesnesi.parts and "audio" in dosya_nesnesi.parts, (
        f"Dosya 'static/audio' dizini altında oluşturulmalıdır. Mevcut yol: {dosya_nesnesi}"
    )

    boyut_kb = dosya_nesnesi.stat().st_size / 1024
    RenkliKonsol.basarili("Edge-TTS Türkçe ses dosyası başarıyla üretildi.")
    RenkliKonsol.detay("Dosya Adı", dosya_nesnesi.name)
    RenkliKonsol.detay("Dosya Boyutu", f"{boyut_kb:.2f} KB")
    RenkliKonsol.detay("Konum", str(dosya_nesnesi))


def test_06_fastapi_sunucu_ve_uc_noktalar():
    """6. Aşama: TestClient kullanarak FastAPI uç noktalarının HTTP 200 doğrulaması."""
    RenkliKonsol.baslik("AŞAMA 6: FastAPI Sunucusu ve REST Uç Noktaları Testi")

    istemci = TestClient(app)

    # 1. Kök Dizin Kontrolü (GET /)
    kok_yanit = istemci.get("/")
    assert kok_yanit.status_code == 200, f"GET / endpoint'i 200 dönmelidir. Alınan: {kok_yanit.status_code}"
    RenkliKonsol.basarili("GET / (Sağlık & API Bilgisi) HTTP 200 OK doğrulandı.")

    # 2. Metin Özetleme Endpoint'i (POST /summarize)
    ozet_istegi = {
        "text": (
            "Yapay zeka sistemleri günümüzde dil işleme, görüntü tanıma ve otonom sistemlerde yaygın olarak kullanılır. "
            "Derin öğrenme modelleri çok katmanlı yapay sinir ağları sayesinde karmaşık örüntüleri öğrenebilmektedir."
        ),
        "max_length": 100,
        "min_length": 20,
    }
    ozet_yanit = istemci.post("/summarize", json=ozet_istegi)
    assert ozet_yanit.status_code == 200, f"POST /summarize başarısız: {ozet_yanit.text}"
    ozet_verisi = ozet_yanit.json()
    assert ozet_verisi["status"] == "success", "Özet durumu 'success' olmalıdır."
    assert "summary" in ozet_verisi and len(ozet_verisi["summary"]) > 0
    RenkliKonsol.basarili("POST /summarize (Metin Özetleme) HTTP 200 OK doğrulandı.")
    RenkliKonsol.detay("Özet Sonucu", ozet_verisi["summary"])

    # 3. Notlara Soru Sorma RAG Endpoint'i (POST /ask-notes)
    rag_istegi = {
        "question": "Newton'un eylemsizlik yasası nedir?",
        "n_results": 2,
    }
    rag_yanit = istemci.post("/ask-notes", json=rag_istegi)
    assert rag_yanit.status_code == 200, f"POST /ask-notes başarısız: {rag_yanit.text}"
    rag_verisi = rag_yanit.json()
    assert rag_verisi["status"] == "success", "RAG sorgu durumu 'success' olmalıdır."
    assert "answer" in rag_verisi and len(rag_verisi["answer"]) > 0
    RenkliKonsol.basarili("POST /ask-notes (RAG Soru Sorma) HTTP 200 OK doğrulandı.")
    RenkliKonsol.detay("RAG Yanıtı", rag_verisi["answer"][:120] + "...")

    # 4. Otomatik Quiz & Flashcard Üretim Endpoint'i (POST /generate-quiz)
    quiz_istegi = {
        "text": (
            "Mitoz Bölünme: Tek hücreli canlılarda üremeyi, çok hücrelilerde büyüme ve onarımı sağlar.\n"
            "Mayoz Bölünme: Eşeyli üreyen canlılarda gamet oluşumunu sağlar ve kromozom sayısını yarıya indirir."
        ),
        "num_questions": 2,
    }
    quiz_yanit = istemci.post("/generate-quiz", json=quiz_istegi)
    assert quiz_yanit.status_code == 200, f"POST /generate-quiz başarısız: {quiz_yanit.text}"
    quiz_verisi = quiz_yanit.json()
    assert quiz_verisi["status"] == "success"
    assert len(quiz_verisi["quiz"]) >= 1
    assert len(quiz_verisi["flashcards"]) >= 1
    RenkliKonsol.basarili("POST /generate-quiz (Quiz & Kart Üretimi) HTTP 200 OK doğrulandı.")
    RenkliKonsol.detay("Üretilen Soru Adedi", str(len(quiz_verisi["quiz"])))
    RenkliKonsol.detay("Üretilen Kart Adedi", str(len(quiz_verisi["flashcards"])))

    # 5. Sesli Özet Endpoint'i (POST /audio-summary)
    ses_istegi = {
        "text": "Ders notu API uç nokta seslendirme testi tamamlanmaktadır.",
        "voice": "tr-TR-EmelNeural",
    }
    ses_yanit = istemci.post("/audio-summary", json=ses_istegi)
    assert ses_yanit.status_code == 200, f"POST /audio-summary başarısız: {ses_yanit.text}"
    ses_verisi = ses_yanit.json()
    assert ses_verisi["status"] == "success"
    assert "audio_url" in ses_verisi and ses_verisi["audio_url"].startswith("/static/audio/")
    RenkliKonsol.basarili("POST /audio-summary (Edge-TTS Sesli Özet) HTTP 200 OK doğrulandı.")
    RenkliKonsol.detay("Statik Ses Bağlantısı", ses_verisi["audio_url"])

    # 6. Üretilen Ses Dosyasının Statik Uç Noktadan İndirilebilirlik Kontrolü
    statik_dosya_yanit = istemci.get(ses_verisi["audio_url"])
    assert statik_dosya_yanit.status_code == 200, (
        f"Statik ses dosyası indirilemedi. Kod: {statik_dosya_yanit.status_code}"
    )
    assert len(statik_dosya_yanit.content) > 0, "İndirilen ses içeriği boş olamaz."
    RenkliKonsol.basarili("GET /static/audio/... (Statik Dosya Erişimi) HTTP 200 OK doğrulandı.")


def test_07_word_ve_sinav_tahmin_motoru():
    """7. Aşama: Word (.docx) ChromaDB indeksleme ve sınav sorusu tahmin motoru testi."""
    RenkliKonsol.baslik("AŞAMA 7: Word (.docx) İndeksleme ve Sınav Soru Tahmin Motoru")

    # 1. Bellek içi sahte DOCX baytı oluştur (python-docx)
    try:
        from docx import Document as DocxDocument
        import io as _io

        sahte_docx = DocxDocument()
        sahte_docx.add_heading("Kimya Ders Notu", level=1)
        sahte_docx.add_paragraph(
            "Mol kavramı: Kimyasal madde miktarının ölçüm birimi olup 6.022x10^23 parçacık içerir."
        )
        sahte_docx.add_paragraph(
            "Periyodik sistem: Elementlerin atom numarasına göre sıralandığı çizelgedir."
        )
        sahte_docx.add_paragraph(
            "Asit ve baz: Asitler H+ iyonu verirken, bazlar OH- iyonu üretir."
        )

        docx_buffer = _io.BytesIO()
        sahte_docx.save(docx_buffer)
        docx_bayt = docx_buffer.getvalue()
        RenkliKonsol.bilgi(f"Sahte DOCX belgesi hafızada oluşturuldu ({len(docx_bayt):,} bayt).")
    except ImportError:
        pytest.skip("python-docx yüklenmemiş, bu test atlanıyor.")
        return

    # 2. DOCX'i ChromaDB'ye indeksle
    eklenen_parca = add_docx_to_rag(docx_bayt, source_name="Kimya_Notu_Word")
    assert eklenen_parca > 0, "Word belgesinden en az bir parça ChromaDB'ye eklenmelidir."
    RenkliKonsol.basarili(f"Word belgesi ChromaDB'ye başarıyla indekslendi: {eklenen_parca} parça.")

    # 3. /index-docx FastAPI Endpoint testi
    istemci = TestClient(app)
    docx_buffer2 = _io.BytesIO()
    sahte_docx.save(docx_buffer2)
    docx_buffer2.seek(0)
    yanit = istemci.post(
        "/index-docx",
        files={"file": ("kimya_notu.docx", docx_buffer2,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert yanit.status_code == 200, f"POST /index-docx başarısız: {yanit.text}"
    yanit_verisi = yanit.json()
    assert yanit_verisi["status"] == "success"
    assert yanit_verisi["chunks_added"] > 0
    RenkliKonsol.basarili(f"POST /index-docx HTTP 200 OK doğrulandı ({yanit_verisi['chunks_added']} parça).")

    # 4. Sınav Sorusu Tahmin Motoru - Doğrudan fonksiyon testi
    tahmin_sonucu = predict_exam_questions(
        past_questions_text="Mol hesabı nasıl yapılır? Asit ve baz tepkimelerini açıklayınız.",
        top_n=3,
    )
    assert isinstance(tahmin_sonucu, dict), "Tahmin sonucu sözlük formatında dönmelidir."
    assert tahmin_sonucu.get("durum") in ("basarili", "hata"), "'durum' alanı bulunmalıdır."
    if tahmin_sonucu["durum"] == "basarili":
        assert "tahmin_sorular" in tahmin_sonucu, "Tahmin soru listesi bulunmalıdır."
        assert len(tahmin_sonucu["tahmin_sorular"]) > 0, "En az bir tahmin sorusu üretilmelidir."
        assert "en_onemli_konular" in tahmin_sonucu, "Konu sıralaması bulunmalıdır."
        RenkliKonsol.basarili(f"{len(tahmin_sonucu['tahmin_sorular'])} adet tahmin sorusu üretildi.")
        ilk_tahmin = tahmin_sonucu["tahmin_sorular"][0]
        RenkliKonsol.detay("En Olası Soru", ilk_tahmin["tahmin_soru"])
        RenkliKonsol.detay("Konu", ilk_tahmin["konu"])
        RenkliKonsol.detay("Önem", ilk_tahmin["onem_seviyesi"])
    else:
        RenkliKonsol.bilgi("ChromaDB boş, tahmin yapılamadı (beklenen durum).")

    # 5. /predict-exam-questions FastAPI Endpoint testi
    api_yanit = istemci.post(
        "/predict-exam-questions",
        json={
            "past_questions": "Mol kavramı nedir? Periyodik sistem nasıl okunur?",
            "top_n": 3,
        },
    )
    assert api_yanit.status_code == 200, f"POST /predict-exam-questions başarısız: {api_yanit.text}"
    api_yanit_verisi = api_yanit.json()
    RenkliKonsol.basarili("POST /predict-exam-questions HTTP 200 OK doğrulandı.")
    RenkliKonsol.detay("API Yanıtı durum", api_yanit_verisi.get("durum", "-") if isinstance(api_yanit_verisi, dict) else "-")


def test_08_powerpoint_sunum_uretimi():
    """8. Aşama: Ders notundan otomatik PowerPoint (.pptx) sunum üretimi ve dosya doğrulama testi."""
    RenkliKonsol.baslik("AŞAMA 8: Otomatik PowerPoint (.pptx) Sunum Üretim Testi")

    sunum_notu = (
        "FOTOSENTEŞ\n"
        "Kloroplast organelinde gerçekleşen ve ışık enerjisinin kimyasal enerjiye dönüştürüldüğü süreçtir.\n"
        "Klorofil \u0131şık enerjisini emer.\n"
        "Su ve karbondioksit hammadde olarak kullanılır.\n\n"
        "HUCRESEL SOLUNUM\n"
        "Glikozun oksijen varlığında parçalanarak ATP üretilmesidir.\n"
        "Mitokondride gerçekleşir.\n"
        "Son ürünler: Su, karbondioksit ve ATP enerjisidir."
    )

    # 1. Doğrudan fonksiyon testi
    uretilen_yol = generate_presentation(
        text=sunum_notu,
        title="Biyoloji Ders Sunumu",
        subtitle="Test Sunumu",
        max_slides=4,
        filename="test_sunumu.pptx",
    )

    assert uretilen_yol.exists(), f"PowerPoint dosyası oluşturulmalıdır: {uretilen_yol}"
    assert uretilen_yol.suffix.lower() == ".pptx", "Dosya uzantısı .pptx olmalıdır."
    assert uretilen_yol.stat().st_size > 0, "PowerPoint dosyası boş olamaz."
    assert "static" in uretilen_yol.parts and "presentations" in uretilen_yol.parts, (
        f"Dosya 'static/presentations' dizininde olmalıdır. Mevcut: {uretilen_yol}"
    )

    boyut_kb = uretilen_yol.stat().st_size / 1024
    RenkliKonsol.basarili("PowerPoint sunumu başarıyla üretildi.")
    RenkliKonsol.detay("Dosya Adı", uretilen_yol.name)
    RenkliKonsol.detay("Dosya Boyutu", f"{boyut_kb:.2f} KB")
    RenkliKonsol.detay("Konum", str(uretilen_yol))

    # 2. PPTX geçerlilik kontrolü (slayt sayısı)
    try:
        from pptx import Presentation
        prs = Presentation(str(uretilen_yol))
        slayt_sayisi = len(prs.slides)
        # Kapak + içerik slaytları + teşekkür = en az 3 slayt
        assert slayt_sayisi >= 3, f"Sunumda en az 3 slayt (kapak + içerik + teşekkür) bulunmalıdır. Mevcut: {slayt_sayisi}"
        RenkliKonsol.basarili(f"Sunum geçerli bir .pptx dosyası: {slayt_sayisi} slayt içeriyor.")
    except ImportError:
        RenkliKonsol.bilgi("python-pptx yüklenmemiş, slayt sayısı doğrulanmadı.")

    # 3. FastAPI Endpoint testi
    istemci = TestClient(app)
    api_yanit = istemci.post(
        "/generate-presentation",
        json={
            "text": sunum_notu,
            "title": "Test API Sunumu",
            "max_slides": 3,
        },
    )
    assert api_yanit.status_code == 200, f"POST /generate-presentation başarısız: {api_yanit.text}"
    api_verisi = api_yanit.json()
    assert api_verisi["status"] == "success"
    assert "download_url" in api_verisi and api_verisi["download_url"].startswith("/static/presentations/")

    RenkliKonsol.basarili("POST /generate-presentation HTTP 200 OK doğrulandı.")
    RenkliKonsol.detay("İndirme Bağlantısı", api_verisi["download_url"])


def test_09_excel_csv_indeksleme():
    """9. Aşama: Excel (.xlsx) ve CSV (.csv) dosyalarını ChromaDB'ye indeksleme testi."""
    RenkliKonsol.baslik("AŞAMA 9: Excel / CSV Veri İndeksleme ve /index-excel API Testi")

    try:
        import pandas as pd
        import io as _io
    except ImportError:
        pytest.skip("pandas yüklenmemiş, bu test atlanıyor.")
        return

    # 1. Bellek içi sahte CSV oluştur
    csv_icerik = (
        "Konu,Açıklama,Örnek\n"
        "Fotosemtez,Kloroplastta gerçekleşen enerji dönüşümü,Yapraklar güneş ışığı alır\n"
        "Solunum,Mitokondride ATP üretimi,Glikoz parçalanır\n"
        "Mitoz,Hücre bölünmesi,Büyüme ve onarım\n"
        "Mayoz,Gamet oluşumu,Sperm ve yumurta"
    )
    csv_bayt = csv_icerik.encode("utf-8")

    # 2. Doğrudan fonksiyon: CSV
    eklenen_csv = add_excel_to_rag(csv_bayt, source_name="Biyoloji_CSV", dosya_uzantisi=".csv")
    assert eklenen_csv > 0, "CSV'den en az bir parça ChromaDB'ye eklenmelidir."
    RenkliKonsol.basarili(f"CSV ChromaDB'ye indekslendi: {eklenen_csv} parça.")

    # 3. Bellek içi sahte XLSX oluştur
    df = pd.DataFrame({
        "Terim": ["Mol", "Atom", "Elektron", "Proton"],
        "Tanım": [
            "6.022x10^23 parçacık içeren miktar",
            "Elementin en küçük birimi",
            "Negatif yüklü parçacık",
            "Pozitif yüklü parçacık",
        ],
        "Birim": ["mol", "-", "e-", "p+"],
    })
    xlsx_buffer = _io.BytesIO()
    df.to_excel(xlsx_buffer, index=False)
    xlsx_bayt = xlsx_buffer.getvalue()

    # 4. Doğrudan fonksiyon: XLSX
    eklenen_xlsx = add_excel_to_rag(xlsx_bayt, source_name="Kimya_Excel", dosya_uzantisi=".xlsx")
    assert eklenen_xlsx > 0, "XLSX'den en az bir parça ChromaDB'ye eklenmelidir."
    RenkliKonsol.basarili(f"XLSX ChromaDB'ye indekslendi: {eklenen_xlsx} parça.")

    # 5. FastAPI /index-excel Endpoint testi (CSV)
    istemci = TestClient(app)
    csv_yanit = istemci.post(
        "/index-excel",
        files={"file": ("biyoloji.csv", _io.BytesIO(csv_bayt), "text/csv")},
    )
    assert csv_yanit.status_code == 200, f"POST /index-excel (CSV) başarısız: {csv_yanit.text}"
    assert csv_yanit.json()["status"] == "success"
    RenkliKonsol.basarili("POST /index-excel (CSV) HTTP 200 OK doğrulandı.")

    # 6. FastAPI /index-excel Endpoint testi (XLSX)
    xlsx_yanit = istemci.post(
        "/index-excel",
        files={"file": ("kimya.xlsx", _io.BytesIO(xlsx_bayt),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert xlsx_yanit.status_code == 200, f"POST /index-excel (XLSX) başarısız: {xlsx_yanit.text}"
    assert xlsx_yanit.json()["status"] == "success"
    RenkliKonsol.basarili("POST /index-excel (XLSX) HTTP 200 OK doğrulandı.")
    RenkliKonsol.detay("XLSX Parça Sayısı", str(xlsx_yanit.json()["chunks_added"]))


def test_09b_pdf_belgesi_indeksleme():
    """9b. Aşama: PDF ders notunu ChromaDB'ye indeksleme ve /index-pdf API testi."""
    RenkliKonsol.baslik("AŞAMA 9b: PDF Ders Notu İndeksleme ve /index-pdf API Testi")

    # 1. Bellek içi geçerli metin içeren PDF oluştur
    import io as _io
    sahte_pdf_metni = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 58 >>
stream
BT
/F1 16 Tf
50 700 Td
(Fizik Ders Notu: Kinetik ve Potansiyel Enerji) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000318 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
426
%%EOF"""

    # 2. Doğrudan fonksiyon testi: add_pdf_to_rag
    eklenen_pdf = add_pdf_to_rag(sahte_pdf_metni, source_name="Fizik_Enerji_Notu.pdf")
    assert eklenen_pdf > 0, "PDF belgesinden en az bir parça ChromaDB'ye eklenmelidir."
    RenkliKonsol.basarili(f"PDF belgesi ChromaDB'ye başarıyla indekslendi: {eklenen_pdf} parça.")

    # 3. FastAPI /index-pdf Endpoint testi
    istemci = TestClient(app)
    pdf_yanit = istemci.post(
        "/index-pdf",
        files={"file": ("enerji_notu.pdf", _io.BytesIO(sahte_pdf_metni), "application/pdf")},
    )
    assert pdf_yanit.status_code == 200, f"POST /index-pdf başarısız: {pdf_yanit.text}"
    pdf_verisi = pdf_yanit.json()
    assert pdf_verisi["status"] == "success"
    assert pdf_verisi["chunks_added"] > 0
    RenkliKonsol.basarili(f"POST /index-pdf HTTP 200 OK doğrulandı ({pdf_verisi['chunks_added']} parça).")
    RenkliKonsol.detay("Eklenen Parça", str(pdf_verisi["chunks_added"]))


def test_10_kavram_sozlugu_uretimi():
    """10. Aşama: Ders notundan Kavram Sözlüğü, Formül ve Anahtar Kelime üretim testi."""
    RenkliKonsol.baslik("AŞAMA 10: Kavram Sözlüğü (Glossary) ve Formül Çıkarma Testi")

    kimya_notu = (
        "Mol: Kimyasal madde miktarının ölçüm birimidir. 1 mol = 6.022x10^23 parçacık içerir.\n"
        "Atom: Elementin kimyasal özelliklerini taşıyan en küçük yapı birimidir.\n"
        "Newton’un İkinci Yasası: F = m * a bığıntısıyla ifade edilir.\n"
        "Klorofil yeşil renkli pigmenttir ve ışık enerjisini soğurarak fotosemtezi başlatmaya yardımcı olur.\n"
        "DNA: Kalıtsal bilgiyi taşıyan ve deoksiriböz şekeri içeren çift sarmal yapılı nkükleik asittir."
    )

    # 1. Doğrudan fonksiyon testi
    sozluk_sonucu = generate_glossary(kimya_notu)
    assert isinstance(sozluk_sonucu, dict), "Sonuç sözlük formatında olmalıdır."
    assert "sozluk" in sozluk_sonucu, "'sozluk' anahtarı bulunmalıdır."
    assert "formuller" in sozluk_sonucu, "'formuller' anahtarı bulunmalıdır."
    assert "anahtar_kelimeler" in sozluk_sonucu, "'anahtar_kelimeler' anahtarı bulunmalıdır."
    assert sozluk_sonucu["toplam_terim"] > 0, "En az bir terim çıkarılmış olmalıdır."

    RenkliKonsol.basarili(f"{sozluk_sonucu['toplam_terim']} terim ve tanım çıkarıldı.")
    RenkliKonsol.detay("Formül Sayısı", str(len(sozluk_sonucu["formuller"])))
    RenkliKonsol.detay("Anahtar Kelime", ", ".join(sozluk_sonucu["anahtar_kelimeler"][:5]))
    if sozluk_sonucu["sozluk"]:
        ilk = sozluk_sonucu["sozluk"][0]
        RenkliKonsol.detay("1. Terim", ilk["terim"])
        RenkliKonsol.detay("Kategori", ilk["kategori"])

    # 2. FastAPI /generate-glossary endpoint testi
    istemci = TestClient(app)
    yanit = istemci.post("/generate-glossary", json={"text": kimya_notu})
    assert yanit.status_code == 200, f"POST /generate-glossary başarısız: {yanit.text}"
    veri = yanit.json()
    assert veri["status"] == "success"
    assert "sozluk" in veri
    assert "anahtar_kelimeler" in veri
    RenkliKonsol.basarili("POST /generate-glossary HTTP 200 OK doğrulandı.")
    RenkliKonsol.detay("API Terim Sayısı", str(veri["toplam_terim"]))


def test_11_zihin_haritasi_uretimi():
    """11. Aşama: Ders notundan Düğüm-Kenar (Zihin Haritası) verisi üretim testi."""
    RenkliKonsol.baslik("AŞAMA 11: Zihin Haritası (MindMap) Düğüm-Kenar Verisi Testi")

    fizik_notu = (
        "Newton kuvvet ve hareket yasalarını tanımlamıştır. "
        "Kuvvet kütle ve ivmenin çarpımına eşittir. "
        "Hareket eden cisimler ivme kazanır. "
        "Eylemsizlik hareket değişikliğine karşı direnme özelliğidir. "
        "Gravitasyon kuvveti kütle ile doğru orantılıdır. "
        "Enerji korunumu yasasına göre toplam mekanik enerji sabittir. "
        "Kinetik enerji hareket eden cismin enerjisidir. "
        "Potansiyel enerji konuma bağlı depolanan enerjidir."
    )

    # 1. Doğrudan fonksiyon testi
    harita_sonucu = generate_mindmap(fizik_notu, max_nodes=15)
    assert isinstance(harita_sonucu, dict), "Sonuç sözlük formatında olmalıdır."
    assert "merkez_konu" in harita_sonucu and harita_sonucu["merkez_konu"], "Merkez konu bulunmalıdır."
    assert "dugumler" in harita_sonucu and len(harita_sonucu["dugumler"]) >= 3, (
        "En az 3 düğüm üretilmelidir."
    )
    assert "kenarlar" in harita_sonucu, "Kenar listesi bulunmalıdır."
    assert "toplam_dugum" in harita_sonucu

    # Düğüm yapısını doğrula
    ilk_dugum = harita_sonucu["dugumler"][0]
    assert "id" in ilk_dugum, "Düğümde 'id' alanı bulunmalıdır."
    assert "etiket" in ilk_dugum, "Düğümde 'etiket' alanı bulunmalıdır."
    assert "seviye" in ilk_dugum, "Düğümde 'seviye' alanı bulunmalıdır."
    assert "agirlik" in ilk_dugum, "Düğümde 'agirlik' alanı bulunmalıdır."

    RenkliKonsol.basarili(f"Zihin haritası üretildi: {harita_sonucu['toplam_dugum']} düğüm.")
    RenkliKonsol.detay("Merkez Konu", harita_sonucu["merkez_konu"])
    RenkliKonsol.detay("Kenar Sayısı", str(len(harita_sonucu["kenarlar"])))
    RenkliKonsol.detay("Örnek Düğümler",
        ", ".join(d["etiket"] for d in harita_sonucu["dugumler"][:5]))

    # 2. FastAPI /generate-mindmap endpoint testi
    istemci = TestClient(app)
    yanit = istemci.post("/generate-mindmap", json={"text": fizik_notu, "max_nodes": 15})
    assert yanit.status_code == 200, f"POST /generate-mindmap başarısız: {yanit.text}"
    veri = yanit.json()
    assert veri["status"] == "success"
    assert "merkez_konu" in veri and veri["merkez_konu"]
    assert len(veri["dugumler"]) >= 3
    RenkliKonsol.basarili("POST /generate-mindmap HTTP 200 OK doğrulandı.")
    RenkliKonsol.detay("API Merkez", veri["merkez_konu"])
    RenkliKonsol.detay("API Düğüm Sayısı", str(veri["toplam_dugum"]))


def test_08b_gelismis_pptx_tablo_ve_iki_sutun():
    """8b. Aşama: Gelişmiş PowerPoint motoru — Tablo ve İki Sütun slayt yapıları testi."""
    RenkliKonsol.baslik("AŞAMA 8b: Gelişmiş PPTX — Tablo + İki Sütun + Callout")

    sunum_metni = (
        "Fotosemtez\nKloroplastta gerçekleşir. Klorofil ışık enerjisi emer.\n\n"
        "Solunum\nMitokondride ATP üretimi sağlanır."
    )
    tablo = [{
        "baslik": "Biyoloji Terimler Tablosu",
        "sutunlar": ["Terim", "Tanım", "Bulunduğu Yer"],
        "satirlar": [
            ["Mitokondri", "ATP üretimi yapılır", "Hücre"],
            ["Kloroplast", "Fotosemtez yapılır", "Bitki hücresi"],
            ["Ribozom", "Protein sentezlenir", "Hücre plazması"],
        ],
    }]
    iki_sutun = [{
        "baslik": "Fotosemtez vs Solunum",
        "sol": ["Kloroplastta yapılır", "İşık gerekir", "CO2 tüketilir"],
        "sag": ["Mitokondride yapılır", "Oksijen gerekir", "CO2 üretilir"],
        "sol_baslik": "Fotosemtez",
        "sag_baslik": "Solunum",
    }]
    calloutlar = [
        {"metin": "Kloroplast sadece bitki hücrelerinde bulunur!", "tip": "bilgi"},
    ]

    yol = generate_presentation(
        text=sunum_metni,
        title="Biyoloji Gelişmiş Sunum",
        max_slides=3,
        filename="test_gelismis_sunum.pptx",
        tablo_verisi=tablo,
        iki_sutun_verisi=iki_sutun,
        callout_listesi=calloutlar,
    )

    assert yol.exists() and yol.stat().st_size > 0, "Dosya oluşturulmalı ve boş olmamaldır."

    from pptx import Presentation
    prs = Presentation(str(yol))
    slayt_sayisi = len(prs.slides)
    # Kapak + Gündem + içerik + tablo + iki-sütun + teşekkür = en az 5
    assert slayt_sayisi >= 5, f"En az 5 slayt bekleniyor. Mevcut: {slayt_sayisi}"

    boyut_kb = yol.stat().st_size / 1024
    RenkliKonsol.basarili(f"Gelişmiş sunum oluşturuldu: {slayt_sayisi} slayt, {boyut_kb:.1f} KB")
    RenkliKonsol.detay("Dosya", yol.name)
    RenkliKonsol.detay("Slayt Sayısı", str(slayt_sayisi))


def test_09_test_verilerini_temizle():
    """9. Aşama: Testler bittikten sonra ChromaDB test verilerini tamamen temizler."""
    RenkliKonsol.baslik("AŞAMA 9: Test Verilerinin Temizlenmesi (Teardown)")
    clear_rag_notes()
    stats = get_rag_stats()
    assert stats["total_chunks"] == 0, "ChromaDB test verileri temizlenmiş olmalıdır."
    RenkliKonsol.basarili("ChromaDB test verileri başarıyla temizlendi, veritabanı sıfırlandı.")
