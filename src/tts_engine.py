import logging
import uuid
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union

import edge_tts

# Loglama ayarları
logger = logging.getLogger(__name__)

# Temel dizin yapılandırmaları
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO_DIR = BASE_DIR / "static" / "audio"
DEFAULT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Projenin kesintisiz çalışmasını garanti eden zorunlu AI sesi
DEFAULT_VOICE = "tr-TR-EmelNeural"


def clean_text_for_tts(text: str) -> str:
    """
    Ses motoruna gönderilmeden önce metni analiz eder ve temizler.
    Markdown işaretlerini, özel karakterleri ve okumayı bozacak
    fazla boşlukları ortadan kaldırır.
    """
    if not text:
        return ""
        
    # Markdown (**, *, #) vb. işaretleri temizle
    text = re.sub(r'[*#_]', '', text)
    
    # Çoklu boşlukları tek boşluğa düşür
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


async def generate_audio_summary(
    text: str,
    output_path: Optional[Union[str, Path]] = None,
    voice: str = DEFAULT_VOICE,
    rate: Optional[str] = None,
    pitch: Optional[str] = None,
) -> Path:
    """
    RAG veya Özet motorundan gelen metinleri konuşmaya çevirir.
    Microsoft sunucularındaki erkek ses kısıtlamalarından dolayı,
    kullanıcı arayüzden ne seçerse seçsin sistemin çökmemesi için
    arka planda %100 kararlı olan Emel sesini (Kadın) kullanmaya zorlar.
    
    Args:
        text (str): Seslendirilecek özet metni.
        output_path (Path|str, optional): Çıktı dosyasının yolu.
        voice (str): Ses modeli (Güvenlik için devre dışı bırakıldı, DEFAULT kullanılır).
        rate (str, optional): Konuşma hızı (örn: '+10%').
        pitch (str, optional): Ses tonu (örn: '+5Hz').
        
    Returns:
        Path: Üretilen .mp3 dosyasının konumu.
    """
    cleaned_text = clean_text_for_tts(text)
    if not cleaned_text:
        raise ValueError("Analiz motorundan gelen seslendirilecek metin boş olamaz.")

    # Dosya kaydetme yolunu belirle
    if output_path is None:
        filename = f"summary_{uuid.uuid4().hex[:10]}.mp3"
        target_file = DEFAULT_AUDIO_DIR / filename
    else:
        target_file = Path(output_path)

    # Klasörlerin var olduğundan emin ol
    target_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Edge-TTS motoru başlatılıyor...")
    logger.info("Zorunlu Güvenli Ses Modeli Kullanılıyor: %s", DEFAULT_VOICE)

    # Edge-TTS API Parametreleri (Dinamik)
    kwargs = {
        "text": cleaned_text,
        "voice": DEFAULT_VOICE  # Gelen voice parametresi eziliyor
    }
    
    if rate and rate != "+0%":
        kwargs["rate"] = rate
    if pitch and pitch != "+0Hz":
        kwargs["pitch"] = pitch

    try:
        # İletişim kur ve MP3 olarak kaydet
        communicate = edge_tts.Communicate(**kwargs)
        await communicate.save(str(target_file))
        logger.info("Ses dosyası başarıyla üretildi: %s", target_file.name)
        
    except Exception as e:
        logger.error("Edge-TTS motoru kritik hata verdi: %s", str(e))
        raise RuntimeError(f"Yapay zekâ ses motoru yanıt vermedi: {e}")

    return target_file


async def get_turkish_voices() -> List[Dict[str, str]]:
    """
    Web arayüzüne (Frontend) sunulacak kullanılabilir Türkçe ses listesi.
    Sistemin kilitlenmesini önlemek adına sadece en garantili model sunulur.
    """
    return [
        {
            "id": "tr-TR-EmelNeural",
            "name": "Emel (Yapay Zekâ Asistanı - Kesintisiz)",
            "gender": "Female",
            "locale": "tr-TR",
            "is_default": True,
        }
    ]


# Sadece bu dosya çalıştırıldığında test yapılması için
if __name__ == "__main__":
    async def run_test():
        test_text = "Merhaba! Ders notu özetleyici asistanınız başarıyla yapılandırıldı."
        print("--- Ses Modülü Testi Başlatılıyor ---")
        try:
            result = await generate_audio_summary(test_text)
            print(f"✅ Test Başarılı! Dosya kaydedildi: {result}")
        except Exception as e:
            print(f"❌ Test Başarısız! Hata: {e}")
            
    asyncio.run(run_test())