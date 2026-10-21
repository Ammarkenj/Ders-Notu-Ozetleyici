"""Özetleme, sözlük ve zihin haritası servisleri.

Mimari kural: Tüm metin üretimi/özetleme %100 Google Gemini API (gemini-2.5-flash) üzerinden
yapılır. Lokal CPU/işlemci asla metin üretimi veya özetleme için kullanılmaz; bu modülde
herhangi bir yerel model (transformers/torch) veya kural tabanlı üretici bulunmaz.
Gemini yanıt vermezse GeminiUnavailableError fırlatılır.
"""
import logging
from typing import Any, Dict, Optional

from src.gemini_service import generate_with_distillation
from src.quiz_generator import _extract_json_block

logger = logging.getLogger(__name__)

# Gemini'ye gönderilecek maksimum metin uzunluğu (karakter)
MAX_INPUT_CHARS = 24000


def summarize_text(
    text: str,
    max_length: int = 1024,
    min_length: int = 30,
    model_name: Optional[str] = None,  # geriye dönük uyumluluk için; yok sayılır (daima gemini-2.5-flash)
) -> str:
    """Metni Gemini (gemini-2.5-flash) ile özetler."""
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return ""

    # Çok kısa metin zaten özet niteliğindedir (üretim gerektirmez)
    if len(cleaned_text.split()) < 20:
        return cleaned_text

    prompt = (
        "Aşağıdaki ders notunu Türkçe, akademik ve öz bir biçimde özetle. "
        "Önemli kavramları ve sonuçları koru; metinde olmayan bilgi ekleme. "
        f"Özet yaklaşık {max(min_length, 30)}-{max(max_length, 60)} kelime aralığında olsun. "
        "Yalnızca özet metnini yaz.\n\n"
        f"DERS NOTU:\n\"\"\"\n{cleaned_text[:MAX_INPUT_CHARS]}\n\"\"\""
    )
    return generate_with_distillation(
        task_type="text_summary",
        instruction=f"Ders notunu özetle (en fazla {max_length} kelime).",
        context=cleaned_text,
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=2048,
        postprocess=lambda raw: raw.strip() or None,
    )


def generate_quiz_and_flashcards(
    text: str,
    num_questions: int = 5,
) -> Dict[str, Any]:
    """Quiz & Flashcard üretimi (Gemini)."""
    from src.quiz_generator import generate_quiz_and_flashcards as _quiz_gen
    return _quiz_gen(text=text, num_questions=num_questions)


def generate_glossary(text: str) -> dict:
    """Ders notundaki terimleri, tanımları ve formülleri Gemini ile çıkarır.

    Returns:
        {
            "sozluk": [{"terim": ..., "tanim": ..., "kategori": ...}],
            "formuller": ["..."],
            "anahtar_kelimeler": ["..."],
            "toplam_terim": int,
        }
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return {"sozluk": [], "formuller": [], "anahtar_kelimeler": [], "toplam_terim": 0}

    prompt = (
        "Aşağıdaki ders notundan önemli terimleri, tanımlarını, formülleri ve anahtar kelimeleri çıkar.\n"
        "SADECE geçerli bir JSON objesi döndür (markdown bloğu yok):\n"
        "{\"sozluk\": [{\"terim\": \"...\", \"tanim\": \"...\", \"kategori\": \"Tanım|Formül|Kural / Kavram|Ölçü / Birim\"}],\n"
        " \"formuller\": [\"F = m * a\"],\n"
        " \"anahtar_kelimeler\": [\"...\"]}\n"
        "Yalnızca metinde geçen bilgileri kullan.\n\n"
        f"DERS NOTU:\n\"\"\"\n{cleaned[:MAX_INPUT_CHARS]}\n\"\"\""
    )

    def _post(raw: str) -> Optional[dict]:
        data = _extract_json_block(raw)
        if not data or not isinstance(data.get("sozluk"), list) or not data["sozluk"]:
            return None
        sozluk = [
            {
                "terim": str(i.get("terim", "")).strip(),
                "tanim": str(i.get("tanim", "")).strip(),
                "kategori": str(i.get("kategori") or "Tanım").strip(),
            }
            for i in data["sozluk"]
            if isinstance(i, dict) and i.get("terim") and i.get("tanim")
        ]
        if not sozluk:
            return None
        formuller = [str(f) for f in (data.get("formuller") or []) if f]
        anahtar = [str(k) for k in (data.get("anahtar_kelimeler") or []) if k]
        return {
            "sozluk": sozluk,
            "formuller": formuller[:15],
            "anahtar_kelimeler": anahtar[:20],
            "toplam_terim": len(sozluk),
        }

    return generate_with_distillation(
        task_type="glossary_json",
        instruction="Ders notundan kavram sözlüğü, formüller ve anahtar kelimeleri JSON olarak çıkar.",
        context=cleaned,
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=3500,
        postprocess=_post,
    )


def generate_mindmap(text: str, max_nodes: int = 30) -> dict:
    """Ders notundaki kavramların ilişki ağını (Zihin Haritası) Gemini ile üretir.

    D3.js / vis.js uyumlu düğüm + kenar yapısı döndürür:
        {"merkez_konu", "dugumler": [{id, etiket, seviye, agirlik}], "kenarlar": [{kaynak, hedef, iliski}], "toplam_dugum"}
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return {"merkez_konu": "", "dugumler": [], "kenarlar": [], "toplam_dugum": 0}

    prompt = (
        f"Aşağıdaki ders notundaki kavramların hiyerarşik zihin haritasını en fazla {max_nodes} düğümle çıkar.\n"
        "SADECE geçerli bir JSON objesi döndür (markdown bloğu yok):\n"
        "{\"merkez_konu\": \"...\",\n"
        " \"dugumler\": [{\"id\": 0, \"etiket\": \"Merkez\", \"seviye\": 0, \"agirlik\": 1.0}, "
        "{\"id\": 1, \"etiket\": \"Kavram\", \"seviye\": 1, \"agirlik\": 0.8}],\n"
        " \"kenarlar\": [{\"kaynak\": 0, \"hedef\": 1, \"iliski\": \"alt_konu\"}]}\n"
        "Kurallar: id 0 merkez düğümdür (seviye 0); seviye 1-3; agirlik 0-1 arası; "
        "her düğüm en az bir kenarla bağlı olsun; kenarlardaki id'ler dugumler içinde bulunsun.\n\n"
        f"DERS NOTU:\n\"\"\"\n{cleaned[:MAX_INPUT_CHARS]}\n\"\"\""
    )

    def _post(raw: str) -> Optional[dict]:
        data = _extract_json_block(raw)
        if not data or not isinstance(data.get("dugumler"), list) or len(data["dugumler"]) < 2:
            return None
        dugumler = []
        for d in data["dugumler"][:max_nodes]:
            if not isinstance(d, dict) or "id" not in d or not d.get("etiket"):
                return None
            dugumler.append({
                "id": d["id"],
                "etiket": str(d["etiket"]).strip(),
                "seviye": int(d.get("seviye", 1)),
                "agirlik": float(d.get("agirlik", 0.5)),
            })
        idler = {d["id"] for d in dugumler}
        kenarlar = [
            {
                "kaynak": k["kaynak"],
                "hedef": k["hedef"],
                "iliski": str(k.get("iliski") or "alt_konu"),
            }
            for k in (data.get("kenarlar") or [])
            if isinstance(k, dict) and k.get("kaynak") in idler and k.get("hedef") in idler
        ]
        if not kenarlar:
            return None
        return {
            "merkez_konu": str(data.get("merkez_konu") or dugumler[0]["etiket"]),
            "dugumler": dugumler,
            "kenarlar": kenarlar,
            "toplam_dugum": len(dugumler),
        }

    return generate_with_distillation(
        task_type="mindmap_json",
        instruction=f"Ders notundan en fazla {max_nodes} düğümlü zihin haritası (düğüm-kenar JSON) üret.",
        context=cleaned,
        prompt=prompt,
        temperature=0.2,
        max_output_tokens=4000,
        postprocess=_post,
    )
