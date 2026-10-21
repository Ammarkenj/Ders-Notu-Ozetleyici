"""Gemini servis katmanı.

Mimari kural: Tüm LLM/zeka tabanlı işlemler (özet, soru-cevap, sözlük, sınav tahmini,
Mermaid akış şeması, quiz...) YALNIZCA Google Gemini API (gemini-2.5-flash) üzerinden
çalışır. Lokal CPU yalnızca PDF okuma, OCR, chunking ve ChromaDB vektör aramasında
kullanılır ("kurye modu").

Her başarılı Gemini yanıtı otomatik olarak data/fine_tuning_dataset.jsonl dosyasına
(instruction, context, response) formatında eklenir (Knowledge Distillation).
"""
import logging
import os
import time
from typing import Any, Callable, Optional

from dotenv import load_dotenv

from src.dataset_manager import (
    get_cached_response,
    log_fine_tuning_sample,
    set_cached_response,
)

logger = logging.getLogger(__name__)

# .env dosyasını yükle
load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"

_GEMINI_CLIENT = None
_API_KEY = None


class GeminiUnavailableError(RuntimeError):
    """Gemini API anahtarı yok veya API yanıt veremedi (lokal fallback YOKTUR)."""


def get_gemini_api_key() -> Optional[str]:
    """Ortam değişkenlerinden GEMINI_API_KEY değerini okur."""
    global _API_KEY
    if not _API_KEY:
        load_dotenv(override=True)
        _API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return _API_KEY


def get_gemini_client():
    """Google GenAI istemcisini başlatır ve döndürür."""
    global _GEMINI_CLIENT
    api_key = get_gemini_api_key()
    if not api_key:
        logger.error("GEMINI_API_KEY bulunamadı (.env dosyasını kontrol edin).")
        return None

    if _GEMINI_CLIENT is None:
        try:
            from google import genai
            from google.genai import types
            _GEMINI_CLIENT = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60000))
        except Exception as exc:
            logger.error("Google GenAI client başlatılamadı: %s", exc)
            return None

    return _GEMINI_CLIENT


# gemini-2.5-flash bazı hesaplarda yeni kullanıcılara kapatılmıştır (404). Bu durumda YİNE Gemini API
# içindeki sıradaki model denenir; lokal modele asla düşülmez.
GEMINI_FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-flash-latest", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
_UNAVAILABLE_MODELS: set = set()


def _is_model_missing(exc: Exception) -> bool:
    msg = str(exc)
    return "404" in msg or "NOT_FOUND" in msg


def call_gemini(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    primary_model: str = GEMINI_MODEL,
    retries: int = 3,
) -> Optional[str]:
    """Gemini API çağrısı. Öncelik gemini-2.5-flash; model hesapta yoksa (404) sıradaki Gemini
    modeline geçilir. Geçici hatalarda aynı modelle kısa beklemeyle tekrar denenir.
    Lokal CPU/modele ASLA düşülmez; hepsi başarısızsa None döner."""
    client = get_gemini_client()
    if client is None:
        return None

    from google.genai import types

    config_kwargs = dict(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    try:
        # Düşünme token'ları çıktı bütçesini tüketip yanıtı kesmesin
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    config = types.GenerateContentConfig(**config_kwargs)

    models = []
    for m in [primary_model] + GEMINI_FALLBACK_MODELS:
        if m not in models and m not in _UNAVAILABLE_MODELS:
            models.append(m)

    for model_name in models:
        for attempt in range(1, retries + 1):
            try:
                logger.info("Gemini API çağrılıyor... Model: %s (deneme %d)", model_name, attempt)
                response = client.models.generate_content(model=model_name, contents=prompt, config=config)
                text = getattr(response, "text", None)
                if not text and getattr(response, "candidates", None):
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        text = "".join(p.text for p in candidate.content.parts if getattr(p, "text", None))
                if text and text.strip():
                    return text.strip()
            except Exception as exc:
                if _is_model_missing(exc):
                    logger.warning("Gemini modeli kullanılamıyor (%s), sıradaki Gemini modeline geçiliyor.", model_name)
                    _UNAVAILABLE_MODELS.add(model_name)
                    break
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    logger.warning("Gemini kotası dolu (%s), sıradaki Gemini modeline geçiliyor.", model_name)
                    break
                logger.warning("Gemini API hatası (%s, deneme %d): %s", model_name, attempt, exc)
            if attempt < retries:
                time.sleep(1.5 * attempt)

    return None


def generate_with_distillation(
    task_type: str,
    instruction: str,
    context: str,
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 2500,
    postprocess: Optional[Callable[[str], Any]] = None,
) -> Any:
    """Tüm LLM görevlerinin ortak giriş noktası.

    1. ai_cache.json kontrolü (aynı görev + soru + bağlam için tekrar API çağrısı yapılmaz).
    2. Gemini (gemini-2.5-flash) çağrısı.
    3. `postprocess` ham metni doğrular/dönüştürür (geçersizse None döndürmelidir).
    4. Başarılı yanıt cache'e ve fine_tuning_dataset.jsonl'e (instruction, context, response) yazılır.

    Gemini yanıt vermezse GeminiUnavailableError fırlatılır; lokal üretim fallback'i yoktur.
    """
    cached = get_cached_response(task_type=task_type, prompt=instruction, context=context)
    if cached:
        logger.info("Yanıt ai_cache.json üzerinden döndürüldü [%s].", task_type)
        return cached

    if get_gemini_client() is None:
        raise GeminiUnavailableError(
            "Gemini API kullanılamıyor: GEMINI_API_KEY tanımlı değil veya istemci başlatılamadı."
        )

    raw = call_gemini(
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    result = postprocess(raw) if (postprocess and raw) else raw
    if not result:
        raise GeminiUnavailableError("Gemini API geçerli bir yanıt döndürmedi. Lütfen tekrar deneyin.")

    set_cached_response(task_type=task_type, prompt=instruction, context=context, response=result)
    log_fine_tuning_sample(instruction=instruction, context=context, response=result, task_type=task_type)
    return result
