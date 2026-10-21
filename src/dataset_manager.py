import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = DATA_DIR / "ai_cache.json"
DATASET_FILE = DATA_DIR / "fine_tuning_dataset.jsonl"


def _compute_cache_key(task_type: str, prompt: str, context: str) -> str:
    """Sorgu, bağlam ve görev türünden benzersiz bir SHA-256 hash üretir."""
    norm_task = (task_type or "").strip().lower()
    norm_prompt = (prompt or "").strip().lower()
    norm_ctx = (context or "").strip()
    raw = f"{norm_task}|||{norm_prompt}|||{norm_ctx}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_ai_cache() -> Dict[str, Any]:
    """Yerel ai_cache.json dosyasını yükler."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("ai_cache.json okuma hatası: %s", exc)
        return {}


def save_ai_cache(cache: Dict[str, Any]) -> None:
    """Yerel ai_cache.json dosyasını günceller."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("ai_cache.json kaydetme hatası: %s", exc)


def get_cached_response(task_type: str, prompt: str, context: str) -> Optional[Any]:
    """1. Katman: Yerel önbellekten (ai_cache.json) yanıtı arar.
    
    Hash anahtarı eşleşirse veya bağlam/soru daha önce kaydedilmişse doğrudan döndürür.
    """
    try:
        cache = load_ai_cache()
        key = _compute_cache_key(task_type, prompt, context)
        if key in cache:
            entry = cache[key]
            # Sözlük ise 'response' anahtarını döndür, değilse doğrudan nesneyi döndür
            if isinstance(entry, dict) and "response" in entry:
                logger.info("Cache HIT [Tam Hash]: %s (%s)", key[:12], task_type)
                return entry["response"]
            return entry

        # Not: Bağlam farklıysa başka belgenin yanıtı dönmesin diye kısmi eşleşme yapılmaz.

    except Exception as exc:
        logger.warning("Cache arama sırasında hata: %s", exc)
    return None


def set_cached_response(task_type: str, prompt: str, context: str, response: Any) -> None:
    """1. Katman: Başarılı yanıtı yerel ai_cache.json içine kaydeder."""
    try:
        cache = load_ai_cache()
        key = _compute_cache_key(task_type, prompt, context)
        cache[key] = {
            "task_type": task_type,
            "prompt": prompt,
            "context_hash": hashlib.sha256(context.encode("utf-8")).hexdigest() if context else "",
            "response": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_ai_cache(cache)
        logger.info("Cache SAVE: %s (%s)", key[:12], task_type)
    except Exception as exc:
        logger.warning("Cache kaydetme sırasında hata: %s", exc)


def log_fine_tuning_sample(
    instruction: str,
    context: str,
    response: Any,
    task_type: str,
) -> None:
    """2. Katman: Model Eğitim Veri Seti (Knowledge Distillation).
    
    Gemini'den gelen her başarılı yanıtı standart Instruction-Tuning formatında
    data/fine_tuning_dataset.jsonl dosyasına yeni satır olarak ekler.
    """
    try:
        payload = {
            "instruction": instruction,
            "context": context,
            "response": response,
        }
        line = json.dumps(payload, ensure_ascii=False)
        with open(DATASET_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        logger.info("Fine-tuning veri setine yeni örnek eklendi [Görev: %s]", task_type)
    except Exception as exc:
        logger.error("Fine-tuning verisi kaydedilirken hata: %s", exc)
