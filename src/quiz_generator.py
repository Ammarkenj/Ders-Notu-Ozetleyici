import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.gemini_service import generate_with_distillation

logger = logging.getLogger(__name__)


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Metin içinden geçerli JSON bloğunu ayıklar."""
    if not text:
        return None

    cleaned = text.strip()
    # ```json ... ``` veya ``` ... ``` bloklarını temizle
    if "```" in cleaned:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()

    # Doğrudan json.loads dene
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # { ... } aralığını bulmayı dene
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = cleaned[start_idx : end_idx + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def _validate_quiz_payload(data: Dict[str, Any], min_items: int = 1) -> bool:
    """Quiz ve flashcard verisinin beklenen şemaya uygun olduğunu doğrular."""
    if not isinstance(data, dict):
        return False
    quiz = data.get("quiz")
    flashcards = data.get("flashcards")
    if not isinstance(quiz, list) or not isinstance(flashcards, list):
        return False
    if len(quiz) < min_items or len(flashcards) < min_items:
        return False

    for q in quiz:
        if not isinstance(q, dict):
            return False
        if "question" not in q or "options" not in q or "correct_answer" not in q:
            return False
        if not isinstance(q["options"], list) or len(q["options"]) != 4:
            return False
        if str(q["correct_answer"]).upper() not in ["A", "B", "C", "D"]:
            return False

    for f in flashcards:
        if not isinstance(f, dict):
            return False
        if "front" not in f or "back" not in f:
            return False

    return True


def generate_gemini_quiz_and_flashcards(
    text: str,
    num_questions: int = 5,
) -> Optional[Dict[str, Any]]:
    """Gemini (gemini-2.5-flash) ile yüksek akademik kalitede çoktan seçmeli quiz ve flashcards üretir."""
    prompt = (
        f"Aşağıdaki ders notu metnini akademik titizlikle analiz et. "
        f"Bu metinden tam olarak {num_questions} adet 4 seçenekli (A, B, C, D) çoktan seçmeli sınav sorusu "
        f"ve en az {num_questions} adet soru-cevap çalışma kartı (flashcard) oluştur.\n\n"
        f"ÖNEMLİ KURALLAR:\n"
        f"1. Her sorunun 'options' listesinde 4 seçenek olmalı ve seçenekler 'A) ...', 'B) ...', 'C) ...', 'D) ...' formatında olmalıdır.\n"
        f"2. 'correct_answer' sadece tek harf olarak 'A', 'B', 'C' veya 'D' olmalıdır.\n"
        f"3. 'explanation' alanında doğru cevabın gerekçesi açıklanmalıdır.\n"
        f"4. Flashcards için 'front' (soru/kavram), 'back' (açıklama/tanım) ve 'topic' alanları bulunmalıdır.\n"
        f"5. SADECE ve SADECE geçerli bir JSON objesi döndür. Başka hiçbir açıklama yazma.\n\n"
        f"İSTENEN JSON ŞEMASI:\n"
        f"{{\n"
        f'  "quiz": [\n'
        f'    {{\n'
        f'      "id": 1,\n'
        f'      "question": "Soru metni?",\n'
        f'      "options": ["A) Seçenek 1", "B) Seçenek 2", "C) Seçenek 3", "D) Seçenek 4"],\n'
        f'      "correct_answer": "A",\n'
        f'      "explanation": "Açıklama metni"\n'
        f'    }}\n'
        f'  ],\n'
        f'  "flashcards": [\n'
        f'    {{\n'
        f'      "id": 1,\n'
        f'      "front": "Kavram / Soru?",\n'
        f'      "back": "Tanım / Cevap",\n'
        f'      "topic": "Konu Başlığı"\n'
        f'    }}\n'
        f'  ]\n'
        f"}}\n\n"
        f"DERS NOTU METNİ:\n\"\"\"\n{text}\n\"\"\""
    )

    def _post(raw: str) -> Optional[Dict[str, Any]]:
        parsed_json = _extract_json_block(raw)
        if not parsed_json or not _validate_quiz_payload(parsed_json, min_items=1):
            logger.warning("Gemini quiz yanıtı beklenen JSON şemasına uymadı.")
            return None

        quiz_items = parsed_json.get("quiz", [])
        flashcard_items = parsed_json.get("flashcards", [])

        for idx, q in enumerate(quiz_items, 1):
            q["id"] = idx
            q["correct_answer"] = str(q["correct_answer"]).strip().upper()
            if "explanation" not in q:
                q["explanation"] = f"Doğru cevap {q['correct_answer']} seçeneğidir."

        for idx, f in enumerate(flashcard_items, 1):
            f["id"] = idx
            if "topic" not in f:
                f["topic"] = "Ders Notu"

        return {
            "quiz": quiz_items[:num_questions],
            "flashcards": flashcard_items[:num_questions],
            "total_questions": len(quiz_items[:num_questions]),
            "total_flashcards": len(flashcard_items[:num_questions]),
        }

    system_instruction = (
        "Sen üniversite düzeyinde sınav hazırlayan uzman bir akademisyensin. "
        "Ders notlarından yüksek kaliteli, pedagojik olarak tutarlı çoktan seçmeli sorular ve çalışma kartları üretirsin. "
        "Yalnızca geçerli JSON formatında yanıt verirsin."
    )

    # Cache -> Gemini -> cache + fine_tuning_dataset.jsonl (merkezi akış)
    return generate_with_distillation(
        task_type="quiz",
        instruction=f"Ders notundan {num_questions} adet 4 seçenekli çoktan seçmeli sınav sorusu ve çalışma kartı üret.",
        context=text,
        prompt=prompt,
        system_instruction=system_instruction,
        temperature=0.2,
        max_output_tokens=4000,
        postprocess=_post,
    )


def generate_quiz_and_flashcards(
    text: str,
    num_questions: int = 5,
) -> Dict[str, Any]:
    """Quiz & Flashcard üretimi: %100 Gemini (gemini-2.5-flash), lokal fallback yoktur.

    Gemini kullanılamazsa GeminiUnavailableError fırlatılır.
    """
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        return {
            "quiz": [],
            "flashcards": [],
            "total_questions": 0,
            "total_flashcards": 0,
            "message": "Metin boş olduğu için soru üretilemedi.",
        }
    return generate_gemini_quiz_and_flashcards(cleaned_text, num_questions=num_questions)
