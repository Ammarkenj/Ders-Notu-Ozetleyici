import logging
import os
import random
import re
from typing import Any, Dict, List, Optional
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

# Varsayılan Türkçe özetleme modeli
DEFAULT_MODEL_NAME = os.getenv(
    "SUMMARIZATION_MODEL", "ozcangundes/mt5-small-turkish-summarization"
)

# Pipeline önbelleği
_SUMMARIZER_PIPELINE = None
_CACHED_MODEL_NAME = None


def get_summarizer_pipeline(model_name: Optional[str] = None):
    """Hugging Face summarization pipeline'ını önbellekten döndürür veya ilklendirir."""
    global _SUMMARIZER_PIPELINE, _CACHED_MODEL_NAME

    chosen_model = model_name or DEFAULT_MODEL_NAME

    if _SUMMARIZER_PIPELINE is None or _CACHED_MODEL_NAME != chosen_model:
        device = 0 if torch.cuda.is_available() else -1
        logger.info(
            "Özetleme modeli yükleniyor: %s (Cihaz: %s)",
            chosen_model,
            "CUDA" if device == 0 else "CPU",
        )
        try:
            _SUMMARIZER_PIPELINE = pipeline(
                "summarization",
                model=chosen_model,
                tokenizer=chosen_model,
                device=device,
            )
            _CACHED_MODEL_NAME = chosen_model
        except Exception as e:
            logger.error("Özetleme pipeline yüklenirken hata oluştu: %s", e)
            raise e

    return _SUMMARIZER_PIPELINE


def summarize_text(
    text: str,
    max_length: int = 1024,
    min_length: int = 30,
    model_name: Optional[str] = None,
) -> str:
    """Hugging Face transformers pipeline kullanarak metni özetler."""
    cleaned_text = text.strip()
    if not cleaned_text:
        return ""

    words = cleaned_text.split()
    if len(words) < 20:
        return cleaned_text

    try:
        pipe = get_summarizer_pipeline(model_name=model_name)
        results = pipe(
            cleaned_text,
            max_length=max_length,
            min_length=min_length,
            truncation=True,
            do_sample=False,
        )
        if results and isinstance(results, list):
            return results[0].get("summary_text", "").strip()
        return ""
    except Exception as e:
        logger.warning(
            "Transformers pipeline ile özetleme yapılamadı (%s), metin çıkarımıyla özet oluşturuluyor...",
            e,
        )
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", cleaned_text)
            if len(s.strip()) > 15
        ]
        if not sentences:
            return cleaned_text[:max_length]
        key_sentences = [
            s
            for s in sentences
            if any(
                k in s.lower()
                for k in [
                    "denir",
                    "tanımlanır",
                    "amacı",
                    "özelliğidir",
                    "sağlar",
                    "önemlidir",
                    "oluşur",
                ]
            )
        ]
        chosen = key_sentences[:3] if key_sentences else sentences[:2]
        return " ".join(chosen)


def extract_key_statements(text: str) -> List[Dict[str, str]]:
    """Ders notundan tanımları, terimleri ve önemli kuralları ayıklar."""
    cleaned = text.strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    statements: List[Dict[str, str]] = []

    # 1. Başlık: Açıklama veya Terim - Açıklama formatları
    for line in lines:
        # İki nokta veya tire ayrımı
        match = re.match(r"^([A-ZÇĞİÖŞÜa-zçğıöşü0-9\s]{3,35})\s*[:\-–]\s*(.{15,})$", line)
        if match:
            term = match.group(1).strip()
            desc = match.group(2).strip()
            statements.append({"term": term, "definition": desc, "raw": line})

    # 2. Cümle bazlı anahtar ifadeler ("... denir", "... tanımlanır", "... sağlar")
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20:
            continue
        # Eğer zaten eklenmediyse ve tanım belirteci içeriyorsa
        if any(w in sent.lower() for w in ["denir", "tanımlanır", "adlandırılır", "amacı", "özelliğidir", "sağlar"]):
            # Cümlenin başındaki terimi bulmaya çalış
            parts = sent.split(",")
            term = parts[0].strip() if len(parts) > 1 and len(parts[0]) < 35 else "Temel Kavram"
            statements.append({"term": term, "definition": sent, "raw": sent})
        elif len(sent) > 30 and len(statements) < 10:
            statements.append({"term": "Ders Notu Bilgisi", "definition": sent, "raw": sent})

    return statements


def generate_quiz_and_flashcards(
    text: str,
    num_questions: int = 5,
) -> Dict[str, Any]:
    """Ders notundan otomatik çoktan seçmeli Soru-Cevap (Quiz) ve Soru Kartları (Flashcards) üretir.

    Akıllı Not benzeri çalışma kartları ve test modülü sağlar.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return {
            "quiz": [],
            "flashcards": [],
            "total_questions": 0,
            "total_flashcards": 0,
            "message": "Metin boş olduğu için soru üretilemedi.",
        }

    statements = extract_key_statements(cleaned_text)

    # Yeterli tanım bulunamazsa cümleleri kullan
    if len(statements) < 3:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_text) if len(s.strip()) > 20]
        for idx, s in enumerate(sentences):
            statements.append({
                "term": f"Kavram {idx + 1}",
                "definition": s,
                "raw": s,
            })

    flashcards: List[Dict[str, Any]] = []
    quiz: List[Dict[str, Any]] = []

    # 1. Flashcards (Soru Kartları) Üretimi
    for idx, item in enumerate(statements[: max(num_questions, len(statements))], 1):
        term = item["term"]
        definition = item["definition"]

        # Kartın ön yüzü: Soru formatı
        if term not in ("Ders Notu Bilgisi", "Temel Kavram") and not term.startswith("Kavram"):
            front = f"{term} nedir / neyi ifade eder?"
        else:
            front = f"Bu ders notundaki önemli bilgi nedir? (Kart #{idx})"

        flashcards.append({
            "id": idx,
            "front": front,
            "back": definition,
            "topic": term if len(term) < 30 else "Ders Notu",
        })

    # 2. Quiz (Çoktan Seçmeli Soru) Üretimi
    pool_definitions = [item["definition"] for item in statements]
    target_count = min(num_questions, len(statements))

    for q_idx in range(target_count):
        curr = statements[q_idx]
        term = curr["term"]
        correct_desc = curr["definition"]

        # Çeldiricileri havuzdan seç (farklı tanımlar)
        distractors = [d for d in pool_definitions if d != correct_desc]
        random.shuffle(distractors)
        chosen_distractors = distractors[:3]

        # Yetersiz çeldirici varsa genel alternatifler ekle
        generic_distractors = [
            "Yukarıdaki ifadenin tam tersi geçerlidir.",
            "Bu süreç yalnızca laboratuvar ortamında yapay olarak gerçekleşebilir.",
            "Ders notunda belirtilen konudan bağımsız ikincil bir etmendir.",
            "Bu durum yalnızca istisnai koşullarda ortaya çıkar.",
        ]
        while len(chosen_distractors) < 3:
            cand = generic_distractors[len(chosen_distractors) % len(generic_distractors)]
            if cand not in chosen_distractors:
                chosen_distractors.append(cand)

        # Şıkları karıştır ve doğru şıkkı belirle
        options_raw = [correct_desc] + chosen_distractors[:3]
        random.shuffle(options_raw)

        option_letters = ["A", "B", "C", "D"]
        correct_letter = option_letters[options_raw.index(correct_desc)]
        formatted_options = [
            f"{letter}) {opt}" for letter, opt in zip(option_letters, options_raw)
        ]

        if term not in ("Ders Notu Bilgisi", "Temel Kavram") and not term.startswith("Kavram"):
            question_text = f"Ders notlarına göre '{term}' hakkında aşağıdakilerden hangisi doğrudur?"
        else:
            question_text = f"Ders notundaki bilgilere göre aşağıdakilerden hangisi doğru bir ifadedir?"

        quiz.append({
            "id": q_idx + 1,
            "question": question_text,
            "options": formatted_options,
            "correct_answer": correct_letter,
            "explanation": f"Doğru cevap {correct_letter} şıkkıdır: {correct_desc}",
        })

    return {
        "quiz": quiz[:num_questions],
        "flashcards": flashcards[:num_questions],
        "total_questions": len(quiz[:num_questions]),
        "total_flashcards": len(flashcards[:num_questions]),
    }


def generate_glossary(text: str) -> dict:
    """Ders notundaki terimleri, tanımları ve formülleri çıkarır.

    Returns:
        {
            "sozluk": [{"terim": ..., "tanim": ..., "kategori": ...}],
            "formuller": ["..."],
            "anahtar_kelimeler": ["..."],
            "toplam_terim": int,
        }
    """
    cleaned = text.strip()
    if not cleaned:
        return {"sozluk": [], "formuller": [], "anahtar_kelimeler": [], "toplam_terim": 0}

    sozluk: list = []
    formuller: list = []
    anahtar_kelimeler: list = []
    eklenen_terimler: set = set()

    # 1. Formül tespiti (sayısal ve sembolik ifadeler)
    formul_patternler = [
        r"[A-ZÇĞİÖŞÜa-zçğıöşü]+\s*=\s*[^,;\n]{3,50}",       # X = ...
        r"\b[A-Z]{1,3}\s*[=<>]\.?\s*[\w\s\.\+\-\*/\^\(\)]{2,40}",  # F = ma
        r"\d+[,\.]\d+\s*[a-zçğıöşü]{1,5}(?:\.?\^?\d*)",             # 6.022x10^23
    ]
    for fp in formul_patternler:
        for m in re.finditer(fp, cleaned):
            f = m.group().strip()
            if f and f not in formuller and len(f) > 3:
                formuller.append(f)

    # 2. Terim:Taınım veya Terim - Tanım formatı
    satirlar = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    for satir in satirlar:
        # "Terim: tanım" veya "Terim - tanım"
        m = re.match(r"^([A-ZÇĞİÖŞÜa-zçğıöşü][\w\sçğıöşü]{1,40})\s*[:\-–]\.?\s*(.{15,})", satir)
        if m:
            terim = m.group(1).strip()
            tanim = m.group(2).strip()
            if terim.lower() not in eklenen_terimler:
                kategori = "Tanım"
                # Formül içeriyorsa
                if re.search(r"[=<>\+\-\*\/\^]", tanim):
                    kategori = "Formül"
                # Özel alan bildiren keyword
                elif any(k in tanim.lower() for k in ["oran", "birim", "löş", "katsay", "değer"]):
                    kategori = "Ölçü / Birim"
                sozluk.append({"terim": terim, "tanim": tanim[:200], "kategori": kategori})
                eklenen_terimler.add(terim.lower())

    # 3. Cümle bazlı terim tespiti ("X ... denir", "X ... adı verilir")
    cumleler = re.split(r"(?<=[.!?])\s+", cleaned)
    for cumle in cumleler:
        cumle = cumle.strip()
        if len(cumle) < 20:
            continue
        # "X'e Y denir" veya "Y olarak adlandırılır" formatı
        m = re.match(
            r"^([A-ZÇĞİÖŞÜa-zçğıöşü][\wçğıöşü\s]{1,35})",
            cumle
        )
        if m:
            aday_terim = m.group(1).strip().rstrip(",")
            if (len(aday_terim) > 3
                    and aday_terim.lower() not in eklenen_terimler
                    and any(w in cumle.lower() for w in ["denir", "tanımlanır", "adlandırılır", "özellikler", "sonuçtur"])):
                sozluk.append({
                    "terim": aday_terim,
                    "tanim": cumle[:200],
                    "kategori": "Kural / Kavram",
                })
                eklenen_terimler.add(aday_terim.lower())

    # 4. Anahtar kelimeler (4+ karakter, en çok geçen anlamı kelimeler)
    DURAK = {"olan", "veya", "ile", "bir", "bu", "da", "de", "ki", "ne",
             "için", "gibi", "ise", "ama", "çok", "en", "her", "daha", "kadar"}
    kelimeler = re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]{4,}", cleaned)
    from collections import Counter
    sayac = Counter(k.lower() for k in kelimeler if k.lower() not in DURAK)
    anahtar_kelimeler = [k for k, _ in sayac.most_common(20)]

    return {
        "sozluk": sozluk,
        "formuller": list(dict.fromkeys(formuller))[:15],
        "anahtar_kelimeler": anahtar_kelimeler,
        "toplam_terim": len(sozluk),
    }


def generate_mindmap(text: str, max_nodes: int = 30) -> dict:
    """Ders notundaki anahtar kavramların ilişki ağını (Zihin Haritası) üretir.

    D3.js / vis.js uyumlu düğüm + kenar yapısı döndürür.

    Returns:
        {
            "merkez_konu": str,
            "dugumler": [{"id": ..., "etiket": ..., "seviye": ..., "agirlik": ...}],
            "kenarlar":  [{"kaynak": ..., "hedef": ..., "iliski": ...}],
            "toplam_dugum": int,
        }
    """
    cleaned = text.strip()
    if not cleaned:
        return {"merkez_konu": "", "dugumler": [], "kenarlar": [], "toplam_dugum": 0}

    from collections import Counter

    DURAK = {"olan", "veya", "ile", "bir", "bu", "da", "de", "ki", "ne",
             "için", "gibi", "ise", "ama", "çok", "en", "her", "daha", "kadar",
             "the", "and", "of", "in", "to", "is", "are", "was"}

    kelimeler = re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]{4,}", cleaned)
    sayac: Counter = Counter(k.lower() for k in kelimeler if k.lower() not in DURAK)

    # Merkez konu: En sık geçen kelime
    en_cok = sayac.most_common(max_nodes)
    if not en_cok:
        return {"merkez_konu": "", "dugumler": [], "kenarlar": [], "toplam_dugum": 0}

    merkez = en_cok[0][0].capitalize()
    maks_frekans = en_cok[0][1]

    # Düğümler
    dugumler = [
        {
            "id": 0,
            "etiket": merkez,
            "seviye": 0,
            "agirlik": 1.0,
            "frekans": maks_frekans,
        }
    ]
    dugum_id_map = {en_cok[0][0]: 0}

    for i, (kelime, frekans) in enumerate(en_cok[1:], 1):
        norm_agirlik = round(frekans / maks_frekans, 3)
        # Seviye: ağırlığa göre
        seviye = 1 if norm_agirlik >= 0.5 else (2 if norm_agirlik >= 0.2 else 3)
        dugumler.append({
            "id": i,
            "etiket": kelime.capitalize(),
            "seviye": seviye,
            "agirlik": norm_agirlik,
            "frekans": frekans,
        })
        dugum_id_map[kelime] = i

    # Kenarlar: Aynı cümlede geçen kavramlar arasında bağlantı
    cumleler = re.split(r"(?<=[.!?])\s+", cleaned)
    kenar_seti: set = set()
    kenarlar = []

    for cumle in cumleler:
        cumle_kelimeleri = [
            k.lower() for k in re.findall(r"[A-ZÇĞİÖŞÜa-zçğıöşü]{4,}", cumle)
            if k.lower() in dugum_id_map and k.lower() not in DURAK
        ]
        for i in range(len(cumle_kelimeleri)):
            for j in range(i + 1, min(i + 4, len(cumle_kelimeleri))):
                k1, k2 = cumle_kelimeleri[i], cumle_kelimeleri[j]
                if k1 == k2:
                    continue
                anahtar = tuple(sorted([dugum_id_map[k1], dugum_id_map[k2]]))
                if anahtar not in kenar_seti:
                    kenar_seti.add(anahtar)
                    kenarlar.append({
                        "kaynak": anahtar[0],
                        "hedef": anahtar[1],
                        "iliski": "beraber_gecme",
                        "guc": 1,
                    })

    # Merkez düğümün bağlı olmayan üst düğümlere bağlanması
    bagli_idler = {k["kaynak"] for k in kenarlar} | {k["hedef"] for k in kenarlar}
    for d in dugumler[1:]:
        if d["id"] not in bagli_idler and d["seviye"] == 1:
            kenarlar.append({
                "kaynak": 0,
                "hedef": d["id"],
                "iliski": "alt_konu",
                "guc": round(d["agirlik"] * 2, 2),
            })

    return {
        "merkez_konu": merkez,
        "dugumler": dugumler[:max_nodes],
        "kenarlar": kenarlar[:max_nodes * 3],
        "toplam_dugum": len(dugumler[:max_nodes]),
    }
