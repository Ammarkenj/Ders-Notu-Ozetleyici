import logging
from pathlib import Path
from typing import List, Optional, Union
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Global cache for EasyOCR Reader to avoid reloading weights on every call
_EASYOCR_READER = None


def load_image(image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """Girdi olarak verilen farklı türlerdeki görseli OpenCV BGR formatına dönüştürür.

    Windows dosya yollarındaki Türkçe/Unicode karakter sorunlarını önlemek için
    np.fromfile kullanılır.
    """
    if isinstance(image_input, (str, Path)):
        path_str = str(image_input)
        if not Path(path_str).exists():
            raise FileNotFoundError(f"Görsel dosyası bulunamadı: {path_str}")

        # Unicode/Türkçe yol desteği
        img_array = np.fromfile(path_str, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Görsel dosyası okunamadı: {path_str}")
        return img

    elif isinstance(image_input, bytes):
        img_array = np.frombuffer(image_input, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Görsel byte verisi çözümlenemedi.")
        return img

    elif isinstance(image_input, Image.Image):
        # PIL Image -> NumPy array (RGB -> BGR)
        rgb_array = np.array(image_input)
        if len(rgb_array.shape) == 2:  # Grayscale
            return cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2BGR)
        elif rgb_array.shape[2] == 4:  # RGBA
            return cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:  # Grayscale
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        return image_input

    else:
        raise TypeError(f"Desteklenmeyen görsel girdi türü: {type(image_input)}")


def preprocess_image(
    image: np.ndarray,
    apply_threshold: bool = True,
    apply_blur: bool = True,
    method: str = "otsu",
) -> np.ndarray:
    """OpenCV kullanarak görseli OCR için ön işlemden geçirir.

    - Grayscale dönüşümü
    - Gürültü giderme (Gaussian Blur)
    - Thresholding / İkiliğe dönüştürme (Otsu veya Adaptive Threshold)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Gürültü azaltma (Blur)
    processed = gray
    if apply_blur:
        processed = cv2.GaussianBlur(processed, (5, 5), 0)

    # Thresholding (İkili görüntü)
    if apply_threshold:
        if method == "adaptive":
            processed = cv2.adaptiveThreshold(
                processed,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )
        else:
            # Otsu Thresholding
            _, processed = cv2.threshold(
                processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

    return processed


def get_easyocr_reader(languages: Optional[List[str]] = None, gpu: bool = False):
    """EasyOCR Reader nesnesini önbellekten döndürür veya ilklendirir."""
    global _EASYOCR_READER
    if languages is None:
        languages = ["tr", "en"]

    if _EASYOCR_READER is None:
        try:
            import easyocr

            logger.info("EasyOCR Reader yükleniyor (diller: %s)...", languages)
            _EASYOCR_READER = easyocr.Reader(languages, gpu=gpu)
        except Exception as e:
            logger.error("EasyOCR yüklenirken hata oluştu: %s", e)
            raise e
    return _EASYOCR_READER


def extract_text_from_image(
    image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
    languages: Optional[List[str]] = None,
    use_preprocessing: bool = True,
    engine: str = "auto",
) -> str:
    """Görselden metin çıkarır (OpenCV ön işleme ve EasyOCR / Tesseract mantığı).

    Args:
        image_input: Dosya yolu, byte verisi, PIL görseli veya numpy dizisi.
        languages: OCR dil listesi (varsayılan: ['tr', 'en']).
        use_preprocessing: OpenCV ön işlemesi uygulansın mı?
        engine: 'auto', 'easyocr' veya 'tesseract'.

    Returns:
        Çıkarılan metin (str).
    """
    if languages is None:
        languages = ["tr", "en"]

    img_bgr = load_image(image_input)

    # İsteğe bağlı OpenCV ön işleme
    if use_preprocessing:
        processed_img = preprocess_image(img_bgr, apply_threshold=True, apply_blur=True)
    else:
        processed_img = img_bgr

    extracted_lines: List[str] = []

    # EasyOCR Denemesi
    if engine in ("auto", "easyocr"):
        try:
            reader = get_easyocr_reader(languages=languages, gpu=False)
            results = reader.readtext(processed_img, detail=0)
            if results:
                extracted_lines = [line.strip() for line in results if line.strip()]
                return "\n".join(extracted_lines)
        except Exception as ocr_err:
            logger.warning("EasyOCR ile metin çıkarma başarısız oldu: %s", ocr_err)
            if engine == "easyocr":
                raise ocr_err

    # Tesseract Yedek / Alternatif Denemesi
    if engine in ("auto", "tesseract") and not extracted_lines:
        try:
            import pytesseract

            lang_str = "+".join(languages) if languages else "tur+eng"
            # PIL Image formatı pytesseract ile daha uyumlu
            pil_img = Image.fromarray(processed_img)
            text = pytesseract.image_to_string(pil_img, lang=lang_str)
            return text.strip()
        except Exception as tess_err:
            logger.warning("Tesseract ile metin çıkarma başarısız oldu: %s", tess_err)
            if engine == "tesseract":
                raise tess_err

    return "\n".join(extracted_lines).strip()
