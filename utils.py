"""Image encoding helpers."""
import base64
import cv2
import numpy as np


def encode_image_to_base64(img: np.ndarray, quality: int = 90) -> str:
    """Encode a BGR numpy image to a base64 JPEG data URL."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def decode_base64_to_image(data_url: str) -> np.ndarray | None:
    """Decode a base64 data URL back to a BGR numpy image."""
    if not data_url:
        return None
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    buf = np.frombuffer(base64.b64decode(data_url), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
