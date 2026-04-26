import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import tempfile
import os
import time
import re

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PlateVision AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;700&display=swap');
:root {
  --bg:      #0a0e14;
  --card:    #111720;
  --panel:   #161d28;
  --accent:  #00d4ff;
  --muted:   #6a7d96;
  --border:  #1e2d3d;
  --green:   #00ff9d;
}
html, body, [class*="css"] {
  font-family: 'Exo 2', sans-serif;
  background-color: var(--bg);
  color: #e2eaf5;
}
.hero {
  background: linear-gradient(135deg,#0a0e14,#0d1b2a,#0a1628);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2.5rem 2rem;
  margin-bottom: 2rem;
  position: relative; overflow: hidden;
}
.hero::before {
  content:''; position:absolute; top:-40%; left:-10%;
  width:60%; height:200%;
  background: radial-gradient(ellipse,rgba(0,212,255,.08) 0%,transparent 70%);
}
.hero-title {
  font-family:'Rajdhani',sans-serif; font-size:2.8rem; font-weight:700;
  letter-spacing:3px; color:var(--accent);
  text-shadow:0 0 30px rgba(0,212,255,.4); margin:0;
}
.hero-sub { font-size:1rem; color:var(--muted); margin-top:.5rem; max-width:700px; line-height:1.7; }
.badge {
  display:inline-block; background:rgba(0,212,255,.12);
  border:1px solid rgba(0,212,255,.35); color:var(--accent);
  font-family:'Share Tech Mono',monospace; font-size:.72rem;
  padding:3px 10px; border-radius:4px; margin-right:8px; margin-top:10px;
}
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:1.5rem; margin-bottom:1.2rem; }
.card-title {
  font-family:'Rajdhani',sans-serif; font-size:1.1rem; font-weight:600;
  letter-spacing:2px; color:var(--accent); text-transform:uppercase;
  margin-bottom:.8rem; border-bottom:1px solid var(--border); padding-bottom:.5rem;
}
.plate-box {
  background:#000; border:2px solid var(--accent); border-radius:8px;
  padding:1rem 2rem; text-align:center; margin:1rem 0;
  box-shadow:0 0 20px rgba(0,212,255,.25),inset 0 0 20px rgba(0,212,255,.05);
}
.plate-text {
  font-family:'Share Tech Mono',monospace; font-size:2.6rem;
  letter-spacing:.3em; color:var(--accent); text-shadow:0 0 12px var(--accent);
}
.plate-label { font-size:.75rem; color:var(--muted); letter-spacing:2px; text-transform:uppercase; }
.metric-row { display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }
.metric-chip { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:.6rem 1.2rem; min-width:110px; }
.metric-val { font-family:'Share Tech Mono',monospace; font-size:1.4rem; color:var(--green); }
.metric-lbl { font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }
.step { display:flex; gap:1rem; align-items:flex-start; margin-bottom:1rem; }
.step-num {
  background:var(--accent); color:#000; font-family:'Rajdhani',sans-serif;
  font-weight:700; font-size:1.1rem; width:30px; height:30px;
  border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.step-text { font-size:.9rem; color:#e2eaf5; padding-top:4px; line-height:1.6; }
.stButton > button {
  background:linear-gradient(135deg,var(--accent) 0%,#0099bb 100%);
  color:#000 !important; font-family:'Rajdhani',sans-serif;
  font-weight:700; letter-spacing:2px; font-size:1rem;
  border:none; border-radius:8px; padding:.55rem 1.5rem; transition:all .2s;
}
.stButton > button:hover { opacity:.85; box-shadow:0 0 18px rgba(0,212,255,.4); }
div[data-testid="stFileUploader"] { background:var(--card); border:1px dashed var(--border); border-radius:10px; }
div[data-testid="stFileUploader"]:hover { border-color:var(--accent); }
.stProgress > div > div > div { background:var(--accent) !important; }
section[data-testid="stSidebar"] { background:var(--panel) !important; border-right:1px solid var(--border); }
.stTabs [data-baseweb="tab-list"] { background:var(--card); border-radius:8px; padding:4px; }
.stTabs [data-baseweb="tab"] { color:var(--muted); font-family:'Rajdhani',sans-serif; font-weight:600; letter-spacing:1px; }
.stTabs [aria-selected="true"] { background:rgba(0,212,255,.15) !important; color:var(--accent) !important; border-radius:6px; }
#MainMenu, footer { visibility:hidden; }
header { background:transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">🚗 PLATEVISION AI</p>
  <p class="hero-sub">
    Sistema inteligente de <strong>detección y lectura de placas vehiculares colombianas</strong>.
    Carga una imagen, un video o activa la cámara — YOLOv8 localiza cada placa
    y EasyOCR extrae letras y números automáticamente.
    Si hay <strong>varias placas</strong> en la imagen, las detecta todas.
  </p>
  <span class="badge">YOLOv8 Ultralytics</span>
  <span class="badge">EasyOCR</span>
  <span class="badge">Multi-Placa</span>
  <span class="badge">Tiempo Real</span>
</div>
""", unsafe_allow_html=True)

with st.expander("📖 ¿Cómo usar esta aplicación?", expanded=False):
    st.markdown("""
    <div class="card">
      <div class="step"><div class="step-num">1</div><div class="step-text">
        Coloca <code>best.onnx</code> en la misma carpeta que <code>app.py</code>.
      </div></div>
      <div class="step"><div class="step-num">2</div><div class="step-text">
        Elige la fuente: <strong>Imagen</strong>, <strong>Video</strong> o <strong>Cámara Web</strong>.
      </div></div>
      <div class="step"><div class="step-num">3</div><div class="step-text">
        Ajusta el umbral de confianza (recomendado ≥ 0.45).
      </div></div>
      <div class="step"><div class="step-num">4</div><div class="step-text">
        Presiona <strong>Detectar Placa</strong> o toma una foto con la cámara.
      </div></div>
      <div class="step"><div class="step-num">5</div><div class="step-text">
        Ve la imagen anotada, el texto de cada placa (formato ABC-123) y las métricas.
      </div></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
MODEL_PATH = "best.onnx"

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")
    conf_threshold = st.slider("Umbral de confianza", 0.10, 0.99, 0.45, 0.01)
    iou_threshold  = st.slider("Umbral IoU (NMS)",    0.10, 0.99, 0.45, 0.01)
    ocr_langs      = st.multiselect("Idiomas OCR", ["es","en","pt","fr","de"], default=["es","en"])
    draw_conf      = st.checkbox("Mostrar confianza en imagen", value=True)
    enhance_crop   = st.checkbox("Mejorar recorte antes del OCR", value=True)
    st.markdown("---")
    st.markdown("<small style='color:#4a5568'>PlateVision AI · YOLOv8 + EasyOCR</small>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return YOLO(path)

@st.cache_resource(show_spinner=False)
def load_ocr(langs: tuple):
    return easyocr.Reader(list(langs), gpu=False)

# ─────────────────────────────────────────────
#  PREPROCESAMIENTO (pasos del notebook)
# ─────────────────────────────────────────────
def enhance_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocesa el recorte de la placa siguiendo los mismos pasos
    del notebook de entrenamiento:

      Paso 1 — Escala de grises (BGR → GRAY)
      Paso 2 — Upscaling 2x con interpolación CUBIC
      Paso 3 — CLAHE  (clipLimit=2.0, tileGridSize=8×8)
      Paso 4 — Filtro bilateral  (d=11, sigmaColor=17, sigmaSpace=17)

    Retorna imagen BGR de 3 canales (EasyOCR acepta ambos formatos).
    """
    # Paso 1 — escala de grises
    gris = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Paso 2 — upscaling 2x con INTER_CUBIC
    alto, ancho = gris.shape
    gris = cv2.resize(gris, (ancho * 2, alto * 2),
                      interpolation=cv2.INTER_CUBIC)

    # Paso 3 — CLAHE (contraste adaptativo)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gris  = clahe.apply(gris)

    # Paso 4 — filtro bilateral (reduce ruido, preserva bordes de caracteres)
    gris = cv2.bilateralFilter(gris, 11, 17, 17)

    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)


# ─────────────────────────────────────────────
#  LIMPIEZA Y CORRECCIÓN DE PLACA
# ─────────────────────────────────────────────
def clean_plate_text(texts: list) -> str:
    """Une tokens OCR y conserva solo caracteres alfanuméricos."""
    combined = " ".join(texts).upper()
    return re.sub(r"[^A-Z0-9]", "", combined)


def correct_colombian_plate(text: str) -> str:
    """
    Ordena y corrige al formato de placa colombiana: ABC-123

    1. Separa todos los caracteres en LETRAS y DÍGITOS.
    2. Reordena: primero 3 letras, luego 3 dígitos (independientemente
       del orden en que el OCR los devolvió).
    3. Si faltan letras, convierte dígitos confundibles (0→O, 1→I, etc.).
    4. Si faltan dígitos, convierte letras confundibles (O→0, I→1, etc.).
    5. Rellena con '?' si no hay suficientes caracteres de ningún tipo.

    Tablas de confusión OCR:
      Dígito → Letra : 0→O  1→I  8→B  5→S  6→G  2→Z  4→A  7→T
      Letra  → Dígito: O→0  I→1  B→8  S→5  G→6  Z→2  A→4  T→7
                       Q→0  D→0  L→1  E→8  U→0  V→7  F→7
    """
    clean = re.sub(r"[^A-Z0-9]", "", text.upper())

    digit_to_letter = {
        "0": "O", "1": "I", "8": "B", "5": "S",
        "6": "G", "2": "Z", "4": "A", "7": "T",
    }
    letter_to_digit = {
        "O": "0", "I": "1", "B": "8", "S": "5",
        "G": "6", "Z": "2", "A": "4", "T": "7",
        "Q": "0", "D": "0", "L": "1", "E": "8",
        "U": "0", "V": "7", "F": "7",
    }

    # Separar letras y dígitos tal como llegaron del OCR
    letras  = [c for c in clean if c.isalpha()]
    digitos = [c for c in clean if c.isdigit()]

    # Completar letras desde dígitos sobrantes si faltan
    while len(letras) < 3 and digitos:
        candidato = digitos.pop(0)
        letras.append(digit_to_letter.get(candidato, candidato))

    # Completar dígitos desde letras sobrantes si faltan
    while len(digitos) < 3 and len(letras) > 3:
        candidato = letras.pop(3)          # tomar el 4to carácter letra
        digitos.insert(0, letter_to_digit.get(candidato, candidato))

    # Recortar a 3 de cada tipo y rellenar con '?' si faltan
    letras  = (letras[:3]  + ["?"] * 3)[:3]
    digitos = (digitos[:3] + ["?"] * 3)[:3]

    return f"{''.join(letras)}-{''.join(digitos)}"


# ─────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def run_detection(frame_bgr, model, ocr_reader,
                  conf_th=0.45, iou_th=0.45,
                  do_enhance=True, show_conf=True):
    """
    Pipeline completo:
      1. YOLO detecta todas las placas en el frame
      2. Recorta exactamente el bbox (sin margen)
      3. Preprocesa con los 4 pasos del notebook (si do_enhance=True)
      4. EasyOCR lee caracteres alfanuméricos
      5. Ordena letras/dígitos y corrige al formato ABC-123
      6. Anota el frame con cajas y texto
    """
    results = model(frame_bgr, conf=conf_th, iou=iou_th, verbose=False)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    annotated   = frame_bgr.copy()
    plates_info = []

    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Recorte exacto — sin margen
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Preprocesar (4 pasos del notebook)
        crop_ocr = enhance_for_ocr(crop) if do_enhance else crop

        # EasyOCR
        ocr_results = ocr_reader.readtext(
            crop_ocr,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            paragraph=False,
            width_ths=1.0,
            height_ths=1.0,
        )

        print(f"\n[DEBUG] Crop shape: {crop_ocr.shape}")
        print(f"[DEBUG] OCR raw: {ocr_results}")

        texts     = [r[1] for r in ocr_results if r[2] > 0.10]
        confs_ocr = [r[2] for r in ocr_results if r[2] > 0.10]

        raw_text     = clean_plate_text(texts)
        plate        = correct_colombian_plate(raw_text) if raw_text else "— SIN TEXTO —"
        avg_ocr_conf = float(np.mean(confs_ocr)) if confs_ocr else 0.0

        print(f"[DEBUG] raw='{raw_text}'  →  plate='{plate}'")

        # ── Dibujar ────────────────────────────────────────────────────
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 212, 255), 2)
        label = f"{plate}  {conf:.0%}" if show_conf else plate
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
        cv2.rectangle(annotated,
                      (x1, y1 - th - 10), (x1 + tw + 8, y1),
                      (0, 212, 255), -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        plates_info.append({
            "bbox"      : (x1, y1, x2, y2),
            "det_conf"  : float(conf),
            "plate_text": plate,
            "raw_texts" : texts,
            "ocr_conf"  : avg_ocr_conf,
            "crop_rgb"  : cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
        })

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, plates_info

# ─────────────────────────────────────────────
#  CARGAR MODELO Y OCR
# ─────────────────────────────────────────────
with st.spinner("Cargando modelo YOLOv8…"):
    model = load_model(MODEL_PATH)

with st.spinner("Iniciando EasyOCR…"):
    ocr_reader = load_ocr(tuple(ocr_langs) if ocr_langs else ("es", "en"))

if model is None:
    st.error(
        f"⚠️ No se encontró `{MODEL_PATH}` en la carpeta del proyecto. "
        "Asegúrate de que **best.onnx** esté en la misma carpeta que app.py."
    )
    st.stop()
else:
    st.success(f"✅ Modelo cargado — `{MODEL_PATH}`", icon="🤖")

# ─────────────────────────────────────────────
#  RENDER RESULTADOS
# ─────────────────────────────────────────────
def render_results(annotated_rgb, plates_info):
    col_img, col_info = st.columns([3, 2], gap="medium")
    with col_img:
        st.image(annotated_rgb, caption="Detección", use_container_width=True)
    with col_info:
        if not plates_info:
            st.warning("No se detectaron placas en esta imagen.")
            return
        for idx, info in enumerate(plates_info, 1):
            st.markdown(f"<div class='card-title'>Placa #{idx}</div>", unsafe_allow_html=True)
            plate_str = info["plate_text"]
            st.markdown(f"""
            <div class='plate-box'>
              <div class='plate-text'>{plate_str}</div>
              <div class='plate-label'>PLACA DETECTADA · Formato ABC-123</div>
            </div>
            <div class='metric-row'>
              <div class='metric-chip'>
                <div class='metric-val'>{info['det_conf']:.0%}</div>
                <div class='metric-lbl'>Det. YOLO</div>
              </div>
              <div class='metric-chip'>
                <div class='metric-val'>{info['ocr_conf']:.0%}</div>
                <div class='metric-lbl'>OCR Conf.</div>
              </div>
              <div class='metric-chip'>
                <div class='metric-val'>{len(info['plate_text'].replace('-',''))}</div>
                <div class='metric-lbl'>Caracteres</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if info["raw_texts"]:
                with st.expander("Ver tokens OCR crudos"):
                    for t in info["raw_texts"]:
                        st.code(t, language=None)
            st.image(info["crop_rgb"], caption=f"Recorte placa #{idx}", use_container_width=True)
            st.markdown("<hr style='border-color:#1e2d3d;margin:1rem 0'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab_img, tab_vid, tab_cam = st.tabs(["📷  Imagen", "🎬  Video", "📡  Cámara Web"])

# ══════════════════════════════════════════════
#  TAB 1 — IMAGEN
# ══════════════════════════════════════════════
with tab_img:
    st.markdown("<div class='card-title'>Subir Imagen</div>", unsafe_allow_html=True)
    uploaded_img = st.file_uploader(
        "Arrastra o selecciona una imagen",
        type=["jpg","jpeg","png","bmp","webp"], key="img_up"
    )
    if uploaded_img:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        frame_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                     caption="Original", use_container_width=True)
        with col2:
            st.markdown(f"""
            <div class='card'>
              <div class='metric-lbl'>Resolución</div>
              <div class='metric-val' style='font-size:1.1rem'>{frame_bgr.shape[1]}×{frame_bgr.shape[0]}</div>
              <br>
              <div class='metric-lbl'>Archivo</div>
              <div style='color:#e2eaf5;font-size:.85rem'>{uploaded_img.name}</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔍  Detectar Placa", key="btn_img"):
            with st.spinner("Procesando…"):
                t0 = time.time()
                annotated, plates = run_detection(
                    frame_bgr, model, ocr_reader,
                    conf_th=conf_threshold, iou_th=iou_threshold,
                    do_enhance=enhance_crop, show_conf=draw_conf,
                )
                elapsed = time.time() - t0
            st.markdown(
                f"<small style='color:#4a5568'>Inferencia: <b>{elapsed*1000:.0f} ms</b> — "
                f"<b>{len(plates)}</b> placa(s) detectada(s)</small>",
                unsafe_allow_html=True
            )
            render_results(annotated, plates)

# ══════════════════════════════════════════════
#  TAB 2 — VIDEO
# ══════════════════════════════════════════════
with tab_vid:
    st.markdown("<div class='card-title'>Subir Video</div>", unsafe_allow_html=True)
    uploaded_vid = st.file_uploader(
        "Arrastra o selecciona un video",
        type=["mp4","avi","mov","mkv","webm"], key="vid_up"
    )
    if uploaded_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_vid.read())
        tfile.flush()

        cap   = cv2.VideoCapture(tfile.name)
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        st.markdown(f"""
        <div class='card'>
          <div class='metric-row'>
            <div class='metric-chip'><div class='metric-val'>{w}×{h}</div><div class='metric-lbl'>Resolución</div></div>
            <div class='metric-chip'><div class='metric-val'>{fps:.0f}</div><div class='metric-lbl'>FPS</div></div>
            <div class='metric-chip'><div class='metric-val'>{total}</div><div class='metric-lbl'>Frames</div></div>
            <div class='metric-chip'><div class='metric-val'>{total/fps:.1f}s</div><div class='metric-lbl'>Duración</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        frame_step = st.slider("Analizar 1 de cada N frames", 1, 30, 5)
        max_frames  = st.slider("Máximo de frames a mostrar", 1, 30, 10)

        if st.button("▶️  Procesar Video", key="btn_vid"):
            cap        = cv2.VideoCapture(tfile.name)
            progress   = st.progress(0, text="Procesando video…")
            frame_idx  = 0
            shown      = 0
            all_plates = []

            while cap.isOpened() and shown < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_step == 0:
                    annotated, plates = run_detection(
                        frame, model, ocr_reader,
                        conf_th=conf_threshold, iou_th=iou_threshold,
                        do_enhance=enhance_crop, show_conf=draw_conf,
                    )
                    if plates:
                        st.markdown(f"<div class='card-title'>Frame {frame_idx}</div>",
                                    unsafe_allow_html=True)
                        render_results(annotated, plates)
                        all_plates.extend(plates)
                        shown += 1
                frame_idx += 1
                progress.progress(min(frame_idx / max(total, 1), 1.0),
                                  text=f"Frame {frame_idx} / {total}")

            cap.release()
            os.unlink(tfile.name)
            progress.empty()

            if all_plates:
                unique = list({p["plate_text"] for p in all_plates if p["plate_text"]})
                st.markdown("### 📋 Placas únicas detectadas en el video")
                for pl in unique:
                    st.markdown(f"""
                    <div class='plate-box' style='margin:.4rem 0'>
                      <div class='plate-text' style='font-size:1.8rem'>{pl}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No se detectaron placas en los frames analizados.")

# ══════════════════════════════════════════════
#  TAB 3 — CÁMARA WEB
# ══════════════════════════════════════════════
with tab_cam:
    st.markdown("<div class='card-title'>Cámara Web — Detección en Vivo</div>",
                unsafe_allow_html=True)
    st.info("Apunta la cámara hacia una placa y presiona el botón de captura. "
            "La imagen se analiza automáticamente al instante.")

    camera_col, info_col = st.columns([3, 2], gap="medium")

    with camera_col:
        cam_img = st.camera_input("📸 Captura un frame", key="cam_input")

    with info_col:
        if cam_img is not None:
            file_bytes = np.frombuffer(cam_img.getvalue(), np.uint8)
            frame_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("Analizando…"):
                t0 = time.time()
                annotated, plates = run_detection(
                    frame_bgr, model, ocr_reader,
                    conf_th=conf_threshold, iou_th=iou_threshold,
                    do_enhance=enhance_crop, show_conf=draw_conf,
                )
                elapsed = time.time() - t0

            st.markdown(
                f"<small style='color:#4a5568'>Inferencia: <b>{elapsed*1000:.0f} ms</b></small>",
                unsafe_allow_html=True
            )

            if plates:
                for idx, info in enumerate(plates, 1):
                    plate_str = info["plate_text"]
                    st.markdown(f"""
                    <div class='plate-box'>
                      <div class='plate-text'>{plate_str}</div>
                      <div class='plate-label'>PLACA #{idx} · ABC-123</div>
                    </div>
                    <div class='metric-row'>
                      <div class='metric-chip'>
                        <div class='metric-val'>{info['det_conf']:.0%}</div>
                        <div class='metric-lbl'>YOLO</div>
                      </div>
                      <div class='metric-chip'>
                        <div class='metric-val'>{info['ocr_conf']:.0%}</div>
                        <div class='metric-lbl'>OCR</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No se detectó ninguna placa. Intenta bajar el umbral de confianza.")

    if cam_img is not None and "plates" in dir() and plates:
        st.markdown("**Frame anotado:**")
        st.image(annotated, use_container_width=True)