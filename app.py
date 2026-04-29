
Claude ha terminado la respuesta
Has dicho: dame el ssh http://3.
dame el ssh http://3.81.214.53:8080/docs#/default/predict_predict_post ubuntu placas.pem

17:02
Claude ha respondido: Parece que quieres hacer una solicitud SSH o acceder a una API con una llave .
Parece que quieres hacer una solicitud SSH o acceder a una API con una llave .pem. Sin embargo, hay algunas cosas que necesito aclarar:

SSH y HTTP son protocolos diferentes. Una URL http://... no se accede por SSH.
No veo ningún archivo .pem adjunto en tu mensaje.
¿Qué es lo que quieres hacer exactamente?

¿Conectarte por SSH al servidor 3.81.214.53 usando una llave placas.pem?
¿Hacer una petición a la API en http://3.81.214.53:8080/docs#/default/predict_predict_post?
¿Ambas cosas?
Si subes el archivo placas.pem y me dices qué quieres lograr, puedo ayudarte con el comando correcto.

Has dicho: dsolo dame el cmd de sshh
dsolo dame el cmd de sshh

17:02
Claude ha respondido: Si da error de permisos en la llave:
bash
ssh -i placas.pem ubuntu@3.81.214.53
Si da error de permisos en la llave:

bash
chmod 400 placas.pem
Has dicho: activa el venv
activa el venv

17:02
bash
source venv/bin/activate
Has dicho: como puedo copiar el app.
como puedo copiar el [app.py](http://app.py) con cat

17:03
bash
cat app.py
Has dicho: quiero que alaices todo lo q hace este codigo como la verificacion de placa que son 3 letras 3 numerosd todo eso pero todo es todo y lo agrugues a este codigo …

from fastapi import FastAPI, File, UploadFile, Form from fastapi.responses import JSONResponse from ultralytics import YOLO import easyocr import numpy as np import cv2 import re import io import base64 from PIL import Image from typing import Optional app = FastAPI() model = YOLO("best.pt") reade

pasted


import streamlit as st import cv2 import numpy as np import easyocr from ultralytics import YOLO import tempfile import os import time import re # ───────────────────────────────────────────── # PAGE CONFIG # ───────────────────────────────────────────── st.set_page_config( page_

pasted

quiero que alaices todo lo q hace este codigo como la verificacion de placa que son 3 letras 3 numerosd todo eso pero todo es todo y lo agrugues a este codigo en streamlit

17:04
Claude ha respondido: Voy a analizar ambos códigos a fondo y fusionar toda la lógica del app.
Voy a analizar ambos códigos a fondo y fusionar toda la lógica del app.py (FastAPI) en el Streamlit.


Leer skill de frontend
Leer skill de frontend
Ahora voy a analizar todo en detalle y crear el archivo fusionado:


Archivo creado, leyó un archivo
Archivo creado, leyó un archivo
Acá está el app.py fusionado y mejorado. Todo lo que hace:

Lo que se integró del FastAPI (app.py) al Streamlit:

PALABRAS_IGNORAR — ahora con más palabras (MOTO, PARTICULAR, PUBLICO, OFICIAL)
DIGIT_TO_LETTER y LETTER_TO_DIGIT — tablas de confusión OCR expuestas en el sidebar con explicación visual
enhance_for_ocr — los 4 pasos documentados (gray → 2x → CLAHE → bilateral)
clean_plate_text — filtro de tokens del fondo con lógica exacta
correct_colombian_plate — ahora retorna también un dict con el detalle del proceso (método usado, letras/dígitos encontrados, correcciones aplicadas)
Doble pasada OCR (retry con threshold bajo si el texto tiene <6 chars)
Lo nuevo que se agregó:

validate_plate_format — valida 6 reglas: longitud, guion, posición del guion, 3 letras, 3 dígitos, mayúsculas
Colores diferenciados: 🟢 verde = placa válida, 🔵 cyan = corregida/revisar
Panel de correcciones paso a paso (activar con "Mostrar detalle de corrección OCR" en sidebar)
Tab "ℹ️ Cómo funciona" con documentación de todo el pipeline
En video: placas separadas en válidas vs corregidas
Recorte original + recorte procesado OCR lado a lado

App
PY 
Has dicho: no tienes que usar apis solo las verificaciones y eso no ves que el topma el modelo de la misma carpeta en git
no tienes que usar apis solo las verificaciones y eso no ves que el topma el modelo de la misma carpeta en git

17:09
Claude ha respondido: Tienes razón, no se necesita ninguna API.
Tienes razón, no se necesita ninguna API. Solo es el Streamlit con toda la lógica local.


Se crearon 2 archivos, ejecutó un comando, leyó un archivo
Se crearon 2 archivos, ejecutó un comando, leyó un archivo
Listo, sin APIs ni FastAPI, todo local. Lo que se integró del app.py original al Streamlit:

Lógica de procesamiento fusionada:

PALABRAS_IGNORAR — filtra COLOMBIA, POLICIA, TRANSITO, etc. del texto OCR
DIGIT_TO_LETTER y LETTER_TO_DIGIT — tablas de confusión completas
enhance_for_ocr — los 4 pasos exactos (gray → 2x → CLAHE → bilateral)
clean_plate_text — filtra tokens del fondo por nombre exacto y por contención
correct_colombian_plate — regex primero, luego corrección por tablas, luego relleno
Doble pasada OCR — retry con threshold bajo si el texto tiene <6 chars
Lo nuevo:

validate_plate_format — valida las 6 reglas del formato ABC-123 con chips visuales ✓/✗
Colores diferenciados en la imagen anotada: 🟢 verde = válida, 🔵 cyan = corregida
Panel de debug opcional con las correcciones aplicadas paso a paso
Recorte original + recorte procesado lado a lado para comparar

App
PY 

Se ha quedado sin mensajes gratuitos hasta las 22:00



Claude es IA y puede cometer errores. Por favor, verifica las respuestas.
App · PY
Copiar

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
  --red:     #ff4d6d;
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
.plate-box-valid {
  background:#000; border:2px solid var(--green); border-radius:8px;
  padding:1rem 2rem; text-align:center; margin:1rem 0;
  box-shadow:0 0 20px rgba(0,255,157,.25),inset 0 0 20px rgba(0,255,157,.05);
}
.plate-box-invalid {
  background:#000; border:2px solid var(--accent); border-radius:8px;
  padding:1rem 2rem; text-align:center; margin:1rem 0;
  box-shadow:0 0 20px rgba(0,212,255,.25),inset 0 0 20px rgba(0,212,255,.05);
}
.plate-text-valid {
  font-family:'Share Tech Mono',monospace; font-size:2.6rem;
  letter-spacing:.3em; color:var(--green); text-shadow:0 0 12px var(--green);
}
.plate-text-invalid {
  font-family:'Share Tech Mono',monospace; font-size:2.6rem;
  letter-spacing:.3em; color:var(--accent); text-shadow:0 0 12px var(--accent);
}
.plate-label { font-size:.75rem; color:var(--muted); letter-spacing:2px; text-transform:uppercase; }
.validation-row { display:flex; gap:.5rem; flex-wrap:wrap; margin:.5rem 0; }
.val-chip-ok {
  background:rgba(0,255,157,.1); border:1px solid rgba(0,255,157,.3);
  color:var(--green); font-size:.72rem; padding:3px 10px; border-radius:4px;
  font-family:'Share Tech Mono',monospace;
}
.val-chip-fail {
  background:rgba(255,77,109,.1); border:1px solid rgba(255,77,109,.3);
  color:var(--red); font-size:.72rem; padding:3px 10px; border-radius:4px;
  font-family:'Share Tech Mono',monospace;
}
.metric-row { display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }
.metric-chip { background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:.6rem 1.2rem; min-width:110px; }
.metric-val { font-family:'Share Tech Mono',monospace; font-size:1.4rem; color:var(--green); }
.metric-lbl { font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:1px; }
.pipeline-step {
  background:var(--panel); border-left:3px solid var(--accent);
  border-radius:0 8px 8px 0; padding:.7rem 1rem; margin:.4rem 0;
  font-size:.85rem; color:#e2eaf5;
}
.pipeline-step-label {
  font-family:'Share Tech Mono',monospace; font-size:.7rem;
  color:var(--accent); text-transform:uppercase; letter-spacing:1px;
}
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
    YOLOv8 localiza cada placa, EasyOCR extrae los caracteres, y el motor de corrección
    valida el formato <strong>ABC-123</strong> con tablas de confusión OCR avanzadas.
  </p>
  <span class="badge">YOLOv8</span>
  <span class="badge">EasyOCR</span>
  <span class="badge">Multi-Placa</span>
  <span class="badge">Corrección OCR</span>
  <span class="badge">Validación Formato</span>
</div>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────
 
# Palabras del fondo/borde de la placa colombiana que NO son parte de la matrícula
PALABRAS_IGNORAR = {
    "COLOMBIA", "COLONBIA", "COLOM", "COLOMBI",
    "COL", "POLICIA", "TRANSITO", "DIPLOMATICO",
    "MOTO", "PARTICULAR", "PUBLICO", "OFICIAL",
}
 
# Cuando el OCR lee un dígito donde debería ir una letra
DIGIT_TO_LETTER = {
    "0": "O", "1": "I", "8": "B", "5": "S",
    "6": "G", "2": "Z", "4": "A", "7": "T",
}
 
# Cuando el OCR lee una letra donde debería ir un dígito
LETTER_TO_DIGIT = {
    "O": "0", "I": "1", "B": "8", "S": "5",
    "G": "6", "Z": "2", "A": "4", "T": "7",
    "Q": "0", "D": "0", "L": "1", "E": "8",
    "U": "0", "V": "7", "F": "7",
}
 
# ─────────────────────────────────────────────
#  FUNCIONES DE PROCESAMIENTO
# ─────────────────────────────────────────────
 
def enhance_for_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    """
    4 pasos de preprocesamiento antes del OCR:
      1. BGR → Escala de grises
      2. Upscaling 2x (INTER_CUBIC)
      3. CLAHE (clipLimit=2.0, tileGridSize=8x8)
      4. Filtro bilateral (d=11, sigmaColor=17, sigmaSpace=17)
    """
    gris = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    alto, ancho = gris.shape
    gris = cv2.resize(gris, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gris = clahe.apply(gris)
    gris = cv2.bilateralFilter(gris, 11, 17, 17)
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
 
 
def clean_plate_text(texts: list) -> str:
    """
    Une tokens OCR y filtra palabras del fondo de la placa colombiana.
    Solo conserva A-Z y 0-9.
    """
    filtrados = []
    for t in texts:
        t_upper = t.upper().strip()
        if t_upper in PALABRAS_IGNORAR:
            continue
        if any(palabra in t_upper for palabra in PALABRAS_IGNORAR):
            continue
        filtrados.append(t_upper)
    combined = " ".join(filtrados)
    return re.sub(r"[^A-Z0-9]", "", combined)
 
 
def correct_colombian_plate(text: str) -> tuple:
    """
    Corrige y formatea al estándar colombiano: ABC-123
    Retorna (placa, dict_con_detalle_del_proceso)
 
    Algoritmo:
      1. Busca patrón exacto [A-Z]{3}[0-9]{3} con regex.
      2. Separa letras y dígitos.
      3. Si faltan letras: convierte dígitos con DIGIT_TO_LETTER.
      4. Si faltan dígitos: convierte letras sobrantes con LETTER_TO_DIGIT.
      5. Rellena con X/0 si aún faltan.
    """
    clean = re.sub(r"[^A-Z0-9]", "", text.upper())
    correcciones = []
 
    if not clean:
        return "XXX-000", {"metodo": "texto_vacio", "correcciones": ["Texto vacío → XXX-000"]}
 
    # Intento 1 — patrón exacto regex
    patron = re.search(r'[A-Z]{3}[0-9]{3}', clean)
    if patron:
        match = patron.group()
        return f"{match[:3]}-{match[3:]}", {"metodo": "patron_exacto", "correcciones": []}
 
    # Intento 2 — separar y corregir por tablas de confusión
    letras  = [c for c in clean if c.isalpha()]
    digitos = [c for c in clean if c.isdigit()]
 
    while len(letras) < 3 and digitos:
        candidato = digitos.pop(0)
        convertido = DIGIT_TO_LETTER.get(candidato, "X")
        letras.append(convertido)
        correcciones.append(f"'{candidato}' (dígito) → '{convertido}' (letra)")
 
    while len(digitos) < 3 and len(letras) > 3:
        candidato = letras.pop(3)
        convertido = LETTER_TO_DIGIT.get(candidato, "0")
        digitos.insert(0, convertido)
        correcciones.append(f"'{candidato}' (letra) → '{convertido}' (dígito)")
 
    letras_comunes  = ["X", "A", "B"]
    digitos_comunes = ["0", "1", "2"]
 
    while len(letras) < 3:
        r = letras_comunes[len(letras)]
        letras.append(r)
        correcciones.append(f"Relleno letra '{r}'")
 
    while len(digitos) < 3:
        r = digitos_comunes[len(digitos)]
        digitos.append(r)
        correcciones.append(f"Relleno dígito '{r}'")
 
    placa = f"{''.join(letras[:3])}-{''.join(digitos[:3])}"
    return placa, {"metodo": "correccion_confusion_ocr", "correcciones": correcciones}
 
 
def validate_plate_format(plate: str) -> dict:
    """
    Valida el formato colombiano ABC-123:
      - 7 caracteres totales con guion
      - Guion en posición 4
      - Exactamente 3 letras A-Z
      - Exactamente 3 dígitos 0-9
      - Solo mayúsculas
    """
    partes        = plate.split("-")
    parte_letras  = partes[0] if len(partes) > 0 else ""
    parte_digitos = partes[1] if len(partes) > 1 else ""
 
    checks = {
        "longitud_total"       : len(plate) == 7,
        "tiene_guion"          : "-" in plate,
        "posicion_guion"       : len(parte_letras) == 3,
        "exactamente_3_letras" : len(parte_letras) == 3 and parte_letras.isalpha(),
        "exactamente_3_digitos": len(parte_digitos) == 3 and parte_digitos.isdigit(),
        "solo_mayusculas"      : parte_letras == parte_letras.upper(),
    }
    checks["formato_valido"] = all(checks.values())
    return checks
 
 
# ─────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def run_detection(frame_bgr, model, ocr_reader,
                  conf_th=0.45, iou_th=0.45,
                  do_enhance=True, show_conf=True,
                  ocr_conf_th=0.10):
    results = model(frame_bgr, conf=conf_th, iou=iou_th, verbose=False)[0]
    boxes   = results.boxes.xyxy.cpu().numpy()
    confs   = results.boxes.conf.cpu().numpy()
 
    annotated   = frame_bgr.copy()
    plates_info = []
 
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
 
        crop_ocr = enhance_for_ocr(crop) if do_enhance else crop
 
        ocr_results = ocr_reader.readtext(
            crop_ocr,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1, paragraph=False,
            width_ths=1.0, height_ths=1.0,
        )
 
        texts     = [r[1] for r in ocr_results if r[2] > ocr_conf_th]
        confs_ocr = [r[2] for r in ocr_results if r[2] > ocr_conf_th]
 
        # Retry con umbral más bajo si el texto tiene menos de 6 caracteres
        retry_used = False
        if len(clean_plate_text(texts)) < 6:
            texts_low = [r[1] for r in ocr_results if 0.01 < r[2] <= ocr_conf_th]
            confs_low = [r[2] for r in ocr_results if 0.01 < r[2] <= ocr_conf_th]
            texts     = texts + texts_low
            confs_ocr = confs_ocr + confs_low
            retry_used = True
 
        raw_text     = clean_plate_text(texts)
        plate, corr  = correct_colombian_plate(raw_text)
        validation   = validate_plate_format(plate)
        avg_ocr_conf = float(np.mean(confs_ocr)) if confs_ocr else 0.0
 
        # Verde = válida, Cyan = corregida
        color = (0, 255, 157) if validation["formato_valido"] else (0, 212, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{plate}  {conf:.0%}" if show_conf else plate
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
 
        plates_info.append({
            "bbox"        : (x1, y1, x2, y2),
            "det_conf"    : float(conf),
            "plate_text"  : plate,
            "raw_text"    : raw_text,
            "raw_tokens"  : texts,
            "ocr_conf"    : avg_ocr_conf,
            "corr_info"   : corr,
            "validation"  : validation,
            "retry_used"  : retry_used,
            "crop_rgb"    : cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
            "crop_ocr_rgb": cv2.cvtColor(crop_ocr, cv2.COLOR_BGR2RGB),
        })
 
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), plates_info
 
 
# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
MODEL_PATH = "best.pt"
 
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")
    conf_threshold = st.slider("Umbral confianza YOLO",  0.10, 0.99, 0.45, 0.01)
    iou_threshold  = st.slider("Umbral IoU (NMS)",       0.10, 0.99, 0.45, 0.01)
    ocr_conf_th    = st.slider("Umbral confianza OCR",   0.01, 0.99, 0.10, 0.01)
    ocr_langs      = st.multiselect("Idiomas OCR", ["es","en","pt","fr","de"], default=["es","en"])
    draw_conf      = st.checkbox("Mostrar confianza en imagen", value=True)
    enhance_crop   = st.checkbox("Mejora OCR (gray→2x→CLAHE→bilateral)", value=True)
    show_debug     = st.checkbox("Mostrar detalle de corrección OCR", value=False)
    st.markdown("---")
    st.markdown("<small style='color:#4a5568'>PlateVision AI · YOLOv8 + EasyOCR</small>",
                unsafe_allow_html=True)
 
 
# ─────────────────────────────────────────────
#  LOADERS
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)
 
@st.cache_resource(show_spinner=False)
def load_ocr(langs):
    return easyocr.Reader(list(langs), gpu=False)
 
with st.spinner("Cargando modelo YOLOv8…"):
    model = load_model(MODEL_PATH)
 
with st.spinner("Iniciando EasyOCR…"):
    ocr_reader = load_ocr(tuple(ocr_langs) if ocr_langs else ("es", "en"))
 
if model is None:
    st.error(f"⚠️ No se encontró `{MODEL_PATH}`. Ponlo en la misma carpeta que app.py.")
    st.stop()
else:
    st.success(f"✅ Modelo cargado — `{MODEL_PATH}`", icon="🤖")
 
 
# ─────────────────────────────────────────────
#  RENDER DE RESULTADOS
# ─────────────────────────────────────────────
def render_plate_card(info: dict, idx: int):
    plate_str = info["plate_text"]
    validation = info["validation"]
    corr_info  = info["corr_info"]
    es_valida  = validation["formato_valido"]
 
    box_cls = "plate-box-valid"   if es_valida else "plate-box-invalid"
    txt_cls = "plate-text-valid"  if es_valida else "plate-text-invalid"
    estado  = "✅ FORMATO VÁLIDO" if es_valida else "⚠️ FORMATO CORREGIDO"
 
    st.markdown(f"<div class='card-title'>Placa #{idx}</div>", unsafe_allow_html=True)
 
    st.markdown(f"""
    <div class='{box_cls}'>
      <div class='{txt_cls}'>{plate_str}</div>
      <div class='plate-label'>{estado} · 3 LETRAS + 3 DÍGITOS · ABC-123</div>
    </div>
    """, unsafe_allow_html=True)
 
    # Chips de validación
    v_labels = {
        "longitud_total"       : "7 caracteres",
        "tiene_guion"          : "Guion presente",
        "posicion_guion"       : "Guion en pos. 4",
        "exactamente_3_letras" : "3 letras A-Z",
        "exactamente_3_digitos": "3 dígitos 0-9",
        "solo_mayusculas"      : "Mayúsculas",
    }
    chips = "<div class='validation-row'>"
    for key, label in v_labels.items():
        ok  = validation.get(key, False)
        cls = "val-chip-ok" if ok else "val-chip-fail"
        chips += f"<span class='{cls}'>{'✓' if ok else '✗'} {label}</span>"
    chips += "</div>"
    st.markdown(chips, unsafe_allow_html=True)
 
    # Métricas
    n_corr = len(corr_info.get("correcciones", []))
    st.markdown(f"""
    <div class='metric-row'>
      <div class='metric-chip'>
        <div class='metric-val'>{info['det_conf']:.0%}</div>
        <div class='metric-lbl'>YOLO conf.</div>
      </div>
      <div class='metric-chip'>
        <div class='metric-val'>{info['ocr_conf']:.0%}</div>
        <div class='metric-lbl'>OCR conf.</div>
      </div>
      <div class='metric-chip'>
        <div class='metric-val'>{n_corr}</div>
        <div class='metric-lbl'>Correcciones</div>
      </div>
      <div class='metric-chip'>
        <div class='metric-val'>{'🔁' if info['retry_used'] else '1x'}</div>
        <div class='metric-lbl'>Pasadas OCR</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    raw = info.get("raw_text", "")
    if raw:
        st.markdown(f"**Texto crudo OCR:** `{raw}`")
 
    if show_debug:
        with st.expander(f"🔍 Detalle corrección — Placa #{idx}"):
            metodo = corr_info.get("metodo", "")
            corrs  = corr_info.get("correcciones", [])
            st.markdown(f"""
            <div class='pipeline-step'>
              <div class='pipeline-step-label'>Método</div>
              <code>{metodo}</code>
            </div>
            <div class='pipeline-step'>
              <div class='pipeline-step-label'>Correcciones aplicadas</div>
              {'<br>'.join([f"• {c}" for c in corrs]) if corrs else '• Ninguna — patrón directo encontrado'}
            </div>
            """, unsafe_allow_html=True)
            if info.get("raw_tokens"):
                st.markdown("**Tokens OCR crudos:**")
                for t in info["raw_tokens"]:
                    st.code(t, language=None)
 
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(info["crop_rgb"],     caption=f"Recorte original #{idx}",  use_container_width=True)
    with col_b:
        st.image(info["crop_ocr_rgb"], caption=f"Recorte procesado #{idx}", use_container_width=True)
 
    st.markdown("<hr style='border-color:#1e2d3d;margin:1rem 0'>", unsafe_allow_html=True)
 
 
def render_results(annotated_rgb, plates_info):
    col_img, col_info = st.columns([3, 2], gap="medium")
    with col_img:
        st.image(annotated_rgb, caption="Detección anotada", use_container_width=True)
    with col_info:
        if not plates_info:
            st.warning("No se detectaron placas en esta imagen.")
            return
        for idx, info in enumerate(plates_info, 1):
            render_plate_card(info, idx)
 
 
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
                    ocr_conf_th=ocr_conf_th,
                )
                elapsed = time.time() - t0
            st.markdown(
                f"<small style='color:#4a5568'>Inferencia: <b>{elapsed*1000:.0f} ms</b> · "
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
                        ocr_conf_th=ocr_conf_th,
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
                validas    = list({p["plate_text"] for p in all_plates if p["validation"]["formato_valido"]})
                corregidas = list({p["plate_text"] for p in all_plates if not p["validation"]["formato_valido"]})
                st.markdown("### 📋 Placas únicas detectadas")
                if validas:
                    st.markdown("**✅ Formato válido:**")
                    for pl in validas:
                        st.markdown(f"<div class='plate-box-valid' style='margin:.4rem 0'><div class='plate-text-valid' style='font-size:1.8rem'>{pl}</div></div>", unsafe_allow_html=True)
                if corregidas:
                    st.markdown("**⚠️ Corregidas (revisar):**")
                    for pl in corregidas:
                        st.markdown(f"<div class='plate-box-invalid' style='margin:.4rem 0'><div class='plate-text-invalid' style='font-size:1.8rem'>{pl}</div></div>", unsafe_allow_html=True)
            else:
                st.info("No se detectaron placas en los frames analizados.")
 
# ══════════════════════════════════════════════
#  TAB 3 — CÁMARA WEB
# ══════════════════════════════════════════════
with tab_cam:
    st.markdown("<div class='card-title'>Cámara Web — Detección en Vivo</div>",
                unsafe_allow_html=True)
    st.info("Apunta la cámara hacia una placa y presiona el botón de captura.")
 
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
                    ocr_conf_th=ocr_conf_th,
                )
                elapsed = time.time() - t0
 
            st.markdown(f"<small style='color:#4a5568'>Inferencia: <b>{elapsed*1000:.0f} ms</b></small>",
                        unsafe_allow_html=True)
 
            if plates:
                for idx, info in enumerate(plates, 1):
                    es_valida = info["validation"]["formato_valido"]
                    box_cls   = "plate-box-valid"  if es_valida else "plate-box-invalid"
                    txt_cls   = "plate-text-valid" if es_valida else "plate-text-invalid"
                    n_corr    = len(info["corr_info"].get("correcciones", []))
                    st.markdown(f"""
                    <div class='{box_cls}'>
                      <div class='{txt_cls}'>{info['plate_text']}</div>
                      <div class='plate-label'>PLACA #{idx} · {'✅ VÁLIDA' if es_valida else '⚠️ CORREGIDA'}</div>
                    </div>
                    <div class='metric-row'>
                      <div class='metric-chip'><div class='metric-val'>{info['det_conf']:.0%}</div><div class='metric-lbl'>YOLO</div></div>
                      <div class='metric-chip'><div class='metric-val'>{info['ocr_conf']:.0%}</div><div class='metric-lbl'>OCR</div></div>
                      <div class='metric-chip'><div class='metric-val'>{n_corr}</div><div class='metric-lbl'>Correcciones</div></div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No se detectó ninguna placa. Intenta bajar el umbral de confianza.")
 
    if cam_img is not None and 'plates' in dir() and plates:
        st.markdown("**Frame anotado:**")
        st.image(annotated, use_container_width=True)
 
