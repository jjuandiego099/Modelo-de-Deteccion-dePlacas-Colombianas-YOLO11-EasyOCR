# 🚗 PlateVision AI — Detección de Placas Vehiculares Colombianas

**Autor:** Juan Diego Chaparro García  
**Stack:** YOLOv11 · EasyOCR · PaddleOCR · Streamlit · Roboflow · Google Colab
**Streamlit:** [App](https://modelo-de-deteccion-deplacas-colombianas-yolo11-easyocr-xfzab3.streamlit.app/)


---

## 🇨🇴 ¿Qué son las placas colombianas?

En Colombia, el **Ministerio de Transporte** regula el formato de las placas vehiculares mediante la **Resolución 3500 de 2005** y sus actualizaciones. El estándar es:

```
ABC - 123
───   ───
 │     └─ Tres dígitos numéricos (0–9)
 └──────── Tres letras mayúsculas (A–Z)
```

Las placas colombianas son blancas con letras y números negros, y contienen en la parte inferior el nombre del departamento emisor (ej: `CUNDINAMARCA`, `ANTIOQUIA`, `SANTANDER`). Este proyecto detecta y lee automáticamente esas placas usando visión artificial.

---

## 📁 Estructura del proyecto

```
platevision/
├── PLACA.ipynb        ← Notebook de entrenamiento (Google Colab, GPU T4)
├── app.py             ← Aplicación web Streamlit para inferencia
├── best.pt            ← Pesos del modelo entrenado (generado por el notebook)
└── README.md          ← Este archivo
```

---

## 🧠 Arquitectura del sistema

```
Imagen/Video/Cámara
        │
        ▼
  ┌─────────────┐
  │  YOLOv11-L  │  ← Detecta la región de la placa (bounding box)
  └──────┬──────┘
         │  recorte de la placa
         ▼
  ┌─────────────────────────────┐
  │  Preprocesamiento (4 pasos) │
  │  1. Escala de grises        │
  │  2. Upscaling 2x (CUBIC)   │
  │  3. CLAHE (contraste)       │
  │  4. Filtro bilateral        │
  └──────────────┬──────────────┘
                 │
                 ▼
         ┌──────────────┐
         │  EasyOCR /   │  ← Lee los caracteres alfanuméricos
         │  PaddleOCR   │
         └──────┬───────┘
                │
                ▼
     Corrección formato ABC-123
     (tablas de confusión OCR:
      0↔O, 1↔I, 8↔B, 5↔S…)
                │
                ▼
          🏁 ABC - 123
```

---

## 📓 Notebook: `PLACA.ipynb`

El notebook cubre el ciclo completo de entrenamiento en **Google Colab con GPU T4**:

| Sección | Descripción |
|---|---|
| Instalación | Dependencias: `ultralytics`, `roboflow`, `easyocr` |
| Dataset | Descarga desde Roboflow — placas colombianas anotadas |
| Exploración | Estructura de carpetas, conteo de etiquetas, visualización |
| Configuración | Lectura de `data.yaml` con clases y rutas |
| Entrenamiento | Fine-tuning YOLOv11-Large — 50 épocas, early stopping |
| Validación | mAP@50, Precision, Recall sobre conjunto de validación |
| Métricas | Gráficas de pérdidas, mAP y precisión/recall por época |
| Pipeline OCR | `detect_and_recognize_plate()` — detección + OCR + formato |
| Exportación | Modelo exportado a ONNX para despliegue en `app.py` |

---

## 🖥️ Aplicación web: `app.py`

Interfaz **Streamlit** con diseño oscuro futurista para usar el modelo entrenado:

- **📷 Imagen** — sube un JPG/PNG y detecta todas las placas
- **🎬 Video** — procesa frames seleccionados de un MP4/AVI
- **📡 Cámara web** — captura en tiempo real desde la cámara del dispositivo

### Ejecutar localmente

```bash
# 1. Instalar dependencias
pip install streamlit ultralytics easyocr opencv-python

# 2. Colocar best.pt en la misma carpeta que app.py

# 3. Lanzar la aplicación
streamlit run app.py
```

### Parámetros configurables (sidebar)

| Parámetro | Descripción | Valor recomendado |
|---|---|---|
| Umbral de confianza | Sensibilidad de detección YOLO | ≥ 0.45 |
| Umbral IoU (NMS) | Supresión de detecciones duplicadas | 0.45 |
| Mejorar recorte | Aplica los 4 pasos de preprocesamiento | ✅ Activado |

---

## 📊 Métricas del modelo entrenado

| Métrica | Valor |
|---|---|
| mAP@50 | ~0.95+ |
| Precision | ~0.93+ |
| Recall | ~0.94+ |
| Épocas entrenadas | 50 (con early stopping) |
| Tamaño de imagen | 640×640 px |
| Backbone | YOLOv11-Large (COCO pretrained) |

> Los valores exactos dependen del dataset y la sesión de entrenamiento. Consulta las gráficas generadas en el notebook.

---

## 🔧 Corrección de caracteres OCR

El OCR comete errores predecibles en las placas colombianas. El sistema aplica corrección automática basada en la posición:

```
Posiciones 1-3 → SIEMPRE letras   → dígito confundido se convierte: 0→O, 1→I, 8→B...
Posiciones 4-6 → SIEMPRE números  → letra confundida se convierte:  O→0, I→1, B→8...
```

Esto garantiza que cualquier combinación leída por el OCR (ej: `1BC456`, `ABC4S6`) se corrija al formato estándar `ABC-456`.

---

## 👤 Autor

**Juan Diego Chaparro García**  
Proyecto de visión artificial aplicada a la identificación vehicular en Colombia.

---

## 📄 Licencia

Uso académico y educativo. Dataset provisto por [Roboflow Universe](https://universe.roboflow.com).
