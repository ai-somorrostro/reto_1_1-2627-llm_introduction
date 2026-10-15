# Fundamentos de LLMs con Hugging Face + introducción a LangChain

Notebook práctico (`llm_fundamentos.ipynb`) para aprender a cargar, configurar, usar y adaptar modelos de lenguaje pequeños con las librerías de Hugging Face, con una introducción a LangChain al final.

| # | Sección | Modelos |
|---|---------|---------|
| 1 | Las tres familias de Transformers (encoder, decoder, encoder-decoder) | [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased), [DistilBERT-SST2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), [BERT-NER](https://huggingface.co/dslim/bert-base-NER), [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [Flan-T5-small](https://huggingface.co/google/flan-t5-small) |
| 2 | Adaptación al dominio con el dataset de recetas [corbt/all-recipes](https://huggingface.co/datasets/corbt/all-recipes) | [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), [Flan-T5-small](https://huggingface.co/google/flan-t5-small), [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) |
| 3 | Parámetros de generación y *chat templates* | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| 4 | Modelo con *instruction tuning* vs. modelo base | [T5-base](https://huggingface.co/google-t5/t5-base) vs [Flan-T5-base](https://huggingface.co/google/flan-t5-base), [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) vs [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) |
| 5 | Introducción a LangChain | LLM vía OpenRouter + [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |

Todos los modelos tienen **≤ 0.5B parámetros**. Hay dos formas de ejecutar el notebook:

- **En local**, si tienes una GPU NVIDIA (con ~6 GB de VRAM es suficiente) → [Opción A](#opción-a-ejecución-en-local-con-uv).
- **En Google Colab**, si no tienes GPU → [Opción B](#opción-b-ejecución-en-google-colab).

---

## Opción A: ejecución en local con `uv`

[`uv`](https://docs.astral.sh/uv/) es un gestor de proyectos y paquetes de Python muy rápido. Descarga la versión de Python adecuada, crea el entorno virtual (`.venv`) e instala las dependencias exactas definidas en `pyproject.toml`.

### 1. Instalar uv

**Linux / macOS / WSL**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cierra y vuelve a abrir la terminal y comprueba la instalación:

```bash
uv --version
```

### 2. Crear el entorno e instalar las dependencias

Desde la carpeta del proyecto (donde está `pyproject.toml`):

```bash
uv sync
```

Esto hace tres cosas:

- Descarga Python 3.12 si no lo tienes (versión fijada en `.python-version`).
- Crea el entorno virtual en `.venv/`.
- Instala todas las dependencias. En **Linux y Windows**, PyTorch se instala **con soporte CUDA** (índice `cu128` configurado en `pyproject.toml`); en **macOS**, se instala la versión estándar (CPU / Apple MPS).

> La primera vez tarda unos minutos: PyTorch con CUDA ocupa ~3 GB.

### 3. Comprobar la GPU

```bash
uv run python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

Si aparece `CUDA: True`, todo está listo. Si aparece `False`, consulta [Problemas habituales](#problemas-habituales).

### 4. Configurar la API key de OpenRouter (sección 5)

La sección de LangChain usa [OpenRouter](https://openrouter.ai/), que ofrece modelos gratuitos.

1. Crea una cuenta en <https://openrouter.ai/> y genera una clave en **Keys**.
2. Copia la plantilla y pega tu clave:

   ```bash
   cp .env.example .env
   # edita .env:  OPENROUTER_API_KEY=sk-or-v1-...
   ```

El fichero `.env` está en `.gitignore`: **nunca subas tu clave a GitHub**.

### 5. Abrir el notebook

Elige **una** de estas tres formas:

**a) JupyterLab en el navegador**

```bash
uv run jupyter lab
```

Se abrirá el navegador. Abre `llm_fundamentos.ipynb`; el kernel *Python 3* ya es el del entorno `.venv`.

**b) VS Code**

1. Instala las extensiones **Python** y **Jupyter**.
2. Abre la carpeta del proyecto (`code .`).
3. Abre `llm_fundamentos.ipynb` → **Select Kernel** (arriba a la derecha) → **Python Environments** → `.venv`.

**c) Registrar el entorno como kernel de Jupyter** (para usarlo desde otro Jupyter que ya tengas instalado)

```bash
uv run python -m ipykernel install --user --name llm-fundamentos --display-name "Python (llm-fundamentos)"
```

Después, elige el kernel **Python (llm-fundamentos)** en tu Jupyter.

### Comandos útiles de uv

| Comando | Para qué sirve |
|---|---|
| `uv sync` | Crear o actualizar el entorno según `pyproject.toml` |
| `uv add <paquete>` | Añadir una dependencia (actualiza `pyproject.toml`) |
| `uv remove <paquete>` | Quitar una dependencia |
| `uv run <comando>` | Ejecutar un comando dentro del entorno, sin activarlo |
| `source .venv/bin/activate` (Linux/macOS) / `.venv\Scripts\activate` (Windows) | Activar el entorno manualmente |

---

## Opción B: ejecución en Google Colab

1. Abre <https://colab.research.google.com/> → **Archivo → Subir cuaderno** y sube `llm_fundamentos.ipynb`. También puedes abrirlo desde GitHub (pestaña *GitHub*) o desde Google Drive.
2. **Activa la GPU**: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → Acelerador por hardware: T4 GPU → Guardar**.
3. Ejecuta las celdas en orden. La primera celda detecta que estás en Colab e instala las librerías con `pip` (en local no hace nada).
4. **API key de OpenRouter (sección 5)**: en la barra lateral izquierda, abre **Secretos** (icono de la llave) → **Añadir nuevo secreto**:
   - Nombre: `OPENROUTER_API_KEY`
   - Valor: tu clave `sk-or-v1-...`
   - Activa **Acceso del cuaderno**.

   El notebook la lee automáticamente con `google.colab.userdata`.

Ten en cuenta que:

- Todo lo que hay en la máquina de Colab (modelos descargados y modelos entrenados en `outputs/`) **se borra al cerrar la sesión**. Si quieres conservar los modelos adaptados, monta Google Drive (`from google.colab import drive; drive.mount('/content/drive')`) y cambia `OUTPUT_DIR` a una carpeta de tu Drive.
- La T4 no soporta bf16: el notebook lo detecta y usa fp16 automáticamente (o fp32 para T5, que es inestable en fp16).
- En la versión gratuita, la sesión se desconecta si está inactiva un rato.

---

## Problemas habituales

| Problema | Solución |
|---|---|
| `torch.cuda.is_available()` devuelve `False` en local | Comprueba que `nvidia-smi` funciona y que el driver de NVIDIA está actualizado (necesita soporte de CUDA ≥ 12.8). En WSL2, instala el driver **en Windows**, no dentro de WSL. |
| `CUDA out of memory` | Reinicia el kernel y reduce `BATCH_SIZE` (sección 2). Comprueba que se han ejecutado las celdas `del ...; liberar_memoria()`. |
| El entrenamiento es muy lento (CPU) | Pon `MODO_RAPIDO = True` en la celda de configuración de la sección 2, o usa Colab con GPU. |
| `No se ha encontrado OPENROUTER_API_KEY` | Revisa que el fichero `.env` está en la carpeta del notebook, o el secreto en Colab. Reinicia el kernel después de crearlo. |
| Error 404 o 429 de OpenRouter | El modelo gratuito no está disponible o has superado el límite de peticiones. Espera un poco o cambia `MODELO` por otro de <https://openrouter.ai/models?max_price=0>. |
| Algún modelo del Hub pide autenticación | Ejecuta `uv run huggingface-cli login` (o `hf auth login`) con un token de <https://huggingface.co/settings/tokens>. Los modelos del notebook son públicos y no lo necesitan. |

## Estructura del proyecto

```
.
├── llm_fundamentos.ipynb   # notebook principal
├── pyproject.toml          # dependencias (uv)
├── uv.lock                 # versiones exactas (generado por uv)
├── .python-version         # versión de Python del proyecto
├── .env.example            # plantilla de la API key
└── outputs/                # modelos adaptados (se crea al ejecutar la sección 2)
```
