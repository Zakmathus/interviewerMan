# InterviewerMan

Aplicación de escritorio en Python que abre la cámara y el micrófono al iniciar, acumula el transcript de lo que escucha y usa OpenAI para responder en español con contexto multimodal. Presiona `S` para tomar una foto; la imagen se guarda localmente y se adjunta a la siguiente consulta de IA.

## Requisitos

- Python 3.11+
- Cámara y micrófono disponibles en el equipo
- Una API key de OpenAI configurada en `OPENAI_API_KEY`

> No es posible crear una API key desde la app ni desde este repositorio. Créala en tu cuenta de OpenAI y expórtala como variable de entorno.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
export OPENAI_API_KEY="tu_api_key"
python app.py
```

Opcionalmente puedes cambiar modelos con variables de entorno:

```bash
export OPENAI_CHAT_MODEL="gpt-4.1-mini"
export OPENAI_TRANSCRIBE_MODEL="gpt-4o-mini-transcribe"
```

## Uso

1. Inicia la app con `python app.py`.
2. La cámara se muestra en vivo y el micrófono se transcribe por chunks de 8 segundos.
3. El transcript acumulado aparece en el panel derecho.
4. La respuesta de la IA se actualiza usando el transcript y hasta las últimas 3 fotos.
5. Presiona `S` o el botón **Tomar foto (S)** para capturar una imagen.

Las fotos se guardan en `captures/`.
