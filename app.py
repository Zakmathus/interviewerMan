"""Desktop assistant that listens to microphone audio, watches the camera, and asks OpenAI for help.

Run with:
    OPENAI_API_KEY=sk-... python app.py

Press "s" while the app window is focused to capture a photo and include it in the next AI request.
"""

from __future__ import annotations

import base64
import os
import queue
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import sounddevice as sd
import tkinter as tk
from openai import OpenAI
from PIL import Image, ImageTk
from tkinter import messagebox, scrolledtext, ttk


APP_TITLE = "InterviewerMan - Asistente multimodal"
AUDIO_SAMPLE_RATE = 16_000
AUDIO_CHANNELS = 1
AUDIO_CHUNK_SECONDS = 8
MAX_IMAGES_PER_REQUEST = 3
CAPTURES_DIR = Path("captures")
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
DEFAULT_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")


@dataclass(frozen=True)
class CapturedPhoto:
    path: Path
    data_url: str
    created_at: str


class AudioTranscriber(threading.Thread):
    """Records microphone audio continuously and sends chunks to OpenAI for transcription."""

    def __init__(
        self,
        client: OpenAI,
        on_transcript: Callable[[str], None],
        on_status: Callable[[str], None],
        stop_event: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self.client = client
        self.on_transcript = on_transcript
        self.on_status = on_status
        self.stop_event = stop_event
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()

    def run(self) -> None:
        self.on_status("Micrófono activo")
        try:
            with sd.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=AUDIO_CHANNELS,
                dtype="float32",
                callback=self._audio_callback,
            ):
                self._process_audio_chunks()
        except Exception as exc:
            self.on_status(f"Error de micrófono: {exc}")

    def _audio_callback(self, indata: np.ndarray, _frames: int, _time_info: object, status: sd.CallbackFlags) -> None:
        if status:
            self.on_status(f"Aviso de audio: {status}")
        self.audio_queue.put(indata.copy())

    def _process_audio_chunks(self) -> None:
        frames: list[np.ndarray] = []
        frames_needed = AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECONDS
        collected_frames = 0

        while not self.stop_event.is_set():
            try:
                audio = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            frames.append(audio)
            collected_frames += len(audio)

            if collected_frames >= frames_needed:
                chunk = np.concatenate(frames, axis=0)
                frames.clear()
                collected_frames = 0
                self._transcribe_chunk(chunk)

    def _transcribe_chunk(self, audio: np.ndarray) -> None:
        if self._is_mostly_silence(audio):
            return

        wav_path = self._write_temp_wav(audio)
        try:
            with wav_path.open("rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=DEFAULT_TRANSCRIBE_MODEL,
                    file=audio_file,
                    response_format="text",
                )
            text = str(transcription).strip()
            if text:
                self.on_transcript(text)
        except Exception as exc:
            self.on_status(f"Error transcribiendo: {exc}")
        finally:
            wav_path.unlink(missing_ok=True)

    @staticmethod
    def _is_mostly_silence(audio: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return rms < 0.01

    @staticmethod
    def _write_temp_wav(audio: np.ndarray) -> Path:
        int_audio = np.clip(audio, -1.0, 1.0)
        int_audio = (int_audio * 32767).astype(np.int16)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = Path(temp_file.name)
        temp_file.close()

        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(AUDIO_CHANNELS)
            wav_file.setsampwidth(2)
            wav_file.setframerate(AUDIO_SAMPLE_RATE)
            wav_file.writeframes(int_audio.tobytes())

        return temp_path


class CameraWorker(threading.Thread):
    """Continuously reads frames from the default camera."""

    def __init__(self, on_frame: Callable[[np.ndarray], None], on_status: Callable[[str], None], stop_event: threading.Event) -> None:
        super().__init__(daemon=True)
        self.on_frame = on_frame
        self.on_status = on_status
        self.stop_event = stop_event
        self.latest_frame: np.ndarray | None = None
        self.frame_lock = threading.Lock()

    def run(self) -> None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            self.on_status("No se pudo abrir la cámara")
            return

        self.on_status("Cámara activa")
        try:
            while not self.stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    self.on_status("No se pudo leer un frame de la cámara")
                    time.sleep(0.5)
                    continue

                with self.frame_lock:
                    self.latest_frame = frame.copy()
                self.on_frame(frame)
                time.sleep(1 / 30)
        finally:
            camera.release()
            self.on_status("Cámara detenida")

    def snapshot(self) -> np.ndarray | None:
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()


class OpenAIAssistant:
    """Maintains transcript/photo context and requests concise answers from OpenAI."""

    def __init__(self, client: OpenAI | None, on_answer: Callable[[str], None], on_status: Callable[[str], None]) -> None:
        self.client = client
        self.on_answer = on_answer
        self.on_status = on_status
        self.transcript_parts: list[str] = []
        self.photos: list[CapturedPhoto] = []
        self.lock = threading.Lock()
        self.request_lock = threading.Lock()

    def add_transcript(self, text: str) -> None:
        with self.lock:
            self.transcript_parts.append(text)
        self.ask_async("Nuevo audio transcrito")

    def add_photo(self, photo: CapturedPhoto) -> None:
        with self.lock:
            self.photos.append(photo)
        self.ask_async(f"Foto capturada: {photo.path.name}")

    def ask_async(self, reason: str) -> None:
        worker = threading.Thread(target=self._ask, args=(reason,), daemon=True)
        worker.start()

    def _ask(self, reason: str) -> None:
        if not self.request_lock.acquire(blocking=False):
            self.on_status("IA ocupada; acumularé el contexto para la siguiente respuesta")
            return

        try:
            if self.client is None:
                self.on_status("Falta OPENAI_API_KEY; no puedo consultar transcripción/IA")
                return

            transcript, photos = self._context_snapshot()
            if not transcript and not photos:
                return

            self.on_status(f"Consultando IA: {reason}")
            content: list[dict[str, str]] = [
                {
                    "type": "input_text",
                    "text": (
                        "Analiza el transcript acumulado y las fotos adjuntas. "
                        "Responde en español con ayuda directa para contestar la pregunta o explicar qué se ve. "
                        "Si falta contexto, dilo brevemente y sugiere qué información escuchar o fotografiar.\n\n"
                        f"TRANSCRIPT ACUMULADO:\n{transcript or '[Todavía sin audio transcrito]'}"
                    ),
                }
            ]
            for photo in photos[-MAX_IMAGES_PER_REQUEST:]:
                content.append({"type": "input_image", "image_url": photo.data_url})

            response = self.client.responses.create(
                model=DEFAULT_CHAT_MODEL,
                input=[{"role": "user", "content": content}],
            )
            answer = getattr(response, "output_text", "").strip()
            if not answer:
                answer = "La IA no devolvió texto en esta respuesta."
            self.on_answer(answer)
            self.on_status("Respuesta de IA actualizada")
        except Exception as exc:
            self.on_status(f"Error consultando IA: {exc}")
        finally:
            self.request_lock.release()

    def _context_snapshot(self) -> tuple[str, list[CapturedPhoto]]:
        with self.lock:
            return "\n".join(self.transcript_parts), list(self.photos)


class InterviewerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x780")
        self.minsize(980, 680)
        self.stop_event = threading.Event()
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.camera_worker: CameraWorker | None = None
        self.audio_worker: AudioTranscriber | None = None
        self.assistant = OpenAIAssistant(self.client, self._set_ai_answer, self._set_status)
        self.current_preview: ImageTk.PhotoImage | None = None
        self.last_photo_preview: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._bind_events()
        self._start_devices()
        self._show_api_key_warning_if_needed()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=3)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=12)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(1, weight=1)
        left.rowconfigure(3, weight=1)
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Cámara en vivo", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky="w")
        self.camera_label = ttk.Label(left, text="Iniciando cámara...", anchor="center")
        self.camera_label.grid(row=1, column=0, sticky="nsew", pady=(6, 16))

        ttk.Label(left, text="Última foto enviada", font=("Arial", 14, "bold")).grid(row=2, column=0, sticky="w")
        self.photo_label = ttk.Label(left, text="Presiona S para tomar foto", anchor="center")
        self.photo_label.grid(row=3, column=0, sticky="nsew", pady=(6, 16))

        self.capture_button = ttk.Button(left, text="Tomar foto (S)", command=self.capture_photo)
        self.capture_button.grid(row=4, column=0, sticky="ew")

        right = ttk.Frame(self, padding=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Transcript acumulado", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky="w")
        self.transcript_text = scrolledtext.ScrolledText(right, wrap=tk.WORD, height=14)
        self.transcript_text.grid(row=1, column=0, sticky="nsew", pady=(6, 16))

        ttk.Label(right, text="Respuesta de la IA", font=("Arial", 14, "bold")).grid(row=2, column=0, sticky="w")
        self.ai_text = scrolledtext.ScrolledText(right, wrap=tk.WORD, height=14)
        self.ai_text.grid(row=3, column=0, sticky="nsew", pady=(6, 16))

        self.status_var = tk.StringVar(value="Iniciando...")
        status = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=6)
        status.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _bind_events(self) -> None:
        self.bind("<s>", lambda _event: self.capture_photo())
        self.bind("<S>", lambda _event: self.capture_photo())
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _start_devices(self) -> None:
        self.camera_worker = CameraWorker(self._schedule_frame_update, self._set_status, self.stop_event)
        self.camera_worker.start()

        if self.client is not None:
            self.audio_worker = AudioTranscriber(self.client, self._append_transcript, self._set_status, self.stop_event)
            self.audio_worker.start()
        else:
            self._set_status("Falta OPENAI_API_KEY; cámara activa, pero transcripción/IA deshabilitadas")

    def _show_api_key_warning_if_needed(self) -> None:
        if os.getenv("OPENAI_API_KEY"):
            return
        messagebox.showwarning(
            "OPENAI_API_KEY requerida",
            "No puedo crear una API key por ti. Define OPENAI_API_KEY en tu entorno para habilitar transcripción e IA.",
        )

    def _schedule_frame_update(self, frame: np.ndarray) -> None:
        self.after(0, self._update_camera_preview, frame)

    def _update_camera_preview(self, frame: np.ndarray) -> None:
        image = self._frame_to_tk_image(frame, max_size=(520, 330))
        self.current_preview = image
        self.camera_label.configure(image=image, text="")

    def capture_photo(self) -> None:
        if self.camera_worker is None:
            self._set_status("La cámara todavía no está lista")
            return

        frame = self.camera_worker.snapshot()
        if frame is None:
            self._set_status("No hay frame disponible para capturar")
            return

        CAPTURES_DIR.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = CAPTURES_DIR / f"capture-{timestamp}.jpg"
        cv2.imwrite(str(path), frame)

        self.last_photo_preview = self._frame_to_tk_image(frame, max_size=(520, 240))
        self.photo_label.configure(image=self.last_photo_preview, text="")

        data_url = self._frame_to_data_url(frame)
        self._set_status(f"Foto guardada y enviada: {path}")
        self.assistant.add_photo(CapturedPhoto(path=path, data_url=data_url, created_at=timestamp))

    def _append_transcript(self, text: str) -> None:
        self.after(0, self._append_transcript_on_ui, text)
        self.assistant.add_transcript(text)

    def _append_transcript_on_ui(self, text: str) -> None:
        self.transcript_text.insert(tk.END, f"{time.strftime('%H:%M:%S')}  {text}\n")
        self.transcript_text.see(tk.END)

    def _set_ai_answer(self, text: str) -> None:
        self.after(0, self._set_ai_answer_on_ui, text)

    def _set_ai_answer_on_ui(self, text: str) -> None:
        self.ai_text.delete("1.0", tk.END)
        self.ai_text.insert(tk.END, text)

    def _set_status(self, text: str) -> None:
        self.after(0, self.status_var.set, text)

    def _on_close(self) -> None:
        self.stop_event.set()
        self.destroy()

    @staticmethod
    def _frame_to_tk_image(frame: np.ndarray, max_size: tuple[int, int]) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail(max_size)
        return ImageTk.PhotoImage(image)

    @staticmethod
    def _frame_to_data_url(frame: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("No se pudo codificar la imagen")
        raw = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{raw}"


def main() -> None:
    app = InterviewerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
