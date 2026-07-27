"""VoiceManager - Streamlit integration layer.

This module provides Streamlit-specific UI integration for voice features.
All Streamlit dependencies are isolated here.
"""

import logging
from typing import Any, Optional

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from voice.stt import SpeechToText
from voice.tts import TextToSpeech

logger = logging.getLogger(__name__)


class VoiceManager:
    """Streamlit convenience layer for voice features.

    This class provides Streamlit-specific methods for voice input/output.
    It handles UI feedback (spinners, errors) while delegating actual
    voice processing to STT and TTS modules.

    Example:
        >>> voice = VoiceManager.from_env()
        >>>
        >>> if voice:
        ...     user_input = voice.get_chat_input()
        ...     if user_input:
        ...         with st.chat_message("ai"):
        ...             voice.render_message("Hello!")
    """

    def __init__(self, stt: SpeechToText | None = None, tts: TextToSpeech | None = None):
        """Initialize VoiceManager.

        Args:
            stt: SpeechToText instance (None to disable STT)
            tts: TextToSpeech instance (None to disable TTS)
        """
        self.stt = stt
        self.tts = tts

        logger.info(
            f"VoiceManager: STT={'enabled' if stt else 'disabled'}, "
            f"TTS={'enabled' if tts else 'disabled'}"
        )

    @classmethod
    def from_env(cls) -> Optional["VoiceManager"]:
        """Create VoiceManager from environment variables.

        Reads VOICE_STT_PROVIDER and VOICE_TTS_PROVIDER to configure
        speech-to-text and text-to-speech providers.

        Returns:
            VoiceManager if either STT or TTS is configured, None otherwise

        Example:
            >>> # In .env:
            >>> # VOICE_STT_PROVIDER=openai
            >>> # VOICE_TTS_PROVIDER=openai
            >>>
            >>> voice = VoiceManager.from_env()
            >>> # Returns configured VoiceManager or None if disabled
        """
        # Create STT and TTS from environment
        stt = SpeechToText.from_env()
        tts = TextToSpeech.from_env()

        # If both disabled, return None (no voice features)
        if not stt and not tts:
            logger.debug("Voice features not configured")
            return None

        return cls(stt=stt, tts=tts)

    def _transcribe_audio(self, audio) -> str | None:
        """Transcribe audio with UI feedback.

        Shows spinner during transcription and error message on failure.

        Args:
            audio: Audio file object from Streamlit chat input

        Returns:
            Transcribed text, or None if transcription failed
        """
        # Defensive check (should not happen if called correctly)
        if not self.stt:
            st.error("⚠️ Speech-to-text not configured.")
            return None

        # Show spinner while transcribing
        with st.spinner("🎤 Transcribing audio..."):
            transcribed = self.stt.transcribe(audio)

        # Check if transcription succeeded
        if not transcribed:
            st.error("⚠️ Transcription failed. Please try again or type your message.")
            return None

        return transcribed

    def get_chat_input(
        self,
        placeholder: str = "Your message",
        file_type: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Get chat input with optional voice transcription and file uploads.

        Handles Streamlit UI including audio input widget and transcription
        feedback (spinner, errors). File uploads are always enabled so users
        can attach images/documents regardless of whether voice is configured.

        Args:
            placeholder: Placeholder text for input
            file_type: Optional list of allowed file extensions for uploads.

        Returns:
            Dict with keys "text" (str), "files" (list[UploadedFile]),
            "audio" (UploadedFile | None). Returns None when the user has
            not submitted anything yet.
        """
        kwargs: dict[str, Any] = {"accept_file": "multiple"}
        if file_type:
            kwargs["file_type"] = file_type
        if self.stt:
            kwargs["accept_audio"] = True

        chat_value = st.chat_input(placeholder, **kwargs)

        if chat_value is None:
            return None

        # Older Streamlit returns a bare str when neither accept_file nor
        # accept_audio is set; newer builds always return a ChatInputValue.
        if isinstance(chat_value, str):
            return {"text": chat_value, "files": [], "audio": None}

        text = chat_value["text"] if isinstance(chat_value, dict) else getattr(chat_value, "text", "")
        files: list[UploadedFile] = (
            chat_value["files"] if isinstance(chat_value, dict) else list(getattr(chat_value, "files", []))
        )
        audio: UploadedFile | None = (
            chat_value["audio"] if isinstance(chat_value, dict) else getattr(chat_value, "audio", None)
        )

        # Audio takes precedence over typed text: transcribe the recording and
        # use it as the message text.
        if audio and self.stt:
            transcribed = self._transcribe_audio(audio)
            if transcribed is not None:
                text = transcribed

        return {"text": text or "", "files": list(files), "audio": audio if not self.stt else None}

    def render_message(self, content: str, container=None, audio_only: bool = False) -> None:
        """Render message with optional TTS audio.

        Handles Streamlit UI including text display and audio player.
        Saves generated audio in session state so it persists across reruns.

        Args:
            content: Message content to display
            container: Streamlit container (defaults to current context)
            audio_only: If True, only render audio (text already displayed)
        """
        if container is None:
            container = st

        # Show text unless audio_only mode (for streaming where text is already shown)
        if not audio_only:
            container.write(content)

        # Add audio if TTS enabled and content is not empty
        if self.tts and content.strip():
            # Show placeholder while generating audio
            placeholder = container.empty()
            with placeholder:
                st.caption("🎙️ Generating audio...")

            # Generate TTS audio
            audio = self.tts.generate(content)

            # Save audio in session state for the last AI message
            # This allows it to persist across st.rerun() calls
            if audio:
                st.session_state.last_audio = {"data": audio, "format": self.tts.get_format()}

            # Replace placeholder with audio player or error message
            if audio:
                placeholder.audio(audio, format=self.tts.get_format())
            else:
                placeholder.caption("🔇 Audio generation unavailable")
