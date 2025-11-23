"""Speech processing for input and output."""

import io
from typing import Optional, AsyncGenerator
from datetime import datetime
import speech_recognition as sr
from gtts import gTTS

from src.core.config import settings
from src.core.logger import setup_logger

logger = setup_logger(__name__)


class SpeechInput:
    """Handle speech-to-text conversion."""

    def __init__(self):
        """Initialize speech input."""
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self.logger = setup_logger(__name__)

    def listen_once(
        self, timeout: int = 5, phrase_time_limit: int = 10
    ) -> Optional[str]:
        """
        Listen for a single speech input.

        Args:
            timeout: Listening timeout
            phrase_time_limit: Max phrase duration

        Returns:
            Transcribed text or None
        """
        try:
            with sr.Microphone() as source:
                self.logger.info("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )

                self.logger.info("Processing speech...")

                # Try Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                self.logger.info(f"Recognized: {text}")
                return text

        except sr.WaitTimeoutError:
            self.logger.warning("Listening timed out")
            return None
        except sr.UnknownValueError:
            self.logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error in speech input: {e}")
            return None

    async def listen_for_wake_word(
        self, wake_word: str = None
    ) -> bool:
        """
        Listen for wake word.

        Args:
            wake_word: Wake word to listen for (default from settings)

        Returns:
            True if wake word detected
        """
        wake_word = wake_word or settings.wake_word
        self.logger.info(f"Listening for wake word: {wake_word}")

        text = self.listen_once(timeout=30)
        if text and wake_word.lower() in text.lower():
            self.logger.info("Wake word detected!")
            return True

        return False

    def transcribe_audio_file(self, file_path: str) -> Optional[str]:
        """
        Transcribe audio from file.

        Args:
            file_path: Path to audio file

        Returns:
            Transcribed text
        """
        try:
            with sr.AudioFile(file_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                return text
        except Exception as e:
            self.logger.error(f"Failed to transcribe audio file: {e}")
            return None


class SpeechOutput:
    """Handle text-to-speech conversion."""

    def __init__(self):
        """Initialize speech output."""
        self.logger = setup_logger(__name__)

    def speak(self, text: str, language: str = "en") -> bytes:
        """
        Convert text to speech.

        Args:
            text: Text to convert
            language: Language code

        Returns:
            Audio bytes
        """
        try:
            self.logger.info(f"Converting to speech: {text[:50]}...")

            tts = gTTS(text=text, lang=language, slow=False)

            # Convert to bytes
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

            return audio_bytes.getvalue()

        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}")
            return b""

    def save_speech(self, text: str, output_path: str, language: str = "en"):
        """
        Save speech to file.

        Args:
            text: Text to convert
            output_path: Output file path
            language: Language code
        """
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(output_path)
            self.logger.info(f"Speech saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save speech: {e}")


class VoiceInterface:
    """Complete voice interface with input and output."""

    def __init__(self):
        """Initialize voice interface."""
        self.speech_input = SpeechInput()
        self.speech_output = SpeechOutput()
        self.logger = setup_logger(__name__)
        self.conversation_active = False

    async def activate(self) -> bool:
        """
        Activate voice interface with wake word detection.

        Returns:
            True if activated
        """
        self.logger.info("Voice interface waiting for activation...")
        activated = await self.speech_input.listen_for_wake_word()

        if activated:
            self.conversation_active = True
            # Play activation sound/response
            activation_audio = self.speech_output.speak("Yes, how can I help?")
            self.logger.info("Voice interface activated")

        return activated

    def listen(self) -> Optional[str]:
        """
        Listen for user input.

        Returns:
            User speech as text
        """
        if not self.conversation_active:
            self.logger.warning("Voice interface not active")
            return None

        return self.speech_input.listen_once()

    def respond(self, text: str) -> bytes:
        """
        Generate speech response.

        Args:
            text: Response text

        Returns:
            Audio bytes
        """
        return self.speech_output.speak(text)

    def deactivate(self):
        """Deactivate voice interface."""
        self.conversation_active = False
        goodbye_audio = self.speech_output.speak("Goodbye")
        self.logger.info("Voice interface deactivated")


# Global instances
_speech_input: Optional[SpeechInput] = None
_speech_output: Optional[SpeechOutput] = None
_voice_interface: Optional[VoiceInterface] = None


def get_speech_input() -> SpeechInput:
    """Get global speech input instance."""
    global _speech_input
    if _speech_input is None:
        _speech_input = SpeechInput()
    return _speech_input


def get_speech_output() -> SpeechOutput:
    """Get global speech output instance."""
    global _speech_output
    if _speech_output is None:
        _speech_output = SpeechOutput()
    return _speech_output


def get_voice_interface() -> VoiceInterface:
    """Get global voice interface instance."""
    global _voice_interface
    if _voice_interface is None:
        _voice_interface = VoiceInterface()
    return _voice_interface
