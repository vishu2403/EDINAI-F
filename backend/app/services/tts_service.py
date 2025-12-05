"""Utilities for generating lecture audio via Edge TTS."""
from __future__ import annotations

import logging
import asyncio
from pathlib import Path
from typing import Iterator, Optional, Dict, Any
import aiohttp
import edge_tts
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class EdgeTTSService:
    """Lightweight wrapper around the edge-tts client with retry logic."""

    _CHUNK_CHAR_LIMIT = 2000  # Reduced chunk size for better stability
    _MAX_RETRIES = 3
    _INITIAL_RETRY_DELAY = 1  # seconds
    _MAX_RETRY_DELAY = 10  # seconds

    def __init__(self, storage_root: str = "./storage/chapter_lectures") -> None:
        self._storage_root = Path(storage_root)
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._session = aiohttp.ClientSession()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        if hasattr(self, '_session') and self._session and not self._session.closed:
            await self._session.close()

    @retry(
        stop=stop_after_attempt(_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=_INITIAL_RETRY_DELAY, max=_MAX_RETRY_DELAY),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception)),
        reraise=True
    )
    async def _synthesize_chunk(self, text: str, voice: str, output_file) -> bool:
        """Synthesize a single chunk of text with retry logic."""
        try:
            # Add a small delay between retries
            if self._synthesize_chunk.retry.statistics.get("attempt_number", 0) > 1:  # type: ignore
                delay = min(
                    self._INITIAL_RETRY_DELAY * (2 ** (self._synthesize_chunk.retry.statistics["attempt_number"] - 1)),  # type: ignore
                    self._MAX_RETRY_DELAY
                )
                await asyncio.sleep(delay)

            communicator = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate="+0%",
                volume="+0%",
                pitch="+0%"
            )
            
            audio_received = False
            async for chunk in communicator.stream():
                if chunk["type"] == "audio":
                    output_file.write(chunk["data"])
                    audio_received = True
            
            if not audio_received:
                raise edge_tts.exceptions.NoAudioReceived("No audio data was received")
                
            return True
            
        except Exception as e:
            logger.warning(
                "Attempt %d/%d failed for TTS synthesis: %s",
                getattr(self._synthesize_chunk.retry, "statistics", {}).get("attempt_number", 0),  # type: ignore
                self._MAX_RETRIES,
                str(e)
            )
            raise

    async def synthesize_text(
        self,
        *,
        lecture_id: str,
        text: str,
        language: str,
        filename: str,
        subfolder: str | None = None,
    ) -> Optional[Path]:
        """Generate an MP3 file for the provided text chunk with retry logic."""
        normalized_text = (text or "").strip()
        if not normalized_text:
            logger.info(
                "Skipping TTS for lecture %s (%s) because text is empty.",
                lecture_id,
                filename,
            )
            return None

        target_path = self._build_audio_path(lecture_id, filename, subfolder=subfolder)
        voice = self._voice_for_language(language)

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(target_path, "wb") as audio_file:
                # Process text in smaller chunks
                for chunk_text in self._chunk_text(normalized_text):
                    try:
                        await self._synthesize_chunk(chunk_text, voice, audio_file)
                    except Exception as chunk_error:
                        logger.error(
                            "Failed to process text chunk for %s (%s): %s",
                            lecture_id,
                            filename,
                            str(chunk_error)
                        )
                        # If any chunk fails, the entire synthesis fails
                        raise

            # Verify the file was created and has content
            if not target_path.exists() or target_path.stat().st_size == 0:
                raise ValueError("Generated audio file is empty")

            logger.info("Successfully generated lecture audio at %s", target_path)
            return target_path

        except Exception as exc:
            logger.error(
                "Failed to synthesize audio for lecture %s (%s) after %d attempts: %s",
                lecture_id,
                filename,
                getattr(self._synthesize_chunk.retry, "statistics", {}).get("attempt_number", 1),  # type: ignore
                str(exc),
                exc_info=True
            )
            # Clean up any partially created file
            try:
                if target_path.exists():
                    target_path.unlink(missing_ok=True)
            except Exception as cleanup_error:
                logger.warning("Failed to clean up failed audio file: %s", str(cleanup_error))
            
            return None

    def _build_audio_path(self, lecture_id: str, filename: str, *, subfolder: str | None) -> Path:
        lecture_dir = self._storage_root / str(lecture_id)
        if subfolder:
            lecture_dir = lecture_dir / subfolder
        lecture_dir.mkdir(parents=True, exist_ok=True)
        return lecture_dir / filename

    @staticmethod
    def _voice_for_language(self, language: str) -> str:
        """Get the appropriate voice for the given language with fallback options."""
        # More comprehensive voice mapping with fallbacks
        voice_mapping: Dict[str, str] = {
            # English voices
            "English": "en-US-JennyNeural",
            "en": "en-US-JennyNeural",
            # Indian languages
            "Hindi": "hi-IN-SwaraNeural",
            "hi": "hi-IN-SwaraNeural",
            "Gujarati": "gu-IN-DhwaniNeural",
            "gu": "gu-IN-DhwaniNeural",
            # Add more languages as needed
            "Spanish": "es-ES-ElviraNeural",
            "es": "es-ES-ElviraNeural",
            "French": "fr-FR-DeniseNeural",
            "fr": "fr-FR-DeniseNeural",
            "German": "de-DE-KatjaNeural",
            "de": "de-DE-KatjaNeural",
        }
        
        # Try exact match first
        voice = voice_mapping.get(language)
        if voice:
            return voice
            
        # Try case-insensitive match
        language_lower = language.lower()
        for lang, voice in voice_mapping.items():
            if lang.lower() == language_lower:
                return voice
                
        # Default fallback
        logger.warning("Using default voice for unsupported language: %s", language)
        return "en-US-JennyNeural"  # Default fallback voice

    def _chunk_text(self, text: str) -> Iterator[str]:
        """Split text into chunks, trying to break at sentence boundaries."""
        if not text:
            return
            
        if len(text) <= self._CHUNK_CHAR_LIMIT:
            yield text
            return

        # Try to split at sentence boundaries first
        sentences = self._split_into_sentences(text)
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If a single sentence is too long, we need to split it
            if len(sentence) > self._CHUNK_CHAR_LIMIT:
                # If we have accumulated some text, yield it first
                if current_chunk:
                    yield ' '.join(current_chunk)
                    current_chunk = []
                    current_length = 0
                # Split the long sentence into smaller chunks
                for i in range(0, len(sentence), self._CHUNK_CHAR_LIMIT):
                    yield sentence[i:i + self._CHUNK_CHAR_LIMIT]
                continue
                
            # If adding this sentence would exceed the chunk limit, yield current chunk
            if current_length + len(sentence) > self._CHUNK_CHAR_LIMIT and current_chunk:
                yield ' '.join(current_chunk)
                current_chunk = []
                current_length = 0
                
            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 for the space
        
        # Don't forget the last chunk
        if current_chunk:
            yield ' '.join(current_chunk)
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter that handles common cases."""
        if not text:
            return []
            
        # Split on sentence endings followed by space and capital letter
        import re
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Handle cases where the split might have been too aggressive
        result = []
        i = 0
        while i < len(sentences):
            # If the sentence is too short, it might be part of an abbreviation
            if i < len(sentences) - 1 and len(sentences[i]) < 10 and sentences[i].endswith('.'):
                # Combine with next sentence
                combined = f"{sentences[i]} {sentences[i+1]}"
                result.append(combined)
                i += 2
            else:
                result.append(sentences[i])
                i += 1
                
        return result