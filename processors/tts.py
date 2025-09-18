#!/usr/bin/env python3
"""
Custom Waves TTS processor for Pipecat using Waves (Smallest AI) API.
"""

import asyncio
import base64
import json
import os
import time
from typing import Optional, Generator
from dataclasses import dataclass
from websocket import WebSocketApp
from loguru import logger

from pipecat.frames.frames import Frame, TextFrame, OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.transcriptions.language import Language


@dataclass
class WavesTTSConfig:
    """Configuration for Waves TTS service."""
    api_key: str
    voice_id: Optional[str] = None
    language: str = "en"
    sample_rate: int = 8000
    speed: float = 1.0
    model: str = "lightning"


class WavesTTSClient:
    """WebSocket client for Waves TTS API."""
    
    def __init__(self, config: WavesTTSConfig):
        self.config = config
        # Use the model from config to build the correct URL
        self.ws_url = os.getenv("WAVES_WS_URL", f"wss://waves-api.smallest.ai/api/v1/{config.model}/get_speech/stream")
        self.ws = None
        self.audio_queue = asyncio.Queue()
        self.error_queue = asyncio.Queue()
        self.is_complete = False
        self.is_connected = False
        
    def _get_headers(self):
        """Get WebSocket headers with API key."""
        return [f"Authorization: Bearer {self.config.api_key}"]
    
    def _create_payload(self, text: str) -> dict:
        """Create the payload for Waves API."""
        # Enhanced payload with more options that might be required
        payload = {
            "text": text,
            "voice_id": self.config.voice_id or "ariba",  # Use ariba as default
            "model": self.config.model,
            "sample_rate": self.config.sample_rate,
            "speed": self.config.speed
        }
            
        logger.info(f"Waves TTS payload: {payload}")
        return payload
    
    def _on_open(self, ws):
        """WebSocket connection opened callback."""
        logger.info("Waves WebSocket connected")
        self.is_connected = True
        
    def _on_message(self, ws, message):
        """WebSocket message received callback."""
        try:
            if not message:
                logger.warning("Received empty message from Waves API")
                return
                
            data = json.loads(message)
            logger.info(f"Waves message received: {data}")
            
            # Ensure data is a dictionary
            if not isinstance(data, dict):
                logger.error(f"Invalid message format from Waves API: {type(data)}")
                self.error_queue.put_nowait(Exception(f"Invalid message format: {type(data)}"))
                return
            
            # Check for errors
            if data.get("status") == "error":
                error_msg = data.get("message", "Unknown error")
                logger.error(f"Waves API error: {error_msg}")
                self.error_queue.put_nowait(Exception(error_msg))
                return
            
            # Check for audio data - try multiple possible paths with safe access
            audio_b64 = None
            
            # Safe dictionary access with None checks
            data_section = data.get("data")
            if data_section and isinstance(data_section, dict):
                audio_b64 = data_section.get("audio")
            
            if not audio_b64:
                audio_b64 = data.get("audio")
            
            if not audio_b64:
                chunk_section = data.get("chunk")
                if chunk_section and isinstance(chunk_section, dict):
                    audio_b64 = chunk_section.get("audio")
            
            if audio_b64:
                try:
                    audio_data = base64.b64decode(audio_b64)
                    self.audio_queue.put_nowait(audio_data)
                    logger.info(f"SUCCESS! Received audio chunk: {len(audio_data)} bytes")
                except Exception as e:
                    logger.error(f"Error decoding audio data: {e}")
                    self.error_queue.put_nowait(e)
            else:
                logger.warning(f"No audio data found in message. Message structure: {data}")
            
            # Check if complete
            if data.get("status") == "complete" or data.get("is_final", False):
                self.is_complete = True
                logger.info("Waves synthesis complete")
                self.audio_queue.put_nowait(None)
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}, message: {message}")
            self.error_queue.put_nowait(e)
        except Exception as e:
            logger.error(f"Error processing Waves message: {e}")
            self.error_queue.put_nowait(e)
    
    def _on_error(self, ws, error):
        """WebSocket error callback."""
        logger.error(f"Waves WebSocket error: {error}")
        # Use thread-safe queue put
        self.error_queue.put_nowait(error)
        
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket closed callback."""
        logger.info(f"Waves WebSocket closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        if not self.is_complete:
            # Use thread-safe queue put
            self.audio_queue.put_nowait(None)
    
    async def connect(self):
        """Connect to Waves WebSocket."""
        if self.ws:
            self.ws.close()
            
        logger.info(f"Connecting to Waves: {self.ws_url}")
        logger.info(f"Using API key: {self.config.api_key[:10]}...")
        
        # Create WebSocket connection
        self.ws = WebSocketApp(
            self.ws_url,
            header=self._get_headers(),
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # Run WebSocket in a thread
        def run_websocket():
            self.ws.run_forever()
            
        import threading
        ws_thread = threading.Thread(target=run_websocket, daemon=True)
        ws_thread.start()
        
        # Wait for connection
        timeout = 10.0
        start_time = time.time()
        while not self.is_connected and time.time() - start_time < timeout:
            await asyncio.sleep(0.1)
            
        if not self.is_connected:
            raise Exception(f"Failed to connect to Waves after {timeout}s")
            
        logger.info("Waves WebSocket connected successfully")
    
    async def synthesize(self, text: str) -> Generator[bytes, None, None]:
        """Synthesize text to speech and yield audio chunks."""
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return
            
        try:
            # Reset state
            self.is_complete = False
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    break
            while not self.error_queue.empty():
                try:
                    self.error_queue.get_nowait()
                except:
                    break
            
            # Validate configuration
            if not self.config or not self.config.api_key:
                logger.error("Invalid Waves TTS configuration")
                raise Exception("Invalid Waves TTS configuration")
            
            # Connect if not connected
            if not self.is_connected:
                await self.connect()
            
            # Send text for synthesis
            payload = self._create_payload(text)
            logger.info(f"Sending text to Waves: '{text}' (length: {len(text)})")
            
            if not self.ws or not self.is_connected:
                logger.error("WebSocket not connected, attempting to reconnect...")
                await self.connect()
            
            if not self.ws:
                raise Exception("Failed to establish WebSocket connection")
            
            self.ws.send(json.dumps(payload))
            logger.info("Text sent to Waves API")
            
            # Wait for audio chunks with reduced timeout for faster processing
            timeout = 15.0
            start_time = time.time()
            chunks_received = 0
            
            while True:
                # Check for errors
                if not self.error_queue.empty():
                    try:
                        error = await self.error_queue.get()
                        logger.error(f"Error from Waves API: {error}")
                        raise error
                    except Exception as e:
                        logger.error(f"Error processing error queue: {e}")
                        raise Exception("Error in Waves API processing")
                
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.error(f"Timeout waiting for audio after {timeout}s")
                    raise Exception(f"Timeout waiting for audio after {timeout}s")
                
                try:
                    # Wait for audio chunk with reduced timeout for faster processing
                    chunk = await asyncio.wait_for(self.audio_queue.get(), timeout=0.5)
                    if chunk is None:
                        logger.info(f"Synthesis complete. Total chunks received: {chunks_received}")
                        break  # Synthesis complete
                    
                    if isinstance(chunk, bytes) and len(chunk) > 0:
                        chunks_received += 1
                        logger.info(f"Yielding audio chunk #{chunks_received}: {len(chunk)} bytes")
                        yield chunk
                    else:
                        logger.warning(f"Invalid chunk received: {type(chunk)}")
                    
                except asyncio.TimeoutError:
                    logger.debug(f"Timeout waiting for chunk. Complete: {self.is_complete}")
                    if self.is_complete:
                        logger.info("Synthesis marked as complete, breaking loop")
                        break
                    continue
                except Exception as e:
                    logger.error(f"Error getting chunk from queue: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in Waves synthesis: {e}")
            import traceback
            logger.error(f"Synthesis traceback: {traceback.format_exc()}")
            raise
        finally:
            # Don't close connection immediately - keep it open for reuse
            logger.info("Synthesis method completed")


class WavesTTSService(FrameProcessor):
    """Custom Waves TTS service for Pipecat."""
    
    def __init__(
        self,
        api_key: str,
        voice_id: Optional[str] = None,
        language: Language = Language.EN,
        sample_rate: int = 8000,
        speed: float = 1.0,
        model: str = "lightning-v2",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        
        # Create configuration
        self.config = WavesTTSConfig(
            api_key=api_key,
            voice_id=voice_id,
            language=language.value if hasattr(language, 'value') else str(language),
            sample_rate=sample_rate,
            speed=speed,
            model=model
        )
        
        # Initialize Waves client
        self.waves_client = WavesTTSClient(self.config)
        
        # Text buffering for sentence aggregation
        self.text_buffer = ""
        self.buffer_timeout = 0.5  # 500ms timeout for sentence completion
        self.last_buffer_time = 0
        self.buffer_task = None
        
        logger.info(f"Waves TTS initialized - Voice: {voice_id or 'default'}, Sample Rate: {sample_rate}Hz")
    
    async def _process_buffered_text(self, direction: FrameDirection):
        """Process the buffered text as a complete sentence."""
        if not self.text_buffer.strip():
            return
            
        text_to_process = self.text_buffer.strip()
        self.text_buffer = ""
        
        logger.info(f"Processing complete sentence: '{text_to_process}' (length: {len(text_to_process)})")
        
        try:
            # Validate that we have a proper TTS client
            if not self.waves_client:
                logger.error("Waves client not initialized")
                return
                
            # Generate audio from complete text
            chunk_count = 0
            synthesis_generator = self.waves_client.synthesize(text_to_process)
            
            if synthesis_generator is None:
                logger.error("Synthesis generator returned None")
                return
                
            async for chunk in synthesis_generator:
                if chunk is not None and len(chunk) > 0:
                    chunk_count += 1
                    # Create audio frame for each chunk and push immediately
                    audio_frame = OutputAudioRawFrame(
                        audio=chunk,
                        sample_rate=self.config.sample_rate,
                        num_channels=1
                    )
                    
                    logger.debug(f"Streaming audio chunk #{chunk_count}: {len(chunk)} bytes")
                    await self.push_frame(audio_frame, direction)
                elif chunk is None:
                    logger.debug("Received None chunk, synthesis likely complete")
                    break
            
            logger.info(f"Streamed {chunk_count} audio chunks for complete sentence: '{text_to_process[:50]}...'")
                
        except Exception as e:
            logger.error(f"Error generating speech for complete sentence: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't re-raise, just log and continue

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, converting text to speech with sentence buffering."""
        # Always call super first to handle system frames
        await super().process_frame(frame, direction)
        
        # Handle both TextFrame and TTSSpeakFrame for pipecat_flows compatibility
        text_to_speak = None
        if isinstance(frame, TextFrame):
            text_to_speak = frame.text
        elif hasattr(frame, 'text') and frame.text:  # Handle TTSSpeakFrame and similar
            text_to_speak = frame.text
        else:
            # Not a text frame, pass through
            await self.push_frame(frame, direction)
            return
        
        if text_to_speak:
            # Add text to buffer
            self.text_buffer += text_to_speak
            self.last_buffer_time = time.time()
            
            logger.info(f"Buffered text: '{text_to_speak}' (buffer: '{self.text_buffer}')")
            
            # Check if we have a complete sentence (ends with punctuation)
            if (text_to_speak.strip().endswith(('.', '!', '?', '।', '।', '।')) or 
                len(self.text_buffer) > 200):  # Force process if buffer gets too long
                
                # Process the complete sentence immediately
                await self._process_buffered_text(direction)
            else:
                # Cancel any existing buffer timeout task
                if self.buffer_task and not self.buffer_task.done():
                    self.buffer_task.cancel()
                
                # Set up a timeout to process buffered text if no more text comes
                async def buffer_timeout_handler():
                    await asyncio.sleep(self.buffer_timeout)
                    if self.text_buffer.strip():  # Only process if buffer has content
                        await self._process_buffered_text(direction)
                
                self.buffer_task = asyncio.create_task(buffer_timeout_handler())
    
    async def cleanup(self):
        """Clean up any remaining buffered text."""
        if self.text_buffer.strip():
            logger.info(f"Processing remaining buffered text: '{self.text_buffer}'")
            await self._process_buffered_text(FrameDirection.DOWNSTREAM)
        
        # Cancel any pending buffer task
        if self.buffer_task and not self.buffer_task.done():
            self.buffer_task.cancel()
    
    @classmethod
    def InputParams(cls):
        """Define input parameters for the service."""
        class Params:
            def __init__(self):
                self.api_key = os.getenv("WAVES_API_KEY", "")
                self.voice_id = os.getenv("WAVES_VOICE_ID", "")
                self.language = Language.EN
                self.sample_rate = 8000
                self.speed = 1.0
                self.model = os.getenv("WAVES_MODEL", "lightning-v2")
        
        return Params()