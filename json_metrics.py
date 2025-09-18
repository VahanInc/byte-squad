import json
import time
import os
from datetime import datetime
from typing import Optional
from loguru import logger
from pipecat.processors.metrics.frame_processor_metrics import FrameProcessorMetrics


class JsonFrameProcessorMetrics(FrameProcessorMetrics):
    def __init__(self, session_id="default_session", to_number=None, json_path="metrics.json"):
        super().__init__()
        self.session_id = session_id
        self.to_number = to_number
        self.json_path = json_path
        self.turn_id = 0
        self.buffer = []
        self._service_start_times = {}  
        self._service_ttfb_times = {}   
        self._last_processing_time = None
        self._last_latency = None
        
        os.makedirs(os.path.dirname(self.json_path) if os.path.dirname(self.json_path) else ".", exist_ok=True)
        
        # Initialize with session metadata
        self._initialize_session()

    def _initialize_session(self):
        """Initialize the session with metadata."""
        try:
            session_data = {
                "session_id": self.session_id,
                "to_number": self.to_number,
                "session_start": datetime.now().isoformat(),
                "turns": []
            }
            
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, dict) and "turns" in existing_data:
                        session_data["turns"] = existing_data["turns"]
                    else:
                        session_data["turns"] = existing_data if isinstance(existing_data, list) else []
            
            self.buffer = session_data["turns"]
            logger.info(f"Initialized metrics session: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error initializing session: {e}")
            self.buffer = []

    @property
    def processing_time(self) -> Optional[float]:
        """Get the last recorded processing time in seconds."""
        return self._last_processing_time

    @property
    def latency(self) -> Optional[float]:
        """Get the last recorded latency in seconds."""
        return self._last_latency

    async def stop_ttfb_metrics(self):
        """Override to capture TTFB metrics and save to JSON."""
        result = await super().stop_ttfb_metrics()
        
        # If we got a TTFB measurement, record it
        if self.ttfb is not None:
            self._last_latency = self.ttfb  # TTFB is used as latency for TTFB measurements
            self.turn_id += 1
            turn_data = {
                "turn_id": self.turn_id,
                "service_type": self._get_service_type_from_processor(),
                "timestamp": time.time(),
                "ttfb": self.ttfb,
                "processing_time": None,
                "latency": self.ttfb
            }
            self.buffer.append(turn_data)
            self._flush()
            logger.debug(f"TTFB recorded: {self.ttfb}s for {turn_data['service_type']}")
            
        return result

    async def stop_processing_metrics(self):
        """Override to capture processing metrics and save to JSON."""
        result = await super().stop_processing_metrics()
        
        # Calculate processing time if we have the data
        if hasattr(self, '_start_processing_time') and self._start_processing_time > 0:
            processing_time = time.time() - self._start_processing_time
            self._last_processing_time = processing_time
            self._last_latency = processing_time  # Processing time is used as latency for processing measurements
            self.turn_id += 1
            turn_data = {
                "turn_id": self.turn_id,
                "service_type": self._get_service_type_from_processor(),
                "timestamp": time.time(),
                "ttfb": None,
                "processing_time": processing_time,
                "latency": processing_time
            }
            self.buffer.append(turn_data)
            self._flush()
            logger.debug(f"Processing time recorded: {processing_time}s for {turn_data['service_type']}")
            
        return result

    async def start_llm_usage_metrics(self, tokens):
        """Override to capture LLM usage metrics."""
        result = await super().start_llm_usage_metrics(tokens)
        
        # Record LLM usage
        turn_data = {
            "turn_id": self.turn_id + 1,
            "service_type": "llm",
            "timestamp": time.time(),
            "ttfb": None,
            "processing_time": None,
            "latency": None,
            "tokens": {
                "prompt_tokens": tokens.prompt_tokens,
                "completion_tokens": tokens.completion_tokens,
                "total_tokens": tokens.prompt_tokens + tokens.completion_tokens
            }
        }
        self.buffer.append(turn_data)
        self._flush()
        logger.debug(f"LLM usage recorded: {tokens.prompt_tokens} prompt + {tokens.completion_tokens} completion tokens")
        
        return result

    async def start_tts_usage_metrics(self, text):
        """Override to capture TTS usage metrics."""
        result = await super().start_tts_usage_metrics(text)
        
        # Record TTS usage
        turn_data = {
            "turn_id": self.turn_id + 1,
            "service_type": "tts",
            "timestamp": time.time(),
            "ttfb": None,
            "processing_time": None,
            "latency": None,
            "characters": len(text)
        }
        self.buffer.append(turn_data)
        self._flush()
        logger.debug(f"TTS usage recorded: {len(text)} characters")
        
        return result


    def _get_service_type_from_processor(self):
        """Get service type from the current processor context."""
        if hasattr(self, '_core_metrics_data') and self._core_metrics_data:
            processor_name = self._core_metrics_data.processor.lower()
            if 'stt' in processor_name or 'speech' in processor_name or 'transcription' in processor_name:
                return 'stt'
            elif 'llm' in processor_name or 'openai' in processor_name or 'gpt' in processor_name:
                return 'llm'
            elif 'tts' in processor_name or 'text_to_speech' in processor_name or 'elevenlabs' in processor_name or 'waves' in processor_name:
                return 'tts'
        return 'unknown'


    def record_turn_with_timing(self, service_type, ttfb, processing_time, latency):
        """Record a turn with real calculated timing data."""
        self.turn_id += 1
        current_time = time.time()
        
        # Store the values for property access
        self._last_processing_time = processing_time
        self._last_latency = latency
        
        turn_data = {
            "turn_id": self.turn_id,
            "service_type": service_type,
            "timestamp": current_time,
            "ttfb": round(ttfb, 3),
            "processing_time": round(processing_time, 3),
            "latency": round(latency, 3)
        }
        self.buffer.append(turn_data)
        self._flush()
        logger.info(f"Turn {self.turn_id} recorded for {service_type}: TTFB={ttfb:.3f}s, processing={processing_time:.3f}s, latency={latency:.3f}s")


    def _flush(self):
        """Write buffer to disk."""
        try:
            session_data = {
                "session_id": self.session_id,
                "to_number": self.to_number,
                "session_start": getattr(self, '_session_start', datetime.now().isoformat()),
                "last_updated": datetime.now().isoformat(),
                "turns": self.buffer
            }
            
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"Metrics flushed to {self.json_path}")
            
        except Exception as e:
            logger.error(f"Error flushing metrics to {self.json_path}: {e}")
