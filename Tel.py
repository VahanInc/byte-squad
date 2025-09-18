from os import getenv
from fastapi import WebSocket, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pipecat_flows import FlowArgs, FlowManager, NodeConfig, ContextStrategyConfig, ContextStrategy, FlowResult
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.transcriptions.language import Language
from processors.tts import WavesTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from typing import Literal
from plivo import RestClient
from asyncio import Event
from json import loads, dump
from loguru import logger
from sys import stderr
from dotenv import load_dotenv
import uvicorn
import os
import json
from datetime import datetime

load_dotenv(dotenv_path=".env", stream=None, verbose=True, override=True, interpolate=True, encoding="utf-8")

# Environment variables
DEEPGRAM_API_KEY = getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = getenv("OPENAI_API_KEY")

# For ngrok, we'll get the public URL from the request
def get_public_url(request):
    """Get the public URL from the request, useful for ngrok."""
    # Check if we have a custom public URL in environment
    custom_url = getenv("PUBLIC_URL")
    if custom_url:
        return custom_url
    
    if request.headers.get("x-forwarded-proto"):
        return f"{request.headers['x-forwarded-proto']}://{request.headers['host']}"
    return f"http://{request.headers['host']}"

# Global variables for conversation tracking
conversation_logs = []
callData = {"event": Event(), "userInterest": None, "haveLicense": None, "conversation": []}

# Create c4 folder if it doesn't exist
import os
if not os.path.exists("c4"):
    os.makedirs("c4")

# Single system prompt definition - This is the ONLY place to modify the system prompt
# All other references use this variable to ensure consistency
BLUE_COLLAR_SYSTEM_PROMPT = """You are a helpful AI assistant for a BLUE COLLAR job recruitment system. 

You specialize in blue collar positions like:
- Factory workers, assembly line operators
- Construction workers, laborers
- Warehouse workers, forklift operators
- Machine operators, technicians
- Maintenance workers, electricians
- Drivers, delivery personnel
- Agricultural workers, farm hands

You should:
1. Answer questions about blue collar job opportunities, requirements, and working conditions
2. Be professional, friendly, and use simple, clear language
3. Focus on physical requirements, safety protocols, training, and work environment
4. MUST complete two quick questions before answering FAQs:
   - Q1) "Are you ready to work?" (expect yes/no)
   - Q2) "Do you have an Aadhaar card?" (expect yes/no)
   If the user asks a question before Q1/Q2 are done, politely say you'll answer after the two quick questions.
5. Keep responses under 2-3 sentences for phone conversation. Speak clearly and slightly slower for phone audio, with short, simple sentences and good enunciation.
6. If you don't know something, politely say so and offer to connect them with a human agent

For SALARY questions specifically:
- Provide general salary ranges for the type of work (e.g., "Factory workers typically earn ₹15,000-25,000 per month")
- Mention overtime opportunities and rates (e.g., "Overtime is available at 1.5x regular rate")
- Include benefits like PF, ESI, or other statutory benefits
- Be honest about ranges rather than specific amounts
- If asked for exact salary, suggest discussing with HR during the interview

Common blue collar topics you can help with:
- Physical fitness requirements
- Safety equipment and protocols
- Training and certification needs
- Work hours and shifts
- Pay rates and overtime
- Work location and transportation
- Health and safety benefits
- Equipment and tools provided
- Work environment conditions

Context: This is a phone call for blue collar job recruitment. Your responses will be converted to audio, so avoid special characters. Always use the available functions to progress the conversation naturally."""

def log_conversation(speaker: str, message_type: str, content: str, call_uuid: str = None):
    """Log conversation interactions."""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "role": speaker,
        "type": message_type,
        "text": content
    }
    conversation_logs.append(log_entry)
    callData["conversation"].append(log_entry)
    logger.info(f"Conversation log: {speaker} - {message_type}: {content}")
    
    # Save to c4 folder if call_uuid is provided
    if call_uuid:
        save_conversation_to_c4(call_uuid, log_entry)

def save_conversation_to_c4(call_uuid: str, new_entry: dict):
    """Save conversation to c4 folder in the specified format."""
    try:
        file_path = f"c4/{call_uuid}.json"
        
        # Load existing conversation or create new one
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
        else:
            conversation_data = {
                "call_uuid": call_uuid,
                "flow": "started",
                "q1": None,
                "q2": None,
                "faq_questions": [],
                "_updated_at": datetime.now().isoformat(),
                "transcript": []
            }
        
        # Add new entry to transcript
        conversation_data["transcript"].append(new_entry)
        conversation_data["_updated_at"] = datetime.now().isoformat()
        
        # Update specific fields based on message type
        if new_entry["type"] == "q1_answer":
            conversation_data["q1"] = new_entry["text"]
            conversation_data["last_q1_raw"] = new_entry["text"]
        elif new_entry["type"] == "q2_answer":
            conversation_data["q2"] = new_entry["text"]
            conversation_data["last_q2_raw"] = new_entry["text"]
        elif new_entry["type"] == "faq_intent":
            conversation_data["faq_questions"].append(new_entry["text"])
            conversation_data["last_question"] = new_entry["text"]
        elif new_entry["type"] == "faq_answer":
            conversation_data["last_answer"] = new_entry["text"]
        
        # Save updated conversation
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Conversation saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving conversation to c4: {e}")

# Function handlers
async def check_ready_to_work(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[FlowResult, NodeConfig]:
    """Process readiness to work check."""
    ready = args["ready"]
    logger.info(f"check_ready_to_work handler executing with ready: {ready} and args: {args}")
    
    # Get call_uuid from flow_manager state
    call_uuid = flow_manager.state.get("call_uuid", None)
    
    log_conversation("user", "q1_answer", f"Ready to work: {ready}", call_uuid)
    
    flow_manager.state["ready_to_work"] = ready
    
    if ready:
        next_node = create_aadhaar_check_node()
        response_text = "Great! Do you have an Aadhaar card? Please say yes or no."
        log_conversation("assistant", "system", "Q2 prompt (Aadhaar)", call_uuid)
        logger.info(f"Q1 completed successfully. Moving to Aadhaar check. Response: {response_text}")
        result = FlowResult(ready=ready, response=response_text)
    else:
        next_node = create_end_node()
        response_text = "Thank you for your time. Goodbye!"
        log_conversation("assistant", "system", "Farewell after Q1 = no", call_uuid)
        logger.info(f"Q1 completed. User not ready. Ending conversation.")
        result = FlowResult(ready=ready, response=response_text)
    
    return result, next_node


async def check_aadhaar_card(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[FlowResult, NodeConfig]:
    """Process Aadhaar card check."""
    has_aadhaar = args["has_aadhaar"]
    logger.debug(f"check_aadhaar_card handler executing with has_aadhaar: {has_aadhaar}")
    
    # Get call_uuid from flow_manager state
    call_uuid = flow_manager.state.get("call_uuid", None)
    
    log_conversation("user", "q2_answer", f"Has Aadhaar card: {has_aadhaar}", call_uuid)
    
    # Both yes and no lead to job questions
    next_node = create_job_question_node()
    response_text = "Thank you for that information. Would you like to ask any questions about blue collar jobs? Please say yes or no."
    log_conversation("assistant", "system", "FAQ willingness prompt", call_uuid)
    result = FlowResult(has_aadhaar=has_aadhaar, response=response_text)
    
    return result, next_node


async def handle_job_question(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[FlowResult, NodeConfig]:
    """Handle job-related questions using LLM."""
    question = args.get("question", "")
    willing_to_ask = args.get("willing_to_ask", False)
    
    logger.info(f"handle_job_question handler executing with willing_to_ask: {willing_to_ask}, question: '{question}', args: {args}")
    logger.info(f"Question length: {len(question)}, Question stripped: '{question.strip()}', Is empty: {not question or question.strip() == ''}")
    
    # Get call_uuid from flow_manager state
    call_uuid = flow_manager.state.get("call_uuid", None)
    
    if willing_to_ask:
        # If user wants to ask questions but hasn't asked one yet, prompt them
        # Check if the "question" is actually just a confirmation (like "Yes", "Yeah", etc.)
        if (not question or question.strip() == "" or 
            question.lower().strip() in ["yes", "yeah", "sure", "okay", "ok", "yep", "yes."]):
            response_text = "Great! Please go ahead and ask your question about blue collar jobs. I can help with information about salaries, work hours, safety protocols, training requirements, and working conditions."
            log_conversation("assistant", "system", "Prompting for specific question", call_uuid)
            logger.info(f"PATH 1: User wants to ask questions but provided confirmation instead of question. Prompting: {response_text}")
            logger.info(f"PATH 1: Setting next_node to wait_for_question_node and returning early")
            result = FlowResult(question="", answer="", response=response_text)
            # Move to a new node that waits for the actual question
            next_node = create_wait_for_question_node()
            return result, next_node
        else:
            # User has asked a specific question, process it
            logger.info(f"PATH 2: User has asked a specific question: '{question}'")
            log_conversation("user", "faq_intent", question, call_uuid)
        
        # Try to use LLM service from flow manager state, fallback to predefined responses
        try:
            llm_service = flow_manager.state.get("llm_service")
            if llm_service and hasattr(llm_service, 'run'):
                # Use the global system prompt
                system_prompt = BLUE_COLLAR_SYSTEM_PROMPT
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
                
                # Get LLM response
                response = await llm_service.run(messages)
                answer = response.content if hasattr(response, 'content') else str(response)
            else:
                # Fallback to predefined responses
                if "salary" in question.lower() or "pay" in question.lower() or "money" in question.lower():
                    answer = "For blue collar jobs, salaries typically range from ₹15,000 to ₹35,000 per month depending on experience and skills. Factory workers earn ₹15,000-25,000, construction workers ₹18,000-30,000, and skilled technicians ₹25,000-35,000. Overtime is available at 1.5x rate. Benefits include PF, ESI, and other statutory benefits. For exact salary details, please discuss with HR during your interview."
                elif "work" in question.lower() and "hours" in question.lower():
                    answer = "Blue collar jobs typically have 8-10 hour shifts, 5-6 days a week. Some positions offer flexible shifts including morning, evening, and night shifts. Overtime opportunities are available for additional income."
                elif "safety" in question.lower() or "equipment" in question.lower():
                    answer = "Safety is our top priority. All necessary safety equipment like helmets, gloves, safety shoes, and protective gear are provided. We conduct regular safety training and follow strict safety protocols. All workers must complete safety certification before starting work."
                elif "training" in question.lower() or "experience" in question.lower():
                    answer = "We provide on-the-job training for all positions. While some experience is preferred, we welcome freshers and provide comprehensive training. Training duration varies from 1-4 weeks depending on the role complexity."
                else:
                    answer = "Thank you for your question about blue collar jobs. I can help with information about salaries, work hours, safety protocols, training requirements, and working conditions. Please ask your specific question and I'll provide detailed information."
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            # Fallback to predefined responses
            if "salary" in question.lower() or "pay" in question.lower() or "money" in question.lower():
                answer = "For blue collar jobs, salaries typically range from ₹15,000 to ₹35,000 per month depending on experience and skills. Factory workers earn ₹15,000-25,000, construction workers ₹18,000-30,000, and skilled technicians ₹25,000-35,000. Overtime is available at 1.5x rate. Benefits include PF, ESI, and other statutory benefits. For exact salary details, please discuss with HR during your interview."
            elif "work" in question.lower() and "hours" in question.lower():
                answer = "Blue collar jobs typically have 8-10 hour shifts, 5-6 days a week. Some positions offer flexible shifts including morning, evening, and night shifts. Overtime opportunities are available for additional income."
            elif "safety" in question.lower() or "equipment" in question.lower():
                answer = "Safety is our top priority. All necessary safety equipment like helmets, gloves, safety shoes, and protective gear are provided. We conduct regular safety training and follow strict safety protocols. All workers must complete safety certification before starting work."
            elif "training" in question.lower() or "experience" in question.lower():
                answer = "We provide on-the-job training for all positions. While some experience is preferred, we welcome freshers and provide comprehensive training. Training duration varies from 1-4 weeks depending on the role complexity."
            else:
                answer = "Thank you for your question about blue collar jobs. I can help with information about salaries, work hours, safety protocols, training requirements, and working conditions. Please ask your specific question and I'll provide detailed information."
        
        log_conversation("assistant", "faq_answer", answer, call_uuid)
        
        result = FlowResult(question=question, answer=answer, response=answer)
        next_node = create_end_node()
    else:
        log_conversation("user", "faq_intent", "Not willing to ask questions", call_uuid)
        response_text = "Thank you for your time. Goodbye!"
        log_conversation("assistant", "system", "Farewell after FAQ = no", call_uuid)
        result = FlowResult(question="", answer="", response=response_text)
        next_node = create_end_node()
    
    return result, next_node


async def end_conversation(args: FlowArgs, flow_manager: FlowManager = None) -> tuple[FlowResult, NodeConfig]:
    """Handle conversation end."""
    logger.debug("end_conversation handler executing")
    
    # Get call_uuid from flow_manager state if available
    call_uuid = None
    if flow_manager:
        call_uuid = flow_manager.state.get("call_uuid", None)
    
    # Save conversation logs to file
    try:
        with open("conversation_logs.json", "w") as f:
            json.dump(conversation_logs, f, indent=2)
        logger.info("Conversation logs saved to conversation_logs.json")
    except Exception as e:
        logger.error(f"Error saving conversation logs: {e}")
    
    response_text = "Thank you for using our job counseling service. Have a great day!"
    log_conversation("assistant", "system", "Conversation ended", call_uuid)
    result = FlowResult(status="completed", response=response_text)
    next_node = create_final_end_node()
    return result, next_node

# Node configurations
def create_initial_node() -> NodeConfig:
    """Create the initial node asking if ready to work."""
    return NodeConfig({
        "name": "initial",
        "role_messages": [
            {
                "role": "system",
                "content": BLUE_COLLAR_SYSTEM_PROMPT,
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "You MUST call the check_ready_to_work function when the user responds to the 'Are you ready to work?' question. If they say 'yes', 'yeah', 'sure', or any positive response, call check_ready_to_work with ready=true. If they say 'no', 'nope', or any negative response, call check_ready_to_work with ready=false. Do not proceed without calling this function.",
            }
        ],
        "functions": [
            {
                "type": "function",
                "function": {
                    "name": "check_ready_to_work",
                    "handler": check_ready_to_work,
                    "description": "Check if person is ready to work",
                    "parameters": {
                        "type": "object",
                        "properties": {"ready": {"type": "boolean"}},
                        "required": ["ready"],
                    },
                },
            }
        ],
        "pre_actions": [
            {
                "type": "tts_say",
                "text": "Hello! Welcome to our Vahan Blue Collar job recruitment service. Are you ready to work? Please say yes or no."
            }
        ],
        "respond_immediately": False
    })


def create_aadhaar_check_node() -> NodeConfig:
    """Create node for checking Aadhaar card."""
    return NodeConfig({
        "name": "aadhaar_check",
        "role_messages": [
            {
                "role": "system",
                "content": BLUE_COLLAR_SYSTEM_PROMPT,
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "You MUST call the check_aadhaar_card function when the user responds to the Aadhaar card question. If they say 'yes', 'yeah', 'sure', or any positive response, call check_aadhaar_card with has_aadhaar=true. If they say 'no', 'nope', or any negative response, call check_aadhaar_card with has_aadhaar=false. Do not proceed without calling this function.",
            }
        ],
        "functions": [
            {
                "type": "function",
                "function": {
                    "name": "check_aadhaar_card",
                    "handler": check_aadhaar_card,
                    "description": "Check if person has Aadhaar card - MUST be called with has_aadhaar=true for yes/positive responses, has_aadhaar=false for no/negative responses",
                    "parameters": {
                        "type": "object",
                        "properties": {"has_aadhaar": {"type": "boolean"}},
                        "required": ["has_aadhaar"],
                    },
                },
            }
        ],
        "pre_actions": [
            {
                "type": "tts_say",
                "text": "Great! Do you have an Aadhaar card? Please say yes or no."
            }
        ],
        "respond_immediately": False
    })


def create_job_question_node() -> NodeConfig:
    """Create node for handling job questions."""
    return NodeConfig({
        "name": "job_question",
        "role_messages": [
            {
                "role": "system",
                "content": BLUE_COLLAR_SYSTEM_PROMPT,
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "CRITICAL INSTRUCTIONS: You MUST call the handle_job_question function when the user responds. "
                    "FOR 'YES' RESPONSES: Call handle_job_question(willing_to_ask=true, question=\"\") with EMPTY question string. "
                    "FOR 'NO' RESPONSES: Call handle_job_question(willing_to_ask=false, question=\"\"). "
                    "NEVER pass the user's 'yes' or 'no' response as the question parameter. "
                    "The question parameter should always be empty string for this node. "
                    "Do not proceed without calling this function exactly as specified."
                ),
            }
        ],
        "functions": [
            {
                "type": "function",
                "function": {
                    "name": "handle_job_question",
                    "handler": handle_job_question,
                    "description": "Handle job-related questions - MUST be called with willing_to_ask=true for yes/positive responses (use question='' for empty), willing_to_ask=false for no/negative responses",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "willing_to_ask": {"type": "boolean"},
                            "question": {"type": "string"},
                        },
                        "required": ["willing_to_ask"],
                    },
                },
            }
        ],
        "pre_actions": [
            {
                "type": "tts_say",
                "text": "Thank you for that information. Would you like to ask any questions about blue collar jobs? Please say yes or no."
            }
        ],
        "respond_immediately": False
    })


def create_wait_for_question_node() -> NodeConfig:
    """Create node for waiting for the user's actual question."""
    return NodeConfig({
        "name": "wait_for_question",
        "role_messages": [
            {
                "role": "system",
                "content": BLUE_COLLAR_SYSTEM_PROMPT,
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "The user has already confirmed they want to ask a question. "
                    "Now wait for them to ask their actual question about blue collar jobs. "
                    "When they ask a question, call handle_job_question with willing_to_ask=true and question=their_actual_question. "
                    "Do not repeat the 'would you like to ask questions' prompt. "
                    "IMPORTANT: The user should ask their question within 2 minutes to avoid timeout."
                ),
            }
        ],
        "functions": [
            {
                "type": "function",
                "function": {
                    "name": "handle_job_question",
                    "handler": handle_job_question,
                    "description": "Handle the user's actual question about blue collar jobs",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "willing_to_ask": {"type": "boolean"},
                            "question": {"type": "string"},
                        },
                        "required": ["willing_to_ask", "question"],
                    },
                },
            }
        ],
        "pre_actions": [
            {
                "type": "tts_say",
                "text": "I'm listening. Please go ahead and ask your question about blue collar jobs."
            }
        ],
        "respond_immediately": False
    })


def create_end_node() -> NodeConfig:
    """Create the end node."""
    return NodeConfig({
        "name": "end",
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "Thank the person for their time and end the conversation gracefully."
                ),
            }
        ],
        "post_actions": [{"type": "end_conversation"}],
    })


def create_final_end_node() -> NodeConfig:
    """Create the final end node."""
    return NodeConfig({
        "name": "final_end",
        "task_messages": [
            {
                "role": "system",
                "content": "The conversation has ended.",
            }
        ],
        "post_actions": [{"type": "end_conversation"}],
    })

async def run_bot(websocket: WebSocket, stream_id: str, call_id: str):
    """Run the bot with the given websocket connection."""
    try:
        logger.info(f"Starting bot for stream_id: {stream_id}, call_id: {call_id}")
        
        # Step 1: Initialize services with proper configuration
        stt = DeepgramSTTService(
            api_key=DEEPGRAM_API_KEY,
            live_options=LiveOptions(
                model="nova-2-phonecall",
                language=Language.EN_IN,
                smart_format=True
            )
        )
        
        # Try to initialize TTS with fallback voices
        tts = None
        voice_ids_to_try = [
            "21m00Tcm4TlvDq8ikWAM",  # Rachel voice
            "EXAVITQu4vr4xnSDxMaL",  # Bella voice
            "pNInz6obpgDQGcFmaJgB",  # Adam voice
        ]
        
        try:
            logger.info("Initializing Waves TTS service")
            tts = WavesTTSService(
                api_key=getenv("WAVES_API_KEY"),
                voice_id=getenv("WAVES_VOICE_ID", ""),  # Empty string will use default
                language=Language.EN,
                sample_rate=int(getenv("WAVES_SAMPLE_RATE", "8000")),
                speed=float(getenv("WAVES_SPEED", "1.0")),
                model=getenv("WAVES_MODEL", "lightning")
            )
            logger.info("Successfully initialized Waves TTS service")
            logger.info(f"TTS Configuration: model={getenv('WAVES_MODEL', 'lightning')}, speed={getenv('WAVES_SPEED', '1.0')}, sample_rate={getenv('WAVES_SAMPLE_RATE', '8000')}")
        except Exception as e:
            logger.error(f"Failed to initialize Waves TTS service: {e}")
            raise Exception("Could not initialize Waves TTS service")
        
        if tts is None:
            logger.error("Failed to initialize TTS with any voice ID")
            raise Exception("Could not initialize ElevenLabs TTS service")
        
        llm = OpenAILLMService(
            api_key=OPENAI_API_KEY, 
            model="gpt-4o-mini",
            params=OpenAILLMService.InputParams(temperature=0.75)
        )
        
        context = OpenAILLMContext()
        context_aggregator = llm.create_context_aggregator(context)

        # Create transport with proper serializer configuration
        serializer = PlivoFrameSerializer(
            stream_id=stream_id, 
            call_id=call_id, 
            auth_id=getenv("PLIVO_AUTH_ID"), 
            auth_token=getenv("PLIVO_AUTH_TOKEN")
        )
        
        transport = FastAPIWebsocketTransport(
            websocket, 
            params=FastAPIWebsocketParams(
                audio_out_enabled=True,
                audio_in_enabled=True,
                serializer=serializer,  # Use serializer directly
                vad_analyzer=SileroVADAnalyzer(),
                session_timeout=120
            )
        )

        # Create pipeline with proper task parameters
        pipeline = Pipeline([
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        task = PipelineTask(
            pipeline, 
            params=PipelineParams(
                allow_interruptions=False,  # Disable interruptions to prevent word-by-word speech
                audio_in_sample_rate=8000,
                audio_out_sample_rate=8000,
                enable_metrics=False,
                enable_usage_metrics=False,
                report_only_initial_ttfb=True,
                send_initial_empty_metrics=False
            )
        )

        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.APPEND,
                summary_prompt=BLUE_COLLAR_SYSTEM_PROMPT
            ),
            transport=transport
        )
        
        # Store LLM service in flow manager state for access in handlers
        flow_manager.state["llm_service"] = llm
        flow_manager.state["context_aggregator"] = context_aggregator

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected: {getattr(client, 'id', 'unknown')}")
            try:
                # Store call_uuid in flow manager state for access in handlers
                flow_manager.state["call_uuid"] = call_id
                await flow_manager.initialize(create_initial_node())
                logger.info("Flow manager initialized successfully")
                
                # Log the initial welcome message
                log_conversation("assistant", "system", "Welcome + Q1 prompt", call_id)
            except Exception as e:
                logger.error(f"Error initializing flow manager: {e}")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected: {getattr(client, 'id', 'unknown')}")
            try:
                await task.cancel()
                callData["event"].set()
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")

        @transport.event_handler("on_session_timeout")
        async def on_session_timeout(transport, client):
            logger.info(f"Session timeout: {getattr(client, 'id', 'unknown')}")

        # Run the pipeline
        logger.info("Starting pipeline runner")
        runner = PipelineRunner(handle_sigint=True, handle_sigterm=True, force_gc=True)
        try:
            await runner.run(task)
            logger.info("Pipeline runner completed")
        except Exception as e:
            logger.error(f"Pipeline runner error: {e}")
            import traceback
            logger.error(f"Pipeline traceback: {traceback.format_exc()}")
        
    except Exception as e:
        logger.error(f"Error in run_bot: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        try:
            callData["event"].set()
        except:
            pass

def main():
    app = FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/callStream")
    async def sendCallStream_xml(request: Request):
        """Endpoint for sending the xml configuration to plivo for setting call streaming"""
        public_url = get_public_url(request)
        return HTMLResponse(content=f"""<?xml version="1.0" encoding="UTF-8"?><Response><Stream keepCallAlive="true" bidirectional="true" contentType="audio/x-mulaw;rate=8000"> wss://{public_url.replace("https://","").replace("http://","")}/ws </Stream></Response>""", media_type="application/xml")

  

    @app.websocket(path="/ws", name="call_streaming_endpoint")
    async def callStreaming(websocket: WebSocket):
        """Handle websocket connection for audio streaming"""
        try:
            await websocket.accept()
            logger.info("WebSocket connection accepted")
            
            # Wait for the initial message from Plivo
            try:
                init_message = await websocket.receive_text()
                logger.info(f"Received initial message: {init_message}")
                init_config = loads(init_message).get("start", {})
                streamId = init_config.get("streamId")
                callId = init_config.get("callId")
                
                logger.info(f"Stream ID: {streamId}, Call ID: {callId}")
                
                if not streamId or not callId:
                    logger.error("Missing streamId or callId in initial message")
                    return
                
                await run_bot(websocket=websocket, stream_id=streamId, call_id=callId)
                
            except Exception as e:
                logger.error(f"Error processing initial message: {e}")
                await websocket.close()
                
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            try:
                await websocket.close()
            except:
                pass



    @app.get("/hangup")
    async def hangup_endpoint():
        """Handle call hangup events from Plivo."""
        return HTMLResponse(content="<Response><Hangup/></Response>", media_type="application/xml")

    @app.get("/")
    async def root():
        """Root endpoint to test if server is running."""
        return {
            "message": "Blue Collar Job Recruitment Bot API", 
            "status": "running",
            "version": "2.0",
            "features": {
                "blue_collar_specialization": True,
                "two_question_flow": True,
                "salary_guidance": True,
                "safety_protocols": True
            },
            "websocket_url": "wss://9bd8e08c62b4.ngrok-free.app/ws",
            "env_check": {
                "deepgram_key": "✓" if DEEPGRAM_API_KEY else "✗",
                "openai_key": "✓" if OPENAI_API_KEY else "✗",
                "elevenlabs_key": "✓" if getenv("ELEVENLABS_API_KEY") else "✗",
                "plivo_auth_id": "✓" if getenv("PLIVO_AUTH_ID") else "✗",
                "plivo_auth_token": "✓" if getenv("PLIVO_AUTH_TOKEN") else "✗",
                "from_number": "✓" if getenv("PLIVO_FROM_NUMBER") else "✗",
                "to_number": "✓" if getenv("PLIVO_TO_NUMBER") else "✗"
            },
            "endpoints": {
                "make_call": "/make-call",
                "plivo_answer": "/plivo-answer",
                "websocket": "/ws",
                "test_prompt": "/test-prompt"
            }
        }

    @app.get("/test-prompt")
    async def test_prompt():
        """Test the system prompt with a sample question."""
        try:
            # Test the LLM with the system prompt
            llm = OpenAILLMService(
                api_key=OPENAI_API_KEY, 
                model="gpt-4o-mini",
                params=OpenAILLMService.InputParams(temperature=0.75)
            )
            
            system_prompt = BLUE_COLLAR_SYSTEM_PROMPT
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What is the salary for factory workers?"}
            ]
            
            response = await llm.run(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "status": "success",
                "test_question": "What is the salary for factory workers?",
                "llm_response": answer,
                "system_prompt_length": len(system_prompt),
                "prompt_working": "blue collar" in answer.lower() or "factory" in answer.lower() or "salary" in answer.lower()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to test system prompt"
            }

    @app.get("/fallback")
    async def fallback_endpoint():
        """Handle call fallback events from Plivo."""
        return HTMLResponse(content="<Response><Hangup/></Response>", media_type="application/xml")

    @app.post("/plivo-answer")
    async def plivo_answer(request: Request):
        """Handle Plivo answer webhook."""
        try:
            logger.info("=== PLIVO ANSWER WEBHOOK CALLED ===")
            # Try to get form data, but handle the case where python-multipart is not installed
            try:
                form_data = await request.form()
                call_uuid = form_data.get("CallUUID", form_data.get("call_uuid", form_data.get("Uuid", form_data.get("UUID", "unknown"))))
            except Exception as e:
                logger.warning(f"Could not parse form data: {e}")
                # Try to get call UUID from query parameters as fallback
                call_uuid = request.query_params.get("CallUUID", request.query_params.get("call_uuid", "unknown"))
            
            logger.info(f"Plivo answer (WS) for call: {call_uuid}")
            logger.info(f"Request headers: {dict(request.headers)}")
            
            # Get public URL for ngrok
            public_url = get_public_url(request)
            ws_base = public_url.replace("https://", "wss://").replace("http://", "ws://")
            ws_url = f"{ws_base}/ws"
            logger.info(f"WebSocket URL: {ws_url}")
            
            xml_response = f"""<Response>
  
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">{ws_url}</Stream>
</Response>"""
            return HTMLResponse(content=xml_response, media_type="text/xml")
            
        except Exception as e:
            logger.error(f"Error in plivo-answer: {e}")
            # Return a simple response to prevent call from ending
            return HTMLResponse(content="<Response><Speak>Welcome to our service.</Speak></Response>", media_type="text/xml")

    @app.api_route("/make-call", methods=["GET", "POST"])
    async def make_call_get(request: Request):
        """Make an outbound call using GET request with numbers from .env file."""
        try:
            logger.info("Make-call endpoint called")
            
            # Get Plivo from and to numbers from environment
            from_number = getenv("PLIVO_FROM_NUMBER")
            to_number = getenv("PLIVO_TO_NUMBER")
            
            logger.info(f"From number: {from_number}")
            logger.info(f"To number: {to_number}")
            
            if not from_number:
                logger.error("PLIVO_FROM_NUMBER not found in environment")
                return JSONResponse(
                    content={"success": False, "error": "PLIVO_FROM_NUMBER not configured in environment"},
                    status_code=500,
                    media_type="application/json"
                )
            
            if not to_number:
                logger.error("PLIVO_TO_NUMBER not found in environment")
                return JSONResponse(
                    content={"success": False, "error": "PLIVO_TO_NUMBER not configured in environment"},
                    status_code=500,
                    media_type="application/json"
                )
            
            # Get public URL for ngrok
            public_url = get_public_url(request)
            logger.info(f"Public URL: {public_url}")
            
            # For testing, if we're on localhost, use a placeholder URL
            if "localhost" in public_url or "127.0.0.1" in public_url:
                logger.warning("Running on localhost - using placeholder URL for testing")
                # You can set this to your ngrok URL for testing
                public_url = getenv("PUBLIC_URL")
                if not public_url or "your-ngrok-url" in public_url:
                    logger.error("Please set PUBLIC_URL in .env file to your actual ngrok URL")
                    return JSONResponse(content={
                        "success": False,
                        "error": "PUBLIC_URL not set. Please add your ngrok URL to .env file",
                        "message": "Set PUBLIC_URL=https://your-actual-ngrok-url.ngrok.io in .env"
                    }, status_code=500, media_type="application/json")
                logger.info(f"Using URL from .env: {public_url}")
            
            # Create Plivo client
            plivo_client = RestClient(
                auth_id=getenv("PLIVO_AUTH_ID"), 
                auth_token=getenv("PLIVO_AUTH_TOKEN"), 
                timeout=30
            ).calls
            
            # Create the call
            call_response = plivo_client.create(
                from_=from_number,
                to_=to_number,
                answer_url=f"{public_url}/plivo-answer",
                answer_method="POST",
                hangup_url=f"{public_url}/hangup",
                fallback_url=f"{public_url}/fallback",
                ring_timeout=30,
                time_limit=1800
            )
            
            call_uuid = call_response["request_uuid"]
            logger.info(f"Outbound call initiated - From: {from_number}, To: {to_number}, Call UUID: {call_uuid}")
            
            return JSONResponse(content={
                "success": True,
                "call_uuid": call_uuid,
                "message": "Call initiated successfully",
                "from": from_number,
                "to": to_number
            }, media_type="application/json")
            
        except Exception as e:
            logger.error(f"Error making outbound call: {e}")
            return JSONResponse(content={
                "success": False,
                "error": str(e),
                "message": "Failed to initiate call"
            }, status_code=500, media_type="application/json")

    return app


if __name__ == "__main__":
    app = main()
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
            