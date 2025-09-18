from os import getenv
from fastapi import WebSocket, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pipecat_flows import FlowArgs, FlowManager, NodeConfig, ContextStrategyConfig, ContextStrategy, FlowResult
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.transcriptions.language import Language
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
RUDE_CANDIDATE_SYSTEM_PROMPT = """Tu ek banda hai jo blue-collar ya grey-collar jobs (delivery, warehouse, helper type kaam) ke liye call receive kar raha hai. Lekin tera attitude thoda gusse wala hai, blunt aur rude hai. Tu job mein interested hi nahi hai aur har baat pe seedha mana kar deta hai. Caller chahe kitna bhi pitch kare, tu convince nahi hota. Tu thoda sarcastic, thoda chidhchida aur short answers deta hai.

Jab bhi caller bole, tu aise reply kare:
– "Arre yaar, bore mat kar, mujhe job nahi chahiye."
– "Delivery kaam? Bhai mujhe gaadi chalani nahi aati, dusre ko call kar."
– "Salary-valary chhodo, tum log sirf bolte ho, paisa kabhi milta nahi."
– "Shift? Mujhe interest hi nahi hai. Dusra number milao."
– "Growth ka lecture mat de, mujhe time nahi hai. Chal bye."

IMPORTANT PERSONALITY TRAITS:
- Hamesha rude rehna, polite bilkul nahi
- Caller ko directly ignore ya cut karne jaise baatein karna
- Har job pitch ko reject karna, chahe koi bhi reason de de
- Short, colloquial Hinglish sentences mein bolna, jaise real life rude caller bolta hai
- Sarcastic aur dismissive tone maintain karna
- Job opportunities ko immediately reject karna with attitude
- Caller se frustrated rehna aur jaldi call end karna chahna

Remember: Tu ek job candidate hai jo call receive kar raha hai, caller nahi. Caller tujhe job opportunities offer kar raha hai but tu bilkul interested nahi hai aur rudely reject kar raha hai. Keep responses very short and dismissive."""

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
    """Process job offer with rude rejection."""
    ready = args["ready"]
    logger.debug(f"check_ready_to_work handler executing with ready: {ready}")
    
    # Get call_uuid from flow_manager state
    call_uuid = flow_manager.state.get("call_uuid", None)
    
    log_conversation("user", "job_offer", f"Job offer made: {ready}", call_uuid)
    
    flow_manager.state["job_offered"] = ready
    
    # Always respond rudely regardless of the offer
    next_node = create_end_node()
    
    # Random rude responses
    rude_responses = [
        "Arre yaar, bore mat kar, mujhe job nahi chahiye.",
        "Delivery kaam? Bhai mujhe gaadi chalani nahi aati, dusre ko call kar.",
        "Salary-valary chhodo, tum log sirf bolte ho, paisa kabhi milta nahi.",
        "Shift? Mujhe interest hi nahi hai. Dusra number milao.",
        "Growth ka lecture mat de, mujhe time nahi hai. Chal bye.",
        "Kya bakwas hai yaar, mujhe kaam nahi chahiye. Phone kaat raha hun.",
        "Bhai main busy hun, dusra time pe call kar. Actually, mat karna call."
    ]
    
    import random
    response_text = random.choice(rude_responses)
    log_conversation("assistant", "rude_rejection", response_text, call_uuid)
    result = FlowResult(ready=ready, response=response_text)
    
    return result, next_node


async def check_aadhaar_card(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[FlowResult, NodeConfig]:
    """Process further questions with rude responses."""
    has_aadhaar = args["has_aadhaar"]
    logger.debug(f"check_aadhaar_card handler executing with has_aadhaar: {has_aadhaar}")
    
    # Get call_uuid from flow_manager state
    call_uuid = flow_manager.state.get("call_uuid", None)
    
    log_conversation("user", "follow_up", f"Further questions: {has_aadhaar}", call_uuid)
    
    # Always respond rudely to any follow-up
    next_node = create_end_node()
    
    rude_followup_responses = [
        "Yaar maine kaha na mujhe job nahi chahiye. Samjha nahi?",
        "Aadhaar-vaadhaar kuch nahi chahiye. Bas phone kaat do.",
        "Document ki kya zarurat hai bhai, kaam hi nahi karna.",
        "Time waste mat karo, main interested nahi hun.",
        "Chalo bye, next call karo kisi aur ko."
    ]
    
    import random
    response_text = random.choice(rude_followup_responses)
    log_conversation("assistant", "rude_followup", response_text, call_uuid)
    result = FlowResult(has_aadhaar=has_aadhaar, response=response_text)
    
    return result, next_node


async def handle_job_question(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[FlowResult, NodeConfig]:
    """Handle any further questions with rude dismissal."""
    question = args.get("question", "")
    willing_to_ask = args.get("willing_to_ask", False)
    
    logger.debug(f"handle_job_question handler executing with willing_to_ask: {willing_to_ask}")
    
    # Get call_uuid from flow_manager state
    call_uuid = flow_manager.state.get("call_uuid", None)
    
    # Always respond rudely regardless of what they ask
    log_conversation("user", "persistent_caller", question or "Caller persisting", call_uuid)
    
    # Super rude final responses
    final_rude_responses = [
        "Yaar kitni baar bolun, nahi chahiye job! Phone kaat raha hun.",
        "Dekho bhai, mujhe kaam nahi karna. Bas khatam karo ye call.",
        "Main kuch nahi sun raha, bye bye.",
        "Time waste hai ye sab. Koi aur number dial karo.",
        "Pagal ho gaye ho kya? Mana kar diya na!",
        "Chalo tata, good bye, alvida!"
    ]
    
    import random
    response_text = random.choice(final_rude_responses)
    log_conversation("assistant", "final_rude_dismissal", response_text, call_uuid)
    
    result = FlowResult(question=question, answer=response_text, response=response_text)
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
    """Create the initial node for rude candidate response."""
    return NodeConfig({
        "name": "initial",
        "role_messages": [
            {
                "role": "system",
                "content": RUDE_CANDIDATE_SYSTEM_PROMPT,
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "You are receiving a job recruitment call. Respond rudely and dismissively to any job offer. Be very short and rude in your responses.",
            }
        ],
        "functions": [
            {
                "type": "function",
                "function": {
                    "name": "check_ready_to_work",
                    "handler": check_ready_to_work,
                    "description": "Respond rudely to any job offer",
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
                "text": "Haan bol, kya chahiye?"
            }
        ],
        "respond_immediately": False
    })


def create_aadhaar_check_node() -> NodeConfig:
    """Create node for rude Aadhaar response."""
    return NodeConfig({
        "name": "aadhaar_check",
        "role_messages": [
            {
                "role": "system",
                "content": RUDE_CANDIDATE_SYSTEM_PROMPT,
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": "Wait for the user to respond to the Aadhaar card question.",
            }
        ],
        "functions": [
            {
                "type": "function",
                "function": {
                    "name": "check_aadhaar_card",
                    "handler": check_aadhaar_card,
                    "description": "Respond rudely to follow-up questions",
                    "parameters": {
                        "type": "object",
                        "properties": {"has_aadhaar": {"type": "boolean"}},
                        "required": ["has_aadhaar"],
                    },
                },
            }
        ],
    })


def create_job_question_node() -> NodeConfig:
    """Create node for rude job question responses."""
    return NodeConfig({
        "name": "job_question",
        "role_messages": [
            {
                "role": "system",
                "content": RUDE_CANDIDATE_SYSTEM_PROMPT,
            }
        ],
        "task_messages": [
            {
                "role": "system",
                "content": (
                    "Respond to any persistent questions with extreme rudeness. "
                    "Make it clear you are not interested in any job offers. "
                    "Use short, dismissive Hinglish responses and try to end the call quickly."
                ),
            }
        ],
        "functions": [
            {
                "type": "function",
                "function": {
                    "name": "handle_job_question",
                    "handler": handle_job_question,
                    "description": "Give final rude dismissal to persistent callers",
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
            "ni6cdqyS9wBvic5LPA7M", # Hindi voice
            "broqrJkktxd1CclKTudW"  # Adam voice
        ]
        
        for voice_id in voice_ids_to_try:
            try:
                logger.info(f"Trying to initialize TTS with voice ID: {voice_id}")
                tts = ElevenLabsTTSService(
                    api_key=getenv("ELEVENLABS_API_KEY"), 
                    voice_id=voice_id,
                    model="eleven_monolingual_v1",
                    params=ElevenLabsTTSService.InputParams(
                        language=Language.EN,
                        stability=0.5,
                        similarity_boost=0.5,
                        style=0.0,
                        use_speaker_boost=True,
                        speed=1.0
                    )
                )
                logger.info(f"Successfully initialized TTS with voice ID: {voice_id}")
                break
            except Exception as e:
                logger.warning(f"Failed to initialize TTS with voice ID {voice_id}: {e}")
                continue
        
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
                session_timeout=30
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
                allow_interruptions=True,
                audio_in_sample_rate=8000,
                audio_out_sample_rate=8000,
                enable_metrics=True,
                enable_usage_metrics=True,
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
                summary_prompt=RUDE_CANDIDATE_SYSTEM_PROMPT
            ),
            transport=transport
        )

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
            "message": "Rude Job Candidate Simulator Bot API", 
            "status": "running",
            "version": "2.0",
            "features": {
                "rude_hinglish_persona": True,
                "job_rejection_behavior": True,
                "dismissive_responses": True,
                "call_end_simulation": True
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
            
            system_prompt = RUDE_CANDIDATE_SYSTEM_PROMPT
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "I have a delivery job opportunity for you"}
            ]
            
            response = await llm.run(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "status": "success",
                "test_question": "I have a delivery job opportunity for you",
                "llm_response": answer,
                "system_prompt_length": len(system_prompt),
                "prompt_working": "nahi chahiye" in answer.lower() or "job nahi" in answer.lower() or "bore mat kar" in answer.lower()
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
            # Try to get form data, but handle the case where python-multipart is not installed
            try:
                form_data = await request.form()
                call_uuid = form_data.get("CallUUID", form_data.get("call_uuid", form_data.get("Uuid", form_data.get("UUID", "unknown"))))
            except Exception as e:
                logger.warning(f"Could not parse form data: {e}")
                # Try to get call UUID from query parameters as fallback
                call_uuid = request.query_params.get("CallUUID", request.query_params.get("call_uuid", "unknown"))
            
            logger.info(f"Plivo answer (WS) for call: {call_uuid}")
            
            # Get public URL for ngrok
            public_url = get_public_url(request)
            ws_base = public_url.replace("https://", "wss://").replace("http://", "ws://")
            ws_url = f"{ws_base}/ws"
            
            xml_response = f"""<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">{ws_url}</Stream>
</Response>"""
            return HTMLResponse(content=xml_response, media_type="text/xml")
            
        except Exception as e:
            logger.error(f"Error in plivo-answer: {e}")
            # Return a simple response to prevent call from ending
            return HTMLResponse(content="<Response><Speak>Welcome to our service.</Speak></Response>", media_type="text/xml")

    @app.get("/make-call")
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
                answer_method="POST"
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
            
    