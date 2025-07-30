import os
import warnings
import asyncio
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Suprimir warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules after import of package.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*BaseAuthenticatedTool: This feature is experimental.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*auth_config or auth_config.auth_scheme is missing.*")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from toolbox_core import ToolboxSyncClient
from google.adk.agents import LlmAgent, Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Cargar variables de entorno
load_dotenv()

# Validar que la API key esté presente
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY no encontrada en el entorno ni en el archivo .env")

# Configuración
TARGET_FOLDER_PATH = "/home/cetec/AIProjects/adk_mcp"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "semantic-memory")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Variables globales para almacenar instancias
session_service = None
runner = None
root_agent = None

# Modelos Pydantic
class ChatRequest(BaseModel):
    message: str
    user_id: str = "api_user"
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    user_id: str

class SessionRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    status: str

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]

# Inicialización de agentes y herramientas
def initialize_agents():
    """Inicializar los agentes y herramientas"""
    # Load toolbox tools
    toolbox = ToolboxSyncClient("http://127.0.0.1:7000")
    tools = toolbox.load_toolset('my_bq_toolset')
    
    # Lista de herramientas MCP
    tools_list = [
        # MCP Filesystem Server
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params={
                    'command': 'npx',
                    'args': [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        os.path.abspath(TARGET_FOLDER_PATH),
                    ],
                }
            ),
            tool_filter=['list_directory', 'read_file', 'write_file', 'delete_file', 'create_directory'],
        ),
        # MCP Playwright Server
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params={
                    'command': 'npx',
                    'args': [
                        "@playwright/mcp@latest"
                    ],
                },
                timeout=30
            ),
        ),
        # MCP Qdrant Server
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params={
                    'command': 'uvx',
                    'args': ['mcp-server-qdrant'],
                    'env': {
                        'QDRANT_URL': QDRANT_URL,
                        'QDRANT_API_KEY': QDRANT_API_KEY,
                        'COLLECTION_NAME': COLLECTION_NAME,
                        'EMBEDDING_MODEL': EMBEDDING_MODEL,
                        'TOOL_STORE_DESCRIPTION': 'Store information in the semantic memory database. The information parameter should contain the text/content to store, and metadata can include additional context like source, tags, or categories.',
                        'TOOL_FIND_DESCRIPTION': 'Search for relevant information from the semantic memory database using natural language queries. Returns the most relevant stored information based on semantic similarity.'
                    }
                },
                timeout=30
            ),
            tool_filter=['qdrant-store', 'qdrant-find'],
        )
    ]

    # GCP Agent
    gcp_agent = Agent(
        name="gcp_releasenotes_agent",
        model="gemini-2.5-flash",
        description="Agent to answer questions about Google Cloud Release notes.",
        instruction="You are a helpful agent who can answer user questions about the Google Cloud Release notes. Use the tools to answer the question",
        tools=tools,
    )

    # Root Agent
    root_agent = LlmAgent(
        model='gemini-2.5-flash',
        name='filesystem_playwright_qdrant_sse_assistant_agent',
        instruction='''Help the user manage their files, automate web interactions, work with semantic memory storage, and use additional MCP services. 
You can:
- List files, read files, write files, navigate web pages, interact with web elements, take screenshots
- Store information in Qdrant vector database for semantic search and memory
- Retrieve relevant information from stored memories using natural language queries
- Manage collections in the vector database for organized information storage
- Access additional MCP tools via HTTP SSE connection
''',
        tools=tools_list,
        sub_agents=[gcp_agent]
    )
    
    return root_agent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida de la aplicación"""
    global session_service, runner, root_agent
    
    print("🚀 Iniciando API Backend...")
    
    # Inicializar agentes
    root_agent = initialize_agents()
    
    # Inicializar servicio de sesiones
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="filesystem_api", 
        user_id="default_user", 
        session_id="default_session"
    )
    
    # Inicializar runner
    runner = Runner(
        agent=root_agent, 
        app_name="filesystem_api", 
        session_service=session_service
    )
    
    print("✅ API Backend iniciada correctamente")
    yield
    
    print("🛑 Cerrando API Backend...")

# Crear la aplicación FastAPI
app = FastAPI(
    title="Filesystem + Playwright + Qdrant + SSE Assistant API",
    description="API para gestión de archivos, automatización web, memoria semántica y servicios MCP",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Filesystem + Playwright + Qdrant + SSE Assistant API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verificar el estado de salud de la API y servicios"""
    services = {
        "api": "healthy",
        "session_service": "healthy" if session_service else "unavailable",
        "runner": "healthy" if runner else "unavailable",
        "root_agent": "healthy" if root_agent else "unavailable"
    }
    
    status = "healthy" if all(s == "healthy" for s in services.values()) else "degraded"
    
    return HealthResponse(status=status, services=services)

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """Crear una nueva sesión para un usuario"""
    if not session_service:
        raise HTTPException(status_code=503, detail="Session service not available")
    
    session_id = request.session_id or f"session_{request.user_id}_{asyncio.get_event_loop().time()}"
    
    try:
        await session_service.create_session(
            app_name="filesystem_api",
            user_id=request.user_id,
            session_id=session_id
        )
        
        return SessionResponse(
            session_id=session_id,
            user_id=request.user_id,
            status="created"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Procesar un mensaje del usuario y devolver la respuesta del agente"""
    if not runner:
        raise HTTPException(status_code=503, detail="Runner service not available")
    
    # Generar session_id si no se proporciona
    session_id = request.session_id or f"session_{request.user_id}"
    
    # Crear sesión si no existe
    try:
        await session_service.create_session(
            app_name="filesystem_api",
            user_id=request.user_id,
            session_id=session_id
        )
    except:
        pass  # La sesión puede ya existir
    
    try:
        # Crear contenido del mensaje
        user_content = types.Content(
            role='user', 
            parts=[types.Part(text=request.message)]
        )
        
        # Ejecutar el agente
        response_text = ""
        async for event in runner.run_async(
            user_id=request.user_id,
            session_id=session_id,
            new_message=user_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                response_text = event.content.parts[0].text
                break
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            user_id=request.user_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Obtener información sobre una sesión específica"""
    # Aquí podrías implementar lógica para obtener información de la sesión
    # Por ahora devolvemos información básica
    return {
        "session_id": session_id,
        "status": "active",
        "created_at": "2024-01-01T00:00:00Z"  # Placeholder
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Eliminar una sesión específica"""
    # Aquí podrías implementar lógica para eliminar la sesión
    return {
        "session_id": session_id,
        "status": "deleted"
    }

@app.get("/tools")
async def list_tools():
    """Listar todas las herramientas disponibles"""
    if not root_agent:
        raise HTTPException(status_code=503, detail="Agent not available")
    
    # Esta es una representación básica de las herramientas disponibles
    tools_info = {
        "filesystem": [
            "list_directory",
            "read_file", 
            "write_file",
            "delete_file",
            "create_directory"
        ],
        "playwright": [
            "navigate",
            "screenshot",
            "click",
            "type",
            "wait_for_element"
        ],
        "qdrant": [
            "qdrant-store",
            "qdrant-find"
        ],
        "gcp": [
            "bigquery_tools"
        ]
    }
    
    return tools_info

# Función principal para ejecutar la API
def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Ejecutar la API"""
    uvicorn.run(
        "agent:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    # Ejecutar la API en modo desarrollo
    run_api(host="127.0.0.1", port=8000, reload=True)