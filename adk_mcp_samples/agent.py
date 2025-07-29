import os # Required for path operations
import warnings
import subprocess
import sys
from dotenv import load_dotenv
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules after import of package.*")
# Suprimir UserWarning de BaseAuthenticatedTool experimental y auth_config
warnings.filterwarnings("ignore", category=UserWarning, message=".*BaseAuthenticatedTool: This feature is experimental.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*auth_config or auth_config.auth_scheme is missing.*")
from toolbox_core import ToolboxSyncClient

toolbox = ToolboxSyncClient("http://127.0.0.1:7000")

# Load all the tools
tools = toolbox.load_toolset('my_bq_toolset')

# Cargar variables de entorno desde .env
load_dotenv()

# Validar que la API key esté presente
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY no encontrada en el entorno ni en el archivo .env")

from google.adk.agents import LlmAgent, Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams

TARGET_FOLDER_PATH = "/home/cetec/AIProjects/adk_mcp"

# Configuración de variables de entorno para Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "semantic-memory")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Lista de herramientas
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
            timeout = 30
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
            timeout = 30
        ),
        tool_filter=['qdrant-store', 'qdrant-find'],
    )
]

gcp_agent = Agent(
    name="gcp_releasenotes_agent",
    model="gemini-2.5-flash",
    description=(
        "Agent to answer questions about Google Cloud Release notes."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the Google Cloud Release notes. Use the tools to answer the question"
    ),
    tools=tools,
)

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
    tools= tools_list,
    sub_agents=[gcp_agent]
)

# CLI interactivo
import asyncio
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(app_name="filesystem_cli", user_id="cli_user", session_id="cli_session")
    runner = Runner(agent=root_agent, app_name="filesystem_cli", session_service=session_service)
    session_id = "cli_session"
    user_id = "cli_user"
    
    print("\n=== Filesystem + Playwright + Qdrant + SSE Assistant CLI ===")
    print("🧠 Con memoria semántica Qdrant")
    print("🗂️  Gestión de archivos")
    print("🌐 Automatización web con Playwright")
    print("🔌 Conexión MCP via HTTP SSE")
    
    print("\nEjemplos de comandos:")
    print("- lista los archivos en la carpeta")
    print("- lee el archivo config.txt")
    print("- navega a https://example.com y toma una captura")
    print("- guarda en memoria: 'La configuración del servidor está en config.json'")
    print("- busca información sobre configuración del servidor")
    print("- /quit para salir\n")
    
    while True:
        try:
            user_input = input("Usuario > ").strip()
            if not user_input:
                continue
            if user_input.lower() == "/quit":
                print("👋 Saliendo del asistente.")
                break
            
            user_content = types.Content(role='user', parts=[types.Part(text=user_input)])
            response_text = ""
            
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=user_content):
                if event.is_final_response() and event.content and event.content.parts:
                    response_text = event.content.parts[0].text
                    break
            
            print(f"\n🤖 {response_text}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Saliendo del asistente.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())