# ./adk_agent_samples/mcp_agent/agent.py

import os # Required for path operations
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules after import of package.*")
# Suprimir UserWarning de BaseAuthenticatedTool experimental y auth_config
warnings.filterwarnings("ignore", category=UserWarning, message=".*BaseAuthenticatedTool: This feature is experimental.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*auth_config or auth_config.auth_scheme is missing.*")

# Cargar variables de entorno desde .env
load_dotenv()

# Validar que la API key esté presente
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY no encontrada en el entorno ni en el archivo .env")
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams



TARGET_FOLDER_PATH = "/media/dracero/08c67654-6ed7-4725-b74e-50f29ea60cb2/pythonAI-Others/MCP_ADK"

root_agent = LlmAgent(
    model='gemini-2.5-flash',
    name='filesystem_playwright_assistant_agent',
    instruction='Help the user manage their files and automate web interactions. You can list files, read files, write files, navigate web pages, interact with web elements, take screenshots, etc.',
    tools=[
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
                }
            ),
            # Puedes agregar tool_filter aquí si quieres limitar las herramientas de Playwright
            # tool_filter=['navigate', 'click', 'type', 'screenshot', 'get_page_content'],
        )
    ],
)
# CLI interactivo
import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


async def main():
    session_service = InMemorySessionService()
    # Crear la sesión antes de usarla
    await session_service.create_session(app_name="filesystem_cli", user_id="cli_user", session_id="cli_session")
    runner = Runner(agent=root_agent, app_name="filesystem_cli", session_service=session_service)
    session_id = "cli_session"
    user_id = "cli_user"
    print("\n=== Filesystem Assistant CLI ===")
    print("Escribe comandos para gestionar archivos. Escribe /quit para salir.\n")
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