# MCP Filesystem Assistant CLI

Este proyecto implementa un asistente de línea de comandos (CLI) para gestionar archivos usando agentes MCP y Gemini. Permite listar, leer, escribir, eliminar y crear directorios en una carpeta específica del sistema de archivos, todo mediante comandos naturales.

## Características

- Interfaz interactiva por línea de comandos
- Operaciones soportadas: listar archivos, leer archivos, escribir archivos, eliminar archivos, crear directorios
- Basado en Google ADK, Gemini y MCP Toolset
- Respuestas inteligentes y manejo de errores

## Requisitos

- Python 3.8+
- Node.js y npx instalados
- Una API key de Google Generative AI (Gemini)
- Paquetes: `python-dotenv`, `google-adk` y dependencias del MCP

## Instalación

1. Clona el repositorio:

```bash
git clone https://github.com/dracero/adk_mcp.git
cd adk_mcp/MCP_ADK
```

2. Instala las dependencias:

```bash
pip install uv
uv sync
```


3. Crea el archivo `.env` en la raíz del proyecto:

```env
# .env
GOOGLE_API_KEY=tu_api_key_de_gemini
```

## Ejecución

Ejecuta el asistente desde la carpeta del script MCP_ADK. DE la siguiente manera para trabajar con linea de comandos:

```bash
uvicorn adk_mcp_samples.agent:app --reload
```

o


```bash
adk web
```

Desde la misma carpeta para poder usar la built-in web interface de ADK

Para el caos de la linea de comandos verás un prompt interactivo. Escribe comandos como:

- `lista los archivos en la carpeta`
- `lee el archivo config.txt`
- `escribe en notas.txt: Hola mundo`
- `elimina el archivo temporal.log`
- `/quit` para salir

## Ejemplo de archivo .env

```env
GOOGLE_API_KEY=tu_api_key_de_gemini
```

## Notas

- El asistente opera sobre la carpeta configurada en el script (`TARGET_FOLDER_PATH`).
- Requiere acceso a internet para usar Gemini.
- Los mensajes de advertencia experimentales y de autenticación están suprimidos para una experiencia más limpia.
