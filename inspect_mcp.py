
import asyncio
from mcp_client import client

async def main():
    for server_name in ["aviationstack", "weather"]:
        print(f"\n{'=' * 60}")
        print(f"SERVER: {server_name.upper()}")
        print("=" * 60)

        try:
            tools = await client.get_tools(server_name=server_name)

            for tool in tools:
                print(f"\nTOOL NAME: {tool.name}")
                print(f"DESCRIPTION: {tool.description}")
                
                if hasattr(tool, "args_schema"):
                    try:
                        print("ARGUMENT SCHEMA:")
                        print(tool.args_schema.model_json_schema())
                    except Exception:
                        print(tool.args_schema)

        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")

if __name__ == "__main__":
    asyncio.run(main())
