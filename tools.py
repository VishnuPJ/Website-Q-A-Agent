import os
import asyncio
from crewai_tools import MDXSearchTool
from typing import Optional, Dict, Any

class ToolCreator:
    """A class for creating and managing various RAG (Retrieval-Augmented Generation) tools.
    
    This class provides methods to create different types of RAG tools,
    currently supporting markdown-based RAG implementations.
    """

    def markdown_rag_tool(self, md_path: str) -> Optional[MDXSearchTool]:
        """Creates a markdown-based RAG tool using the specified markdown file.
        Args:
            md_path (str): Path to the markdown file to be used for RAG.
        Returns:
            Optional[MDXSearchTool]: Configured MDXSearchTool instance if successful,
                                   None if an error occurs.
        Raises:
            FileNotFoundError: If the specified markdown file doesn't exist.
            ValueError: If the markdown file path is invalid or empty.
        """
        try:
            # Validate input parameters
            if not md_path:
                raise ValueError("Markdown file path cannot be empty")
            
            if not os.path.exists(md_path):
                raise FileNotFoundError(f"Markdown file not found at: {md_path}")
            
            # Default configuration for the RAG tool
            config: Dict[str, Any] = {
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": "qwen2.5",
                    }
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": "BAAI/bge-small-en-v1.5",
                    }
                }
            }
            
            # Create and return the MDX search tool
            md_tool = MDXSearchTool(
                mdx=md_path,
                config=config
            )
            
            return md_tool
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except ValueError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            return None

def main():
    """Main function to demonstrate the usage of ToolCreator."""
    try:
        tool_creator = ToolCreator()
        markdown_path = "crawl_results.md"
        
        tool = tool_creator.markdown_rag_tool(markdown_path)
        if tool:
            result = tool.run(search_query="What integrations are available?")
            print(result)
        else:
            print("Failed to create RAG tool")
            
    except Exception as e:
        print(f"Error in main: {e}")

# if __name__ == "__main__":
#     main()
