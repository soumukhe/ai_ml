import requests
import json
import argparse
from typing import List, Dict, Any

class LLMClient:
    def __init__(self, host: str, port: int = 8000):
        """
        Initialize the LLM client to connect to your remote Phi-3-Mini model.
        
        Args:
            host: The IP address or hostname of your EC2 instance
            port: The port your model is listening on (default: 8000)
        """
        self.base_url = f"http://{host}:{port}"
        self.headers = {"Content-Type": "application/json"}
        
    def health_check(self) -> Dict[str, Any]:
        """Check if the model server is healthy and responding."""
        response = requests.get(f"{self.base_url}/v1/health/live")
        return response.json()
    
    def get_models(self) -> Dict[str, Any]:
        """Get available models from the server."""
        response = requests.get(f"{self.base_url}/v1/models")
        return response.json()
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "microsoft/phi-3-mini-4k-instruct",
        max_tokens: int = 32,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to the model.
        
        Args:
            messages: List of message objects with 'role' and 'content'
            model: Model identifier
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            stream: Whether to stream the response
            
        Returns:
            The model's response as a dictionary
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self.headers,
            data=json.dumps(payload)
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

def main():
    parser = argparse.ArgumentParser(description="Interact with a remote LLM")
    parser.add_argument("--host", required=True, help="EC2 instance IP address")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--query", required=True, help="Your question for the model")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--system", default="You are a helpful AI assistant.", 
                        help="System message to set context")
    
    args = parser.parse_args()
    
    client = LLMClient(args.host, args.port)
    
    # Check if the server is up
    try:
        health = client.health_check()
        print(f"Server is healthy: {health}")
    except Exception as e:
        print(f"Server health check failed: {e}")
        return
    
    # Send your query to the model
    messages = [{"role": "user", "content": args.query}]
    
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=args.max_tokens
        )
        
        # Print the response in a nice format
        assistant_message = response["choices"][0]["message"]["content"]
        print("\n--- Model Response ---")
        print(assistant_message)
        print("---------------------")
        
        # Print usage statistics
        usage = response.get("usage", {})
        if usage:
            print(f"\nTokens used: {usage.get('prompt_tokens', 0)} (prompt) + "
                  f"{usage.get('completion_tokens', 0)} (completion) = "
                  f"{usage.get('total_tokens', 0)} (total)")
            
    except Exception as e:
        print(f"Error getting response from model: {e}")

if __name__ == "__main__":
    main()