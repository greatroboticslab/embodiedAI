import ollama
import base64
import os

def encode_image_to_base64(image_path):
    """
    Encodes an image file to a base64 string.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def generate_image_desc(image_path: str, model_name: str, prompt: str):
    """
    Generates a description for an image using a multimodal Ollama model (e.g., LLaVA).
    Yields response chunks from the Ollama stream.
    """
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        yield {"error": f"Could not encode image: {image_path}"} # Yield an error message in stream format
        return

    messages = [
        {
            'role': 'user',
            'content': prompt,
            'images': [base64_image]
        }
    ]

    try:
        stream = ollama.chat(
            model=model_name,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            yield chunk
    except Exception as e:
        print(f"Error during Ollama API call for image description ({model_name}): {e}")
        yield {"error": f"Ollama API error: {e}"} # Yield an error message

def generate_response(model_name: str, prompt: str):
    """
    Generates a text response from a text-based Ollama model (e.g., LLaMA3).
    Yields response chunks from the Ollama stream.
    """
    messages = [
        {
            'role': 'user',
            'content': prompt,
        }
    ]
    try:
        stream = ollama.chat(
            model=model_name,
            messages=messages,
            stream=True
        )
        for chunk in stream:
            yield chunk
    except Exception as e:
        print(f"Error during Ollama API call for text generation ({model_name}): {e}")
        yield {"error": f"Ollama API error: {e}"} # Yield an error message

def stream_parser(ollama_stream):
    """
    Parses the streaming output from Ollama chat completions.
    Yields the content string from each message chunk.
    Handles potential errors in the stream.
    """
    full_response_content = []
    try:
        for chunk in ollama_stream:
            if "error" in chunk: # Check for error messages yielded by generator functions
                print(f"Stream error: {chunk['error']}")
                full_response_content.append(f"[Error: {chunk['error']}]")
                break # Stop processing this stream on error
            if 'message' in chunk and 'content' in chunk['message']:
                content_piece = chunk['message']['content']
                full_response_content.append(content_piece)
                yield content_piece # Yield individual pieces for immediate display/processing
            elif chunk.get('done', False) and not chunk['message']['content']: # Handle final empty chunk if any
                pass
            # You might want to log other chunk structures if they appear for debugging
            # else:
            # print(f"Unexpected chunk structure: {chunk}")
    except Exception as e:
        print(f"Error parsing Ollama stream: {e}")
        yield f"[Error parsing stream: {e}]" # Yield error message

    # The main script does ''.join(response_from_stream_parser).
    # If the stream was empty or only had errors, this ensures something is joined.
    if not full_response_content:
        yield "[No content received from model]"