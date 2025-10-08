import tiktoken


def count_tokens(text: str, model: str = "gemini-2.5-flash") -> int:
   
    if not text:
        return 0
    
    try:
        # For Gemini models, use gpt-4 encoding as approximation
        if "gemini" in model.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")
        else:
            encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(str(text)))
    except:
        # Fallback to approximate token count if tiktoken fails
        # Average of 4 characters per token
        return len(str(text)) // 4