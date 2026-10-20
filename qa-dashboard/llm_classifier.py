from google import genai
import time

def classify_with_llm(text, api_key, max_retries=3):
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are an expert sentiment analyzer. Classify the following text into exactly one of these English categories:
    - positive
    - negative
    - neutral
    - mixed
    
    Reply with ONLY the category word. Do not write any explanation.
    
    Text: {text}
    """
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.8-flash',
                contents=prompt
            )
            result = response.text.strip().lower()
            
            if 'positive' in result: return 'positive'
            if 'negative' in result: return 'negative'
            if 'mixed' in result: return 'mixed'
            return 'neutral'
            
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: {str(e)}"
            # انتظر 5 ثوانٍ قبل المحاولة مرة أخرى
            time.sleep(5) 
    
    return "Error: Max retries exceeded"
