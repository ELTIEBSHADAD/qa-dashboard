from google import genai

def classify_with_llm(text, api_key):
    try:
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
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        
        result = response.text.strip().lower()
        
        if 'positive' in result: return 'positive'
        if 'negative' in result: return 'negative'
        if 'mixed' in result: return 'mixed'
        return 'neutral'
        
    except Exception as e:
        return f"Error: {str(e)}"
