import google.generativeai as genai

def classify_with_llm(text, api_key):
    try:
        genai.configure(api_key=api_key)
        # استخدام النموذج السريع والمجاني
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are an expert sentiment analyzer. Classify the following text into exactly one of these English categories:
        - positive
        - negative
        - neutral
        - mixed
        
        Reply with ONLY the category word. Do not write any explanation.
        
        Text: {text}
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip().lower()
        
        # التأكد من أن النتيجة تطابق تصنيفاتنا
        if 'positive' in result: return 'positive'
        if 'negative' in result: return 'negative'
        if 'mixed' in result: return 'mixed'
        return 'neutral'
        
    except Exception as e:
        return f"Error: {str(e)}"
