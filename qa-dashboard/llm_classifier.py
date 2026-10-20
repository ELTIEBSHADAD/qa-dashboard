import google.generativeai as genai

def classify_with_llm(text, api_key):
    try:
        genai.configure(api_key=api_key)
        
        # --- اختيار النموذج تلقائيًا ---
        selected_model = None
        # نبحث عن أول نموذج يدعم generateContent (مثل flash أو pro)
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # نفضل نماذج flash لأنها أسرع وأرخص، وإلا نأخذ أول نموذج متاح
                if 'flash' in m.name.lower():
                    selected_model = m.name
                    break
                elif selected_model is None:
                    selected_model = m.name
        
        if not selected_model:
            return "Error: No suitable Gemini model found for this API key."
            
        model = genai.GenerativeModel(selected_model)
        
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
        
        if 'positive' in result: return 'positive'
        if 'negative' in result: return 'negative'
        if 'mixed' in result: return 'mixed'
        return 'neutral'
        
    except Exception as e:
        return f"Error: {str(e)}"
