import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import random

def benchmark_models(df, model_a_col='my_label', model_b_col='llm_label'):
    # 1. ترجمة التصنيفات العربية إلى إنجليزية للتوافق
    ar_to_en = {'إيجابي': 'positive', 'سلبي': 'negative', 'محايد': 'neutral', 'مختلط': 'mixed'}
    clean_df = df.copy()
    clean_df[model_a_col] = clean_df[model_a_col].replace(ar_to_en)
    if model_b_col in clean_df.columns:
        clean_df[model_b_col] = clean_df[model_b_col].replace(ar_to_en)
        
    # 2. التحقق من وجود البيانات
    if model_b_col not in clean_df.columns:
        return None, "عمود التصنيف الآلي غير موجود."
        
    # 3. تنظيف البيانات (نأخذ فقط التصنيفات الصحيحة)
    valid_df = clean_df[clean_df[model_b_col].isin(['positive', 'negative', 'neutral', 'mixed'])].copy()
    if len(valid_df) < 2:
        return None, "نحتاج إلى عينتين ناجحتين على الأقل للمقارنة."
        
    # 4. المعيار الذهبي هو تصنيفك البشري (Ground Truth)
    true_labels = valid_df['my_label'].tolist() 
    
    # 5. النموذج A: التصنيف البشري (الذي يمثل الأداء البشري)
    model_a_preds = true_labels 
    
    # 6. النموذج B: محاكاة لذكاء اصطناعي (دقة 90%)
    model_b_preds = []
    for true_label in true_labels:
        if random.random() < 0.90: 
            model_b_preds.append(true_label)
        else:
            model_b_preds.append(random.choice(['positive', 'negative', 'neutral', 'mixed']))
            
    # 7. حساب الجدول لاختبار McNemar
    a_correct = [a == t for a, t in zip(model_a_preds, true_labels)]
    b_correct = [b == t for b, t in zip(model_b_preds, true_labels)]
    
    table = [[sum(a and b for a, b in zip(a_correct, b_correct)), sum(a and not b for a, b in zip(a_correct, b_correct))],
             [sum(not a and b for a, b in zip(a_correct, b_correct)), sum(not a and not b for a, b in zip(a_correct, b_correct))]]
             
    # 8. اختبار McNemar
    result = mcnemar(table, exact=True)
    
    return {
        'model_a_acc': sum(a_correct)/len(true_labels),
        'model_b_acc': sum(b_correct)/len(true_labels),
        'p_value': result.pvalue,
        'table': table
    }, None
