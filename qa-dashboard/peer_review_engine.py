import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
import random

def simulate_and_evaluate_peers(df):
    # 1. محاكاة 4 مُقيّمين آخرين بناءً على تصنيفك (مع إضافة 10% أخطاء عشوائية)
    # افتراض: تصنيفك (my_label) هو المعيار الذهبي
    np.random.seed(42)
    labels = ['positive', 'negative', 'neutral', 'mixed']
    
    simulated_data = []
    for _, row in df.head(10).iterrows(): # نأخذ 10 عينات فقط لسرعة العرض
        true_label = row['my_label']
        
        # المُقيّم 1: ممتاز (95% دقة)
        a1 = true_label if random.random() < 0.95 else random.choice(labels)
        # المُقيّم 2: جيد (85% دقة)
        a2 = true_label if random.random() < 0.85 else random.choice(labels)
        # المُقيّم 3: متوسط (70% دقة)
        a3 = true_label if random.random() < 0.70 else random.choice(labels)
        # المُقيّم 4: ضعيف (50% دقة) - نريد اكتشافه
        a4 = true_label if random.random() < 0.50 else random.choice(labels)
        
        simulated_data.append({'item_id': row['id'], 'true_label': true_label, 
                               'Annotator_1 (You)': true_label, 
                               'Annotator_2': a1, 'Annotator_3': a2, 
                               'Annotator_4': a3, 'Annotator_5': a4})
        
    sim_df = pd.DataFrame(simulated_data)
    
    # 2. تجهيز البيانات لحساب Fleiss' Kappa
    # نحتاج مصفوفة: الصفوف = العينات، الأعمدة = الفئات، القيم = عدد المُقيّمين الذين اختاروا هذه الفئة
    annotators_cols = ['Annotator_1 (You)', 'Annotator_2', 'Annotator_3', 'Annotator_4', 'Annotator_5']
    matrix = []
    for _, row in sim_df.iterrows():
        counts = [sum(row[annotators_cols] == label) for label in labels]
        matrix.append(counts)
    
    # 3. حساب Fleiss' Kappa
    kappa = fleiss_kappa(matrix, method='fleiss')
    
    # 4. كشف المُقيّم غير الموثوق (مقارنة بتصنيفك)
    reliability = {}
    for col in annotators_cols:
        accuracy = (sim_df[col] == sim_df['true_label']).mean()
        reliability[col] = accuracy
        
    return sim_df, kappa, reliability, labels
