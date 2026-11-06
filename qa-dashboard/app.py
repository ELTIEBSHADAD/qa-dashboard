import streamlit as st
import pandas as pd
import os 
from alert_engine import check_alerts 
dir_name = os.path.abspath(os.path.dirname(__file__))
csv_path = os.path.join(dir_name, 'annotations.csv')
df = pd.read_csv(csv_path)

import plotly.express as px
from sklearn.metrics import cohen_kappa_score
from peer_review_engine import simulate_and_evaluate_peers
from benchmark_engine import benchmark_models
# ==========================================
# 13. مقارنة النماذج (McNemar Test)
# ==========================================
st.markdown("---")
st.header("⚔️ مقارنة النماذج (McNemar Test)")
st.write("يقارن هذا القسم بين أداء نموذجين (نموذج A ونموذج B) باستخدام **اختبار McNemar** لمعرفة إن كان الفرق بينهما دالًا إحصائيًا أم مجرد صدفة.")

if st.button("⚔️ تشغيل مقارنة النماذج"):
    # إذا لم يوجد عمود llm_label، نقوم بمحاكاة نموذج ذكاء اصطناعي
    if 'llm_label' not in df.columns:
        st.info("ℹ️ لم يتم تشغيل الذكاء الاصطناعي بعد. سيتم محاكاة نموذج ذكاء اصطناعي (دقة 85%) لمقارنته معك.")
        import random
        labels_list = ['positive', 'negative', 'neutral', 'mixed']
        # محاكاة نموذج ذكاء اصطناعي
        df['llm_label'] = [true_label if random.random() < 0.85 else random.choice(labels_list) for true_label in df['my_label']]
    
    # استدعاء دالة المقارنة
    bench_results, error = benchmark_models(df, model_a_col='my_label', model_b_col='llm_label')
    
    if error:
        st.warning(error)
    else:
        st.success("✅ اكتملت المقارنة!")
        col1, col2 = st.columns(2)
        col1.metric("دقة النموذج A (البشري)", f"{bench_results['model_a_acc']:.1%}")
        col2.metric("دقة النموذج B (المحاكي)", f"{bench_results['model_b_acc']:.1%}")
        
        st.metric("P-Value", f"{bench_results['p_value']:.4f}")
        
        if bench_results['p_value'] < 0.05:
            st.success("🎉 الفرق دال إحصائيًا! النموذج B أفضل فعلاً وليس صدفة.")
        else:
            st.info("🤔 لا يوجد فرق دال إحصائيًا. الفرق قد يكون مجرد صدفة.")
