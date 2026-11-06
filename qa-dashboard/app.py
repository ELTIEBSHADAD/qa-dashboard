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

# إعداد الصفحة
st.set_page_config(page_title="لوحة الجودة", page_icon="📊", layout="wide")
st.title("📊 لوحة مراقبة جودة التعليق")
# ============ عرض التنبيهات ============
alerts = check_alerts(df)

if alerts:
    for alert in alerts:
        if alert["level"] == "error":
            st.error(alert["message"])  # صندوق أحمر
        else:
            st.warning(alert["message"]) # صندوق برتقالي
else:
    st.success("✅ جميع المؤشرات ضمن الحدود الطبيعية.") 

# قراءة البيانات

# ============ المؤشرات ============
agreement = (df['my_label'] == df['my_label_2']).mean()
accuracy = (df['my_label'] == df['model_label']).mean()
kappa = cohen_kappa_score(df['my_label'], df['my_label_2'])

col1, col2, col3 = st.columns(3)
col1.metric("الاتساق الذاتي", f"{agreement:.1%}")
col2.metric("دقة النموذج", f"{accuracy:.1%}")
col3.metric("Cohen's Kappa", f"{kappa:.2f}")

# ============ رسم دائري ============
st.subheader("توزيع التصنيفات")
label_counts = df['my_label'].value_counts().reset_index()
label_counts.columns = ['التصنيف', 'العدد']
fig = px.pie(label_counts, names='التصنيف', values='العدد',
             title='توزيع التصنيفات اليدوية')
st.plotly_chart(fig, use_container_width=True)

# ============ مصفوفة الالتباس ============
st.subheader("مصفوفة الالتباس (تصنيفي vs النموذج)")
confusion = pd.crosstab(df['my_label'], df['model_label'])
fig2 = px.imshow(confusion, text_auto=True,
                 labels=dict(x="النموذج", y="أنا", color="العدد"),
                 color_continuous_scale='Blues')
st.plotly_chart(fig2, use_container_width=True)

# ============ البيانات ============
st.subheader("جميع البيانات")
st.dataframe(df, use_container_width=True)
# ============ تحليل الأخطاء ============
st.subheader("❌ تحليل أخطاء النموذج (Error Analysis)")
errors = df[df['my_label'] != df['model_label']]

if not errors.empty:
    st.warning(f"تم العثور على {len(errors)} أخطاء.")
    st.dataframe(errors[['text', 'my_label', 'model_label', 'error_type']], use_container_width=True)
else:
    st.success("🎉 لا توجد أخطاء! النموذج متطابق تمامًا مع تصنيفك.")

# ============ الاتجاه الزمني ============
st.subheader("📅 تطور دقة النموذج عبر الزمن")
# تحويل التاريخ إلى صيغة تاريخ
df['date'] = pd.to_datetime(df['date'])

# حساب الدقة اليومية
daily_acc = df.groupby('date').apply(
    lambda x: (x['my_label'] == x['model_label']).mean()
).reset_index(name='accuracy')

# رسم الخط الزمني
fig3 = px.line(daily_acc, x='date', y='accuracy', markers=True,
               title="Model Accuracy Over Time")
fig3.update_yaxes(tickformat='.0%')
st.plotly_chart(fig3, use_container_width=True)
# ==========================================
# 12. مراجعة الأقران (Peer Review)
# ==========================================
st.markdown("---")
st.header("👥 نظام مراجعة الأقران (محاكاة)")
st.write("يقوم هذا القسم بمحاكاة وجود 5 مُقيّمين (أنت + 4 آخرين) على نفس البيانات، ويحسب مقياس اتفاق **Fleiss' Kappa**، ويكشف من هو المُقيّم غير الموثوق.")

if st.button("👥 تشغيل محاكاة مراجعة الأقران"):
    sim_df, fleiss_k, reliability, labels = simulate_and_evaluate_peers(df)
    
    st.success("✅ اكتملت المحاكاة!")
    st.subheader("📊 نتائج المُقيّمين")
    
    col1, col2 = st.columns(2)
    col1.metric("Fleiss' Kappa", f"{fleiss_k:.2f}", help="فوق 0.6 جيد، فوق 0.8 ممتاز")
    
    # عرض دقة كل مُقيّم
    rel_df = pd.DataFrame(list(reliability.items()), columns=['المُقيّم', 'نسبة الدقة (مقابل المعيار)'])
    st.dataframe(rel_df.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
    
    # اكتشاف غير الموثوق
    for annotator, acc in reliability.items():
        if acc < 0.60:
            st.error(f"⚠️ المُقيّم **{annotator}** غير موثوق! دقته {acc:.1%} فقط. يجب استبعاده.")
            
    st.subheader("📋 عينة من تصنيفات المُقيّمين")
    st.dataframe(sim_df[['item_id', 'Annotator_1 (You)', 'Annotator_2', 'Annotator_4 (Weak)']].head(), use_container_width=True)

# ==========================================
# 13. مقارنة النماذج (Multi-LLM Benchmarking)
# ==========================================
st.markdown("---")
st.header("⚔️ مقارنة النماذج (McNemar Test)")
st.write("يقارن هذا القسم بين أداء نموذجين (نموذج A ونموذج B) باستخدام **اختبار McNemar** لمعرفة إن كان الفرق بينهما دالًا إحصائيًا أم مجرد صدفة.")

if st.button("⚔️ تشغيل مقارنة النماذج"):
    # نستخدم التصنيف البشري كمعيار
    if 'llm_label' in df.columns:
        bench_results, error = benchmark_models(df, model_a_col='my_label', model_b_col='llm_label')
        
        if error:
            st.warning(error)
        else:
            st.success("✅ اكتملت المقارنة!")
            col1, col2 = st.columns(2)
            col1.metric("دقة النموذج A (البشري)", f"{bench_results['model_a_acc']:.1%}")
            col2.metric("دقة النموذج B (المحاكي)", f"{bench_results['model_b_acc']:.1%}")
            
            st.subheader("📊 اختبار McNemar")
            st.metric("P-Value", f"{bench_results['p_value']:.4f}")
            
            if bench_results['p_value'] < 0.05:
                st.success("🎉 الفرق دال إحصائيًا! النموذج B أفضل فعلاً وليس صدفة.")
            else:
                st.info("🤔 لا يوجد فرق دال إحصائيًا. الفرق قد يكون مجرد صدفة.")
    else:
        st.warning("الرجاء تشغيل قسم الذكاء الاصطناعي أولاً لإنشاء عمود llm_label.")
