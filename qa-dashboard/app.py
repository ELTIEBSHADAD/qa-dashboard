import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from alert_engine import check_alerts
import os
from stats_engine import calculate_statistics
from llm_classifier import classify_with_llm, extract_aspects_with_llm
from peer_review_engine import simulate_and_evaluate_peers
from benchmark_engine import benchmark_models
import numpy as np

# ==========================================
# 1. قراءة البيانات (مع المسار الديناميكي)
# ==========================================
dir_name = os.path.abspath(os.path.dirname(__file__))
csv_path = os.path.join(dir_name, 'annotations.csv')
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])

# ==========================================
# 2. إعدادات الصفحة
# ==========================================
st.set_page_config(page_title="QA Dashboard", page_icon="📊", layout="wide")
st.title("📊 QA Dashboard - Quality Monitoring")

# ==========================================
# 3. الفلاتر التفاعلية
# ==========================================
st.sidebar.header("🔍 تصفية البيانات (Filters)")
all_labels = df['my_label'].unique().tolist()
selected_labels = st.sidebar.multiselect("اختر التصنيفات:", options=all_labels, default=all_labels)
min_date = df['date'].min().date()
max_date = df['date'].max().date()
selected_date = st.sidebar.date_input("اختر الفترة الزمنية:", value=(min_date, max_date), min_value=min_date, max_value=max_date)

if len(selected_labels) > 0 and len(selected_date) == 2:
    start_date, end_date = selected_date
    df = df[(df['my_label'].isin(selected_labels)) & (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]

# ==========================================
# 4. التنبيهات
# ==========================================
alerts = check_alerts(df)
if alerts:
    for alert in alerts:
        if alert["level"] == "error":
            st.error(alert["message"])
        else:
            st.warning(alert["message"])
else:
    st.success("✅ جميع المؤشرات ضمن الحدود الطبيعية.")

# ==========================================
# 5. المؤشرات (KPIs)
# ==========================================
agreement = (df['my_label'] == df['my_label_2']).mean()
accuracy = (df['my_label'] == df['model_label']).mean()
kappa_self = cohen_kappa_score(df['my_label'], df['my_label_2'])
kappa_model = cohen_kappa_score(df['my_label'], df['model_label'])

col1, col2, col3, col4 = st.columns(4)
col1.metric("الاتساق الذاتي", f"{agreement:.1%}")
col2.metric("دقة النموذج", f"{accuracy:.1%}")
col3.metric("Kappa (ذاتي)", f"{kappa_self:.2f}")
col4.metric("Kappa (النموذج)", f"{kappa_model:.2f}")

# ==========================================
# 6. الرسم الدائري
# ==========================================
st.subheader("توزيع التصنيفات")
label_counts = df['my_label'].value_counts().reset_index()
label_counts.columns = ["التصنيف", "العدد"]
fig = px.pie(label_counts, names="التصنيف", values="العدد", title="توزيع التصنيفات اليدوية")
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 7. مصفوفة الالتباس
# ==========================================
st.subheader("مصفوفة الالتباس (تصنيفي vs النموذج)")
labels_list = ['إيجابي', 'سلبي', 'محايد', 'مختلط']
confusion = pd.crosstab(df['my_label'], df['model_label']).reindex(index=labels_list, columns=labels_list, fill_value=0)
fig2 = px.imshow(confusion, text_auto=True, labels=dict(x="النموذج", y="تصنيفي", color="العدد"), color_continuous_scale="Blues")
st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# 8. البيانات
# ==========================================
st.subheader("جميع البيانات")
st.dataframe(df, use_container_width=True)

# ==========================================
# 9. تحليل الأخطاء
# ==========================================
st.subheader("❌ تحليل الأخطاء")
errors = df[df['my_label'] != df['model_label']]
if not errors.empty:
    st.warning(f"تم العثور على {len(errors)} أخطاء.")
    st.dataframe(errors[['text', 'my_label', 'model_label', 'error_type']], use_container_width=True)
else:
    st.success("🎉 لا توجد أخطاء!")

# ==========================================
# 10. الاتجاه الزمني
# ==========================================
st.subheader("📅 تطور دقة النموذج عبر الزمن")
daily_acc = df.groupby('date').apply(lambda x: (x['my_label'] == x['model_label']).mean()).reset_index(name='accuracy')
if len(daily_acc) > 1:
    fig3 = px.line(daily_acc, x='date', y='accuracy', markers=True, title="Model Accuracy Over Time")
    fig3.update_yaxes(tickformat='.0%', range=[0, 1])
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("نحتاج إلى تاريخين مختلفين على الأقل لعرض الرسم البياني.")

# ==========================================
# 11. التحليل الإحصائي المتقدم
# ==========================================
st.markdown("---")
st.header("🧮 التحليل الإحصائي المتقدم")
stats = calculate_statistics(df)
col_stat1, col_stat2 = st.columns(2)
with col_stat1:
    st.subheader("🎯 فترات الثقة (95%)")
    st.metric("دقة النموذج", f"{stats['accuracy']:.1%}")
    st.info(f"نحن واثقون بنسبة 95% أن الدقة الحقيقية تقع بين **{stats['ci_low']:.1%}** و **{stats['ci_high']:.1%}**.")
with col_stat2:
    st.subheader("⚖️ اختبار التحيز (Chi-Square)")
    st.metric("P-Value", f"{stats['p_value']:.4f}")
    if stats['p_value'] < 0.05:
        st.error("⚠️ النموذج **منحاز إحصائياً**.")
    else:
        st.success("✅ لا يوجد انحياز إحصائي دال.")
st.subheader("📊 مقارنة التوزيعات (أنت vs النموذج)")
fig_bias = px.bar(stats['bias_data'], barmode='group', labels={'value': 'النسبة', 'index': 'التصنيف'}, title="هل يُفرط النموذج في توقع فئة معينة؟")
st.plotly_chart(fig_bias, use_container_width=True)

# ==========================================
# 12. اتفاق الإنسان-الآلة (محاكاة)
# ==========================================
st.markdown("---")
st.header("🤖 اتفاق الإنسان-الآلة (GPT vs You)")
st.write("يقارن هذا القسم بين تصنيفك البشري وتصنيف الذكاء الاصطناعي.")
st.info("ℹ️ يتم عرض نتائج محاكاة لأن مفتاح API غير متوفر حالياً.")

if st.button("🚀 تشغيل المقارنة (محاكاة)"):
    import random
    sample_df = df.head(5).copy()
    labels = ['positive', 'negative', 'neutral', 'mixed']
    sample_df['llm_label'] = [true_label if random.random() < 0.85 else random.choice(labels) for true_label in sample_df['my_label']]
    
    st.success("✅ اكتملت المحاكاة!")
    st.dataframe(sample_df[['text', 'my_label', 'llm_label']], use_container_width=True)
    
    valid_df = sample_df
    agreement_ai = (valid_df['my_label'] == valid_df['llm_label']).mean()
    try:
        kappa_ai = cohen_kappa_score(valid_df['my_label'], valid_df['llm_label'])
    except:
        kappa_ai = 0.0
        
    col1, col2 = st.columns(2)
    col1.metric("نسبة الاتفاق", f"{agreement_ai:.1%}")
    col2.metric("Cohen's Kappa", f"{kappa_ai:.2f}")
# ==========================================
# 13. مراجعة الأقران
# ==========================================
st.markdown("---")
st.header("👥 نظام مراجعة الأقران (محاكاة)")
st.caption("ملاحظة: المقيّمون الآخرون محاكاة (Simulated)، وليسوا بشراً حقيقيين، لذا فإن Fleiss' Kappa هنا هو لأغراض العرض فقط.")
if st.button("👥 تشغيل محاكاة مراجعة الأقران"):
    sim_df, fleiss_k, reliability, labels = simulate_and_evaluate_peers(df)
    st.success("✅ اكتملت المحاكاة!")
    col1, col2 = st.columns(2)
    col1.metric("Fleiss' Kappa", f"{fleiss_k:.2f}")
    rel_df = pd.DataFrame(list(reliability.items()), columns=['المُقيّم', 'نسبة الدقة'])
    st.dataframe(rel_df, use_container_width=True)
    for annotator, acc in reliability.items():
        if acc < 0.60:
            st.error(f"⚠️ المُقيّم **{annotator}** غير موثوق!")
    st.subheader("📋 عينة من تصنيفات المُقيّمين")
    st.dataframe(sim_df[['item_id', 'Annotator_1 (You)', 'Annotator_2', 'Annotator_5']].head(), use_container_width=True)

# ==========================================
# 14. مقارنة النماذج (McNemar Test)
# ==========================================
st.markdown("---")
st.header("⚔️ مقارنة النماذج (McNemar Test)")
st.write("يقارن هذا القسم بين أداء نموذجين باستخدام **اختبار McNemar** لمعرفة إن كان الفرق بينهما دالًا إحصائيًا أم مجرد صدفة.")
if st.button("⚔️ تشغيل مقارنة النماذج"):
    if 'llm_label' not in df.columns:
        st.info("ℹ️ لم يتم تشغيل الذكاء الاصطناعي بعد. سيتم محاكاة نموذج ذكاء اصطناعي (دقة 85%) لمقارنته معك.")
        import random
        labels_list = ['positive', 'negative', 'neutral', 'mixed']
        df['llm_label'] = [true_label if random.random() < 0.85 else random.choice(labels_list) for true_label in df['my_label']]
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

# ==========================================
# 15. تحليل الجوانب (ABSA)
# ==========================================
st.markdown("---")
st.header("🎯 تحليل الجوانب (ABSA) - متقدم")
st.write("يقوم هذا القسم بتفكيك النص إلى جوانب (مثل: الطعام، الخدمة) وتحديد مشاعر كل جانب على حدة.")
absa_text = st.text_input("أدخل نصاً لتحليله (مثال: Food is good but service is bad):", key="absa_input")
if st.button("🔍 تحليل الجوانب"):
    if not api_key:
        st.warning("الرجاء إدخال مفتاح Google API في القسم السابق أولاً.")
    elif not absa_text:
        st.warning("الرجاء إدخال نص.")
    else:
        with st.spinner("جاري تحليل الجوانب..."):
            aspects_result = extract_aspects_with_llm(absa_text, api_key)
            if isinstance(aspects_result, list) and len(aspects_result) > 0 and "error" not in aspects_result[0]:
                st.success("✅ اكتمل التحليل!")
                for item in aspects_result:
                    if "aspect" in item and "sentiment" in item:
                        if item['sentiment'].lower() == 'positive':
                            st.success(f"**{item['aspect']}**: {item['sentiment']} ✅")
                        elif item['sentiment'].lower() == 'negative':
                            st.error(f"**{item['aspect']}**: {item['sentiment']} ❌")
                        else:
                            st.info(f"**{item['aspect']}**: {item['sentiment']} ⚖️")
            else:
                st.error(f"حدث خطأ: {aspects_result}")
