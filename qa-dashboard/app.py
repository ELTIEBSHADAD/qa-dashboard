import streamlit as st
import pandas as pd
from alert_engine import check_alerts 
df = pd.read_csv('annotations.csv')

import plotly.express as px
from sklearn.metrics import cohen_kappa_score

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