import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import cohen_kappa_score
from alert_engine import check_alerts
import os

# ==========================================
# 1. قراءة البيانات (بمسار ديناميكي)
# ==========================================
dir_name = os.path.abspath(os.path.dirname(__file__))
csv_path = os.path.join(dir_name, 'annotations.csv')
df = pd.read_csv(csv_path)

# ==========================================
# 2. إعدادات الصفحة
# ==========================================
st.set_page_config(page_title="QA Dashboard", page_icon="📊", layout="wide")
st.title("📊 QA Dashboard - Quality Monitoring")

# ==========================================
# 3. التنبيهات
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
# 4. المؤشرات (KPIs)
# ==========================================
agreement = (df['my_label'] == df['my_label_2']).mean()
accuracy = (df['my_label'] == df['model_label']).mean()
kappa = cohen_kappa_score(df['my_label'], df['my_label_2'])

col1, col2, col3 = st.columns(3)
col1.metric("الاتساق الذاتي", f"{agreement:.1%}")
col2.metric("دقة النموذج", f"{accuracy:.1%}")
col3.metric("Cohen's Kappa", f"{kappa:.2f}")

# ==========================================
# 5. الرسم الدائري (توزيع التصنيفات)
# ==========================================
st.subheader("توزيع التصنيفات")
label_counts = df['my_label'].value_counts().reset_index()
label_counts.columns = ["التصنيف", "العدد"]
fig = px.pie(label_counts, names="التصنيف", values="العدد", title="توزيع التصنيفات اليدوية")
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. مصفوفة الالتباس (Confusion Matrix)
# ==========================================
st.subheader("مصفوفة الالتباس (تصنيفي vs النموذج)")
confusion = pd.crosstab(df['my_label'], df['model_label'])
fig2 = px.imshow(confusion, 
                 text_auto=True, 
                 labels=dict(x="النموذج", y="تصنيفي", color="العدد"),
                 color_continuous_scale="Blues")
st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# 7. البيانات الكاملة
# ==========================================
st.subheader("جميع البيانات")
st.dataframe(df, use_container_width=True)

# ==========================================
# 8. تحليل الأخطاء
# ==========================================
st.subheader("❌ تحليل الأخطاء")
errors = df[df['my_label'] != df['model_label']]

if not errors.empty:
    st.warning(f"تم العثور على {len(errors)} أخطاء.")
    st.dataframe(errors[['text', 'my_label', 'model_label', 'error_type']], use_container_width=True)
else:
    st.success("🎉 لا توجد أخطاء! النموذج متطابق تمامًا مع تصنيفك.")

# ==========================================
# 9. الاتجاه الزمني
# ==========================================
st.subheader("📅 تطور دقة النموذج عبر الزمن")
df['date'] = pd.to_datetime(df['date'])

daily_acc = df.groupby('date').apply(
    lambda x: (x['my_label'] == x['model_label']).mean()
).reset_index(name='accuracy')

fig3 = px.line(daily_acc, x='date', y='accuracy', markers=True, title="Model Accuracy Over Time")
fig3.update_yaxes(tickformat='.0%')
st.plotly_chart(fig3, use_container_width=True)
