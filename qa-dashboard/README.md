# QA Dashboard - Data Annotation Quality Monitoring

لوحة تفاعلية لمراقبة جودة بيانات التعليق (Data Annotation) تعرض مؤشرات الأداء الرئيسية، مصفوفة الالتباس، وتحليل الأخطاء.

## المميزات
- حساب الاتساق الذاتي (Self-Consistency)
- حساب Cohen's Kappa
- عرض مصفوفة الالتباس (Confusion Matrix)
- تحليل أخطاء النموذج والاتجاه الزمني

## التشغيل المحلي
```bash
pip install -r requirements.txt
streamlit run app.py