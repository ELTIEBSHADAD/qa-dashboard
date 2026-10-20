from sklearn.metrics import cohen_kappa_score

def check_alerts(df):
    """
    هذه الدالة تفحص البيانات وتُرجع قائمة بالتنبيهات.
    """
    alerts = []
    
    # 1. حساب المؤشرات
    agreement = (df['my_label'] == df['my_label_2']).mean()
    kappa = cohen_kappa_score(df['my_label'], df['my_label_2'])
    accuracy = (df['my_label'] == df['model_label']).mean()

    # 2. القاعدة الأولى: إذا كان Kappa أقل من 0.80
    if kappa < 0.80:
        alerts.append({
            "level": "error",
            "message": f"⚠️ Cohen's Kappa منخفض ({kappa:.2f}). يجب مراجعة إرشادات التصنيف."
        })

    # 3. القاعدة الثانية: إذا كان الاتساق الذاتي أقل من 90%
    if agreement < 0.90:
        alerts.append({
            "level": "warning",
            "message": f"⚠️ الاتساق الذاتي أقل من 90% ({agreement:.1%}). قد تكون هناك حالات غامضة."
        })

    # 4. القاعدة الثالثة: إذا كانت دقة النموذج أقل من 85%
    if accuracy < 0.85:
        alerts.append({
            "level": "warning",
            "message": f"⚠️ دقة النموذج منخفضة ({accuracy:.1%}). يحتاج النموذج إلى تدريب إضافي."
        })

    # إذا لم تتحقق أي قاعدة، نرجع قائمة فارغة
    return alerts