import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.proportion import proportion_confint

from alert_engine import check_alerts
from peer_review_engine import simulate_and_evaluate_peers
from benchmark_engine import benchmark_models

MIN_N = 30  # below this, significance tests are not meaningful

st.set_page_config(page_title="لوحة الجودة", page_icon="📊", layout="wide")
st.title("📊 لوحة مراقبة جودة التعليق")


@st.cache_data
def load_data():
    data = pd.read_csv("annotations.csv")
    data["date"] = pd.to_datetime(data["date"])
    return data


df = load_data()
n = len(df)

# All labels that appear in any column, so every chart uses the same classes
LABELS = sorted(
    set(df["my_label"]) | set(df["my_label_2"]) | set(df["model_label"])
)

# ============ التنبيهات ============
for alert in check_alerts(df):
    (st.error if alert["level"] == "error" else st.warning)(alert["message"])

if n < MIN_N:
    st.info(f"⚠️ حجم العينة صغير ({n} عنصرًا). اعتبر النتائج تجريبية وليست حكمًا على النموذج.")

# ============ المؤشرات ============
agreement = (df["my_label"] == df["my_label_2"]).mean()
self_kappa = cohen_kappa_score(df["my_label"], df["my_label_2"])
accuracy = (df["my_label"] == df["model_label"]).mean()
model_kappa = cohen_kappa_score(df["my_label"], df["model_label"])

stable = df[df["my_label"] == df["my_label_2"]]  # items where my two passes agree
stable_acc = (stable["my_label"] == stable["model_label"]).mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("الاتساق الذاتي", f"{agreement:.1%}")
c2.metric("Kappa (أنا vs أنا)", f"{self_kappa:.2f}")
c3.metric("دقة النموذج", f"{accuracy:.1%}")
c4.metric("Kappa (النموذج vs أنا)", f"{model_kappa:.2f}")
c5.metric("الدقة على العناصر المستقرة", f"{stable_acc:.1%}",
          help=f"{len(stable)} عناصر حيث تطابق تصنيفاك")

# العناصر التي تغيّر فيها تصنيفك
unstable = df[df["my_label"] != df["my_label_2"]]
if not unstable.empty:
    st.warning(f"{len(unstable)} عنصر/عناصر غير مستقرة في التصنيف الذاتي (المعيار نفسه غير حاسم):")
    st.dataframe(unstable[["text", "my_label", "my_label_2", "model_label"]],
                 use_container_width=True)

# ============ رسم دائري ============
st.subheader("توزيع التصنيفات")
label_counts = df["my_label"].value_counts().reset_index()
label_counts.columns = ["التصنيف", "العدد"]
st.plotly_chart(
    px.pie(label_counts, names="التصنيف", values="العدد", title="توزيع التصنيفات اليدوية"),
    use_container_width=True,
)

# ============ مصفوفة الالتباس (مربعة دائمًا) ============
st.subheader("مصفوفة الالتباس (تصنيفي vs النموذج)")
confusion = pd.crosstab(df["my_label"], df["model_label"]).reindex(
    index=LABELS, columns=LABELS, fill_value=0
)
st.plotly_chart(
    px.imshow(confusion, text_auto=True,
              labels=dict(x="النموذج", y="تصنيفي", color="العدد"),
              color_continuous_scale="Blues"),
    use_container_width=True,
)

# ============ البيانات ============
st.subheader("جميع البيانات")
st.dataframe(df, use_container_width=True)

# ============ تحليل الأخطاء ============
st.subheader("❌ تحليل أخطاء النموذج")
errors = df[df["my_label"] != df["model_label"]]
if not errors.empty:
    st.warning(f"تم العثور على {len(errors)} أخطاء.")
    st.dataframe(errors[["text", "my_label", "model_label", "error_type"]],
                 use_container_width=True)
else:
    st.success("🎉 لا توجد أخطاء!")

# ============ الاتجاه الزمني ============
st.subheader("📅 تطور دقة النموذج عبر الزمن")
daily_acc = (
    df.assign(correct=df["my_label"] == df["model_label"])
    .groupby(df["date"].dt.date)["correct"]
    .mean()
    .reset_index(name="accuracy")
    .rename(columns={"date": "day"})
)
if len(daily_acc) < 2:
    st.info("يوجد يوم واحد فقط من البيانات، لذا لا يمكن رسم اتجاه زمني بعد.")
    st.metric(f"الدقة بتاريخ {daily_acc['day'].iloc[0]}", f"{daily_acc['accuracy'].iloc[0]:.1%}")
else:
    fig3 = px.line(daily_acc, x="day", y="accuracy", markers=True,
                   title="Model Accuracy Over Time")
    fig3.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig3, use_container_width=True)

# ============ التحليل الإحصائي ============
st.markdown("---")
st.header("🔬 التحليل الإحصائي")

st.subheader("🎯 فترة الثقة (95%) لدقة النموذج — Wilson")
correct = int((df["my_label"] == df["model_label"]).sum())
low, high = proportion_confint(correct, n, alpha=0.05, method="wilson")
st.metric("دقة النموذج", f"{correct / n:.1%}")
st.info(f"فترة الثقة 95%: من {low:.1%} إلى {high:.1%} (على {n} عنصرًا)")

st.subheader("⚖️ هل يُفرط النموذج في توقع فئة معينة؟")
human_dist = df["my_label"].value_counts(normalize=True).reindex(LABELS, fill_value=0)
model_dist = df["model_label"].value_counts(normalize=True).reindex(LABELS, fill_value=0)
dist_df = pd.DataFrame({"تصنيفي": human_dist, "النموذج": model_dist}).reset_index()
dist_df = dist_df.rename(columns={"index": "التصنيف", dist_df.columns[0]: "التصنيف"})
st.plotly_chart(
    px.bar(dist_df.melt(id_vars="التصنيف", var_name="المصدر", value_name="النسبة"),
           x="التصنيف", y="النسبة", color="المصدر", barmode="group"),
    use_container_width=True,
)

# Test of homogeneity of the two label distributions (human vs model counts).
# It compares DISTRIBUTIONS. It is not the same as testing whether labels agree.
counts = pd.DataFrame({
    "human": df["my_label"].value_counts().reindex(LABELS, fill_value=0),
    "model": df["model_label"].value_counts().reindex(LABELS, fill_value=0),
})
counts = counts[counts.sum(axis=1) > 0]
if n < MIN_N:
    st.info(f"لا يُجرى اختبار الانحياز: العينة ({n}) أقل من {MIN_N}، والتكرارات المتوقعة صغيرة جدًا.")
else:
    chi2, p, _, expected = chi2_contingency(counts.values)
    if (expected < 5).any():
        st.info("بعض التكرارات المتوقعة أقل من 5؛ نتيجة كاي-تربيع غير موثوقة.")
    else:
        st.metric("P-Value (تجانس التوزيعين)", f"{p:.4f}")
        if p < 0.05:
            st.warning("توزيع تنبؤات النموذج يختلف إحصائيًا عن توزيع تصنيفاتك.")
        else:
            st.success("لا يوجد دليل على اختلاف التوزيعين.")

# ============ مراجعة الأقران (محاكاة) ============
st.markdown("---")
st.header("👥 نظام مراجعة الأقران (محاكاة)")
st.caption("⚠️ المُقيّمون الآخرون بيانات مُولَّدة بالمحاكاة وليسوا أشخاصًا حقيقيين، فلا تعتبر Fleiss' Kappa دليلًا على اتفاق بشري فعلي.")

if st.button("👥 تشغيل محاكاة مراجعة الأقران"):
    sim_df, fleiss_k, reliability, labels = simulate_and_evaluate_peers(df)
    st.success("✅ اكتملت المحاكاة!")
    st.metric("Fleiss' Kappa", f"{fleiss_k:.2f}", help="فوق 0.6 جيد، فوق 0.8 ممتاز")
    rel_df = pd.DataFrame(list(reliability.items()),
                          columns=["المُقيّم", "نسبة الدقة (مقابل المعيار)"])
    st.dataframe(rel_df.style.background_gradient(cmap="RdYlGn"), use_container_width=True)
    for annotator, acc in reliability.items():
        if acc < 0.60:
            st.error(f"⚠️ المُقيّم **{annotator}** غير موثوق! دقته {acc:.1%} فقط.")
    st.dataframe(
        sim_df[["item_id", "Annotator_1 (You)", "Annotator_2", "Annotator_4 (Weak)"]].head(),
        use_container_width=True,
    )

# ============ مقارنة النماذج (McNemar) ============
st.markdown("---")
st.header("⚔️ مقارنة النماذج (McNemar Test)")
st.write("يقارن نموذجين مقابل تصنيفك البشري كمعيار.")

if st.button("⚔️ تشغيل مقارنة النماذج"):
    if "llm_label" in df.columns:
        # FIX: the original passed my_label (the gold standard) as model A, so A
        # scored 100% against itself. Compare the two MODELS against my_label.
        bench_results, error = benchmark_models(df, model_a_col="model_label",
                                                model_b_col="llm_label")
        if error:
            st.warning(error)
        else:
            col1, col2 = st.columns(2)
            col1.metric("دقة النموذج A (model_label)", f"{bench_results['model_a_acc']:.1%}")
            col2.metric("دقة النموذج B (llm_label)", f"{bench_results['model_b_acc']:.1%}")
            st.metric("P-Value", f"{bench_results['p_value']:.4f}")
            if n < MIN_N:
                st.info("العينة صغيرة؛ لا تعتمد على هذه القيمة.")
            elif bench_results["p_value"] < 0.05:
                st.success("الفرق دال إحصائيًا.")
            else:
                st.info("لا يوجد فرق دال إحصائيًا.")
    else:
        st.warning("لا يوجد عمود llm_label بعد. شغّل قسم الذكاء الاصطناعي أولًا.")
