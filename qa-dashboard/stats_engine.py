import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import chi2_contingency

def calculate_statistics(df):
    results = {}
    n = len(df)
    
    # 1. فترات الثقة (Wilson Confidence Intervals)
    correct = (df['my_label'] == df['model_label']).sum()
    ci_low, ci_high = proportion_confint(correct, n, method='wilson')
    results['accuracy'] = correct / n
    results['ci_low'] = ci_low
    results['ci_high'] = ci_high
    
    # 2. كشف التحيز (Bias Detection)
    true_dist = df['my_label'].value_counts(normalize=True)
    pred_dist = df['model_label'].value_counts(normalize=True)
    
    # تجميع التوزيعات في DataFrame
    bias_df = pd.DataFrame({
        'True Label (Human)': true_dist,
        'Predicted (Model)': pred_dist
    }).fillna(0)
    
    results['bias_data'] = bias_df
    
    # 3. اختبار Chi-Square (هل النموذج منحاز إحصائياً؟)
    contingency_table = pd.crosstab(df['my_label'], df['model_label'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    results['chi2'] = chi2
    results['p_value'] = p_value
    
    return results
