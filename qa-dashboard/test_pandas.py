import pandas as pd

df = pd.read_csv('annotations.csv')
print("أول 5 صفوف:")
print(df.head())
print(f"\nعدد الصفوف: {len(df)}")
print(f"عدد الأعمدة: {len(df.columns)}")

agreement = (df['my_label'] == df['my_label_2']).mean()
print(f"\nنسبة الاتفاق الذاتي: {agreement:.2%}")