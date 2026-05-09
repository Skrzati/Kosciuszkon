import pandas as pd
df = pd.read_csv('blue_shield_meteo_data.csv')
print(f"Prawdziwa liczba wierszy w pliku to: {len(df)}")