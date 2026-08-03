import pandas as pd

from inference import HybridModel

# Load the model once, then predict on the held-out test set.
model = HybridModel().load()
df = pd.read_csv("feature_extract/bank_nifty_test.csv")

print(model.predict(df))
