"""
Digital Marketing Attribution Modeling Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# =========================
# DATA GENERATION
# =========================
class AttributionDataGenerator:
    def __init__(self, n_customers=1000, n_days=90):
        self.n_customers = n_customers
        self.n_days = n_days

    def generate(self):
        channels = ['Email', 'Social Media', 'Display Ads', 'Search', 'Organic', 'Direct']
        data = []

        for cid in range(self.n_customers):
            n_touch = np.random.randint(2, 8)
            days = sorted(np.random.choice(self.n_days, n_touch, replace=False))

            for i, d in enumerate(days):
                channel = np.random.choice(channels)
                data.append({
                    'customer_id': cid,
                    'touch': i + 1,
                    'channel': channel,
                    'day': d,
                    'engagement': np.random.uniform(10, 100),
                    'clicked': np.random.randint(0, 2),
                    'cost': np.random.uniform(1, 5)
                })

        df = pd.DataFrame(data)

        conversions = []
        for cid in df['customer_id'].unique():
            cdata = df[df['customer_id'] == cid]
            score = (cdata['engagement'].mean() / 100) + (cdata['clicked'].sum() / 5)
            converted = np.random.random() < score
            revenue = np.random.uniform(50, 500) if converted else 0

            conversions.append({
                'customer_id': cid,
                'converted': int(converted),
                'revenue': revenue
            })

        conv_df = pd.DataFrame(conversions)
        return df.merge(conv_df, on='customer_id')


# =========================
# ATTRIBUTION MODELS
# =========================
class AttributionModel:
    def __init__(self, data):
        self.data = data

    def allocate(self, df):
        raise NotImplementedError

    def run(self):
        results = []

        for cid in self.data['customer_id'].unique():
            cdata = self.data[self.data['customer_id'] == cid].sort_values('day')
            revenue = cdata['revenue'].iloc[0]
            weights = self.allocate(cdata)

            if revenue > 0:
                weights = weights * (revenue / weights.sum())

            for i, row in cdata.iterrows():
                results.append({
                    'channel': row['channel'],
                    'allocated_revenue': weights[row.name - cdata.index[0]],
                    'cost': row['cost']
                })

        return pd.DataFrame(results)


class FirstTouch(AttributionModel):
    def allocate(self, df):
        w = np.zeros(len(df))
        w[0] = 1
        return w


class LastTouch(AttributionModel):
    def allocate(self, df):
        w = np.zeros(len(df))
        w[-1] = 1
        return w


class Linear(AttributionModel):
    def allocate(self, df):
        return np.ones(len(df)) / len(df)


class TimeDecay(AttributionModel):
    def allocate(self, df):
        n = len(df)
        weights = np.array([0.5 ** (n - i - 1) for i in range(n)])
        return weights / weights.sum()


class MLAttribution(AttributionModel):
    def fit(self):
        X, y = [], []

        for cid in self.data['customer_id'].unique():
            cdata = self.data[self.data['customer_id'] == cid]
            X.append([len(cdata), cdata['engagement'].mean(), cdata['clicked'].sum()])
            y.append(cdata['converted'].iloc[0])

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.model = GradientBoostingRegressor()
        self.model.fit(X_train, y_train)

        pred = self.model.predict(X_test)
        print("ML R2:", r2_score(y_test, pred))

    def allocate(self, df):
        weights = df['engagement'].values
        return weights / weights.sum()


# =========================
# MAIN
# =========================
def main():
    print("Generating data...")
    data = AttributionDataGenerator().generate()

    models = {
        "First": FirstTouch(data),
        "Last": LastTouch(data),
        "Linear": Linear(data),
        "Decay": TimeDecay(data)
    }

    results = {}

    for name, model in models.items():
        print(f"Running {name}")
        results[name] = model.run()

    ml = MLAttribution(data)
    ml.fit()
    results["ML"] = ml.run()

    print("\nSummary:")
    for name, df in results.items():
        print(name)
        print(df.groupby('channel')['allocated_revenue'].sum())

    # Save outputs
    pd.concat(results.values()).to_csv("results.csv")

    plt.figure()
    results["Linear"].groupby('channel')['allocated_revenue'].sum().plot(kind='bar')
    plt.title("Linear Attribution")
    plt.savefig("attribution.png")

    # Excel export (fixed indentation)
    summary_stats = {
        'Linear': results["Linear"].groupby('channel')['allocated_revenue'].sum()
    }

    with pd.ExcelWriter('summary.xlsx', engine='openpyxl') as writer:
        for name, df in summary_stats.items():
            df.to_excel(writer, sheet_name=name)


if __name__ == "__main__":
    main()