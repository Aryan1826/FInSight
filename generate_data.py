import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_cloud_billing_data():
    # Set up the timeline (last 180 days)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    
    # Define our cloud providers and their standard daily baseline costs
    services = [
        {'provider': 'AWS', 'service': 'Compute (EC2)', 'baseline': 500, 'volatility': 50},
        {'provider': 'AWS', 'service': 'Storage (S3)', 'baseline': 150, 'volatility': 10},
        {'provider': 'GCP', 'service': 'BigQuery', 'baseline': 300, 'volatility': 80},
        {'provider': 'Azure', 'service': 'Networking', 'baseline': 200, 'volatility': 30}
    ]

    # Generate normal daily spend
    for date in dates:
        for s in services:
            # Add some random noise to the baseline
            daily_cost = max(0, np.random.normal(s['baseline'], s['volatility']))
            
            # Add a weekly seasonality effect (weekends are cheaper)
            if date.weekday() >= 5: 
                daily_cost *= 0.7 
                
            data.append({
                'Date': date,
                'Provider': s['provider'],
                'Service': s['service'],
                'Cost': round(daily_cost, 2)
            })

    df = pd.DataFrame(data)

    # --- INJECT ANOMALIES ---
    # We will simulate a massive EC2 Auto-Scaling failure 30 days ago
    anomaly_date = end_date - timedelta(days=30)
    
    # Find the index for AWS Compute on that date and spike the cost by 400%
    anomaly_mask = (df['Date'] == anomaly_date) & (df['Service'] == 'Compute (EC2)')
    df.loc[anomaly_mask, 'Cost'] = df.loc[anomaly_mask, 'Cost'] * 4.5 
    
    # Inject a slow creep drift in GCP BigQuery over the last 15 days
    creep_start = end_date - timedelta(days=15)
    creep_mask = (df['Date'] >= creep_start) & (df['Service'] == 'BigQuery')
    # Gradually increase cost by 15% every day for the last 15 days
    df.loc[creep_mask, 'Cost'] = df.loc[creep_mask, 'Cost'] * np.linspace(1.1, 2.5, sum(creep_mask))

    # Save to CSV
    filename = 'synthetic_cloud_billing.csv'
    df.to_csv(filename, index=False)
    print(f"Success! Generated {len(df)} rows of multi-cloud billing data.")
    print(f"File saved as: {filename}")
    print("Anomalies injected on AWS Compute (Sudden Spike) and GCP BigQuery (Gradual Drift).")

if __name__ == "__main__":
    generate_cloud_billing_data()