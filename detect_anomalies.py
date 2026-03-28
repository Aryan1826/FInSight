import pandas as pd
from sklearn.ensemble import IsolationForest

def run_anomaly_detection():
    print("Loading synthetic billing data...")
    try:
        df = pd.read_csv('synthetic_cloud_billing.csv')
    except FileNotFoundError:
        print("Error: Could not find 'synthetic_cloud_billing.csv'. Run Step 1 first!")
        return

    # Ensure Date is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    # We will collect the processed dataframes here
    processed_groups = []

    print("Training Isolation Forest models per service...")
    
    # Group by Provider and Service to train context-specific models
    for (provider, service), group_df in df.groupby(['Provider', 'Service']):
        
        # Initialize the Isolation Forest
        # contamination=0.03 means we expect roughly 3% of the data to be anomalies
        model = IsolationForest(contamination=0.03, random_state=42)
        
        # The model needs the data in a 2D array format
        X = group_df[['Cost']]
        
        # Fit the model and make predictions
        # Isolation Forest outputs 1 for normal, -1 for anomalies
        group_df['Anomaly_Label'] = model.fit_predict(X)
        
        # Get the raw anomaly score (lower negative numbers = more severe anomaly)
        group_df['Severity_Score'] = model.decision_function(X)
        
        # Create a clean boolean column for our Streamlit dashboard
        group_df['Is_Anomaly'] = group_df['Anomaly_Label'] == -1
        
        processed_groups.append(group_df)

    # Recombine all the processed groups back into one dataframe
    final_df = pd.concat(processed_groups).sort_values(by=['Date', 'Provider'])
    
    # Drop the scikit-learn specific label (-1/1) to keep the data clean
    final_df = final_df.drop(columns=['Anomaly_Label'])

    # Save the results
    output_filename = 'anomalies_detected.csv'
    final_df.to_csv(output_filename, index=False)
    
    # Print a summary of what the model found
    total_anomalies = final_df['Is_Anomaly'].sum()
    print(f"\nSuccess! Analyzed {len(final_df)} records.")
    print(f"The ML Engine detected {total_anomalies} anomalies across all clouds.")
    print(f"Results saved to: {output_filename}")
    
    # Show a quick sneak peek of the worst anomalies
    print("\n--- Top 3 Most Severe Anomalies Detected ---")
    worst_anomalies = final_df[final_df['Is_Anomaly']].sort_values(by='Severity_Score').head(3)
    print(worst_anomalies[['Date', 'Provider', 'Service', 'Cost', 'Severity_Score']])

if __name__ == "__main__":
    run_anomaly_detection()