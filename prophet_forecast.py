import pandas as pd
from prophet import Prophet
import warnings

# Suppress prophet's non-critical warnings to keep the terminal clean
warnings.filterwarnings('ignore')

def run_forecasting():
    print("Loading synthetic billing data for forecasting...")
    try:
        df = pd.read_csv('synthetic_cloud_billing.csv')
    except FileNotFoundError:
        print("Error: Could not find 'synthetic_cloud_billing.csv'. Run Step 1 first!")
        return

    # Prophet STRICTLY requires the date column to be named 'ds' and the target variable 'y'
    df = df.rename(columns={'Date': 'ds', 'Cost': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    all_forecasts = []

    print("Training Prophet models and forecasting 90 days into the future...")
    
    # We train a separate forecasting model for each cloud service
    for (provider, service), group_df in df.groupby(['Provider', 'Service']):
        print(f"  -> Forecasting for {provider} - {service}...")
        
        # Initialize Prophet with a 90% confidence interval (approximating P5 and P95/P10 and P90)
        # We also tell it to pay attention to weekly seasonality (e.g., weekends are cheaper)
        model = Prophet(interval_width=0.80, daily_seasonality=False, yearly_seasonality=False)
        
        # Fit the model to our historical data
        model.fit(group_df[['ds', 'y']])
        
        # Ask Prophet to create a dataframe that extends 90 days into the future
        future_dates = model.make_future_dataframe(periods=90)
        
        # Make the prediction
        forecast = model.predict(future_dates)
        
        # We only care about the date (ds), the prediction (yhat), and the lower/upper bounds
        clean_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        clean_forecast['Provider'] = provider
        clean_forecast['Service'] = service
        
        # Rename columns to make them business-friendly for our dashboard
        clean_forecast = clean_forecast.rename(columns={
            'ds': 'Date',
            'yhat': 'Forecasted_Cost_P50',
            'yhat_lower': 'Best_Case_Cost_P10',
            'yhat_upper': 'Worst_Case_Cost_P90'
        })
        
        # Ensure we don't predict negative costs
        clean_forecast['Forecasted_Cost_P50'] = clean_forecast['Forecasted_Cost_P50'].clip(lower=0)
        clean_forecast['Best_Case_Cost_P10'] = clean_forecast['Best_Case_Cost_P10'].clip(lower=0)
        clean_forecast['Worst_Case_Cost_P90'] = clean_forecast['Worst_Case_Cost_P90'].clip(lower=0)

        all_forecasts.append(clean_forecast)

    # Combine all forecasts
    final_forecast_df = pd.concat(all_forecasts).sort_values(by=['Date', 'Provider'])

    # Save to CSV
    output_filename = 'forecast_results.csv'
    final_forecast_df.to_csv(output_filename, index=False)
    
    print("\nSuccess! Forecast generated.")
    print(f"Results saved to: {output_filename}")
    print("Your data now contains 7-day, 30-day, and 90-day future projections.")

if __name__ == "__main__":
    run_forecasting()