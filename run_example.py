"""Example script to demonstrate the KPI Standardizer system.

This script loads the sample data and runs it through the full standardization pipeline.
"""

import os
from dotenv import load_dotenv
from main import KPIStandardizer

# Load environment variables from .env file
load_dotenv()

def run_example():
    """Run an example of the KPI Standardizer system."""
    print("KPI Standardizer Example")
    print("=======================")
    
    # Check if API keys are set
    if "GEMINI_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ:
        print("Error: Neither GEMINI_API_KEY nor OPENAI_API_KEY environment variable is set.")
        print("Please create a .env file using the .env.example template.")
        return
    
    # Show which model provider we're using
    model_provider = "Google Gemini" if "GEMINI_API_KEY" in os.environ else "OpenAI"
    print(f"Using {model_provider} as the model provider")
    
    # Initialize the system
    print("Initializing KPI Standardizer system...")
    standardizer = KPIStandardizer()
    
    # Process the sample CSV file
    sample_file = "sample_data.csv"
    print(f"Processing sample data from {sample_file}...")
    kpis_df = standardizer.process_csv_file(sample_file)
    
    if kpis_df is None:
        print("Failed to process sample data. Exiting.")
        return
    
    # Process the KPIs through the full pipeline
    print("\nRunning KPIs through standardization pipeline...")
    results = standardizer.process_kpis(kpis_df)
    
    # Save the results
    output_file = "standardized_kpis.csv"
    print(f"\nSaving results to {output_file}...")
    standardizer.save_results(results, output_file)
    
    print("\nKPI standardization process complete!")
    print(f"Results saved to {output_file}")
    print("\nExample findings:")
    print("- Similar KPIs detected between Sales and Marketing teams")
    print("- Standard definitions created for customer acquisition metrics")
    print("- Team-specific translations provided for each standardized KPI")

if __name__ == "__main__":
    run_example()