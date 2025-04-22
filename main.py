"""Main module for the KPI Standardizer system.

This module provides the entry point for the KPI Standardizer system.
"""

import os
import pandas as pd
from dotenv import load_dotenv

from knowledge_base import KnowledgeBase
from analyzer_agent import AnalyzerAgent
from standardizer_agent import StandardizerAgent
from translator_agent import TranslatorAgent

# Load environment variables from .env file
load_dotenv()

# Ensure API key is set (either Gemini or OpenAI)
if "GEMINI_API_KEY" not in os.environ and "OPENAI_API_KEY" not in os.environ:
    raise ValueError("No API key found. Please set either GEMINI_API_KEY or OPENAI_API_KEY in a .env file.")

# Set model provider based on available keys
USE_GEMINI = "GEMINI_API_KEY" in os.environ
MODEL_PROVIDER = "gemini" if USE_GEMINI else "openai"

# Configure Google Gemini if using it
if USE_GEMINI:
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class KPIStandardizer:
    """Main KPI Standardizer system class."""
    
    def __init__(self, db_path=None):
        """Initialize the KPI Standardizer system.
        
        Args:
            db_path: Path to the ChromaDB directory (optional)
        """
        # Initialize components
        self.knowledge_base = KnowledgeBase(persist_directory=db_path)
        self.analyzer = AnalyzerAgent(self.knowledge_base)
        self.standardizer = StandardizerAgent(self.knowledge_base)
        self.translator = TranslatorAgent()
        
        print("KPI Standardizer system initialized successfully.")
    
    def process_csv_file(self, file_path, team_name=None):
        """Process a CSV file with KPI definitions.
        
        Args:
            file_path: Path to the CSV file
            team_name: Optional team name (if not included in the CSV)
            
        Returns:
            DataFrame with processed KPIs
        """
        # Read the CSV file
        try:
            kpis_df = pd.read_csv(file_path)
            print(f"Loaded {len(kpis_df)} KPIs from {file_path}")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None
        
        # Check required columns
        required_columns = ["kpi_name", "definition"]
        if team_name is None:
            required_columns.append("team")
        
        missing_columns = [col for col in required_columns if col not in kpis_df.columns]
        if missing_columns:
            print(f"CSV file is missing required columns: {missing_columns}")
            return None
        
        # Add team name if provided
        if team_name is not None and "team" not in kpis_df.columns:
            kpis_df["team"] = team_name
        
        # Add KPIs to knowledge base
        self.knowledge_base.add_kpis(kpis_df)
        print(f"Added KPIs to knowledge base")
        
        return kpis_df
    
    def analyze_kpis(self, kpis_df):
        """Analyze KPIs to identify patterns and similarities.
        
        Args:
            kpis_df: DataFrame with KPIs
            
        Returns:
            DataFrame with analysis results
        """
        print("Analyzing KPIs...")
        analyzed_kpis = self.analyzer.analyze_kpi_set(kpis_df)
        print(f"Analysis complete for {len(analyzed_kpis)} KPIs")
        return analyzed_kpis
    
    def standardize_kpis(self, analyzed_kpis_df):
        """Create standardized definitions for KPIs.
        
        Args:
            analyzed_kpis_df: DataFrame with analyzed KPIs
            
        Returns:
            DataFrame with standardized definitions
        """
        print("Standardizing KPIs...")
        standardized_kpis = self.standardizer.standardize_all_kpis(analyzed_kpis_df)
        print(f"Standardization complete for {len(standardized_kpis)} KPIs")
        return standardized_kpis
    
    def translate_kpis(self, standardized_kpis_df):
        """Translate standardized KPIs back to team-specific language.
        
        Args:
            standardized_kpis_df: DataFrame with standardized KPIs
            
        Returns:
            DataFrame with team-specific translations
        """
        print("Translating KPIs for teams...")
        translated_kpis = self.translator.translate_all_kpis(standardized_kpis_df)
        print(f"Translation complete for {len(translated_kpis)} KPIs")
        return translated_kpis
    
    def process_kpis(self, kpis_df):
        """Process KPIs through the full standardization pipeline.
        
        Args:
            kpis_df: DataFrame with KPIs
            
        Returns:
            DataFrame with processed KPIs
        """
        # Step 1: Analyze the KPIs
        analyzed_kpis = self.analyze_kpis(kpis_df)
        
        # Step 2: Standardize the KPIs
        standardized_kpis = self.standardize_kpis(analyzed_kpis)
        
        # Step 3: Translate the standardized KPIs back to team-specific language
        translated_kpis = self.translate_kpis(standardized_kpis)
        
        return translated_kpis
    
    def save_results(self, results_df, output_path):
        """Save results to a CSV file.
        
        Args:
            results_df: DataFrame with results
            output_path: Path to save the results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving results: {e}")
            return False

def main():
    """Main entry point for the KPI Standardizer system."""
    print("KPI Standardizer System")
    print("=======================")
    
    # Initialize the system
    standardizer = KPIStandardizer()
    
    # Example usage - process a CSV file
    file_path = input("Enter path to CSV file with KPI definitions: ")
    team_name = input("Enter team name (leave blank if included in CSV): ").strip() or None
    
    # Process the CSV file
    kpis_df = standardizer.process_csv_file(file_path, team_name)
    if kpis_df is None:
        print("Failed to process CSV file. Exiting.")
        return
    
    # Process the KPIs
    results = standardizer.process_kpis(kpis_df)
    
    # Save the results
    output_path = input("Enter path to save results: ")
    standardizer.save_results(results, output_path)
    
    print("\nKPI standardization process complete!")

if __name__ == "__main__":
    main()