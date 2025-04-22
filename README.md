# KPI Standardizer

A system for harmonizing KPI definitions across different teams using LangChain agents and LLMs.

## Overview

The KPI Standardizer is a specialized system that helps organizations standardize their performance metrics across different teams and departments. It uses LLM-powered agents to analyze team-specific KPI definitions, identify similarities, create standardized definitions, and translate these standards back into team-friendly language.

This system solves a common organizational challenge: different teams often measure similar concepts using different metrics, definitions, and calculation methods, making it difficult to compare performance across the organization.

## System Architecture

The system consists of four main components:

1. **Knowledge Base**: A persistent storage system using ChromaDB that accumulates KPI definitions as more input is processed
2. **Analyzer Agent**: Identifies patterns, similarities, and categories in KPI definitions
3. **Standardizer Agent**: Creates unified definitions for related metrics
4. **Translator Agent**: Adapts standardized metrics to team-specific contexts

## Features

- **Semantic Similarity Detection**: Identifies KPIs that measure similar concepts even when they use different terminology
- **Unified Definition Creation**: Generates clear, precise standard definitions for metrics
- **Team-Specific Translation**: Adapts standardized metrics to team contexts while maintaining consistency
- **Persistent Knowledge**: Maintains a growing database of KPI definitions for continuous improvement
- **Transparent Processing**: Provides visibility into the analysis and standardization process

## Installation

1. Clone this repository:
   ```
   git clone [repository-url]
   cd KPI_Standardizer
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file

## Usage

### Basic Usage

1. Prepare a CSV file with your KPI definitions following this format:
   ```
   team,kpi_name,definition
   Sales,Monthly Revenue,Total monetary value of sales made in a calendar month
   Marketing,CAC,Total marketing spend divided by number of new customers acquired
   ```

2. Run the main script:
   ```
   python main.py
   ```

3. Follow the prompts to:
   - Enter the path to your CSV file
   - Specify a team name (if not included in the CSV)
   - Enter a path to save the results

### Example Script

For a quick demonstration, run the example script:
```
python run_example.py
```

This script uses the included sample data to demonstrate the full standardization pipeline.

## Input Format

Your input CSV must include the following columns:
- `kpi_name`: The name of the KPI
- `definition`: A text description of what the KPI measures and how it's calculated
- `team`: The team or department that uses this KPI (optional if providing team name via command line)

Additional columns will be preserved in the output.

## Output Format

The system generates a CSV file with the following information:
- All original KPI data
- Analysis results for each KPI
- Standardized definitions for groups of related KPIs
- Team-specific translations of standardized KPIs

## Example

Input:
```
team,kpi_name,definition
Sales,Customer Acquisition Cost,Total sales and marketing spend divided by number of new customers acquired
Marketing,Cost Per Acquisition,Total marketing spend divided by number of customers acquired in the period
```

Output:
- Analysis of KPI components (metric type, time period, objects measured)
- Identification of semantic similarity between the two metrics
- Creation of a standardized "Customer Acquisition Cost" definition
- Team-specific translations for Sales and Marketing with appropriate terminology

## Requirements

- Python 3.8+
- LangChain
- pandas
- chromadb
- An OpenAI API key

## License

[Include license information here]

## Contributing

[Include contribution guidelines here]