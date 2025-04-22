"""Analyzer Agent module for the KPI Standardizer system.

This agent identifies patterns, similarities, and categories in KPI definitions.
"""

import os
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.agents import AgentExecutor

# Import appropriate model based on configuration
if "GEMINI_API_KEY" in os.environ:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import create_structured_chat_agent as create_agent
else:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.agents import create_openai_functions_agent as create_agent

from knowledge_base import KnowledgeBase

class AnalyzerAgent:
    """Agent for analyzing KPI definitions."""
    
    def __init__(self, knowledge_base, model_name=None):
        """Initialize the analyzer agent.
        
        Args:
            knowledge_base: KnowledgeBase instance
            model_name: LLM model to use (optional)
        """
        self.knowledge_base = knowledge_base
        
        # Choose appropriate model based on available API keys
        if "GEMINI_API_KEY" in os.environ:
            # Default to Gemini Pro for text generation
            if model_name is None:
                model_name = "gemini-pro"
            self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
        else:
            # Default to GPT-4 for OpenAI
            if model_name is None:
                model_name = "gpt-4"
            self.llm = ChatOpenAI(model=model_name, temperature=0.2)
        
        self.tools = [
            self.find_similar_kpis,
            self.categorize_kpi,
            self.identify_key_components
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a KPI Analyzer specialized in understanding metrics across different business teams.
                Your task is to analyze KPI definitions, find patterns, and identify key components.
                You have access to a knowledge base of existing KPI definitions.
                
                When analyzing a KPI:
                1. Find similar existing KPIs across teams
                2. Identify the core business concept being measured
                3. Determine the calculation method (ratio, count, percentage, etc.)
                4. Identify time periods involved (daily, monthly, quarterly, etc.)
                5. Note any qualifiers or conditions in the definition
            """),
            ("human", "{input}")
        ])
        
        self.agent = create_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    @tool
    def find_similar_kpis(self, kpi_definition):
        """Find KPIs similar to the given definition.
        
        Args:
            kpi_definition: The KPI definition text to compare
            
        Returns:
            DataFrame with similar KPIs
        """
        results = self.knowledge_base.search_similar_kpis(kpi_definition)
        
        # Convert results to DataFrame
        similar_kpis = []
        for doc, score in results:
            similar_kpis.append({
                "team": doc.metadata["team"],
                "kpi_name": doc.metadata["kpi_name"],
                "definition": doc.page_content,
                "similarity_score": score
            })
        
        return pd.DataFrame(similar_kpis)
    
    @tool
    def categorize_kpi(self, kpi_definition):
        """Categorize a KPI definition by business domain.
        
        Args:
            kpi_definition: The KPI definition text to categorize
            
        Returns:
            Dictionary with categories and confidence scores
        """
        # This would be implemented with the LLM to determine categories
        # For now we'll use a simple prompt
        prompt = f"""Analyze this KPI definition and categorize it into business domains.
        Return a JSON object with categories as keys and confidence scores (0-1) as values.
        
        KPI Definition: {kpi_definition}
        
        Categories should be chosen from: Financial, Customer, Operations, Marketing, Sales, HR, IT, Product
        """
        
        # For both models, we need to handle the predict method appropriately
        if isinstance(self.llm, ChatGoogleGenerativeAI):
            response = self.llm.invoke(prompt).content
        else:
            response = self.llm.predict(prompt)
        return response
    
    @tool
    def identify_key_components(self, kpi_definition):
        """Identify key components of a KPI definition.
        
        Args:
            kpi_definition: The KPI definition text to analyze
            
        Returns:
            Dictionary with key components
        """
        prompt = f"""Analyze this KPI definition and extract these key components:
        1. Metric type (count, ratio, percentage, average, etc.)
        2. Time period (daily, weekly, monthly, quarterly, yearly)
        3. Business objects being measured
        4. Calculation method
        5. Any qualifiers or conditions
        
        KPI Definition: {kpi_definition}
        
        Return a JSON object with these components as keys.
        """
        
        # For both models, we need to handle the predict method appropriately
        if isinstance(self.llm, ChatGoogleGenerativeAI):
            response = self.llm.invoke(prompt).content
        else:
            response = self.llm.predict(prompt)
        return response
    
    def analyze_kpi(self, team, kpi_name, kpi_definition):
        """Analyze a single KPI definition.
        
        Args:
            team: Team that owns this KPI
            kpi_name: Name of the KPI
            kpi_definition: Text definition of the KPI
            
        Returns:
            Dictionary with analysis results
        """
        input_text = f"""Analyze this KPI definition from the {team} team:
        
        KPI Name: {kpi_name}
        Definition: {kpi_definition}
        
        Provide a comprehensive analysis.
        """
        
        return self.agent_executor.invoke({"input": input_text})
    
    def analyze_kpi_set(self, kpis_df):
        """Analyze a set of KPI definitions.
        
        Args:
            kpis_df: Pandas DataFrame with columns team, kpi_name, definition
            
        Returns:
            DataFrame with the original data and analysis results
        """
        results = []
        
        for _, row in kpis_df.iterrows():
            analysis = self.analyze_kpi(
                row["team"], 
                row["kpi_name"], 
                row["definition"]
            )
            
            result_row = row.to_dict()
            result_row["analysis"] = analysis
            results.append(result_row)
        
        return pd.DataFrame(results)