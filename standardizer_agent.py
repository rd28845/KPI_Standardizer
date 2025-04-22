"""Standardizer Agent module for the KPI Standardizer system.

This agent creates unified definitions for related metrics.
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

class StandardizerAgent:
    """Agent for standardizing KPI definitions."""
    
    def __init__(self, knowledge_base, model_name=None):
        """Initialize the standardizer agent.
        
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
            self.create_standard_definition,
            self.harmonize_calculation_method,
            self.suggest_naming_convention
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a KPI Standardization Expert whose goal is to create unified, clear, and consistent KPI definitions.
                When standardizing KPIs across teams:
                1. Create clear, unambiguous definitions that capture the essence of what's being measured
                2. Harmonize calculation methods to ensure consistent measurements
                3. Suggest standardized naming conventions
                4. Retain the business context and purpose of the original metrics
                5. Document any important differences that should be preserved from team-specific definitions
            """),
            ("human", "{input}")
        ])
        
        self.agent = create_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    @tool
    def create_standard_definition(self, kpi_definitions):
        """Create a standardized definition from multiple team-specific KPI definitions.
        
        Args:
            kpi_definitions: List of dictionaries, each containing team, kpi_name, and definition
            
        Returns:
            Dictionary with the standardized definition and metadata
        """
        # Format the definitions for the prompt
        definitions_text = ""
        for kpi in kpi_definitions:
            definitions_text += f"Team: {kpi['team']}\n"
            definitions_text += f"KPI Name: {kpi['kpi_name']}\n"
            definitions_text += f"Definition: {kpi['definition']}\n\n"
        
        prompt = f"""Based on these related KPI definitions from different teams:

        {definitions_text}
        
        Create a single standardized definition that:
        1. Captures the core concept measured by all these KPIs
        2. Is clear and unambiguous
        3. Specifies the calculation method precisely
        4. Includes necessary time periods and context
        5. Accommodates the needs of all teams
        
        Return a JSON object with:
        - standard_name: A proposed standard name for this KPI
        - standard_definition: The unified definition
        - calculation_method: Specific formula or method for calculating this KPI
        - notes: Any important considerations or team-specific variations to be aware of
        """
        
        # For both models, we need to handle the prediction appropriately
        if "GEMINI_API_KEY" in os.environ:
            response = self.llm.invoke(prompt).content
        else:
            response = self.llm.predict(prompt)
        return response
    
    @tool
    def harmonize_calculation_method(self, calculation_methods):
        """Harmonize different calculation methods into a standard approach.
        
        Args:
            calculation_methods: List of dictionaries with team and method
            
        Returns:
            Dictionary with the standardized calculation method
        """
        # Format the methods for the prompt
        methods_text = ""
        for method in calculation_methods:
            methods_text += f"Team: {method['team']}\n"
            methods_text += f"Method: {method['method']}\n\n"
        
        prompt = f"""Review these different calculation methods for what appears to be the same KPI:

        {methods_text}
        
        Create a harmonized calculation method that:
        1. Is mathematically sound
        2. Captures the intent of all the team-specific methods
        3. Is clear and implementable
        4. Notes any potential data or implementation challenges
        
        Return a JSON object with:
        - standard_calculation: The recommended calculation formula or method
        - rationale: Why this approach was chosen
        - implementation_notes: Any notes on implementing this calculation
        """
        
        # For both models, we need to handle the prediction appropriately
        if "GEMINI_API_KEY" in os.environ:
            response = self.llm.invoke(prompt).content
        else:
            response = self.llm.predict(prompt)
        return response
    
    @tool
    def suggest_naming_convention(self, kpi_names):
        """Suggest a standardized naming convention based on existing KPI names.
        
        Args:
            kpi_names: List of dictionaries with team and kpi_name
            
        Returns:
            Dictionary with naming convention suggestions
        """
        # Format the names for the prompt
        names_text = ""
        for item in kpi_names:
            names_text += f"Team: {item['team']}\n"
            names_text += f"KPI Name: {item['kpi_name']}\n\n"
        
        prompt = f"""Based on these different names for what appears to be the same KPI:

        {names_text}
        
        Suggest a standardized naming convention that:
        1. Is clear and descriptive
        2. Follows best practices for KPI naming
        3. Balances brevity with clarity
        4. Could be applied consistently across the organization
        
        Return a JSON object with:
        - standard_name: The recommended standard name
        - naming_pattern: A general pattern that could be applied to other KPIs
        - rationale: Why this naming approach was chosen
        """
        
        # For both models, we need to handle the prediction appropriately
        if "GEMINI_API_KEY" in os.environ:
            response = self.llm.invoke(prompt).content
        else:
            response = self.llm.predict(prompt)
        return response
    
    def standardize_kpi_group(self, related_kpis):
        """Standardize a group of related KPIs.
        
        Args:
            related_kpis: DataFrame with KPIs that have been determined to be related
            
        Returns:
            Dictionary with standardized definition and metadata
        """
        kpi_list = related_kpis.to_dict(orient="records")
        
        input_text = f"""Create a standardized definition for this group of related KPIs:
        
        {kpi_list}
        
        These KPIs appear to be measuring the same concept across different teams.
        Create a unified definition that all teams can adopt.
        """
        
        return self.agent_executor.invoke({"input": input_text})
    
    def standardize_all_kpis(self, analyzed_kpis_df, similarity_threshold=0.85):
        """Standardize all KPIs by grouping similar ones and creating unified definitions.
        
        Args:
            analyzed_kpis_df: DataFrame with KPIs and their analysis
            similarity_threshold: Threshold for considering KPIs as related
            
        Returns:
            DataFrame with standardized definitions
        """
        # This is a simplified approach - in a real implementation, we would use the analysis
        # to group similar KPIs more effectively
        
        # For now, we'll create a simplified grouping based on similarity searches
        standardized_kpis = []
        processed_indices = set()
        
        for idx, row in analyzed_kpis_df.iterrows():
            if idx in processed_indices:
                continue
                
            # Find similar KPIs using the knowledge base
            similar_results = self.knowledge_base.search_similar_kpis(row["definition"])
            
            # Filter based on similarity threshold
            related_indices = [idx]
            for doc, score in similar_results:
                if score >= similarity_threshold:
                    # Find the index in our DataFrame
                    for i, r in analyzed_kpis_df.iterrows():
                        if (r["team"] == doc.metadata["team"] and 
                            r["kpi_name"] == doc.metadata["kpi_name"]):
                            related_indices.append(i)
                            break
            
            # Create a group of related KPIs
            related_kpis = analyzed_kpis_df.iloc[related_indices]
            
            # Standardize this group
            standardized_definition = self.standardize_kpi_group(related_kpis)
            
            # Add to results
            for _, related_row in related_kpis.iterrows():
                result = related_row.to_dict()
                result["standardized_definition"] = standardized_definition
                standardized_kpis.append(result)
                processed_indices.add(_)
        
        return pd.DataFrame(standardized_kpis)