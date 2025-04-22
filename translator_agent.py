"""Translator Agent module for the KPI Standardizer system.

This agent adapts standardized metrics to team-specific contexts.
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

class TranslatorAgent:
    """Agent for translating standardized KPIs to team-specific language."""
    
    def __init__(self, model_name=None):
        """Initialize the translator agent.
        
        Args:
            model_name: LLM model to use (optional)
        """
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
            self.create_team_specific_version,
            self.generate_implementation_guide,
            self.compare_with_original
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a KPI Translation Expert who helps bridge the gap between standardized KPI definitions
                and team-specific implementations. Your job is to:
                
                1. Translate standardized KPI definitions into team-specific language and context
                2. Generate implementation guides for teams to adopt standardized KPIs
                3. Highlight differences between standardized definitions and original team definitions
                4. Ensure teams understand how to correctly implement the standardized KPIs
                5. Maintain the integrity of the standardized definition while making it accessible to each team
            """),
            ("human", "{input}")
        ])
        
        self.agent = create_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    @tool
    def create_team_specific_version(self, standardized_kpi, team_context):
        """Create a team-specific version of a standardized KPI.
        
        Args:
            standardized_kpi: Dictionary with standard_name, standard_definition, calculation_method
            team_context: Dictionary with team, original_kpi_name, original_definition, team_vocabulary
            
        Returns:
            Dictionary with the team-specific version
        """
        prompt = f"""Translate this standardized KPI definition into language and context appropriate for the {team_context['team']} team.

        Standardized KPI:
        - Name: {standardized_kpi['standard_name']}
        - Definition: {standardized_kpi['standard_definition']}
        - Calculation Method: {standardized_kpi['calculation_method']}
        
        Team Context:
        - Team: {team_context['team']}
        - Original KPI Name: {team_context['original_kpi_name']}
        - Original Definition: {team_context['original_definition']}
        - Team Vocabulary: {team_context.get('team_vocabulary', 'No specific vocabulary provided')}
        
        Create a version that:
        1. Uses terminology familiar to this team
        2. Maintains the exact same meaning as the standard definition
        3. References team-specific systems or processes where relevant
        4. Is clear and actionable for the team
        
        Return a JSON object with:
        - team_kpi_name: Recommended name for this team's version
        - team_definition: The team-specific definition
        - team_calculation: How this team should calculate the KPI
        - implementation_notes: Any specific notes for this team
        """
        
        # For both models, we need to handle the prediction appropriately
        if "GEMINI_API_KEY" in os.environ:
            response = self.llm.invoke(prompt).content
        else:
            response = self.llm.predict(prompt)
        return response
    
    @tool
    def generate_implementation_guide(self, standardized_kpi, team_context):
        """Generate an implementation guide for a team to adopt a standardized KPI.
        
        Args:
            standardized_kpi: Dictionary with standard_name, standard_definition, calculation_method
            team_context: Dictionary with team, original_kpi_name, original_definition, data_sources
            
        Returns:
            Dictionary with implementation guide
        """
        prompt = f"""Create an implementation guide for the {team_context['team']} team to adopt this standardized KPI.

        Standardized KPI:
        - Name: {standardized_kpi['standard_name']}
        - Definition: {standardized_kpi['standard_definition']}
        - Calculation Method: {standardized_kpi['calculation_method']}
        
        Team Context:
        - Team: {team_context['team']}
        - Original KPI Name: {team_context['original_kpi_name']}
        - Original Definition: {team_context['original_definition']}
        - Data Sources: {team_context.get('data_sources', 'No specific data sources provided')}
        
        Create an implementation guide that includes:
        1. Steps to transition from the current KPI to the standardized version
        2. Data requirements and sources
        3. Calculation instructions with examples
        4. Reporting frequency and format recommendations
        5. Common pitfalls to avoid
        6. FAQ section addressing likely team questions
        
        Return a detailed implementation guide in JSON format with appropriate sections.
        """
        
        # For both models, we need to handle the prediction appropriately
        if "GEMINI_API_KEY" in os.environ:
            response = self.llm.invoke(prompt).content
        else:
            response = self.llm.predict(prompt)
        return response
    
    @tool
    def compare_with_original(self, standardized_kpi, original_kpi):
        """Compare a standardized KPI with the original team-specific version.
        
        Args:
            standardized_kpi: Dictionary with standard name and definition
            original_kpi: Dictionary with team, kpi_name, definition
            
        Returns:
            Dictionary with comparison analysis
        """
        prompt = f"""Compare this standardized KPI with the original team-specific KPI:

        Standardized KPI:
        - Name: {standardized_kpi['standard_name']}
        - Definition: {standardized_kpi['standard_definition']}
        - Calculation Method: {standardized_kpi['calculation_method']}
        
        Original KPI:
        - Team: {original_kpi['team']}
        - Name: {original_kpi['kpi_name']}
        - Definition: {original_kpi['definition']}
        
        Provide an analysis that includes:
        1. Key similarities and differences
        2. Any information that exists in one but not the other
        3. Differences in calculation approach
        4. Potential challenges in transitioning
        5. Benefits the team will gain from the standardized version
        
        Return a detailed comparison in JSON format with appropriate sections.
        """
        
        # For both models, we need to handle the prediction appropriately
        if "GEMINI_API_KEY" in os.environ:
            response = self.llm.invoke(prompt).content
        else:
            response = self.llm.predict(prompt)
        return response
    
    def translate_for_team(self, standardized_kpi, team, original_kpi):
        """Translate a standardized KPI for a specific team.
        
        Args:
            standardized_kpi: Dictionary with standardized KPI information
            team: Team name
            original_kpi: Dictionary with the team's original KPI information
            
        Returns:
            Dictionary with translated KPI and implementation guide
        """
        team_context = {
            "team": team,
            "original_kpi_name": original_kpi["kpi_name"],
            "original_definition": original_kpi["definition"],
            # Additional team context could be provided here
        }
        
        input_text = f"""Translate this standardized KPI for the {team} team:
        
        Standardized KPI: {standardized_kpi}
        
        Original team KPI: {original_kpi}
        
        Create a team-specific version that will be easy for the team to adopt while
        maintaining consistency with the standard.
        """
        
        return self.agent_executor.invoke({"input": input_text})
    
    def translate_all_kpis(self, standardized_kpis_df):
        """Translate all standardized KPIs back to team-specific versions.
        
        Args:
            standardized_kpis_df: DataFrame with standardized KPIs and original info
            
        Returns:
            DataFrame with translations for each team
        """
        results = []
        
        for _, row in standardized_kpis_df.iterrows():
            standardized_kpi = row["standardized_definition"]
            team = row["team"]
            original_kpi = {
                "team": team,
                "kpi_name": row["kpi_name"],
                "definition": row["definition"]
            }
            
            translation = self.translate_for_team(standardized_kpi, team, original_kpi)
            
            result = row.to_dict()
            result["team_translation"] = translation
            results.append(result)
        
        return pd.DataFrame(results)