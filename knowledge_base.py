"""Knowledge base module for the KPI Standardizer system.

This module implements a persistent storage system using ChromaDB
that accumulates KPI definitions as more input is processed.
"""

import pandas as pd
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

class KnowledgeBase:
    """A persistent knowledge base for KPI definitions."""
    
    def __init__(self, persist_directory=None):
        """Initialize the knowledge base.
        
        Args:
            persist_directory: Directory where ChromaDB will store its data (optional)
        """
        model_name = "all-MiniLM-L6-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize Chroma with or without persistence
        if persist_directory:
            self.db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
        else:
            self.db = Chroma(
                embedding_function=self.embeddings
            )
    
    def add_kpis(self, kpis_df):
        """Add KPI definitions to the knowledge base.
        
        Args:
            kpis_df: Pandas DataFrame containing KPI definitions
                     with columns: team, kpi_name, definition
        """
        documents = []
        for _, row in kpis_df.iterrows():
            doc = Document(
                page_content=row["definition"],
                metadata={
                    "team": row["team"],
                    "kpi_name": row["kpi_name"]
                }
            )
            documents.append(doc)
        
        self.db.add_documents(documents)
        self.db.persist()
    
    def search_similar_kpis(self, query, n_results=5):
        """Search for KPIs similar to the query.
        
        Args:
            query: Definition or description to search for
            n_results: Number of results to return
            
        Returns:
            List of tuples (Document, similarity_score)
        """
        return self.db.similarity_search_with_score(query, k=n_results)
    
    def get_all_kpis(self):
        """Get all KPIs in the knowledge base.
        
        Returns:
            List of Documents representing all KPIs
        """
        return self.db.get()
    
    def get_kpis_by_team(self, team):
        """Get all KPIs for a specific team.
        
        Args:
            team: Team name to filter by
            
        Returns:
            List of Documents representing KPIs for the given team
        """
        return self.db.get(where={"team": team})