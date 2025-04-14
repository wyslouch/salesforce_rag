from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
import json
import os
from typing import Dict, Any, List


class TimeConstraintExtractor:
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        model_provider: str = "openai",
        temperature: float = 0.2,
        openai_api_key: str = None,
        vector_store_type: str = "qdrant",
    ):
        self.vector_store_type = vector_store_type
        self.llm = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            temperature=temperature,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
        )

        self.prompt = ChatPromptTemplate.from_template("""
        You are analyzing a question about Salesforce earnings call transcripts to detect any time-based constraints.
        Extract any time periods, fiscal years, quarters, or date ranges mentioned in the question. Current year is 2025.
        Oldest information is from 2023 FY.
                                                       
        Available metadata fields in our documents:
        - fiscal_quarter: The fiscal quarter (e.g., "Q1", "Q2", "Q3", "Q4")
        - fiscal_year: The fiscal year (e.g., "2022", "2023")

        Examples:
        - "What did Salesforce say about AI in 2023?" → {{"fiscal_year": "2023"}}
        - "How did their stock perform between 2021 and 2023?" → {{"start_year": "2021", "end_year": "2023"}}
        - "What's the latest/most recent news?" → {{"fiscal_year": "2025"}}
        Question: {question}
        
        Return a JSON object containing the detected time constraints. Only include fields that are explicitly mentioned.
        Valid fields are:
        - fiscal_year: A single year (string)
        - fiscal_quarter: A single quarter (e.g., "Q1", "Q2")
        - start_year: Start year for a year range
        - end_year: End year for a year range
        
        Return only the JSON object without any explanation. Return an empty JSON object if no time constraints are detected.
        """)

    def detect_time_constraints(self, question: str) -> Dict[str, Any]:
        messages = self.prompt.invoke({"question": question})
        response = self.llm.invoke(messages)

        try:
            result = json.loads(response.content.strip())
            return result
        except json.JSONDecodeError:
            # If parsing fails, return empty dict
            return {}

    def convert_to_filters(self, time_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Convert time constraints to appropriate format for the vector store"""
        if self.vector_store_type == "qdrant":
            return self._convert_to_qdrant_filters(time_constraints)
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store_type}")

    def _convert_to_qdrant_filters(
        self, time_constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert time constraints to Qdrant filter format"""
        filters = []

        # Handle single fiscal year
        if "fiscal_year" in time_constraints:
            filters.append(
                {
                    "field": "fiscal_year",
                    "value": time_constraints["fiscal_year"],
                    "operator": "eq",
                }
            )

        # Handle single fiscal quarter
        if "fiscal_quarter" in time_constraints:
            filters.append(
                {
                    "field": "fiscal_quarter",
                    "value": time_constraints["fiscal_quarter"],
                    "operator": "eq",
                }
            )

        # Handle year range with start and end years
        if "start_year" in time_constraints:
            try:
                start_year = int(time_constraints["start_year"])
                end_year = int(time_constraints.get("end_year", 2025))
                filters.append(
                    {
                        "field": "fiscal_year",
                        "value": [str(start_year), str(end_year)],
                        "operator": "range",
                    }
                )
            except ValueError:
                # If conversion fails, try exact match on start_year
                filters.append(
                    {
                        "field": "fiscal_year",
                        "value": time_constraints["start_year"],
                        "operator": "eq",
                    }
                )

        return filters
