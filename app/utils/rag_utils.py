class RAGQuery_Retry_Fallback_Handler:
    def __init__(self, query_engine, max_retries=3):
        self.query_engine = query_engine
        self.max_retries = max_retries

    def execute_with_retries(self, query: str) -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.query_engine.query(query)
                if response and str(response).strip():
                    print(f"Query executed successfully on attempt {attempt}")
                    print("Response:", response)
                    return str(response)
                else:
                    print(f"Attempt {attempt}: Empty response.")
            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")

        # After max retries
        fallback_message = "No relevant information found for your query."
        print(f"All {self.max_retries} attempts failed or returned no data.")
        return fallback_message