import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableBranch
from langchain.schema.output_parser import StrOutputParser
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore
from langchain.memory import ConversationSummaryMemory
from transformers import AutoTokenizer

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
COLLECTION_NAME = "chat_history"
MODEL_NAME = os.getenv("MODEL_NAME")
HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN")
MAX_MESSAGE_TOKEN_SIZE=1200
client = firestore.Client(project=PROJECT_ID)


class FinanceService:
    def __init__(self, user_id: str):
        self.llm = ChatGroq(
            model_name=MODEL_NAME,
            temperature=0.3,
            max_tokens=None
        )

        self.summary_llm = ChatGroq(
            model_name=MODEL_NAME,
            temperature=0,
            max_tokens=None
        )

        self.chat_history = FirestoreChatMessageHistory(
            session_id=user_id,
            client=client,
            collection=COLLECTION_NAME
        )

    def measure_token_usage(text: str) -> int:
        model_name = "meta-llama/Meta-Llama-3-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_TOKEN)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        return len(token_ids)

    def get_summary_memory(self, chat_history: str) -> str:
        memory = ConversationSummaryMemory(llm=self.llm)
        memory.save_context({"input": chat_history}, {"output": ""})
        summary = memory.load_memory_variables({})["history"]
        return summary

    def combine_results(self, budget: dict, advice: str) -> dict:
        print("budget")
        print(advice)
        return {
            "budget": budget,
            "advice": advice
        }

    def get_remaining_details(self, query):
        conversation_template = ChatPromptTemplate.from_messages([
            ("system", """
                You are a Budgeting Assistant tasked with gathering any missing details necessary to generate a complete budget in the specified JSON format. When the previous check determines that sufficient information is not available (i.e., the response is "false"), follow these instructions:

                1. Identify which details are missing from the current conversation. This may include:
                - Income sources: each with 'source' (string), 'amount' (number), and 'frequency' (string).
                - Expense items: each with 'category' (string), 'amount' (number), 'frequency' (string), and 'necessity' (either 'essential' or 'discretionary').
                - Saving goals: each with 'name' (string), 'targetAmount' (number), 'timeframe' (string), and 'priority' (string).
                - Sufficient details needed to compute the summary (totalMonthlyIncome, totalMonthlyExpenses, monthlySavings, and savingsRate).

                2. Interact with the user in a conversational manner to clarify and request the missing information. Ask follow-up questions that are clear, friendly, and focused on obtaining the required details.

                3. Guide the user step-by-step to ensure that all the necessary information is collected. Confirm with the user when you believe that all required data has been provided.

                4. Do not output the final JSON object until you have collected all required information.

                Your task is to interact conversationally with the user to fill in any missing budget details.
            """),
            ("human", """{query}""")
        ])

    # todo: check token size and reduce summarize history if larger than max-token-usage per message
        history_token_size = self.measure_token_usage()

        if history_token_size > MAX_MESSAGE_TOKEN_SIZE:
            pass
        else:
            self.chat_history.add_user_message(query)

        conversation_chain = conversation_template | self.llm | StrOutputParser()

        try:
            response = conversation_chain.invoke(self.chat_history.messages)
            # Check if response is valid (not empty and is a string)
            if response and isinstance(response, str):
                self.chat_history.add_ai_message(response)
                return response
            else:
                error_msg = "Invalid response received from AI"
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            return {"error": error_msg}

    def process_query(self, query):
        # Add the current query to chat history
        self.chat_history.add_user_message(query)
        messages = self.chat_history.messages
        print(messages)

        # Helper to format chat history as text
        def format_chat_history(messages):
            return "\n".join([f"{m.type}: {m.content}" for m in messages])

        # Create the budget chain using the full conversation context (as a formatted string)
        budget_chain = (
                ChatPromptTemplate.from_messages([
                    ("system", """
                    Parse the user's input to extract income sources, expense items, and saving goals.
                    For each income entry, create an object with the keys: source (string), amount (number), and frequency (string).
                    For each expense, create an object with the keys: category (string), amount (number), frequency (string), and necessity (either 'essential' or 'discretionary').
                    For each saving goal, create an object with the keys: name (string), targetAmount (number), timeframe (string), and priority (string).
                    Compute the summary:
                    - totalMonthlyIncome: Sum of all income amounts.
                    - totalMonthlyExpenses: Sum of all expense amounts.
                    - monthlySavings: totalMonthlyIncome minus totalMonthlyExpenses.
                    - savingsRate: (monthlySavings / totalMonthlyIncome) * 100, rounded to one decimal place.
                    If totalMonthlyIncome is not known or provided return [] for totalMonthlyIncome.
                    If totalMonthlyExpenses is not known or provided return [] for totalMonthlyExpenses.
                    If monthlySavings is not known or provided return [] for monthlySavings.
                    If savingsRate is not known or provided return [] for savingsRate.
                    Output only the JSON object exactly in the structure above with no extra text.
                    Your task is to convert the user's prompt into the budget JSON following these guidelines.
                """),
                    ("human", "Create a detailed budget based on: {query}")
                ])
                | self.llm
                | StrOutputParser()
        )

        # Create the advice chain using the full conversation context as well
        advice_chain = (
                ChatPromptTemplate.from_messages([
                    ("system",
                     "You are an experienced financial advisor. Provide actionable financial advice based on the user's input; don't tell the user that you are an AI assistant."),
                    ("human", "Provide specific financial advice based on: {query}")
                ])
                | self.llm
                | StrOutputParser()
        )

        # Parallel chain to run both chains concurrently and combine their outputs
        parallel_chain = RunnableParallel(
            budget=budget_chain,
            advice=advice_chain
        ) | RunnableLambda(lambda x: self.combine_results(x["budget"], x["advice"]))

        # Validation chain checks if there is enough detail in the entire conversation.
        validation_chain = (
                ChatPromptTemplate.from_messages([
                    ("system", """
                    You are a Budgeting Assistant responsible for determining if there is enough information 
                    from the entire conversation (past messages and the current user prompt) 
                    to generate a complete budget in the specified JSON format.

                    Instructions:
                    1. Review the conversation context (including past messages and the current prompt).
                    2. Check whether the provided details cover:
                    - Income sources: each with 'source' (string), 'amount' (number), and 'frequency' (string).
                    - Expense items: each with 'category' (string), 'amount' (number), 'frequency' (string), and 'necessity' (either 'essential' or 'discretionary').
                    - Saving goals: each with 'name' (string), 'targetAmount' (number), 'timeframe' (string), and 'priority' (string).
                    - Sufficient details to compute the summary (totalMonthlyIncome, totalMonthlyExpenses, monthlySavings, and savingsRate).
                    3. If the conversation covers at least 60% of the required information, output exactly "true".
                    4. If less than 60% is provided, output exactly "false".
                    5. If the user is updating the last budget, consider that as sufficient and output "true".

                    Your response must be either "true" or "false".
                """),
                    ("human", "{query}")
                ])
                | self.llm
                | StrOutputParser()
        )

        # Combine the validation result with the original chat history and raw query
        combined_chain = validation_chain | RunnableLambda(lambda validation_result: {
            "validation": validation_result,
            "chat_history": messages,
            "raw_query": query
        })

        # Branch based on validation result:
        branches = RunnableBranch(
            (
                lambda x: "true" in x["validation"],
                RunnableLambda(lambda x: parallel_chain.invoke({"query": format_chat_history(x["chat_history"])}))
            ),
            (
                lambda x: "false" in x["validation"],
                RunnableLambda(lambda x: self.get_remaining_details(x["raw_query"]))
            ),
            # Default case if validation returns an unexpected result
            RunnableLambda(lambda x: "Invalid response from validation")
        )

        # Chain everything together and invoke
        chain = combined_chain | branches
        result = chain.invoke({"query": format_chat_history(messages)})
        return result
