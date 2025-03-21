import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableBranch
from langchain.schema.output_parser import StrOutputParser
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
COLLECTION_NAME = "chat_history"
MODEL_NAME = os.getenv("MODEL_NAME")
client = firestore.Client(project=PROJECT_ID)


class FinanceService:
    def __init__(self, user_id: str):
        self.llm = ChatGroq(
            model_name=MODEL_NAME, 
            temperature=0.3,
            max_tokens=None
        )
        
         # Separate LLM definition with temperature=0 for regular_conversation_chain
        self.llm_fixed = ChatGroq(
            model_name=MODEL_NAME, 
            temperature=0, 
            max_tokens=None
        )

        self.chat_history = FirestoreChatMessageHistory(
            session_id=user_id,
            client=client,
            collection=COLLECTION_NAME
        )

    def print_step(self, step_name):
        def inner(x):
            print(f"Step {step_name}: {x}")
            return x
        return inner

        
    def combine_results(self, budget, advice: str) -> dict:
        if isinstance(budget, str):
            first_brace = budget.find("{")
            last_brace = budget.rfind("}")
            if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
                raise ValueError("Budget string does not contain a valid JSON object.")
            trimmed = budget[first_brace:last_brace+1]
            # Parse the trimmed JSON string into a dictionary
            budget = json.loads(trimmed)
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
        self.chat_history.add_user_message(query)
        messages = self.chat_history.messages

        def format_chat_history(messages):
            return "\n".join([f"{m.type}: {m.content}" for m in messages])
        # Confirm if the user's prompt is part of a regular converstaion or budget-related
        conversation_type_chain = (
            ChatPromptTemplate.from_messages([
                ("system", """
                    Determine if the user's query is related to finance/budgeting or if it is a regular conversation.
                    If the query is related to finance, budgeting, income, expenses, savings, or any financial advice, respond with "finance".
                    Otherwise, respond with "regular".
                """),
                ("human", "{query}")
            ])
            | self.llm
            | StrOutputParser()
        )
        
        # Regular conversation chain
        regular_conversation_chain = (
            ChatPromptTemplate.from_messages([
                ("system", """
                    You are a helpful assistant. If this is the first message in the conversation, start your response with:
                    "Hi, I am FinAlpha, your personal budgeting guide. Feel free to give me details about your income and expenses so I can create a budget for you and offer financial advice."
                    Otherwise, respond to the user's query in a friendly and informative manner.
                """),
                ("human", "{query}")
            ])
            | self.llm_fixed
            | StrOutputParser()
        )
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
                    Output only the JSON object exactly in the structure above with no extra text, 
                    the keys in the json should always be income, expense, savings then summary
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
                ("system", "You are an experienced financial advisor. Provide actionable financial advice based on the user's input; don't tell the user that you are an AI assistant."),
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
    
        finance_chain = combined_chain | branches
        
        # Chain everything together and invoke
        chain = conversation_type_chain | RunnableLambda(lambda validation_result: {
            "validation": validation_result,
            "raw_query": query
        }) | RunnableBranch(
            (
                lambda x: "finance" in x["validation"],
                RunnableLambda(lambda x: finance_chain.invoke({"query": x["raw_query"]}))
            ),
            (
                lambda x: "regular" in x["validation"],
                RunnableLambda(lambda x: regular_conversation_chain.invoke({"query": x["raw_query"]}))
            ),
            RunnableLambda(lambda x: "Invalid response from validation")
        )

        result = chain.invoke({"query": format_chat_history(messages)})
        if isinstance(result, dict):
            content = json.dumps(result)
        else:
            content = result

        self.chat_history.add_ai_message(content)
        return result
