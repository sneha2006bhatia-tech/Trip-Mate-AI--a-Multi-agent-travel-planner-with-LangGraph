from tools.tavily import tavily_search
from tools.flights import search_flights
from backend import run_travel_agent
# res=search_flights("plan a 7 days india trip from nepal")
# print(res)
user_input = input("Enter travel request: ")

response = run_travel_agent(
    user_input=user_input,
    thread_id="test_user"
)

print("\nFINAL RESPONSE:\n")
print(response["answer"])