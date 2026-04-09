import time

from mistral.client import Mistral
from pydantic import BaseModel
import httpx

API_KEY = "something"

def draw_random_card():
    """Draw a random card from the deck of cards API and return its suit and value."""
    with httpx.Client(timeout=5) as client:
        response = client.get("https://deckofcardsapi.com/api/deck/new/draw/?count=1")
        response.raise_for_status()
        data = response.json()
        return { "suit": data["cards"][0]["suit"], "value": data["cards"][0]["value"] }
    
def get_tool_def():
    return {
        "type": "function",
        "function": {
            "name": draw_random_card.__name__,
            "description": draw_random_card.__doc__,
            # "parameters": {
            #     "type": "object",
            #     "properties": {
            #         "transaction_id": {
            #             "type": "string",
            #             "description": "The transaction id.",
            #         }
            #     },
            #     "required": ["transaction_id"],
            # },
        }
    }


class GameResponseFormat(BaseModel):
    user_won: bool


def main():
    client = Mistral(api_key=API_KEY, timeout=10)
    system_prompt = """
You are an agent which plays a simple card guessing gamae with the user. The rules of the game are as follows:
1. You will draw a random card using the tool provided, the user will try to guess the suit and value of the card.
2. The user can guess till the get it right. 
3. Once the user guesses the card correctly they can either choose to play again or end the game."""

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    user_prompt = input("Guess the suite and value of the card drawn or press q to quit: ")
    messages.append({"role": "user", "content": user_prompt})

    tool_call_dict = {draw_random_card.__name__: draw_random_card}

    while user_prompt != "q" and len(messages) < 10:

        response = client.chat.parse(
            model="mistral-large-latest",
            messages=messages,
            tools=[get_tool_def()],
            tool_choice="auto", 
            response_format = GameResponseFormat
        )
        messages.append(response.choices[0].message)
        if response.choices[0].message.finish_reason == "stop":


            if response.choices[0].message.parsed.user_won:
                print("You won!")
                messages = [
                                {"role": "system", "content": system_prompt},
                            ]
                user_prompt = input("Guess the suite and value of the card drawn or press q to quit: ")
                messages.append({"role": "user", "content": user_prompt})

##################### Missing
            else:
                user_prompt = input("Wrong guess, try again or press q to quit: ")
                messages.append({"role": "user", "content": user_prompt})

########### End of missing

        elif response.choices[0].message.finish_reason == "tool_calls":
            for tool_call in response.choices[0].message.tool_calls:
                tool_response = tool_call_dict[tool_call.function.name]()
                messages.append({"role": "tool", "name": tool_call.function.name, "content": str(tool_response), "tool_call_id": tool_call.id})

    print("Thanks for playing! We ran out of turns or you chose to quit.")

if __name__ == "__main__":
    main()


