import json
import yaml
import requests

def load_config(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)

def ask_llm(messages):

    r = requests.post(
        config["url"],
        json={"model": "llama3", "messages": messages, "stream": True},
        stream=True
        ) 
    r.raise_for_status()
    output = ""

    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            print(content, end="", flush=True)

        if body.get("done", False):
            message["content"] = output
            return message


def ask_openapi(user_prompt):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config['api_key']}"}
    response = requests.post(
        config['openai_url'],
        headers=headers,
        json={"model": "gpt-3.5-turbo", "messages": [{"role":"user", "content": user_prompt}]},
    )
    response.raise_for_status()

    json_response = response.json()
    plain_response = ""

    if json_response["choices"] and len(json_response["choices"]) > 0:
        first_choice = json_response["choices"][0]
        messages = []
        if "message" in first_choice:
            message = first_choice["message"]
            if message['role'] == 'assistant':
                messages.append(message['content'])
        plain_response = " ".join(messages)

    print(plain_response)
    return json_response

def ask(prompt):
    msg = [{"role":"user", "content":prompt}]
    if config["target"] == "openai":
        message = ask_openapi(msg)
    else:
        message = ask_llm(msg)

    return message

def get_agent(user_input):
    prompt = f'''Write a system prompt that takes in a user's question and determines whether 
        it should be sent to a 'math teacher' or a 'history teacher'. 
        The prompt should analyze the content of the question to decide which agent is more relevant based on the subject matter discussed. 
- If the question involves mathematical concepts, computations, or problems (like algebra, geometry, calculus, etc.), the prompt should select the "math teacher". 
- If the question pertains to historical events, figures, dates, or interpretations of past events (like World War II, the Renaissance, ancient civilizations, etc.), the prompt should select the "history teacher".

The response should be in JSON format, indicating the chosen agent. The json output should have only the target agent only as key,
For example, {{"agent": "math teacher"}} or {{"agent": "history teacher"}}.

user question: {user_input}'''

    result = ask(prompt)

    return result["agent"]


def ask_math_teacher(user_input):
    prompt = f'''You're a math teacher. You've been asked to help a student with a math problem. {user_input}'''
    messages = [{"role": "user", "content": prompt}]
    message = ask_llm(messages)
    return message

def ask_history_teacher(user_input):
    prompt = f'''You're a history teacher. You've been asked to help a student with a history question. {user_input}'''
    messages = [{"role": "user", "content": prompt}]
    message = ask_llm(messages)
    return message

def main():

    while True:
        user_input = input("Enter a prompt: ")
        if not user_input:
            exit()
        print()

        agent = get_agent(user_input)

        if agent == "math teacher":
            ask_math_teacher(user_input)
        else:
            ask_history_teacher(user_input)

        print(f"choosen agent: {agent}")
        break


if __name__ == "__main__":
    config = load_config('config.yaml')

    main()