import os

from environs import Env
from openai import AzureOpenAI

from jinja2 import Template


"""
    DESCRIPTION
    Simple prompt engineering approach to identifying overlapping m365 licenses applied to users.
    Up to date information and metadata for m365 licenses are provided via context.
    Querying overlapping licenses is achieved using a structured prompt.
"""


# ==================== [ CONFIG ] ==================== #

DEPLOYMENT = "gpt-40"
API_VERSION = "2025-01-01-preview"

CONTEXT_TEMPLATE_PATH = "./context.jinja"
LICENSE_MATRIX_FILE_PATH = "./license-matrix.csv"


# ==================== [ SETUP ] ==================== #

# Allow use of relative paths

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Set cwd to '{os.getcwd()}'.")

# Uses environment variables if they are set. Otherwise checks for the variable definition in the .env file. Throws error if variable is defined in neither.

print(f"Reading env vars ...")

env = Env()
env.read_env(path="./.env", recurse=False)

endpoint = env.str("AZURE_OPENAI_ENDPOINT_URL")
subscription_key = env.str("AZURE_OPENAI_API_KEY")


# ==================== [ PROMPTING ] ==================== #

def get_client() -> AzureOpenAI:

    print(f"Initialising client ...")

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=API_VERSION,
    )

def create_context() -> str:

    with open(CONTEXT_TEMPLATE_PATH, "r") as f:
        template = Template(f.read())

    with open(LICENSE_MATRIX_FILE_PATH, "r") as f:
        license_matrix_contents = f.read()

    data = {
        "license_matrix_file_contents" : license_matrix_contents
    }

    return template.render(data)

def create_query() -> str:

    return "I have a user that has the following licenses: Office 365 E5, Microsoft 365 Business standard, Microsoft 365 Frontline F5 Sec+Comp, Microsoft 365 Frontline F3, and Microsoft 365 Education A3"

def create_prompt() -> str:

    print(f"Creating prompt ...")

    prompt = [
        {
            "role": "system",
            "content": create_context()
        },
        {
            "role": "user",
            "content": create_query()
        }
    ]

    return prompt

def run_query(prompt):

    client = get_client()

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=prompt,
        max_tokens=6553,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )

    client.close()

    return response.to_dict()

def main():

    prompt = create_prompt()

    response = run_query(prompt)

    result = response["choices"][0]["message"]["content"]

    print(result)


# ==================== [ RUN ] ==================== #

if __name__ == "__main__":
    main()
