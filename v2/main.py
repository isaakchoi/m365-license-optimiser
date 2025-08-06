import os
import time

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

DEPLOYMENT = "gpt-4o" # gpt-4o-2024-11-20
API_VERSION = "2025-01-01-preview"

CONTEXT_TEMPLATE_FILE_PATH = "./input/context.jinja"
LICENSE_MATRIX_FILE_PATH = "./input/license-matrix.csv"

OUTPUT_FILE_PATH = "./output/report.md"


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

    print(f"Reading context template from '{CONTEXT_TEMPLATE_FILE_PATH}'.")

    with open(CONTEXT_TEMPLATE_FILE_PATH, "r") as f:
        template = Template(f.read())

    print(f"Reading context data from '{CONTEXT_TEMPLATE_FILE_PATH}'.")

    with open(LICENSE_MATRIX_FILE_PATH, "r") as f:
        license_matrix_contents = f.read()

    data = {
        "license_matrix_file_contents" : license_matrix_contents
    }

    print(f"Context length (characters): {len(license_matrix_contents)}")

    return template.render(data)

def create_query() -> str:

    query = "A user has a license for administrative units, conditional access, A3 education and entra ID plan 2. Which licenses have overlapping features? Can I get rid of any licenses, why or why not?"

    print(f"Query length (characters): {len(query)}")

    return query

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

    print(f"Running query ...")

    client = get_client()

    start_ns = time.time_ns()

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

    end_ns = time.time_ns()

    client.close()

    response = response.to_dict()

    print(f"Query took {(end_ns - start_ns) // 1000 // 1000}ms.")
    print(f"Response ID: {response["id"]}")
    print(f"Token useage breakdown:")

    for k, v in response["usage"].items():
        print(f"  {k} : {v}")

    return response

def main():

    prompt = create_prompt()

    response = run_query(prompt)

    result = response["choices"][0]["message"]["content"]

    print(f"Writing results to '{OUTPUT_FILE_PATH}'.")

    with open(OUTPUT_FILE_PATH, 'w') as f:
        f.write(result)

    print(f"Done.")

# ==================== [ RUN ] ==================== #

if __name__ == "__main__":
    main()
