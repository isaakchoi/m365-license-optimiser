import os
import time
import json

import pandas as pd
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

MODEL_NAME = "gpt-4o" # gpt-4o-2024-11-20
DEPLOYMENT = "gpt-4o"
API_VERSION = "2025-01-01-preview"

DATA_FILE_PATH = "./input/data.csv"

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


# ==================== [ AZURE UTILITIES ] ==================== #

def get_client() -> AzureOpenAI:

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=API_VERSION,
    )

def query_azure_openai(client: AzureOpenAI, query):

    return client.chat.completions.create(
        model=DEPLOYMENT,
        messages=query,
        max_tokens=6553,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False
    )


# ==================== [ DATA CONNECTION UTILITIES ] ==================== #

def get_context_template() -> Template:

    print(f"Reading context template from '{CONTEXT_TEMPLATE_FILE_PATH}'.")

    with open(CONTEXT_TEMPLATE_FILE_PATH, "r") as f:
        return Template(f.read())

def get_license_matrix_contents() -> str:

    print(f"Reading license matrix data from '{LICENSE_MATRIX_FILE_PATH}'.")

    with open(LICENSE_MATRIX_FILE_PATH, "r") as f:
        return f.read()

def get_user_data() -> list[dict]:

    print(f"Reading user data from '{DATA_FILE_PATH}'.")

    csv = pd.read_csv(DATA_FILE_PATH)

    entries = []
    for _, row in csv.iterrows():
        entries.append({
            "user_id" : row["Object Id"],
            "licenses" : row["Licenses"].split("+")
        })

    return entries


# ==================== [ PROMPT GENERATION ] ==================== #

def create_context() -> str:

    template = get_context_template()

    license_matrix_contents = get_license_matrix_contents()

    data = {
        "license_matrix_file_contents" : license_matrix_contents
    }

    context = template.render(data)

    print(f"Context length (characters): {len(context)}")

    return context

def create_query() -> str:

    data = get_user_data()

    query = "\n".join([
        f"The following is a JSON object that contains a list of user IDs and the licenses that they have allocated to them. For each user, concisely answer the following:",
        f"-> Which licenses have overlapping features?",
        f"-> Can I get rid of any licenses, why or why not?",
        f"",
        f"``` users-and-licenses.json",
        f"{json.dumps(data)}",
        f"```",
        f""
    ])

    print(f"Query length (characters): {len(query)}")

    print(f"[ START QUERY SOURCE ] ------------------------------------------------")
    print(query)
    print(f"[ END QUERY SOURCE ] --------------------------------------------------")

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


# ==================== [ PROMPTING ] ==================== #

def run_query(prompt):

    print(f"Initialising client ...")

    client = get_client()

    print(f"Running query ...")

    start_ns = time.time_ns()

    response = query_azure_openai(client, prompt)

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
