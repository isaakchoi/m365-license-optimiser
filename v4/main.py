import os
import time
import json
from typing import Iterable
from io import StringIO

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


BATCHED_QUERY_TEMPLATE_FILE_PATH = "./input/batched-query-template.jinja"


# ==================== [ SETUP ] ==================== #

# Allow use of relative paths

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"Set working directory to '{os.getcwd()}'.")

# Uses environment variables if they are set. Otherwise checks for the variable definition in the .env file. Throws error if variable is defined in neither.

print(f"Reading env vars ...")

env = Env()
env.read_env(path="./.env", recurse=False)

endpoint = env.str("AZURE_OPENAI_ENDPOINT_URL")
subscription_key = env.str("AZURE_OPENAI_API_KEY")

# Preload Jinja templates

print (f" Loading templates ...")

with open(BATCHED_QUERY_TEMPLATE_FILE_PATH, "r") as f:
    batched_query_template: Template = Template(f.read())

with open(CONTEXT_TEMPLATE_FILE_PATH, "r") as f:
    context_template: Template = Template(f.read())

# ================================================================================

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

# ================================================================================

class Prompt:

    def _get_new_id(start=0):
        while True:
            yield start
            start += 1

    _id_gen = _get_new_id()

    def __init__(self, context: str, query: str):

        self.id = next(Prompt._id_gen)

        self.context = context
        self.query = query

    def as_azure_compatible(self):

        return [
            {
                "role": "system",
                "content": self.context
            },
            {
                "role": "user",
                "content": self.query
            }
        ]

# ================================================================================

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

# ================================================================================

def generate_context() -> str:

    license_matrix_contents = get_license_matrix_contents()

    data = {
        "license_matrix_file_contents" : license_matrix_contents
    }

    context = context_template.render(data)

    return context

def generate_batched_query(license_groups: list[tuple[str]]) -> str:

    return batched_query_template.render({
        "license_groups" : license_groups
    })

def generate_batched_prompt(license_groups: list[tuple[str]]) -> str:

    return Prompt(
        context=generate_context(),
        query=generate_batched_query(license_groups)
    )

# ================================================================================

def dispatch_prompt(client, prompt: Prompt) -> str:

    print(f"Dispatching prompt [{prompt.id}] ...")
    print(f"  Context length (characters): {len(prompt.context)}")
    print(f"  Query length (characters): {len(prompt.query)}")
    print(f"  Query: '''{prompt.query}'''")

    start_ns = time.time_ns()

    response = query_azure_openai(client, prompt.as_azure_compatible())

    end_ns = time.time_ns()

    response = response.to_dict()
    response_str = response["choices"][0]["message"]["content"]

    print(f"  Query took {(end_ns - start_ns) // 1000 // 1000}ms.")
    print(f"  Azure Response ID: {response["id"]}")
    print(f"  Response: '''{response_str}'''")
    print(f"  Token useage breakdown:")

    for k, v in response["usage"].items():
        print(f"    {k} : {v}")

    return response_str

def main():

    # Read in source data

    user_data_csv = pd.read_csv(DATA_FILE_PATH)

    # Extract users and licenses

    USER_ID_FIELD = "Object Id"
    LICENSES_FIELD = "Licenses"

    users = dict()
    for _, row in user_data_csv.iterrows():
        applied_licenses = row[LICENSES_FIELD].split("+")
        applied_licenses = list(set(applied_licenses)) # Drop duplicates
        applied_licenses = tuple(sorted(applied_licenses)) # Make them hashable
        users[row[USER_ID_FIELD]] = applied_licenses

    unique_license_groupings = list(set(users.values())) # Drop duplicates

    # Check cache
    # TODO

    # Generate batched prompts based on license sets

    MAX_BATCH_SIZE = 50

    prompts = []
    for i in range(0, 11, MAX_BATCH_SIZE):
        prompts.append(generate_batched_prompt(unique_license_groupings[i:i+MAX_BATCH_SIZE]))

    # Dispatch prompts

    print(f"Initialising client ...")

    client = get_client()

    def validate_response(response: str) -> bool:

        # Validate can be loaded as a csv
        try:
            csv = pd.read_csv(StringIO(response))
        except pd.errors.ParserError:
            return False

        # Validate contains all correct columns
        REQUIRED_COLUMNS = ["all_licenses", "redundant_licenses", "redundant_licenses_justification", "overlapping_licenses", "overlapping_licenses_justification", "unsure", "unsure_justification"]
        actual_columns = csv.columns.to_list()
        print(actual_columns)
        for col in REQUIRED_COLUMNS:
            if col not in actual_columns:
                return False

        # Validate contains only correct columns and no duplicated
        if len(actual_columns) != len(REQUIRED_COLUMNS):
            return False

        # Passed
        return True

    for prompt in prompts:

        response = dispatch_prompt(client, prompt)

        if not validate_response(response):
            print("INVALID RESPONSE")

        with open('out.csv', 'w') as f:
            f.write(response)

        # Unpack and validate responses - Resend if fails validation?

        # TODO

    client.close()

    # Aggregate response data

    # TODO

    # Join back on users

    # TODO

    # Format for output

    # TODO

# ==================== [ RUN ] ==================== #

if __name__ == "__main__":
    main()
