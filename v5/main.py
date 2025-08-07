import os
import time
import json
from typing import List, Iterable
from io import StringIO

import pandas as pd
from pydantic import BaseModel
from environs import Env
from openai import AzureOpenAI, Client
from jinja2 import Template


"""
    DESCRIPTION
    Simple prompt engineering approach to identifying overlapping m365 licenses applied to users.
    Up to date information and metadata for m365 licenses are provided via context.
    Querying overlapping licenses is achieved using a structured prompt.
"""


# ==================== [ CONFIG ] ==================== #

MAX_BATCH_SIZE = 50

MODEL_NAME = "gpt-4o" # gpt-4o-2024-11-20
DEPLOYMENT = "gpt-4o"
API_VERSION = "2025-01-01-preview"

USER_DATA_FILE_PATH = "./input/data.csv"
LICENSE_MATRIX_FILE_PATH = "./input/license-matrix.csv"

CONTEXT_TEMPLATE_FILE_PATH = "./input/context.jinja"
BATCHED_QUERY_TEMPLATE_FILE_PATH = "./input/batched-query-template.jinja"

OUTPUT_FILE_PATH = "./output/report.md"


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

def dedupe_and_normalise(strings: Iterable[str]) -> tuple[str]:

    return tuple(sorted(list(set(strings))))

# ================================================================================

class UserData:

    def __init__(self, id: str, licenses: tuple[str]):

        # Required
        self.id = id
        self.licenses = licenses

        # Calculated later
        self.output = None

    def as_dict(self) -> dict:

        return {
            "id": self.id,
            "licenses": self.licenses,
            "output": self.output,
        }

def get_license_matrix_contents() -> str:

    print(f"Reading license matrix data from '{LICENSE_MATRIX_FILE_PATH}'.")

    with open(LICENSE_MATRIX_FILE_PATH, "r") as f:
        return f.read()

def get_user_data() -> dict[str, UserData]:

    USER_ID_FIELD = "Object Id"
    LICENSES_FIELD = "Licenses"

    print(f"Reading user data from '{USER_DATA_FILE_PATH}'.")

    user_data_csv = pd.read_csv(USER_DATA_FILE_PATH)

    users = dict()
    for _, row in user_data_csv.iterrows():

        user = UserData(
            id=row[USER_ID_FIELD],
            licenses=dedupe_and_normalise(row[LICENSES_FIELD].split("+"))
        )

        users[user.id] = user

    return users

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

def generate_context() -> str:

    return context_template.render({
        "license_matrix_file_contents" : get_license_matrix_contents()
    })

def generate_batched_query(license_groups: list[tuple[str]]) -> str:

    return batched_query_template.render({
        "license_groups" : license_groups
    })

def generate_batched_prompt(license_groups: list[tuple[str]]) -> Prompt:

    return Prompt(
        context=generate_context(),
        query=generate_batched_query(license_groups)
    )

# ================================================================================

'''
https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/structured-outputs?tabs=python%2Cdotnet-entra-id&pivots=programming-language-python
-> A schema may have up to 100 object properties total, with up to five levels of nesting.
'''

class LicenseGroup(BaseModel):

    license_set: list[str] # The set of licenses

class ResponseSchema(BaseModel):

    responses: list[LicenseGroup] # Batching responses into an array

def dispatch_prompt(client: Client, prompt: Prompt) -> dict:

    print(f"Dispatching prompt [{prompt.id}] ...")
    print(f"  Context length (characters): {len(prompt.context)}")
    print(f"  Query length (characters): {len(prompt.query)}")
    print(f"  Query: '''{prompt.query}'''")

    start_ns = time.time_ns()

    completion = client.chat.completions.parse(
        model=DEPLOYMENT,
        messages=prompt.as_azure_compatible(),
        response_format=ResponseSchema,
        max_tokens=6553,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    end_ns = time.time_ns()

    completion = completion.to_dict()

    print(f"  Query took {(end_ns - start_ns) // 1000 // 1000}ms.")
    print(f"  Azure Response ID: {completion["id"]}")
    print(f"  Response: {completion["choices"][0]["message"]["parsed"]}")
    print(f"  Token useage breakdown:")

    # TODO - Cost analytics based on token usage. https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/#pricing

    for k, v in completion["usage"].items():
        print(f"    {k} : {v}")

    return completion

def main():

    # Load data

    users = get_user_data()
    unique_license_groupings = dedupe_and_normalise([user.licenses for user in users.values()])
    unique_license_groupings = unique_license_groupings[:5] # TEMP

    # TODO - Check cache for hits

    # Generate batched prompts

    prompts = []
    for i in range(0, len(unique_license_groupings), MAX_BATCH_SIZE):
        prompts.append(generate_batched_prompt(unique_license_groupings[i:i+MAX_BATCH_SIZE]))

    # Dispatch prompts

    print(f"Initialising client ...")

    client = get_client()

    for prompt in prompts:

        completion = dispatch_prompt(client, prompt)

        # Ensure finish_reason is valid - TODO

        result = completion["choices"][0]
        finish_reason = result["finish_reason"]

        # Unpack results

        batch_output = result["message"]["parsed"]["responses"]

        batch_results = [LicenseGroup.model_validate(entry) for entry in batch_output]
        for entry in batch_results:
            entry.license_set = dedupe_and_normalise(entry.license_set)

        # TODO - Data validation

        # TODO - Ensure all given license groups were covered in response

        # Join back on users

        for res in batch_results:
            for user in users.values():
                if user.licenses == res.license_set:
                    user.output = res.license_set

    client.close()

    for _, user_data in users.items():
        print(user_data.as_dict())

    # Format and save

    output = [user_data.as_dict() for user_data in users.values()]

    with open(OUTPUT_FILE_PATH, 'w') as f:
        json.dump(output, f)


# ==================== [ RUN ] ==================== #

if __name__ == "__main__":
    main()
