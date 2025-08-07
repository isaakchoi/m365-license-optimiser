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

DEPLOYMENT = "gpt-4o" # gpt-4o-2024-11-20
API_VERSION = "2025-01-01-preview"

USER_DATA_FILE_PATH = "./input/data.csv"
LICENSE_MATRIX_FILE_PATH = "./input/license-matrix.csv"

CONTEXT_TEMPLATE_FILE_PATH = "./input/context-template.jinja"
BATCHED_QUERY_TEMPLATE_FILE_PATH = "./input/batched-query-template.jinja"

OUTPUT_FILE_PATH = "./output/results.json"


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

print (f"Loading templates ...")

with open(BATCHED_QUERY_TEMPLATE_FILE_PATH, "r") as f:
    batched_query_template: Template = Template(f.read())

with open(CONTEXT_TEMPLATE_FILE_PATH, "r") as f:
    context_template: Template = Template(f.read())


# ==================== [ UTILITIES ] ==================== #

def get_client() -> AzureOpenAI:

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=subscription_key,
        api_version=API_VERSION,
    )

def dedupe_and_normalise(strings: Iterable) -> tuple:

    return tuple(sorted(list(set(strings))))

def get_counts(items: Iterable) -> dict:

    counts = dict()
    for item in items:
        counts[item] = counts.get(item, 0) + 1

    return counts

# ==================== [ DATA LOADERS ] ==================== #

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


# ==================== [ PROMPT GENERATION ] ==================== #

class Prompt:

    def __init__(self, context: str, query: str):

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


# ==================== [  ] ==================== #

'''
https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/structured-outputs?tabs=python%2Cdotnet-entra-id&pivots=programming-language-python
-> A schema may have up to 100 object properties total, with up to five levels of nesting.
'''

class LicenseGroup(BaseModel):

    license_group: list[str] # The set of licenses

class ResponseSchema(BaseModel):

    responses: list[LicenseGroup] # Batching responses into an array

def dispatch_prompt(client: Client, prompt: Prompt) -> dict:

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

    users = get_user_data()

    license_groups = dedupe_and_normalise([user.licenses for user in users.values() if len(user.licenses) > 1]) # Keep only unique license groups of size > 1

    print(f"Unique license groups being considered: {len(license_groups)}")

    print(f"Checking cache for hits ...")

    cache_hit_license_groups = set() # TODO
    cache_miss_license_groups = license_groups # TODO

    print(f"Cache hits: {len(cache_hit_license_groups)}")
    print(f"Cache misses: {len(cache_miss_license_groups)}")

    print(f"Initialising client ...")

    client = get_client()

    print(f"Processing license groups from cache misses ...")

    for i in range(0, len(cache_miss_license_groups), MAX_BATCH_SIZE):

        print(f"Processing batch number '{(i // MAX_BATCH_SIZE) + 1}' ...")

        batch = cache_miss_license_groups[i:i+MAX_BATCH_SIZE]

        print(f"Batch size: {len(batch)}")

        print (f"Generating prompt ...")

        prompt = generate_batched_prompt(batch)

        print(f"Dispatching prompt ...")

        completion = dispatch_prompt(client, prompt)

        result = completion["choices"][0]

        finish_reason = result["finish_reason"]
        if finish_reason != "stop":
            raise RuntimeError(f"Completion returned with finish reason '{finish_reason}' (expected 'stop'). Some problem likely occurred.")

        print(f"Unpacking results ...")

        batch_results = [LicenseGroup.model_validate(entry) for entry in result["message"]["parsed"]["responses"]]
        for entry in batch_results:
            entry.license_group = dedupe_and_normalise(entry.license_group)

        print(f"Validating returned data ...")

        counts = get_counts((entry.license_group for entry in batch_results))

        failed = set(batch)
        succeeded = []

        for entry in batch_results:

            valid = True

            if entry.license_group not in batch:
                valid = False

            if counts[entry.license_group] > 1:
                valid = False # There were multiple entries returned with the same ID. Discard all as something probably went wrong.

            if valid:

                failed.remove(entry.license_group)
                succeeded.append(entry)

        for license_group in failed:

            print(f"WARNING: License group '{license_group}' failed. Putting up for retry.")

            # TODO - Put invalid entries up for retry. Also needs a retry limit.

        print(f"Updating cache ...")

        # TODO

        print(f"Joining results back on user data ...")

        for entry in succeeded:
            for user in users.values():
                if user.licenses == entry.license_group:
                    user.output = entry.license_group

    print(f"Closing client ...")

    client.close()

    print(f"Writing output to '{OUTPUT_FILE_PATH}'.")

    output = [user_data.as_dict() for user_data in users.values()]

    with open(OUTPUT_FILE_PATH, 'w') as f:
        f.write(json.dumps(output))


# ==================== [ RUN ] ==================== #

if __name__ == "__main__":
    main()
