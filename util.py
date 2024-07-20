
from git import Repo

import json


def load_types(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['types']

def construct_prompt(types, locale='en', character_limit=100, changelog_length=200):
    prompt = f"""
    You are an AI that strictly conforms to responses in JSON formatted strings in the locale {locale}.
    Your responses consist of valid JSON syntax, with no other comments, explanations, reasoning, or dialogue not consisting of valid JSON.
    You will be given a git diff, which you need to infer the following fields: 
    1. `commit-message` Generate a concise commit-message not exceeding {character_limit} characters.
    2. `change-log` Generate a descriptive change log at least {changelog_length} characters long.
    3. `type` that best describes the git diff change type."""
    
    # Adding indexing to each type description
    for idx, (type_key, description) in enumerate(types.items(), start=1):
        prompt += f'\n        3.{idx}. "{type_key}": "{description}"'

    prompt += """
    The git diff will be provided in the following format:
    ```
    diff --git a/<file_path> b/<file_path>
    <diff_metadata>
    @@ <hunk_header> @@
    <diff_content>
    ```

    Here is another example of a git diff:
    ```
    diff --git a/example.py b/example.py
    index 83db48f..f735c2d 100644
    --- a/example.py
    +++ b/example.py
    @@ -1,4 +1,4 @@
    -print("Hello, world!")
    +print("Hello, universe!")
    ```
    If you cannot interpret the text for any of these fields, return the field with a null value in the JSON.
    """
    return prompt

def get_git_diff(repository_path):
    """
    Get the git diff of the current working directory in the given repository.
    Args:
    repository_path (str): Path to the local git repository.

    Returns:
    str: The git diff as a string.
    """
    # Initialize the repository object
    repo = Repo(repository_path)

    # Check if there are uncommitted changes
    if repo.is_dirty(untracked_files=True):
        # Get the diff of the working directory against the HEAD commit
        diff = repo.git.diff('HEAD', '--', cached=False, unified=0)
        return diff
    else:
        return "No changes detected."
