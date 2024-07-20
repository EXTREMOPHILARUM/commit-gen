import os
from llama_cpp import Llama
import pprint

# from util import generate_prompt

# Define the model repository name on Hugging Face
# MODEL_REPO = "TheBloke/deepseek-coder-1.3b-instruct-GGUF"
MODEL_REPO = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
# MODEL_FILE = "deepseek-coder-1.3b-instruct.Q2_K.gguf"  # Adjust this to the correct model file name
MODEL_FILE = "Meta-Llama-3-8B-Instruct.Q2_K.gguf"  # Adjust this to the correct model file name


# Define the local path to save the model
LOCAL_MODEL_PATH = os.path.join("models", MODEL_REPO.replace("/", "_"))


# Function to load and run the model using llama-cpp-python
def run_model(model_path, diff):
    # Download the model if not already present
    llama = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        device="cpu",  # Change to "cuda" if you have a GPU
        chat_format="llama-3",
        n_ctx=4096,
        n_predict=-1
    )

    # generated_prompt = generate_prompt("en", 50, "conventional")
    # print("Generated Prompt:\n", generated_prompt)
    # Run the model
    # output = llama("Following is the git diff generate a git commit message \n"+diff, echo=True, max_tokens=150)
    main_prompt = """
    You are an AI that strictly conforms to responses in JSON formatted strings in the locale en.
    Your responses consist of valid JSON syntax, with no other comments, explanations, reasoning, or dialogue not consisting of valid JSON.
    You will be given a git diff, which you need to infer the following fields:
    1. `type` that best describes the git diff change type.
    2. `commit-message` Generate a concise git commit message written in present tense not exceeding 100 characters.
    3. `change-log`  Generate a descriptive change log atleast 200 characters long.
    If you cannot interpret the text for any of these fields, return the field with a null value in the JSON.
    """
    # output = llama(
    #     f"{main_prompt} \n {diff}",
    #     temperature=0.8,
    #     echo=True,
    #     max_tokens=500,
    #     top_k=40,
    #     repeat_penalty=1.1,
    #     min_p=0.05,
    #     top_p=0.95,        
    # )
    # print("Model Output:\n")
    # pprint.pprint(output)
    # output = llama(generated_prompt+" \n"+diff, echo=True, max_tokens=150)
    # output = llama.create_chat_completion(
    #     messages=[
    #         {"role": "system", "content": generated_prompt},
    #         {"role": "user", "content": diff},
    #     ],
    #     max_tokens=200,
    #     temperature=0.7,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0,
    #     repeat_penalty=1
    # )
    output = llama.create_chat_completion(
        messages=[
            {"role": "system", "content": main_prompt},
            {"role": "user", "content": diff},
        ],
        temperature=0.8,
        max_tokens=500,
        top_k=40,
        repeat_penalty=1.1,
        min_p=0.05,
        top_p=0.95,
    )

    # Print the output
    print("Model Chat Output:\n")
    pprint.pprint(output["choices"][0]["message"]["content"])

# Main function
def main():
    # Define the input prompt for the model
    diff = """
diff --git a/example.py b/example.py
index 83b6444..e4b04c7 100644
--- a/example.py
+++ b/example.py
@@ -1,5 +1,5 @@
 def add(a, b):
-    return a + b
+    return a + b + 1

 def subtract(a, b):
     return a - b
"""

    # Run the model
    run_model(MODEL_FILE, diff)


if __name__ == "__main__":
    main()
