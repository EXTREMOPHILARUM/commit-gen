import os
from llama_cpp import Llama

from util import generate_prompt

# Define the model repository name on Hugging Face
# MODEL_REPO = "TheBloke/deepseek-coder-1.3b-instruct-GGUF"
MODEL_REPO = "lmstudio-ai/gemma-2b-it-GGUF"
# MODEL_FILE = "deepseek-coder-1.3b-instruct.Q2_K.gguf"  # Adjust this to the correct model file name
MODEL_FILE = "gemma-2b-it-q4_k_m.gguf"  # Adjust this to the correct model file name


# Define the local path to save the model
LOCAL_MODEL_PATH = os.path.join("models", MODEL_REPO.replace("/", "_"))


# Function to load and run the model using llama-cpp-python
def run_model(model_path, input_prompt):
    # Download the model if not already present
    llama = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        device="cpu",  # Change to "cuda" if you have a GPU
        chat_format="llama-2",
        n_ctx=2048,
    )

    generated_prompt = generate_prompt("en", 50, "conventional")
    print("Generated Prompt:\n", generated_prompt)
    # Run the model
    # output = llama("Following is the git diff generate a git commit message \n"+input_prompt, echo=True, max_tokens=150)
    # output = llama(generated_prompt+" \n"+input_prompt, echo=True, max_tokens=150)
    output = llama.create_chat_completion(
        messages=[
            {"role": "system", "content": generated_prompt},
            {"role": "user", "content": input_prompt},
        ],
        max_tokens=200,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        repeat_penalty=1
    )

    # Print the output
    print("Model Output:\n", output)


# Main function
def main():

    # Define the input prompt for the model
    input_prompt = """
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
    run_model(MODEL_FILE, input_prompt)


if __name__ == "__main__":
    main()
