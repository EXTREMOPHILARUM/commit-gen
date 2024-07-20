
from pydantic import BaseModel
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
import instructor
import typer

from util import construct_prompt, load_types, get_git_diff


# Define the model repository name on Hugging Face
MODEL_REPO = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
MODEL_FILE = "Meta-Llama-3-8B-Instruct.Q2_K.gguf"

# Function to load and run the model using llama-cpp-python
def run_model(diff):
    # Download the model if not already present
    llama = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        device="cpu",
        chat_format="llama-3",
        draft_model=LlamaPromptLookupDecoding(num_pred_tokens=2),
        n_ctx=8192,
        n_predict=-1,
        logits_all=True,
        verbose=False,
    )

    class CommitModel(BaseModel):
        type: str
        commit_message: str
        change_log: str

    create = instructor.patch(
        create=llama.create_chat_completion_openai_v1,
        mode=instructor.Mode.JSON_SCHEMA,  # (2)!
    )

    # Load types from JSON
    types = load_types('types.json')

    # Construct the prompt
    main_prompt = construct_prompt(types)

    commit_data = create(
        response_model=CommitModel,
        messages=[
            {"role": "system", "content": main_prompt},
            {"role": "user", "content": f"Here is the git diff:\n{diff}"},
        ],
    )
    print(f"{commit_data.type}: {commit_data.commit_message}")
    print(f"{commit_data.change_log}")

# Main function
def main():
    repository_path = "./"
    diff = get_git_diff(repository_path)
    # print(diff.splitlines().__len__())

    # Run the model
    run_model(diff)


if __name__ == "__main__":
    typer.run(main)
