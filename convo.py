import subprocess  # To run terminal commands from Python

# A mapping from user-friendly model names to their actual Ollama names
MODEL_MAP = {
    "deepseek": "deepseek-r1",
    "llama": "llama3.2"
}

# This function sends a prompt to a given LLM via Ollama and returns a summarized response
def query_ollama(model: str, prompt: str, temperature=0.7):
    try:
        # Run the model with the prompt using Ollama CLI
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180  # Give the model up to 3 minutes to respond
        )
        
        # Decode the response output into readable text
        output = result.stdout.decode("utf-8").strip()

        # Clean up and summarize the response: take only the first 5 non-empty lines
        lines = [line.strip() for line in output.split('\n') if line.strip()]
        summary = "\n".join(lines[:5])
        return summary if summary else "[No summary available.]"

    # If the model takes too long, let the user know
    except subprocess.TimeoutExpired:
        return f"[Error] Response from model '{model}' timed out after 180 seconds."
    
    # Catch any other errors and return them
    except Exception as e:
        return f"[Error] {e}"

# Helper to format a model's name nicely for printing
def label(model_name):
    return f"[{model_name}]"

# Converts internal model ID into a more readable display name
def get_model_name(choice):
    return "Deepseek-r1" if choice == "deepseek" else "Llama 3.2"

# The main function where everything happens
def main():
    print("=== LLM Dialogue Orchestration ===\n")
    print("Available Models: deepseek, llama")

    # Ask the user to choose two models
    model_a = input("Choose Model A: ").strip().lower()
    model_b = input("Choose Model B: ").strip().lower()
    
    # Make sure both model names are valid
    if model_a not in MODEL_MAP or model_b not in MODEL_MAP:
        print("Invalid model choice.")
        return

    # Ask user whether to enable dynamic switching and debate mode
    dynamic_switch = input("Enable Dynamic Switching per round? (y/n): ").strip().lower() == 'y'
    debate_mode = input("Enable Debate Mode? (y/n): ").strip().lower() == 'y'

    # Get the starting question for the conversation
    user_question = input("\nEnter your initial question: ").strip()
    current_prompt = user_question
    current_model = model_a
    alt_model = model_b

    # Prepare the log file to save the entire conversation
    log_file = "llm-conversation.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== LLM Dialogue Log ===\n")
        f.write(f"Initial Question: {user_question}\n\n")

    # Run the dialogue for exactly 10 turns
    for i in range(1, 11):
        model_key = current_model
        model_label = get_model_name(model_key)

        # Display which model is speaking this round
        print(f"\n--- Round {i} | {model_label} ---")
        print(f" Question: {current_prompt}\n")

        # Get the model's response to the current prompt
        response = query_ollama(MODEL_MAP[model_key], current_prompt)

        # Print the summarized response to the terminal
        print(f" {label(model_label)} {response}\n")

        # Save this round’s question and answer to the log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"--- Round {i} | {label(model_label)} ---\n")
            f.write(f"Question: {current_prompt}\n")
            f.write(f"Answer:\n{response}\n\n")

        # Save these for context in the next round
        previous_question = current_prompt
        previous_response = response

        # Decide what the next prompt should be
        if debate_mode:
            current_prompt = f"Argue against this: {previous_response}"
        else:
            current_prompt = (
                f"I asked someone this question: {previous_question}. "
                f"Their answer was: {previous_response}. "
                "Propose a sharp follow-up question or counterpoint to continue the discussion."
            )

        # If dynamic switching is on, ask who should go next
        if dynamic_switch:
            current_model = input("Who should respond next? (deepseek/llama): ").strip().lower()
            if current_model not in MODEL_MAP:
                print("Invalid input, defaulting to alternate model.")
                current_model = alt_model
        else:
            # Otherwise, just alternate models
            current_model, alt_model = alt_model, current_model

    # Let the user know where the dialogue is saved
    print(f"\n Conversation saved to: {log_file}")

# This ensures the script runs only when directly executed (not when imported)
if __name__ == "__main__":
    main()
