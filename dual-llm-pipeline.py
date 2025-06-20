import subprocess
import threading
import pandas as pd
import textwrap
from io import StringIO

# Load your Titanic dataset (adjust path if needed)
df = pd.read_csv("c:/Users/MYSEL/d drive ry/GEN AI/titanic.csv/titanic.csv")

# Dictionary to store the outputs of the two models
model_outputs = {}

############################################
#        LLM COMMAND & THREADING          #
############################################

def run_model_command(model: str, prompt: str) -> str:
    print(f"\nRunning {model} with prompt:\n'{prompt}'")
    process = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        capture_output=True
    )
    return process.stdout.decode().strip()


def model_thread_entry(model_name, prompt, result_holder):
    result = run_model_command(model_name, prompt)
    result_holder[model_name] = result

############################################
#         CODE SNIPPET PROCESSING         #
############################################

def clean_code_snippet(raw_output: str) -> str:
    """
    Extracts and returns the final Pandas expression line from the output.
    """
    lines = raw_output.strip().splitlines()
    for line in reversed(lines):
        if "df.loc[" in line:
            return line.strip()
    return ""


def fix_llama_slice_issues(snippet: str) -> str:
    """
    If Llama3.2 code tries invalid slice like df['PassengerId'] == 1:10,
    transform it to a valid .between(1,10) expression.
    Also unify multiple .loc calls.
    """
    import re
    snippet_fixed = snippet

    # Fix 'df['PassengerId'] == 1:10' -> 'df['PassengerId'].between(1,10)'
    if "== 1:10" in snippet_fixed:
        snippet_fixed = snippet_fixed.replace(
            "df['PassengerId'] == 1:10", 
            "df['PassengerId'].between(1,10)"
        )

    # Unify double .loc usage if present
    pattern = r"df\.loc\[(.*?)\]\.(loc\[.*?\])"
    match = re.search(pattern, snippet_fixed)
    if match:
        first_condition = match.group(1)  # e.g. df['PassengerId'].between(1,10), 'Name'
        second_loc = match.group(2)       # e.g. loc[df['Sex'] == 'female']
        parts = first_condition.split(",")
        if len(parts) == 2:
            conditionA = parts[0].strip()
            colA = parts[1].strip()
            pattern2 = r"loc\[(.*?)\]"
            match2 = re.search(pattern2, second_loc)
            if match2:
                conditionB = match2.group(1).strip()  # e.g. df['Sex'] == 'female'
                snippet_fixed = (
                    f"df.loc[({conditionA}) & ({conditionB}), {colA}]"
                )

    return snippet_fixed


def execute_pandas_query(code_snippet: str):
    try:
        local_context = {"df": df}
        result = eval(code_snippet, {}, local_context)
        return result
    except Exception as exc:
        return f"Error executing query: {exc}"

############################################
#        COMPARISON & SUMMARIZATION       #
############################################

def compare_and_merge(model_name: str, dataA: str, dataB: str, description: str):
    if dataA.strip() == dataB.strip():
        return dataA, "No discrepancy."
    else:
        compare_prompt = textwrap.dedent(f"""
        We have two different {description} outputs.
        Compare them and merge into a single final version.
        Point out any important differences.

        Output A:
        {dataA}

        Output B:
        {dataB}
        """)
        merged_result = run_model_command(model_name, compare_prompt)
        return merged_result, "Discrepancy found and resolved."


def data_preview_to_string(obj, rows=5):
    if isinstance(obj, pd.DataFrame):
        return obj.head(rows).to_string(index=False)
    return str(obj)


def short_summary_prompt(raw_data: str) -> str:
    """
    Create a short summary prompt that discourages chain-of-thought.
    """
    return textwrap.dedent(f"""
    Summarize the following data in a brief paragraph.
    Do not reveal your reasoning or chain-of-thought.

    Data:
    {raw_data}
    """).strip()

############################################
#           PROMPT BUILDING & LOOP        #
############################################

def build_single_line_prompt(user_query: str) -> str:
    instruction = textwrap.dedent(f"""
    We have a Titanic dataset in a Pandas DataFrame named df.
    The columns are: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'].

    Return exactly ONE line of valid Python code (no comments, no explanation) that fetches the desired rows/columns.
    Use only these columns exactly as they appear, if relevant.

    The user's request is: "{user_query}"

    To filter a range of PassengerId from X to Y, use df['PassengerId'].between(X, Y).
    If you want to also filter by another condition, combine them with & in a single .loc.

    For example:
    df.loc[df['PassengerId'].between(1,10) & (df['Sex'] == 'female'), 'Name']

    Do not include any variable assignments or multiline statements.
    Just a single-line Pandas expression referencing df.
    """)
    return instruction.strip()


def start_chat():
    print(
        "\n"
        "====================================================================\n"
        "               Titanic Dataset Chat Interface\n"
        "--------------------------------------------------------------------\n"
        " This dataset contains information about Titanic passengers, such\n"
        " as their names, ages, ticket fares, passenger class, survival, etc.\n"
        "--------------------------------------------------------------------\n"
        "   Ask anything like:\n"
        "    - 'What is the name and age of passenger 16?'\n"
        "    - 'List all female passengers in first class.'\n"
        "    - 'Show me the oldest passengers.'\n"
        "--------------------------------------------------------------------\n"
        "   Type 'exit' to quit.\n"
        "====================================================================\n"
    )

    while True:
        question = input("\nYour request (or 'exit'): ")
        if question.strip().lower() == "exit":
            break

        # 1) Build prompt for single-line code
        prompt_text = build_single_line_prompt(question)

        # 2) Generate queries from both local models in parallel
        threads = []
        for model in ["deepseek-r1", "llama3.2"]:
            t = threading.Thread(
                target=model_thread_entry,
                args=(model, prompt_text, model_outputs)
            )
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        code_model1 = clean_code_snippet(model_outputs.get("deepseek-r1", ""))
        code_model2 = clean_code_snippet(model_outputs.get("llama3.2", ""))

        # Attempt to fix known slice issues automatically for Llama3.2
        code_model2 = fix_llama_slice_issues(code_model2)

        print("\n============== Generated Single-Line Code ==============")
        print(f"deepseek-r1:\n{code_model1}")
        print(f"llama3.2 (possibly fixed):\n{code_model2}")

        # 3) Execute both code snippets
        result1 = execute_pandas_query(code_model1)
        result2 = execute_pandas_query(code_model2)

        # Check for errors
        err1 = (isinstance(result1, str) and result1.startswith("Error executing query"))
        err2 = (isinstance(result2, str) and result2.startswith("Error executing query"))

        if err1:
            print("\n[deepseek-r1 error]", result1)
        if err2:
            print("\n[llama3.2 error]", result2)
        if err1 or err2:
            continue

        # 4) Count rows (use len() for consistent results)
        count1 = len(result1) if isinstance(result1, pd.DataFrame) else 0
        count2 = len(result2) if isinstance(result2, pd.DataFrame) else 0

        print(f"\nRows from deepseek-r1: {count1}")
        print(f"Rows from llama3.2:   {count2}")

        # If either is > 10, ask for refinement
        if count1 > 10 or count2 > 10:
            print("\nToo many rows returned. Please refine your query.")
            continue

        # 5) Compare & unify raw data
        rawA = data_preview_to_string(result1, rows=5)
        rawB = data_preview_to_string(result2, rows=5)
        merged_raw, raw_status = compare_and_merge("deepseek-r1", rawA, rawB, "raw data")

        # 6) Summaries - short, no chain-of-thought
        promptA = short_summary_prompt(rawA)
        promptB = short_summary_prompt(rawB)

        summA = run_model_command("deepseek-r1", promptA)
        summB = run_model_command("llama3.2", promptB)

        merged_summary, summary_status = compare_and_merge("deepseek-r1", summA, summB, "summary")

        # 7) Final structured output
        print("\n==================== FINAL OUTPUT ====================")

        print("RAW DATA (deepseek-r1):")
        print(rawA)
        print("\nRAW DATA (llama3.2):")
        print(rawB)
        print("\n--- MERGED RAW DATA (up to 5 rows) ---")
        print(merged_raw)
        print(f"Raw Data Status: {raw_status}")

        print("\n======================================================")
        print("SUMMARY (deepseek-r1):")
        print(summA)
        print("\nSUMMARY (llama3.2):")
        print(summB)
        print("\n--- MERGED SUMMARY ---")
        print(merged_summary)
        print(f"Summary Status: {summary_status}")
        print("======================================================")

if __name__ == "__main__":
    start_chat()
