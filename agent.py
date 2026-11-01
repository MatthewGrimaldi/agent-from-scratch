from dotenv import load_dotenv
import anthropic
import ast
import csv
import operator
import os

load_dotenv()
client = anthropic.Anthropic()

MODEL = "claude-haiku-5-5"

# --- Half 1: the actual function (the model never sees this) ---
FAKE_PRICES = {"ICE": 172.40, "AAPL": 231.15, "MSFT": 418.90}

def get_stock_price(ticker):
    ticker = ticker.upper()
    if ticker in FAKE_PRICES:
        return f"{ticker} is trading at ${FAKE_PRICES[ticker]}"
    return f"No price found for {ticker}"

# --- Calculator: safely evaluates arithmetic only ---
OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Only basic arithmetic is allowed")

def calculator(expression):
    result = _eval(ast.parse(expression, mode="eval").body)
    return str(round(result, 4))

# --- CSV reader: only reads .csv files in this folder ---
def read_csv(filename):
    filename = os.path.basename(filename)  # blocks paths like ../../secrets
    if not filename.endswith(".csv"):
        return "Error: only .csv files can be read"
    if not os.path.exists(filename):
        return f"Error: {filename} not found"
    with open(filename, newline="") as f:
        rows = list(csv.reader(f))
    note = ""
    if len(rows) > 51:
        rows = rows[:51]
        note = "\n(Truncated to the first 50 data rows)"
    return "\n".join(",".join(r) for r in rows) + note

# --- Half 2: the description the model DOES see ---
tools = [
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol, e.g. ICE or AAPL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The stock ticker symbol",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression, e.g. '(170.05 - 161.20) / 161.20 * 100'. "
                       "Always use this for math instead of calculating in your head.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The arithmetic expression"}
            },
            "required": ["expression"],
        },
    },
    {
        "name": "read_csv",
        "description": "Read a CSV file from the project folder and return its contents as text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Name of the CSV file, e.g. prices.csv"}
            },
            "required": ["filename"],
        },
    }

]

# Maps the name the model uses to the actual Python function
TOOL_FUNCTIONS = {
    "get_stock_price": get_stock_price,
    "calculator": calculator,
    "read_csv": read_csv,
}

def get_text(content):
    return "\n".join(b.text for b in content if b.type == "text")


def run_agent(question, max_turns=10, verbose=True):
    messages = [{"role": "user", "content": question}]
    total_in = total_out = 0

    for turn in range(1, max_turns + 1):
        last_turn = turn == max_turns
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            tools=tools,
            messages=messages,
            tool_choice={"type": "none"} if last_turn else {"type": "auto"},
        )
        total_in += response.usage.input_tokens
        total_out += response.usage.output_tokens
        messages.append({"role": "assistant", "content": response.content})

        if verbose:
            for block in response.content:
                if block.type == "thinking":
                    print(f"  [turn {turn} reasoning] {block.thinking}")
                elif block.type == "text" and response.stop_reason == "tool_use":
                    print(f"  [turn {turn}] {block.text}")

        # Finished: return the final answer
        if response.stop_reason != "tool_use":
            if verbose:
                print(f"  ({turn} turn(s), {total_in} input / {total_out} output tokens)")
            return get_text(response.content)

        # Run requested tools
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = TOOL_FUNCTIONS[block.name](**block.input)
                    is_error = False
                except Exception as e:
                    result = f"Error: {e}"
                    is_error = True
                if verbose:
                    print(f"  [tool] {block.name}({block.input}) -> {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                })

        # Warn the model when the next turn is its last
        if turn == max_turns - 1:
            tool_results.append({
                "type": "text",
                "text": "You are out of steps. On your next reply, answer using only "
                        "the information you already have, and say if anything is incomplete.",
            })

        messages.append({"role": "user", "content": tool_results})

    return "Error: agent ended without an answer."


if __name__ == "__main__":
    print("Agent ready. Ask a question, or type 'quit' to exit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        print("\nAgent:", run_agent(question))