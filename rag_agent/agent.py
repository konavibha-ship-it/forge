"""
Phase 5, part 2: An agent loop — decide, act, observe, answer.

Real LLM-powered agents (coding assistants, research agents, etc.) all
run some version of this loop:

    1. THOUGHT   - look at the question, decide what to do next
    2. ACTION    - if a tool is needed, call it (search, calculator, etc.)
    3. OBSERVATION - look at what the tool returned
    4. ANSWER    - use everything gathered to respond

This pattern is often called "ReAct" (Reason + Act) in the research
literature. In a real system, an LLM itself performs the THOUGHT step —
deciding which tool to use based on understanding the question. Here we
stand in for that decision-making with simple rules, so you can see the
exact shape of the loop without needing a downloaded model. Swapping the
rule-based "decide_action" function for a real LLM call later (see the
notes at the bottom) turns this into a genuine LLM agent.

Requires retrieval.py in the same folder.
"""

from __future__ import annotations
from retrieval import TFIDFVectorizer, SimpleVectorStore


# ---------------------------------------------------------------------------
# Tools available to the agent. Each tool is just a function that takes a
# query and returns a result. Real agents often have many of these:
# web search, code execution, calculators, file readers, APIs, etc.
# ---------------------------------------------------------------------------

def tool_calculator(expression: str) -> str:
    """A basic calculator tool — evaluates simple arithmetic safely."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}})  # restricted eval
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


def make_search_tool(store: SimpleVectorStore):
    """Returns a search tool bound to a specific knowledge base (vector store)."""

    def tool_search(query: str) -> str:
        results = store.search(query, top_k=1)
        if not results or results[0][1] < 0.05:
            return "No relevant information found in the knowledge base."
        best_doc, score = results[0]
        return best_doc

    return tool_search


# ---------------------------------------------------------------------------
# Decision logic: given a question, decide which tool (if any) to use.
# This stands in for "the LLM reasons about what to do" — in a real agent,
# an actual model reads the question and picks the tool + input itself.
# ---------------------------------------------------------------------------

import re

# Finds an arithmetic expression EMBEDDED in a sentence, e.g. pulls
# "12 * (4 + 3)" out of "What is 12 * (4 + 3)?". Requires at least one
# operator so plain numbers ("in 1889") don't get misdetected as math.
MATH_PATTERN = re.compile(r"[\d\s+\-*/().]*\d[\d\s+\-*/().]*[+\-*/][\d\s+\-*/().]*\d[\d\s+\-*/().]*")


def decide_action(question: str) -> tuple[str, str]:
    """Return (tool_name, tool_input) for a given question."""
    match = MATH_PATTERN.search(question)

    # Rule 1: if an arithmetic expression is embedded anywhere in the
    # question, use the calculator on just that part.
    if match:
        return "calculator", match.group().strip()

    # Rule 2: otherwise, assume it's a knowledge question -> search.
    return "search", question


# ---------------------------------------------------------------------------
# The agent loop itself.
# ---------------------------------------------------------------------------

def run_agent(question: str, tools: dict) -> str:
    print(f"\nQuestion: {question}")

    # 1. THOUGHT
    tool_name, tool_input = decide_action(question)
    print(f"  Thought: this looks like a '{tool_name}' task.")

    # 2. ACTION
    print(f"  Action: calling {tool_name}({tool_input!r})")
    observation = tools[tool_name](tool_input)

    # 3. OBSERVATION
    print(f"  Observation: {observation}")

    # 4. ANSWER — combine the observation into a final response.
    if tool_name == "calculator":
        answer = f"The answer is {observation}."
    else:
        answer = f"Based on what I found: {observation}"

    print(f"  Answer: {answer}")
    return answer


if __name__ == "__main__":
    # Set up the same knowledge base from retrieval.py's demo.
    knowledge_base = [
        "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
        "Python is a popular programming language known for readability.",
        "Rust is a systems programming language focused on memory safety.",
        "The Great Wall of China stretches over 13,000 miles.",
        "Transformers are a neural network architecture built around attention.",
        "Mount Everest is the tallest mountain above sea level on Earth.",
        "PyTorch is a deep learning framework widely used for research.",
        "The Amazon rainforest produces a significant share of Earth's oxygen.",
    ]

    vectorizer = TFIDFVectorizer()
    vectorizer.fit(knowledge_base)
    store = SimpleVectorStore(vectorizer)
    for doc in knowledge_base:
        store.add(doc)

    tools = {
        "calculator": tool_calculator,
        "search": make_search_tool(store),
    }

    # Mixed questions — some need search, some need math — the agent
    # decides which tool fits each one.
    questions = [
        "What is 12 * (4 + 3)?",
        "Tell me about a famous tower in France.",
        "What is the tallest mountain?",
        "127 - 58",
    ]

    for q in questions:
        run_agent(q, tools)

    print("\nIf each question routed to the RIGHT tool (math -> calculator,")
    print("knowledge -> search) and the final answers make sense, the agent")
    print("loop is working correctly.")