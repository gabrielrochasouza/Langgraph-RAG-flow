from typing import Dict, Any

from graph.chain.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context":  documents, "question": question})
    return {"documents": documents, "question": question,"generation": generation}
