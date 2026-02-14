"""
Search knowledge base for answers to research questions
Usage: python scripts/search_knowledge.py "regime detection"
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple

KNOWLEDGE_BASE = Path("docs/knowledge")

def search_knowledge(query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
    """
    Search knowledge base for relevant Q&A
    Returns: List of (file_path, question, relevance_score)
    """
    results = []
    query_lower = query.lower()
    query_words = set(re.findall(r'\w+', query_lower))

    # Search all markdown files in knowledge base
    for md_file in KNOWLEDGE_BASE.rglob("*.md"):
        if md_file.name == "KNOWLEDGE_STRUCTURE.md":
            continue

        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract Q&A sections
        qa_pattern = r'## (Q\d+:.*?)\n\n\*\*Answer\*\*:(.*?)(?=\n\n##|\Z)'
        matches = re.findall(qa_pattern, content, re.DOTALL)

        for question, answer in matches:
            # Calculate relevance score
            text = (question + " " + answer).lower()
            text_words = set(re.findall(r'\w+', text))

            # Word overlap score
            overlap = len(query_words & text_words)
            score = overlap / max(len(query_words), 1)

            # Boost if query appears as substring
            if query_lower in text:
                score += 0.5

            if score > 0:
                results.append((str(md_file), question.strip(), score))

    # Sort by relevance
    results.sort(key=lambda x: x[2], reverse=True)

    return results[:top_k]

def print_results(results: List[Tuple[str, str, float]]):
    """Pretty print search results"""
    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} relevant Q&A:\n")

    for i, (file_path, question, score) in enumerate(results, 1):
        file_rel = Path(file_path).relative_to(KNOWLEDGE_BASE)
        print(f"{i}. [{file_rel}]")
        print(f"   {question}")
        print(f"   Relevance: {score:.2f}\n")

def show_full_answer(file_path: str, question: str):
    """Show full Q&A content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the Q&A section
    pattern = re.escape(question) + r'\n\n(.*?)(?=\n\n---\n\n##|\Z)'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        print(f"\n{'='*80}")
        print(f"File: {Path(file_path).relative_to(KNOWLEDGE_BASE)}")
        print(f"{'='*80}\n")
        print(f"## {question}\n")
        print(match.group(1).strip())
        print(f"\n{'='*80}\n")
    else:
        print("Could not extract full answer.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/search_knowledge.py <query> [--show N] [--type TYPE]")
        print("\nExamples:")
        print("  python scripts/search_knowledge.py 'regime detection'")
        print("  python scripts/search_knowledge.py 'window length' --show 1")
        print("  python scripts/search_knowledge.py 'MLP autoencoder' --type evaluation")
        print("  python scripts/search_knowledge.py 'Gate 2 FAIL' --type evaluation")
        sys.exit(1)

    query = sys.argv[1]
    show_index = None
    search_type = None

    # Parse flags
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--show" and i + 1 < len(sys.argv):
            show_index = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--type" and i + 1 < len(sys.argv):
            search_type = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    # Filter by type if specified
    if search_type == "evaluation":
        # Search only in evaluations/ directory
        old_kb = KNOWLEDGE_BASE
        KNOWLEDGE_BASE = Path("docs/knowledge/evaluations")
        results = search_knowledge(query)
        KNOWLEDGE_BASE = old_kb
    else:
        results = search_knowledge(query)

    if show_index is not None and 0 < show_index <= len(results):
        file_path, question, _ = results[show_index - 1]
        show_full_answer(file_path, question)
    else:
        print_results(results)

        if results:
            print("Tip: Use '--show N' to see full answer for result N")
            print(f"Example: python scripts/search_knowledge.py '{query}' --show 1")
