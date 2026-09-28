"""
Terminal chat loop against the same retrieval/generation logic as app.py —
no HTTP server needed. Now keeps conversation history across turns, so
follow-up questions work the same way they would through /ask.

Run:
    python query.py
Type 'exit' to quit.
"""

from app import AskRequest, ChatTurn, _resolve_question, _retrieve_and_format, _generation_chain


def main() -> None:
    history: list[ChatTurn] = []
    print("Nimbus Docs RAG — type a question ('exit' to quit)\n")

    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        standalone_question = _resolve_question(question, history)
        context, sources = _retrieve_and_format(standalone_question)
        answer = _generation_chain.invoke({"context": context, "question": standalone_question})

        print(f"\n{answer}")
        print(f"[sources: {', '.join(sources) or 'none'}]\n")

        history.append(ChatTurn(question=question, answer=answer))


if __name__ == "__main__":
    main()
