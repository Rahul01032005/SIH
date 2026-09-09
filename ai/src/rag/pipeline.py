class RAGPipeline:
    def __init__(self, retriever, llm_client=None):
        self.retriever = retriever
        self.llm_client = llm_client

    def ask(
        self,
        question: str,
        top_k: int = 5,
        project_id: str | None = None,
        risk_context: dict | None = None,
        project_data: list[dict] | None = None,
    ) -> dict:

        if project_id and project_data:
            chunks = []
        else:
            chunks = self.retriever.search(
                question,
                top_k,
                project_id
            )

        context = "\n\n".join(
            [
                f"[{i+1}] "
                f"source={c['metadata'].get('source_file', c['metadata'].get('source'))}; "
                f"month={c['metadata'].get('snapshot_month')}\n"
                f"{c['text']}"
                for i, c in enumerate(chunks)
            ]
        )

        project_context = ""

        if project_data:
            project_lines = []

            for row in project_data:
                project_lines.append(
                    f"""
Snapshot: {row.get('snapshot_month')}
Project ID: {row.get('project_id')}
Project Name: {row.get('project_name')}
Agency: {row.get('agency')}
State: {row.get('state')}
Start Date: {row.get('start_date')}
Target Completion Date: {row.get('target_completion_date')}
Revised Completion Date: {row.get('revised_completion_date')}
Original Cost: {row.get('original_cost')} crore
Revised Cost: {row.get('revised_cost')} crore
Cumulative Expenditure: {row.get('cumulative_expenditure')} crore
Physical Progress: {row.get('physical_progress')}%
""".strip()
                )

            project_context = "\n\n".join(project_lines)

        risk = (
            f"\nML risk context (prediction, not source fact): {risk_context}"
            if risk_context
            else ""
        )

        prompt = f"""
You are a project-monitoring intelligence assistant.

Answer the user's question using ONLY the supplied information.

IMPORTANT RULES:
- For a specific project, prioritize the structured PROJECT DATA.
- Use the PAIMANA DOCUMENT CONTEXT as supporting information.
- Do not invent facts.
- If the supplied information is insufficient, clearly say so.
- Distinguish PAIMANA source facts from ML predictions.
- Do not claim that an ML prediction proves causation.
- When the question asks for the latest information, use the latest available snapshot.

PROJECT DATA:
{project_context if project_context else "No specific project data supplied."}

PAIMANA context:
{context if context else "No document context retrieved."}

{risk}

QUESTION:
{question}

ANSWER:
""".strip()

        answer = (
            self.llm_client.generate(prompt)
            if self.llm_client
            else "No LLM configured. Retrieved context is returned for review."
        )

        sources = [
            {
                key: c["metadata"].get(key)
                for key in (
                    "source_file",
                    "snapshot_month",
                    "chunk_id",
                    "source",
                )
            }
            | {"similarity_score": c["similarity_score"]}
            for c in chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": chunks,
            "project_data": project_data,
            "risk_context": risk_context,
            "prompt": prompt,
        }


def answer_question(
    question: str,
    project_id: str | None = None,
    risk_context: dict | None = None,
    project_data: list[dict] | None = None,
    pipeline: RAGPipeline | None = None,
) -> dict:

    if pipeline is None:
        raise ValueError("Pass a configured RAGPipeline instance.")

    return pipeline.ask(
        question,
        project_id=project_id,
        risk_context=risk_context,
        project_data=project_data,
    )