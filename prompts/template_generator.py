from langchain_core.prompts import PromptTemplate

raw_prompt = """
Please explain the research topic: {paper}

Explanation Style: {style}
Explanation Length: {length}

1. Mathematical Details:
- Include relevant equations where appropriate.
- Explain concepts using intuitive examples or code snippets if helpful.

2. Analogies:
- Use relatable analogies to simplify complex ideas.

Ensure the explanation is clear, accurate, and aligned with the selected style and length.
"""

template = PromptTemplate(
    template=raw_prompt,
    input_variables=['paper', 'style', 'length'],
    validate_template=True,
)

template.save('template.json')
