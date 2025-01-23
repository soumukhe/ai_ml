{"context": retriever, "question": "how to report BOI?"}

1.	"context":
•	The retriever processes the input to find relevant documents.
2.	"question":
•	RunnablePassthrough() receives "how to report BOI?" and passes it as-is to the next step.
3.	Combined Input to the Prompt:
•	After retrieving the context, the prompt receives:
•	{context: retrieved_documents, question: "how to report BOI?"}
4.	LLM:
•	The prompt templates this data and feeds it into the LLM for the final answer generation.
---
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

•	The input to the chain is expected to be a dictionary with keys that map to specific subcomponents in the chain.
•	"context" gets passed to retriever.
•	"question" gets passed to RunnablePassthrough().
•	The chain automatically routes the input to the corresponding keys in this mapping.

---
What is MultiQueryRetriever?

MultiQueryRetriever is a retriever class in LangChain that enhances the document retrieval process by generating multiple variations of a user query. These variations help to increase recall and overcome some limitations of traditional vector-based similarity searches.