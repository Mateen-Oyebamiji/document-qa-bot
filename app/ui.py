
import streamlit as st

from app.llm.rag_answer import answer_question

st.set_page_config(
    page_title="Document QA",
    page_icon="📄",
    layout="wide"
)

st.title("Document QA")
st.caption(
    "Ask questions about the LEGO Group Annual Report 2025. "
    "Answers are generated using retrieved document passages."
)

question = st.text_area(
    "Ask a question about the report",
    placeholder="Example: What was LEGO Group's revenue in 2025?",
    height=100
)

if st.button("Ask question", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            with st.spinner("Searching the report and generating an answer..."):
                result = answer_question(question.strip())

            st.subheader("Answer")
            st.markdown(result["answer"])

            with st.expander("View retrieved sources"):
                for index, source in enumerate(
                    result["sources"], start=1
                ):
                    st.markdown(
                        f"**Source {index} — Page "
                        f"{source['page']}, "
                        f"Chunk {source['chunk']}**"
                    )

                    vector_rank = source["vector_rank"] or "–"
                    keyword_rank = source["keyword_rank"] or "–"

                    st.caption(
                        f"RRF score: {source['rrf_score']:.4f} · "
                        f"vector rank: {vector_rank} · "
                        f"keyword rank: {keyword_rank}"
                    )

                    st.write(source["text"])
                    st.divider()

        except Exception as error:
            error_message = str(error)
            if "503" in error_message or "high demand" in error_message.lower():
                st.warning("The AI model is temporarily busy. "
                    "Your document is still indexed. Please try again "
                    "in a little while."
                    )
            elif "429" in error_message:
                st.warning(
                    "The AI service's request limit has been reached. "
                    "Please wait before trying again."
                )
            else:
                st.error(
                    "Something went wrong while processing your question. "
                    "Check the Qdrant connection and API configuration."
                )

            