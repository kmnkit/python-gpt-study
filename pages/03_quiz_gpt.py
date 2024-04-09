import json
import streamlit as st

from langchain.document_loaders import UnstructuredFileLoader
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

from langchain.schema import BaseOutputParser, output_parser


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuitGPT",
    page_icon="📝",
)

if "key" not in st.session_state:
    st.session_state["key"] = None


st.title("QuizGPT")


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a helpful assistant that is role playing as a teacher.
                    
                Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
                
                Each question should have 4 answers, three of them must be incorrect and one should be correct.
                    
                Use (o) to signal the correct answer.
                    
                Question examples:
                    
                Question: What is the color of the ocean?
                Answers: Red|Yellow|Green|Blue(o)
                    
                Question: What is the capital or Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut
                    
                Question: When was Avatar released?
                Answers: 2007|2001|2009(o)|1998
                    
                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o)|Painter|Actor|Model
                    
                Your turn!
                    
                Context: {context}
            """,
        )
    ]
)


questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a powerful formatting algorithm.
                
                You format exam questions into JSON format.
                Answers with (o) are the correct ones.
                
                Example Input:
                Question: What is the color of the ocean?
                Answers: Red|Yellow|Green|Blue(o)
                    
                Question: What is the capital or Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut
                    
                Question: When was Avatar released?
                Answers: 2007|2001|2009(o)|1998
                    
                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o)|Painter|Actor|Model
                
                
                Example Output:
                
                ```json
                {{ "questions": [
                        {{
                            "question": "What is the color of the ocean?",
                            "answers": [
                                    {{
                                        "answer": "Red",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Yellow",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Green",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Blue",
                                        "correct": true
                                    }},
                            ]
                        }},
                                    {{
                            "question": "What is the capital or Georgia?",
                            "answers": [
                                    {{
                                        "answer": "Baku",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Tbilisi",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "Manila",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Beirut",
                                        "correct": false
                                    }},
                            ]
                        }},
                                    {{
                            "question": "When was Avatar released?",
                            "answers": [
                                    {{
                                        "answer": "2007",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "2001",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "2009",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "1998",
                                        "correct": false
                                    }},
                            ]
                        }},
                        {{
                            "question": "Who was Julius Caesar?",
                            "answers": [
                                    {{
                                        "answer": "A Roman Emperor",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "Painter",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Actor",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Model",
                                        "correct": false
                                    }},
                            ]
                        }}
                    ]
                }}
                ```
                Your turn!
                Questions: {context}
            """,
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Loading File...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


def save_api_key(api_key):
    st.session_state["key"] = api_key
    st.session_state["api_key_bool"] = True


docs = None
topic = None

with st.sidebar:

    api_key = st.text_input(
        "자신의 OPENAI_API_KEY를 입력해 주세요.",
        disabled=st.session_state["key"] is not None,
    ).strip()

    if api_key:
        save_api_key(api_key)
        st.write("API_KEY가 저장되었습니다.")

    if button := st.button("저장"):
        save_api_key(api_key)
        if api_key == "":
            st.write("API_KEY를 넣어주세요.")

    if (st.session_state["api_key_bool"] == True) and (st.session_state["key"] != None):

        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
        )
        if choice == "File":
            if file := st.file_uploader(
                "Upload a .docx, .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            ):
                docs = split_file(file)

        else:
            if topic := st.text_input("Name of the article"):
                docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
            Welcome to QuizGPT.

            I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
            Get started by uploading a file or searching on Wikipedia in the sidebar.
        """
    )
else:

    start = st.button("Generate Quiz")

    if start:
        response = run_quiz_chain(docs, topic if topic else file.name)
        with st.form("questions_form"):
            st.write(response)
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                )
                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                elif value is not None:
                    st.error("Wrong!")
            button = st.form_submit_button()
