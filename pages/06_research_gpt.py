import json
import streamlit as st
import openai as client
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.schema import SystemMessage


st.set_page_config(
    page_title="ResearchGPT",
    page_icon="🔍🔎",
)

st.markdown(
    """
        # ResearchGPT

        Welcome to ResearchGPT.

        Write down the name of a theme and our Agent will do the resarch for you.
    """
)

# Wikipedia
# class TopicWikipediaSearchToolArgsSchema(BaseModel):
#     query: str = Field(description="The query you will search in Wikipedia for.")


# class TopicWikipediaSearchTool(BaseTool):
#     name = "TopicWikipediaSearchTool"
#     description = """
#         Use this tool to research for a theme.
#         It takes a query as an argument.
#         Example query: Research about the XZ backdoor
#     """

#     args_schema: Type[TopicWikipediaSearchToolArgsSchema] = (
#         TopicWikipediaSearchToolArgsSchema
#     )

#     def _run(self, query):
#         wpd = WikipediaAPIWrapper()
#         return wpd.run(query)


# class TopicDuckDuckGoSearchToolArgsSchema(BaseModel):
#     query: str = Field(description="The query you will search in DuckDuckGo for.")


# DuckDuckGo
# class TopicDuckDuckGoSearchTool(BaseTool):
#     name = "TopicDuckDuckGoSearchTool"
#     description = """
#         Use this tool to research for a theme.
#         It takes a query as an argument.
#         Example query: Research about the XZ backdoor
#     """

#     args_schema: Type[TopicDuckDuckGoSearchToolArgsSchema] = (
#         TopicDuckDuckGoSearchToolArgsSchema
#     )


#     def _run(self, query):
#         ddg = DuckDuckGoSearchAPIWrapper()
#         return ddg.run(query)
def get_wpd_result(inputs):
    wpd = WikipediaAPIWrapper()
    theme = inputs["theme"]
    return wpd.run(theme)


def get_ddg_result(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    theme = inputs["theme"]
    return ddg.run(theme)


functions = [
    {
        "type": "function",
        "function": {
            "name": "get_wpd_result",
            "description": "Returns content searched in Wikipedia with the given theme.",
            "parameters": {
                "type": "object",
                "properties": {
                    "theme": {
                        "type": "string",
                        "description": "The theme that user wants to search",
                    }
                },
                "required": [
                    "theme",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ddg_result",
            "description": "Returns content searched in DuckDuckGo with the given theme.",
            "parameters": {
                "type": "object",
                "properties": {
                    "theme": {
                        "type": "string",
                        "description": "The theme that user wants to search",
                    }
                },
                "required": [
                    "theme",
                ],
            },
        },
    },
]

functions_map = {
    "get_wpd_result": get_wpd_result,
    "get_ddg_result": get_ddg_result,
}

assistant_id = "asst_xOBOeZm04lCOfUlMHaJ10BwZ"


# agent = initialize_agent(
#     llm=llm,
#     verbose=True,
#     agent=AgentType.OPENAI_FUNCTIONS,
#     handle_parsing_errors=True,
#     tools=[
#         TopicWikipediaSearchTool(),
#         TopicDuckDuckGoSearchTool(),
#     ],
#     agent_kwargs={
#         "system_message": SystemMessage(
#             content="""
#             You are a web surfer.
#             You want to research a theme in Wikipedia or DuckDuckGo.
#             If there is a website, go to the website, and write down the content in .txt file.
#         """
#         )
#     },
# )


def get_run(run_id, thread_id):
    return client.beta.threads.runs.retrieve(run_id=run_id, thread_id=thread_id)


def send_message(thread_id, content):
    return client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content,
    )


def get_messages(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id).data
    messages.reverse()
    for message in messages:
        print(f"{message.role}: ", end=" ")
        if message.content:
            print(message.content[0].text.value)


def get_tool_outputs(run_id, thread_id):
    run = get_run(run_id, thread_id)
    outputs = []
    for action in run.required_action.submit_tool_outputs.tool_calls:
        action_id = action.id
        function = action.function
        print(f"Calling function: {function.name} with arg {function.arguments}")
        outputs.append(
            {
                "output": functions_map[function.name](json.loads(function.arguments)),
                "tool_call_id": action_id,
            }
        )
    return outputs


def submit_tool_outputs(run_id, thread_id):
    outputs = get_tool_outputs(run_id, thread_id)
    return client.beta.threads.runs.submit_tool_outputs(
        run_id=run_id,
        thread_id=thread_id,
        tool_outputs=outputs,
    )


def save_api_key(api_key):
    st.session_state["key"] = api_key
    st.session_state["api_key_bool"] = True


with st.sidebar:
    if "key" not in st.session_state:
        st.session_state["key"] = None

    api_key = st.text_input(
        "자신의 OPENAI_API_KEY를 입력해 주세요.",
        disabled=st.session_state["key"] is not None,
    ).strip()

    if api_key:
        save_api_key(api_key)
        st.write("API_KEY가 저장되었습니다.")

    if button := st.button("저장", disabled=st.session_state["key"] is not None):
        save_api_key(api_key)
        if api_key == "":
            st.write("API_KEY를 넣어주세요.")

    theme = st.text_input("Write the name of the theme you are interested on.")

if theme:
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "I want to search the theme i provided in DuckDuckGo and Wikipedia",
            }
        ]
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    get_messages(thread.id)
    submit_tool_outputs(run.id, thread.id)
    get_tool_outputs(run.id, thread.id)
    # while get_run(run.id, thread.id).status != "completed":
    #     send_message(thread.id, "Go ahead!")
    # get_messages(thread.id)
    # get_tool_outputs(run.id, thread.id)
    # with open("research.txt", "wb") as file:
    #     encoded = result["output"].encode("utf-8")
    #     file.write(encoded)

    # st.write(result)
