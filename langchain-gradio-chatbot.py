from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.retrievers import WikipediaRetriever
from dotenv import load_dotenv 
import gradio as gr
import time
import os

load_dotenv()

title = """<h1 align="center">chatBot 🤖 </h1>"""
hf_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

repo_id = "tiiuae/falcon-7b-instruct" 
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.1, "max_new_tokens": 200})
# other parameters: https://huggingface.co/docs/transformers/main_classes/text_generation

# Make template
template = """
You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Question: {question}
Answer:
"""
# Make chat chain
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# make wikipedia retriever
retriever = WikipediaRetriever()
#docs = retriever.get_relevant_documents(query='HUNTER X HUNTER')


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history
    
def user(user_message, history):
    return gr.update(value="", interactive=False), history + [[user_message, None]]

def bot(history):
    user_message = history[-1][0]
    bot_message = llm_chain.run(user_message)
    history[-1][1] = ""
    for character in bot_message:
        history[-1][1] += character
        time.sleep(0.025)
        yield history

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def gen_retriever():
    pass
    return ""

with gr.Blocks() as demo:
    gr.HTML(title)
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)
    with gr.Row():
        with gr.Column(scale=0.85):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter, or upload your file",
            ).style(container=False)
        with gr.Column(scale=0.15, min_width=0):
            chk = gr.CheckboxGroup(["Wikipedia", "Documents"], label="Expand my knowledge"),
            btn = gr.UploadButton("📁", file_types=["pdf", "doc"])
    
    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )

    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

demo.queue()
demo.launch()