import streamlit as st
import os
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()
def ui():
    st.markdown(
        '<link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.19.1/css/mdb.min.css" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" '
        'integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" '
        'crossorigin="anonymous">',
        unsafe_allow_html=True,
    )
    
    hide_streamlit_style = """
                <style>
                    header{visibility:hidden;}
                    .main {
                        margin-top: -20px;
                        padding-top:10px;
                    }
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    .reportview-container {
                        padding-top: 0;
                    }
                    .loan-summary {
                        background-color: white;
                        padding: 20px;
                        color:black;
                        border-radius: 5px;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    }
                    .loan-summary h1 {
                        color: #4267B2;
                        font-size: 24px;
                        margin-bottom: 20px;
                    }
                    .loan-summary h2 {
                        color: #4267B2;
                        font-size: 18px;
                        margin-top: 15px;
                        margin-bottom: 10px;
                    }
                    .loan-summary hr {
                        margin: 15px 0;
                    }
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.markdown(
        """
        <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #4267B2;">
        <a class="navbar-brand" href="#" style="margin-left:30px"  target="_blank">Loan Document Analyzer</a>  
        </nav>
    """,
        unsafe_allow_html=True,
    )
ui()
# Set up OpenAI API key (ensure your .env file contains the OPENAI_API_KEY variable)
openai_api_key = st.secrets["OPENAI_KEY"]


# Initialize OpenAI LLM with the API key
@st.cache_resource
def openai_llm():
    return OpenAI(api_key=openai_api_key, temperature=0.3)

def extract_text_from_multiple_pdfs(uploaded_files):
    """Extract text from multiple uploaded PDFs using PyMuPDF."""
    extracted_texts = []
    for uploaded_file in uploaded_files:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")  # Open PDF from uploaded file
        text = ""
        for page in doc:
            text += page.get_text()  # Extract text from each page
        extracted_texts.append(text)
    return " ".join(extracted_texts)

def process_summary(extracted_text):
    """Generate a summary from the extracted text using OpenAI LLM."""
    prompt_template = """
    Summarize the following text in concise and clear language:
    {text}
    """
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    llm = openai_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(text=extracted_text)
    return summary

def process_template(extracted_text):
    """Generate a template analysis from the extracted text using OpenAI LLM."""
    prompt_template = """
    The first visit date was {first_visit_date} and the last visit date was {last_visit_date}. 
    Analyze the following text:
    {text}
    """
    first_visit_date = "placeholder for first visit date"
    last_visit_date = "placeholder for last visit date"
    prompt = PromptTemplate(
        input_variables=["first_visit_date", "last_visit_date", "text"], 
        template=prompt_template
    )
    llm = openai_llm()
    chain = LLMChain(llm=llm, prompt=prompt)
    template_analysis = chain.run(first_visit_date=first_visit_date, last_visit_date=last_visit_date, text=extracted_text)
    return template_analysis

def main():
    """Main function to run the Streamlit app."""
    st.title("Medical Document Analyzer")

    # Upload multiple PDF files
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        # Extract text from multiple PDFs using PyMuPDF
        with st.spinner("Extracting Text..."):
            extracted_text = extract_text_from_multiple_pdfs(uploaded_files)

        st.success("Text Extracted")

        col1, col2 = st.columns([1, 2])

        # Button to process documents
        if st.button('Process Documents'):
            with col1:
                with st.spinner("Summarizing"):
                    summary = process_summary(extracted_text)
                st.success("Summarized")
                st.write(summary)
            
            with col2:
                with st.spinner("Generating Template"):
                    template = process_template(extracted_text)
                st.success("Template Generated")
                st.write(template)

    st.write("Upload your files")

if __name__ == "__main__":
    main()