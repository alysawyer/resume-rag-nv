# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a simple standalone implementation showing rag pipeline using Nvidia AI Foundational models.
# It uses a simple Streamlit UI and one file implementation of a minimalistic RAG pipeline.

SAMPLE_JOB_DESCRIPTION = """NVIDIA is looking for a senior technical program manager to lead new product introduction for hardware for NVIDIA Infrastructure Specialists (NVIS) team. We want you to collaborate with cross-functional teams, including professional services, solutions architects, development engineers, hardware and software engineering, data center operations, project managers, product managers, and go-to-market strategy teams. We want your primary focus is to ensure NVIS' readiness as NVIDIA introduces new hardware including deployment, provisioning and validation for early customers. You will be working with and have the support of the global NVIS team and in turn supporting the team as delivery transitions to production deployment.
What will you be doing:
	•	Leading end-to-end execution of service programs related to new hardware product introduction and other related programs, ensuring adherence to project timelines, budgets, and quality standards. This includes applying your expertise to drive technical strategy, planning, and execution with the team, partners and customers.
	•	Developing comprehensive program delivery plans to achieve successful project outcomes, including scoping, resource allocation, task sequencing, and risk management strategies.
	•	Engaging and building internal and external customer relationships, understanding their needs and expectations, and effectively communicating program status, risks, and mitigation plans to ensure customer satisfaction. This includes engaging executives, engineering teams, and external partners and ensuring visibility and informed decision-making.
	•	You will work with partners, decomposing requirements into technical execution plans, tracking progress towards goals, and reporting status to customers and technological leadership.
	•	Establishing and maintaining project metrics and key performance indicators to track progress, evaluate program success, Identify areas for process improvement, and drive initiatives to improve service program.
What we need to see:
	•	BS/MS Engineering or Computer Science (or equivalent experience)
	•	12+ years of experience in project delivery management
	•	Minimum 5 years of experience in providing field services and/or customer support for hardware & software products
	•	In-depth knowledge of data center environments, servers, and network equipment
	•	Strong interpersonal skills and the ability to work directly with customers
	•	Supreme leadership skills across broad and diverse functional teams
	•	Strong ability prioritize/multi-task easily with limited supervision
	•	Experience leading global projects
NVIDIA is widely considered to be one of the technology world’s most desirable employers. We have some of the most forward-thinking and hardworking people in the world working for us. If you're creative and autonomous, we want to hear from you!
"""

############################################
# Component #0 - UI / Header
############################################

import streamlit as st
import os
from PIL import Image

# load icon
im = Image.open('assets/icon.png')

# page settings and page title 
st.set_page_config(
    layout="wide",
    page_title="Resume Evaluation Assistant", 
    page_icon = "🤖",
    initial_sidebar_state="expanded")

# Custom CSS
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")


# title and description to the top of the page
st.title("Resume Evaluation Assistant 📑")
st.markdown('''Job listings currently receive hundreds of resumes. 
This system streamlines that process through leveraging NVIDIA AI Foundational models to 
evaluate resumes via a RAG (Retrieval-Augmented Generation) pipeline.
Upload resumes, enter a job description, and get AI-powered recommendations 
for top applicants. ''')
st.warning("This is a proof of concept and should only be used to supplement traditional evaluation methods.")



############################################
# Component #1 - Document Loader
############################################

with st.sidebar:
    st.subheader("Upload Applicant Information")

    DOCS_DIR = os.path.abspath("./uploaded_docs")
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    
    with st.form("my-form", clear_on_submit=True):        
        uploaded_files = st.file_uploader("Upload Resumes:", accept_multiple_files = True)
        submitted = st.form_submit_button("Upload!")

    if submitted:
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.success(f"File {uploaded_file.name} uploaded successfully!")
                with open(os.path.join(DOCS_DIR, uploaded_file.name),"wb") as f:
                    f.write(uploaded_file.read())




############################################
# Component #2 - Embedding Model and LLM
############################################

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# make sure to export your NVIDIA AI Playground key as NVIDIA_API_KEY!
llm = ChatNVIDIA(model="ai-llama3-70b")
document_embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", model_type="passage")
query_embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", model_type="query")

############################################
# Component #3 - Vector Database Store
############################################

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
import pickle

with st.sidebar:
    # Option for using an existing vector store
    use_existing_vector_store = st.radio("Use existing vector store if available", ["Yes", "No"], horizontal=True)

# Path to the vector store file
vector_store_path = "vectorstore.pkl"

# Load raw documents from the directory
raw_documents = DirectoryLoader(DOCS_DIR).load()


# Check for existing vector store file
vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None
if use_existing_vector_store == "Yes" and vector_store_exists:
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    with st.sidebar:
        st.success("Existing vector store loaded successfully.")
else:
    with st.sidebar:
        if raw_documents:
            with st.spinner("Splitting documents into chunks..."):
                text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                documents = text_splitter.split_documents(raw_documents)

            with st.spinner("Adding document chunks to vector database..."):
                vectorstore = FAISS.from_documents(documents, document_embedder)

            with st.spinner("Saving vector store"):
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vectorstore, f)
            st.success("Vector store created and saved.")
        else:
            st.warning("No documents available to process!", icon="⚠️")

# [Previous imports and setup code remain unchanged]

############################################
# Component #4 - LLM Response Generation
############################################

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Based on the given job description, identify the top applicants for the job. Explain your reasoning for the ranking."),
    ("user", "Job Description: {input}\n\nAvailable Resumes:\n{context}\n\nPlease provide a numbered list of the top applicants, including their names:")
])

job_description = st.text_area("Enter the job description:", value=SAMPLE_JOB_DESCRIPTION, height=350)
llm = ChatNVIDIA(model="ai-llama3-70b")

# Create a reranker
compressor = LLMChainExtractor.from_llm(llm)

if st.button("Evaluate Resumes") and vectorstore is not None:
    if job_description:
        with st.spinner("Fetching resumes..."):
            # Create a retriever with reranking
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})  # Retrieve more documents initially
            retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
            
            # Retrieve and rerank documents
            docs = retriever.invoke(job_description)
            
            context = ""
            for doc in docs:
                context += doc.page_content + "\n\n"

            chain = prompt_template | llm | StrOutputParser()
            augmented_input = {"input": job_description, "context": context}

        with st.spinner("Evaluating resumes..."):
            response = chain.invoke(augmented_input)
            st.markdown("### Top Applicants:")
            st.markdown(response)
    else:
        st.warning("Please enter a job description.")
    
st.markdown("---")
st.markdown("<div class='footer'>Powered by NVIDIA | © 2024 <a href='https://www.linkedin.com/in/alysawyer/' style='color: #666; text-decoration: none;' class='hover-link'>Alyssa Sawyer</a></div>", unsafe_allow_html=True)