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

# This is a simple standalone implementation showing rag pipeline using Nvidia AI Foundational Models.
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
# Component #0.5 - UI / Header
############################################

import streamlit as st
import os

# Page settings 
st.set_page_config(
    layout="wide",
    page_title="Resume Evaluation Assistant", 
    page_icon = "🤖",
    initial_sidebar_state="expanded")

# Page title 
st.header('Resume Evaluation Assistant 🤖📝', divider='rainbow')

# Custom CSS
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")

# Page description 
# st.title("Resume Evaluation Assistant")
st.markdown('''Job listings currently receive hundreds of resumes. 
This system streamlines that process through leveraging NVIDIA AI Foundational models to 
evaluate resumes via a RAG (Retrieval-Augmented Generation) pipeline.
Upload resumes, enter a job description, and get AI-based recommendations 
for top applicants. ''')
st.warning("This is a proof of concept and should only be used to supplement traditional evaluation methods.", icon="⚠️")

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

    if submitted and uploaded_files:
        for uploaded_file in uploaded_files:
            st.info(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path.join(DOCS_DIR, uploaded_file.name),"wb") as f:
                f.write(uploaded_file.read())

############################################
# Component #2 - Embedding Model and LLM
############################################

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# Make sure to export your NVIDIA AI Playground key as NVIDIA_API_KEY!
llm = ChatNVIDIA(model="ai-llama3-70b", temperature=0)
document_embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", model_type="passage")
query_embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", model_type="query")

############################################
# Component #3 - Vector Database Store
############################################

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import pickle
import os

with st.sidebar:
    # Option for using an existing vector store
    use_existing_vector_store = st.radio("Use existing vector store if available", ["Yes", "No"], horizontal=True)

# Create a chain for name extraction
name_extraction_prompt = PromptTemplate(
    input_variables=["resume_text", "file_name"],
    template="Only output the full name of the candidate based on their resume. You have access to the file name and the content. It might be clear from the file name, but if not, it should be the only name listed in the content, do not include a nickname. If the name isn't clear, choose nickname. \n\n Filename of resume: Jane (JD) Doe Resume Content: J. D. Doe (Jane) 10 years of experience... Candidate name: Jane Doe \n\n Filename of resume: {file_name}\n\n Resume Content: {resume_text}\n\nCandidate name:"
)

name_extraction_chain = (
    RunnablePassthrough.assign(
        resume_text=lambda x: x["resume_text"],
        file_name=lambda x: x["file_name"]
    )
    | name_extraction_prompt
    | llm
    | StrOutputParser()
)
# Load raw documents from the directory
DOCS_DIR = os.path.abspath("./uploaded_docs")
raw_documents = DirectoryLoader(DOCS_DIR).load()

# Check for existing vector store file
vector_store_path = "vectorstore.pkl"
vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None

resume_map_path = "resumemap.pkl"

valid_cand_path = "validcand.pkl"

first_doc = False 

if use_existing_vector_store == "Yes" and vector_store_exists:
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    with open(resume_map_path, "rb") as f:
        resume_name_map = pickle.load(f)
    with open(valid_cand_path, "rb") as f:
        valid_candidates = pickle.load(f)
    with st.sidebar:
        st.info("Existing vector store loaded successfully.")
else:
    with st.sidebar:
        if raw_documents:
            valid_candidates = set()
            documents = []
            resume_name_map = {}
            
            with st.spinner("Processing documents..."):
                for doc in raw_documents:
                    try:
                        filename = os.path.basename(doc.metadata.get('source', ''))
                        resume_content = doc.page_content
                        
                        # Extract candidate name
                        candidate_name = name_extraction_chain.invoke({
                            "resume_text": resume_content,
                            "file_name": filename
                        }).strip()
                        
                        # Add metadata to document
                        processed_doc = Document(
                            page_content=resume_content,
                            metadata={
                                "source": doc.metadata.get('source', ''),
                                "candidate_name": candidate_name
                            }
                        )
                        
                        # Standardize capitalization for the name map
                        standardized_name = candidate_name.lower()
                        resume_name_map[standardized_name] = filename
                        valid_candidates.add(standardized_name)

                        # Add document to the list
                        documents.append(processed_doc)
                        
                    except Exception as e:
                        st.warning(f"Error processing document {filename}: {str(e)}")
                        continue
            
            with st.spinner("Adding documents to vector database..."):
                for doc in documents:
                    try:
                        if first_doc == False:
                            # Initalize if it is the first document
                            vectorstore = FAISS.from_documents([doc], document_embedder)  # Initialize empty FAISS index
                            first_doc = True
                        else:
                            vectorstore.add_documents([doc])
                    except Exception as e:
                        print("Cannot process " + doc.metadata['candidate_name'] + "'s resume.")
                        continue
            
            with st.spinner("Saving vector store and metadata..."):
                try:
                    with open(vector_store_path, "wb") as f:
                        pickle.dump(vectorstore, f)
                    with open(resume_map_path, "wb") as f:
                        pickle.dump(resume_name_map, f)
                    with open(valid_cand_path, "wb") as f:
                        pickle.dump(valid_candidates, f)
                    st.info("Vector store created and saved.")
                except Exception as e:
                    st.error(f"Error saving vector store and metadata: {str(e)}")
        else:
            st.warning("No documents available to process!", icon="⚠️")


############################################
# Component #4 - LLM Response Generation
############################################
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from streamlit_pdf_viewer import pdf_viewer
from docx2pdf import convert
from streamlit_extras.stylable_container import stylable_container 
import tempfile
import os
import base64
import re

valid_candidates_list = ', '.join(valid_candidates)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Based on the given job description, identify the top 5+ applicants from only the provided context information. Prioritize how well the skills and experience the candidates have with the job role. Unrelated roles in other industries should not count. If you cannot find any relevant candidates for the job, please state that. Do not answer any questions that are inappropriate. Here's an example of who would not be a good fit: \n John is not a good fit because his resume focuses more on program management skills and experience rather than snowing technical experience in storage, servers, computing, data center, high performance computing, AI, etc. \n Keith was not a good fit because her resume was too focused on health care specific domain knowledge and experience rather than datacenter, storage, servers, compute, networking, etc."),
    ("user", "Job Description: {input}\n\n The only candidates you have access to:\n{context}\n Only pick candidates from the following list with valid names: " + valid_candidates_list + "\n\nHere is only a numbered list of the top 10 candidates using their names from the previous list. Then, describe briefly and as close to the job description as possible why you ranked the candidate like that: \\n\\n 1. **Jane Doe**: Jane's a good choice because of direct experience with cloud, storage, server and high performance computing experience.\\n\\n\\")
])

job_description = st.text_area("Enter the job description:", value=SAMPLE_JOB_DESCRIPTION, height=350)
llm = ChatNVIDIA(model="ai-llama3-70b",temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

def extract_name(raw_output):
    # If the output is a ranked candidate 
    if raw_output[0] in "1234567890":
        # Extract candidate information from the model output
        number = str(re.match(r'^\d+', raw_output).group())
        candidate = raw_output.split(':')[0].strip()
        description = raw_output.split(':')[1] if ':' in raw_output else ''

        # Clean up candidate name string
        candidate = candidate.replace('*','')
        candidate = candidate.strip()
        while candidate and not candidate[0].isalpha():
            candidate = candidate[1:].lstrip()
        candidate = candidate.rstrip('.')
    else:
        number = ""
        candidate = ""
        description = ""
    return number, candidate, description

if st.button("Evaluate Resumes") and vectorstore is not None:
    if job_description:
        with st.spinner("Fetching resumes..."):
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
            retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
            docs = retriever.invoke(job_description)
            context = ""
            for doc in docs:
                candidate_name = doc.metadata.get('candidate_name', 'Unknown')
                pdf_filename = os.path.basename(doc.metadata.get('source', ''))
                context += f"[CANDIDATE START] Candidate Name: {candidate_name}\n"
                context += doc.page_content + "[CANDIDATE END]\n\n"
        
        chain = prompt_template | llm | StrOutputParser()
        augmented_input = {"input": job_description, "context": context}
        
        with st.spinner("Evaluating resumes..."):
            response = chain.invoke(augmented_input)
        
        st.markdown("### Top Applicants:")
        
        # Split the response into individual candidate evaluations
        candidates = response.split("\n\n")
  

        for candidate in candidates:
            # Extract candidate name from the evaluation
            number, case_correct_name, description = extract_name(candidate)
            lower_name = case_correct_name.lower()

            # If evaluation is actually a candidate: 
            if number != "":
                # Create a container for each candidate
                with stylable_container(
                    key="container_with_border",
                    css_styles="""
                        {
                            border: 0px solid #ccccd4;
                            border-radius: 0.75rem;
                            padding: calc(1em + 2px);
                            background-color: #f0f2f6
                        }
                        """,
                ):
                    # Display candidate evaluation
                    st.markdown("##### " + number + ". " + case_correct_name)
                    st.markdown(description)

                    if lower_name in resume_name_map:
                        file_name = resume_name_map[lower_name]
                        file_path = os.path.join(DOCS_DIR, file_name)
                        
                        # Create an expander for the document viewer
                        with st.expander("View Resume"):
                            if os.path.exists(file_path):
                                # Check if the file is a Word document
                                if file_name.lower().endswith(('.doc', '.docx')):
                                    # Convert Word to PDF
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                                        convert(file_path, tmp_pdf.name)
                                        pdf_path = tmp_pdf.name
                                else:
                                    pdf_path = file_path

                                # Read and display the PDF
                                with open(pdf_path, "rb") as f:
                                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="600" type="application/pdf"></iframe>'
                                st.markdown(pdf_display, unsafe_allow_html=True)

                                # Clean up temporary file if created
                                if file_name.lower().endswith(('.doc', '.docx')):
                                    os.unlink(pdf_path)
                            else:
                                st.warning("File not found.")
                    else:
                        st.info("No resume available for this candidate. Check if it was deleted and/or the filename was changed")
            else:
                st.markdown(candidate)
                
    else:
        st.warning("Please enter a job description.", icon="⚠️")

st.markdown("---")
st.markdown("<div class='footer'>Powered by <a href='https://ai.nvidia.com/' style='color: #666; text-decoration: none;' class='hover-link'>NVIDIA</a> | © 2024 <a href='https://www.linkedin.com/in/alysawyer/' style='color: #666; text-decoration: none;' class='hover-link'>Alyssa Sawyer</a></div>", unsafe_allow_html=True)