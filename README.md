# RAG Assistant for Hiring Managers

RAG web application that will return the best resumes for a given job description. 

## Acknowledgements

 - [Rag in 5 Minutes](https://github.com/NVIDIA/GenerativeAIExamples/tree/4e86d75c813bcc41d4e92e430019053920d08c94/community/5_mins_rag_no_gpu)

## Deployment

1. Create a python virtual environment and activate it:

   ```comsole
   python3 -m virtualenv genai
   source genai/bin/activate
   ```

1. From the root of this repository, install the requirements:

   ```console
   pip install -r requirements.txt
   ```

1. Add your NVIDIA API key as an environment variable:

   ```console
   export NVIDIA_API_KEY="nvapi-*"
   ```

   If you don't already have an API key, visit the [NVIDIA API Catalog](https://build.ngc.nvidia.com/explore/), select on any model, then click on `Get API Key`.

1. Run the example using Streamlit:

   ```console
   streamlit run main.py
   ```

1. Test the deployed example by going to `http://<host_ip>:8501` in a web browser.

   Click **Browse Files** and select all resumes that need to be ranked.
   After selecting, click **Upload!** to complete the ingestion process.

   Then, upload a job description to return a ranking of candidates.


