from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from env_config import load_env

load_env()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv('groq_api_key'),
            model_name = os.getenv('llama_model_name')
            )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPPED TEXT FROM WEBSITE:
            {page_data}
            ###  INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the 
            following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE): 
            """
            )

        chain_extract = prompt_extract | self.llm
        response = chain_extract.invoke(input={'page_data':cleaned_text})
        try:
            json_parser = JsonOutputParser()
            response = json_parser.parse(response.content)
        except OutputParserException:
            raise OutputParserException("Content too big. Unable to parse jobs.")
        return response if isinstance(response, list) else [response]
    
    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are Shreyash, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools. 
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
        process optimization, cost reduction, and heightened overall efficiency. 
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ 
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
        Remember you are Shreyash, Associate Consultant at Infosys. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):        
        """
        )

        chain_email = prompt_email | self.llm
        email_response = chain_email.invoke({'job_description':str(job),
                                            'link_list':links})
        return email_response.content

if __name__ == "__main__":
    print(os.getenv("groq_api_key"))
