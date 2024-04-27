from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

import os
from Bio import Entrez
from datetime import datetime
import re

os.environ['OPENAI_API_KEY'] = "Your_API"
llm = ChatOpenAI(model='gpt-4-1106-preview', temperature=1)

def get_entrez(keywords: list):
    """
    This function takes in a list of keywords and searches the pubmed database for related articles
    It would then return the abstract of articles found.
    :param keywords:
    :return: output
    """
    # Set your email for PubMed requests (required)
    Entrez.email = "your_email"

    # Keywords list to search in PubMed
    keywords_list = keywords

    # Combine keywords with OR operator for PubMed query. you can also use AND
    keywords_query = ' OR '.join(keywords_list)

    # Get today's date in for the text file's name (YYYY/MM/DD)
    today_date = datetime.today().strftime('%Y-%m-%d')

    # All you need to search is the keywords query
    search_query = f'({keywords_query})'
    search_results = Entrez.read(
        Entrez.esearch(db="pubmed", term=search_query, retmax=20, datetype="pdat", reldate=90, usehistory="y"))
    webenv = search_results['WebEnv']
    query_key = search_results['QueryKey']
    id_list = search_results['IdList']
    all_summaries = []
    # Step 2: EFetch to retrieve titles based on the UIDs
    for i in id_list:
        fetch_handle = Entrez.efetch(db="pubmed", id=i, rettype="abstract", retmode="text", webenv=webenv,
                                     query_key=query_key)
        fetch_content = fetch_handle.read()
        all_summaries.append(fetch_content)  # Store title along with summary
    output = ''.join(all_summaries)
    return output

def parse_keywords(output):
    """
    Parses the output list from the llm so the output keywords could be seareched with get_entrez function
    :param output:
    :return: list(unique_keywords)
    """
    # Initialize an empty set to store unique keywords
    unique_keywords = set()

    # Split the output into lines
    lines = output['question'].strip().split('\n')

    # Iterate through each line
    for line in lines:
        # Extract keywords using regular expression
        keywords = re.findall(r'"([^"]*)"', line)

        # Add keywords to the set
        unique_keywords.update(keywords)

    # Convert the set to a list and return
    return list(unique_keywords)

def collapse_list_of_lists(list_of_lists):
    """
    Takes in a list of texts and joins them together
    :param list_of_lists:
    :return: content
    """
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)

def split_embeder_retrieve(text:str,question:str):
    """
    Takes in a text and query, splits the text, embeds it and then seareches the database for a related content
    :param text:
    :param question:
    :return: related documents
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=300,chunk_overlap=50)
    splits=splitter.create_documents([text])
    documents = splitter.split_documents(splits)
    ids = [str(i) for i in range(1, len(documents) + 1)]
    vectorstore=Chroma.from_documents(documents=documents,embedding=OpenAIEmbeddings(),ids=ids)
    print(documents)
    print(len(vectorstore))
    # retriever=vectorstore.as_retriever()
    # retrieved=retriever.get_relevant_documents(question)
    retrieved= vectorstore.similarity_search(question)
    return retrieved

def format_docs(docs):
    """
    creates a final document based on the relevant data
    :param docs:
    :return: document
    """
    return "\n\n".join(doc.page_content for doc in docs)


"""
Lets Set The Prompts Now!
"""


WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501
# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """Information:
--------
{text}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)


KEY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "extract 3 important keywords from  the following question: "
            "{question}\n"
            "You must respond with a list of strings in the following format: "
            '["keywords1", "keywords2","keywords3"]'
            "for example if the question was What are the effects of protein intake on hypertrophy?"
            "then you could return such a list:"
            '["muscle hypertrophy","protein intake"]',
        ),
    ]
)

question=input('aks your question:\n')

"""
The chain below takes in the question and then creates a list of keywords which are used to create a context which the actual question would be answered upon.
"""
key_search = KEY_PROMPT | ChatOpenAI(model='gpt-4-1106-preview', temperature=1) | {
    'question': StrOutputParser()} | RunnablePassthrough.assign(
    text=lambda x: get_entrez(parse_keywords(x)))

output1=key_search.invoke(
        {
            "question":question
        }
    )
retrieved_docs= split_embeder_retrieve(output1['text'],question)



formatted_docs=format_docs(retrieved_docs)


"""
rag_chain takes in the question and context so that it can answer the question fully. It will output an article with a list of references.
"""
rag_chain =  prompt | llm | StrOutputParser()
answer= rag_chain.invoke(
        {
            "question":question,
            "text":formatted_docs
        }
    )


print(answer)