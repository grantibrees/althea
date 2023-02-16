import openai
import os
import json
import numpy as np
import textwrap
import re
from time import time,sleep

# .env
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.environ.get("api_key")

from fileopener import open_file, save_file
from embedder import gpt3_embedding


def similarity(v1, v2):
    # Calculate the dot product of two vectors - The dot product of two vectors is a scalar value that indicates how similar the two vectors are. The higher the dot product value, the more similar the two vectors are. The function returns this value as the result. 
    return np.dot(v1, v2)

#The data argument in the search_brain function is a list of dictionaries, where each dictionary contains information about a piece of text, including its content and vector (the embedding vector representation of the text).

#The text argument is the input text for which the function will find the closest matches in the data list. The function will first generate the embedding vector representation of the input text using the gpt3_embedding function.

#The "count" parameter determines the number of results to be returned when searching the "data" for similar texts to the input "text". The higher the value of "count", the more results will be returned, and the lower the value of "count", the fewer results will be returned.
def search_brain(text, data, count=10): 
    # Generate the embedding for the input text
    vector = gpt3_embedding(text)

    # Create an empty list to store the points for each item in data
    points = list()

    # Loop through each item in data
    for i in data:
        # Calculate the similarity points between the embedding of text and the embedding of the current item
        point = similarity(vector, i['vector'])
        # Add the current item's content and its points to the points list
        points.append({'content': i['content'], 'points': point})

    # Sort the points list in descending order based on the points
    ordered = sorted(points, key=lambda d: d['points'], reverse=True)

    # Return the first count items of the sorted list
    return ordered[0:count]


#GPT-3 Function        
def gpt_3 (prompt):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    text = response['choices'][0]['text'].strip()
    return text


if __name__ == '__main__':
    lorequery = "__empty__"
    
    firstquery = input("Are you searching all, specific, or comparing? \n >>> ")
    if "all" in firstquery:
        print("go")
    elif "specific" in firstquery:
        # Open the '.json' file and read the data into a variable 'data'
        lorequery = input("Which lore are you accessing? \n >>> ")
    
        with open(lorequery + '.json', 'r') as infile:
            data = json.load(infile)
        
    elif "comparing" or "compare" in firstquery:
        print("3")
    

    # Run an infinite loop to take user input and return results
    while True:
        # Prompt the user for input
        query = input("What would you like to know about " + lorequery + "? \n >>> ")
        
        # Call the 'search_brain' function to find the best match for the user input
        results = search_brain(query, data)

        # Initialize a list to store the answers to the user's query
        answers = list()
        empty = []

        # Iterate over the results and generate an answer for each one
        for result in results:
            # Format the prompt for the OpenAI API
            prompt = open_file('text/qsprompt.txt').replace('<<INFO>>', result['content']).replace('<<QS>>', query)

            # Call the OpenAI API to generate an answer to the user's query
            answer = gpt_3(prompt)

            if "is not mentioned" not in answer:
            # Print the answer
                print('\n', answer, end='\n')
            # Add the answer to the list of answers
                answers.append(answer)
        
        if answers is empty:
            print("No answers available. \n")

        # # Join all the answers together into a single string
        # all_answers = '\n\n'.join(answers)

        # # Split the answers into smaller chunks, if necessary
        # chunks = textwrap.wrap(all_answers, 10000)

        # # Initialize a list to store the summaries of the answers
        # end = list()

        # # Generate a summary for each chunk of answers
        # for chunk in chunks:
        #     # Format the prompt for the OpenAI API
        #     prompt = open_file('text/sumanswer.txt').replace('<<SUM>>', chunk)

        #     # Call the OpenAI API to generate a summary of the answers
        #     summary = gpt_3(prompt)

        #     # Add the summary to the list of summaries
        #     end.append(summary)

        # # Print the final summaries
        # print('=========\n\n', '\n\n'.join(end))
        
        # print('\n\n=========\n\n', '\n\n'.join(all_answers))
        