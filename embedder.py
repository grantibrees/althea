import os
import openai
import json
import textwrap

# This fn takes 2 args, "content" and "engine".
# "content" is the input text for which we want to generate the embedding vector.
# "engine" is the engine we want to use for the embedding generation (default is 'text-embedding-ada-002')
# The new model (default) replaces five separate models for the text search, text similarity, 
# and code search, and outperforms the Davinci mode at most tasks.

def gpt3_embedding(content, engine='text-embedding-ada-002'):
    # OpenAI's Embedding API is used to create the embedding vector for the given input text
    response = openai.Embedding.create(input=content,engine=engine)
    
    # The embedding vector is extracted from the API response
    # The 'data' key in the response contains a list of dictionaries, each with an 'embedding' key.
    # The first dictionary in the list is selected (index 0) and its 'embedding' value is extracted
    vector = response['data'][0]['embedding']
    
    # The extracted embedding vector is returned as the function's output
    return vector