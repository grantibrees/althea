import os

def open_file(filepath):
    with open(filepath, 'r', encoding='UTF-8') as infile:
        return infile.read()
    
def save_file(filepath, content):
    with open(filepath, 'w', encoding='UTF-8') as outfile:
        return outfile.write(content)