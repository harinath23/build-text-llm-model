import requests
from bs4 import BeautifulSoup
import os
import re

# Function to fetch book content from Project Gutenberg
def fetch_gutenberg_book(book_id):
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    response = requests.get(url)
    return response.text

# Function to clean and preprocess book text
def clean_book_text(text):
    # Remove Project Gutenberg header and footer
    text = re.sub(r'^\s*Project Gutenberg.*?\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*$', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*.*?$', '', text, flags=re.DOTALL)
    
    # Normalize text
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove excessive newlines
    
    return text.strip()

# Function to save cleaned text to a file
def save_cleaned_text(text, dest_folder, file_name):
    os.makedirs(dest_folder, exist_ok=True)
    with open(os.path.join(dest_folder, file_name), 'w',encoding='utf-8') as file:
        file.write(text)

# Example book IDs from Project Gutenberg (add more as needed)
book_ids = ['236', '215', '76','1661','164','73515','2591','910','61262','74','103','19033','1184','1257','13951','521','2147','17989']  # Frankenstein, Pride and Prejudice, Alice's Adventures in Wonderland

# Destination folder to save the cleaned texts
dest_folder = 'cleaned_books'

# Fetch, clean, and save each book
for book_id in book_ids:
    raw_text = fetch_gutenberg_book(book_id)
    cleaned_text = clean_book_text(raw_text)
    file_name = f"book_{book_id}.txt"
    save_cleaned_text(cleaned_text, dest_folder, file_name)

print("Books fetched, cleaned, and saved.")
