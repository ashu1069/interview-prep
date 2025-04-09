import nltk
import spacy

'''
1. Named Entity Recognition (NER):
   - The process of identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, dates, etc.
   - NER is a crucial step in many NLP tasks, including information extraction, question answering, and text summarization.
2. Relation Extraction:
   - The task of identifying and classifying relationships between named entities in text.
   - It is often used in conjunction with NER to extract structured information from unstructured text.
'''

def ner_and_re(text):
   # Load the English NLP model
   nlp = spacy.load("en_core_web_sm")
    
   # Process the text
   doc = nlp(text)
    
   if doc.ents:
      print(f'Identified named entities in the text:')

      for ent in doc.ents:
         print(f"Entity: {ent.text}, Type: {ent.label_}, Start: {ent.start_char}, End: {ent.end_char}")
   else:
      print('No named entities found in the text.')

