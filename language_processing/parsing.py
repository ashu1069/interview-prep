import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('treebank')

# define a simple grammar using context-free grammar (CFG)
'''
S: start symbol, for a sentence
NP: noun phrase
VP: verb phrase
Det: determiner
N: noun
V: verb
'''
grammar = nltk.CFG.fromstring("""
    S -> NP VP 
    NP -> Det N | N
    VP -> V | V NP
    Det -> 'the' | 'a'
    N -> 'cat' | 'dog' | 'ball'
    V -> 'chased' | 'saw' | 'ate'
""")

# ChartParser uses dynamic programming to efficiently find all possible parse trees for a given sentence according to the grammar
parser = nltk.ChartParser(grammar)

# Tokenize a sentence
sentence1 = "the cat chased the ball".split()

# Parse the sentence 
try:
    trees = parser.parse(sentence1) # returns an iterator of parse trees if the sentence is grammatically valid according to the grammar
    # print the parse trees
    for tree in trees:
        print(tree)
        tree.draw()
except ValueError as e:
    print(f"Error parsing sentence: {e}")

print("\nParsing a sentence that is not in the grammar:")

# an example where the sentence is grammatically wrong
sentence2 = "cat chased ball the".split()

try:
    trees = parser.parse(sentence2) # returns an iterator of parse trees if the sentence is grammatically valid according to the grammar
    # print the parse trees
    parsed = False
    for tree in trees:
        print(tree)
        tree.draw()
    if not parsed:
        print(f"Sentence '{' '.join(sentence2)}' is not in the grammar.")
except ValueError as e:
    print(f"Error parsing sentence: '{''.join(sentence2)}': {e}")

'''
Statistical parsing:
learn the probabilities of different grammatical structures occurring based on the training data. 
When presented with a new sentence, the parser calculates the likelihood of various possible parse 
trees and selects the one with the highest probability.

1. Probabilistic Context-Free Grammar (PCFG):
   - Assigns probabilities to each production rule in a context-free grammar.
2. Treebanks:
   - Annotated corpora that provide examples of sentences and their corresponding parse trees.
   - Used to train the parser to learn the probabilities of different structures.
3. Parsing algorithms like CKY(Cocke-Younger-Kasami) or Earley parser:
4. Dependency parsing:
   - Focuses on the relationships between words in a sentence rather than their hierarchical structure.
   - Represents sentences as directed graphs, where nodes are words and edges represent dependencies.
   - Useful for tasks like information extraction, sentiment analysis, and machine translation.
'''