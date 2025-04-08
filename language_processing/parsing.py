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

