import nltk
from collections import Counter
import math

'''
Basic tokenization methods
'''
class WhitespaceTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        return text.split()
    
class PunctuationTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        return nltk.word_punct_tokenize(text)
    
'''
Advance tokenization methods
'''
class NLTKTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        return nltk.word_tokenize(text)
    
class RuleBasedTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        return nltk.tokenize.TreebankWordTokenizer().tokenize(text)
    
class BPETokenizer:
    '''
    Explanation:
    1. Intialization: Starts with each unique character in the training data as a token. 
                      A special end-of-word token </w> is added to distinguish between word 
                      prefixes and complete words.
    2. Pair Counting: iteratively counts the frequency of adjacent character pairs in the training data.
    3. Merging: the most frequent pair is merged into a new token.
    4. Vocabulary Update: The vocabulary is updated with the new token, and the process continues until it
                          reaches the desired vocabulary size.
    5. Tokenization: To tokenize a new text, the algorithm tries to apply the learned merges greedily. 
                     It looks for the longest possible subword (formed by merges) starting from the beginning of a word.
    '''
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocabulary = {}
        self.merges = {}

    def _get_pair_counts(self, words):
        pairs = Counter()
        for word in words:
            symbols = list(word)
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += 1
        return pairs

    def train(self, text):
        # initialize the vocabulary with individual characters
        self.vocabulary = Counter(text) # gives the count of each character in the text
        self.vocabulary = {char: count for char, count in self.vocabulary.items()} # convert to dict

        # Add special end of word token <EOW>
        self.vocabulary['<EOW>'] = self.vocabulary.get('<EOW>', 0) + len(text.split()) # count of end of the word tokens
        self.vocabulary = {key + '<EOW>': value for key, value in self.vocabulary.items()} # put the end of the word token in the vocabulary

        # convert the vocabulary to a list of words with counts
        words = []
        for word, count in Counter(text.split()).items():
            words.extend([word + '<EOW>'] * count)

        # intialize merges
        self.merges = {}

        # Perform merge operations
        while len(self.vocabulary) < len(self.vocab_size):
            pairs = self._get_pair_counts(words)
            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1]

            new_words = []
            for word in words:
                new_word = word.replace(best_pair[0], '$TEMP$').replace(best_pair[0] + best_pair[1], 1).replace('$TEMP$', best_pair[0]) # replace the best pair with a temporary token
                new_words.append(new_word)

            words = [word.replace(best_pair[0], new_token) for word in new_words] # replace the temporary token with the new token
            self.vocabulary[new_token] = sum(pairs.values()) # add the new token to the vocabulary
            self.merges[best_pair] = new_token # add the new token to the merges

        # Final vocabulary
        self.vocabulary = {token: i for i, token in enumerate(sorted(self.vocabulary.keys()))} # convert the vocabulary to a dict with token as key and index as value

    def tokenize(self, text):
        tokens = []
        for word in text.split():
            word += '<EOW>'
            while len(word) > 1:
                best_pair = None
                best_index = -1
                for pair, merged_token in self.merges.items():
                    try: 
                        index = word.index(pair[0] + pair[1])
                        if index != -1 and best_index == -1 or index < best_index:
                            best_pair = pair
                            best_index = index
                    except ValueError:
                        continue
                if best_pair:
                    word = word[:best_index] + self.merges[best_pair] + word[best_index + len(best_pair):] # replace the best pair with the merged token
                else:
                    break

        return [token for token in tokens if token]
    
class WordPieceTokenizer:
    '''
    Explanation:
    1. Initialization: Starts with each character as a token in the vocabulary.
    2. Pair Scoring: Iteratively considers all possible pairs of adjacent tokens in the current vocabulary. 
                     A score is calculated for each pair, often based on the likelihood of the combined token. 
                     A common scoring method used in the original WordPiece involves calculating the probability 
                     of the pair occurring together divided by the product of their individual probabilities.
    3. Merging: The pair with the highest score is merged into a new token, which is added to the vocabulary.
    4. Iteration: The process continues until the desired vocabulary size is reached or no more pairs can be merged.
    5. Tokenization: To tokenize a new word, the algorithm tries to find the longest subword starting from the beginning 
                     of the word that is present in the vocabulary. If a full word is not in the vocabulary, 
                     it's broken down into subwords. If a character is not in the vocabulary, it's often replaced with a 
                     special "[UNK]" (unknown) token.
    '''
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocabulary = {}
        self.subwords_probab = {}

    def train(self, text):
        words = Counter(text.split())
        self.vocabulary = {word: count for word, count in words.items() if count > 1} # filter out the words with count less than 1
        initial_vocab_size = len(self.vocabulary)

        if initial_vocab_size >= self.vocab_size:
            self.vocabulary = {token: i for i, token in enumerate(sorted(self.vocabulary.keys()))} # convert the vocabulary to a dict with token as key and index as value
            return 
        
        current_vocab = list(self.vocabulary.keys()) # initialize the current vocabulary with the words in the vocabulary

        for i in range(self.vocab_size - initial_vocab_size):
            pair_probs = Counter()
            for word, count in words.items():
                chars = list(words) # convert the word to a list of characters
                for j in range(len(chars)-1):
                    pair = (chars[j], chars[j+1]) # get the pair of characters
                    if pair[0] in current_vocab and pair[1] in current_vocab:
                        combined = pair[0] + pair[1] # combine the pair of characters
                        pair_probs[combined] += count # add the count of the word to the pair
            
            if not pair_probs:
                break

            best_subword = max(pair_probs, key=pair_probs.get)
            current_vocab.append(best_subword)

        self.vocabulary = {token: i for i, token in enumerate(sorted(current_vocab))} # convert the vocabulary to a dict with token as key and index as value

    def tokenize(self, text):
        tokens = []
        for word in text.split():
            sub_tokens = []
            start = 0
            while start < len(word):
                end = len(word)
                best_match = None
                while start < end:
                    subword = word[start:end]
                    if subword in self.vocabulary:
                        best_match = subword
                        break
                    end -= 1
                if best_match:
                    sub_tokens.append(best_match)
                    start += len(best_match)
                else:
                    # handle out of vocabulary words
                    sub_tokens.append("[UNK]")
                    start += 1
            tokens.extend(sub_tokens)
        return tokens


