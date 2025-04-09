import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer

'''
1. Porter Stemmer:
   - Uses a set of rules to iteratively reduce words to their root form.
   - It is simple and fast but can be less accurate for some words.
2. Snowball Stemmer:
   - An improvement over the Porter stemmer.
   - It has a more sophisticated algorithm and is available for multiple languages.
3. Lancaster Stemmer:
    - A more aggressive stemming algorithm than Porter.
    - It can produce shorter stems and is faster but may be less accurate.
'''

# Sample words
words = ["running", "runs", "ran", "easily", "easy", "connection", "connected", "connecting", "connections", "argue", "argued", "arguing"]

porter = PorterStemmer()
print("Porter Stemmer:")
for word in words:
    print(f"{word} -> {porter.stem(word)}")

print("\nSnowball Stemmer (English):")
snowball = SnowballStemmer("english")
for word in words:
    print(f"{word} -> {snowball.stem(word)}")

print("\nLancaster Stemmer:")
lancaster = LancasterStemmer()
for word in words:
    print(f"{word} -> {lancaster.stem(word)}")