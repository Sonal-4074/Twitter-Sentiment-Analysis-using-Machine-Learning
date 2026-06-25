from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

sentence = "i am very happy because i learned deep learning today"

words = sentence.split()

filtered_words = [word for word in words if word not in stop_words]

print("Original:")
print(sentence)

print("\nAfter Stop Word Removal:")
print(" ".join(filtered_words))