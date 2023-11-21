from transformers import pipeline
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Initialisation du texte à traiter
text = """I ordered a used book from your online site. \
The book is Les mystères de Paris from Victor Hugo. \
It was indicated on the site that it was in good condition. \
I received it a week after ordering. \
And while unpacking the package I realized that it was damaged (damaged pages, writings). \
I contacted customer service who proceeded to the immediate refund."""

# Traduction

translator = pipeline('translation_en_to_fr')
outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
print(outputs[0]['translation_text'])

# Text Classification

classifier = pipeline("text-classification")
outputs = classifier(text)
pd.DataFrame(outputs)

# Name Entity Recognition

ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger(text)
pd.DataFrame(outputs)

ner_tagger = pipeline("ner", aggregation_strategy="simple")
outputs = ner_tagger("Donald Trump, the former president, is now playing golf all days in Florida with his friend Bill")
pd.DataFrame(outputs)

# Question Answering

reader = pipeline("question-answering")
results = []
questions = ["What was ordered?",
"What did the customer service?"]

for question in questions:
    outputs = reader(question=question, context=text)

results.append(outputs)
pd.DataFrame(results)

# Summarization

summarizer = pipeline("summarization")
outputs = summarizer(text, max_length=150, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])

# Text Generation

generator = pipeline("text-generation")
response = "Dear friends. Let me tell you my last experience with this online store where I ordered a book."
prompt = text + "\n\nStory I told to my friends:\n\n" + response
outputs = generator(prompt, max_length=200)
print(outputs[0]['generated_text'])