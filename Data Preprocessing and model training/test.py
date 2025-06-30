import spacy
nlp = spacy.load("en_core_web_trf")
text = "I visited Dharnai, a solar-powered village in Bihar, and Nagercoil near Kanyakumari."
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.label_)