from deep_translator import GoogleTranslator, PonsTranslator, LingueeTranslator, MyMemoryTranslator

sentence = 'Hello My name is Youssef'
words = sentence.split(' ')
print(words)
translated_google = GoogleTranslator(source='en', target='ar').translate(sentence)
print('Google: ', translated_google)

translated_mymemory = MyMemoryTranslator(source='english', target='arabic').translate(sentence)
print('MyMemory: ', translated_mymemory)

# Errors when using the following

# translated_pons=PonsTranslator(source='english',target='arabic').translate(word)
# print('Pons: ',translated_pons)

# translated_linguee=LingueeTranslator(source='english',target='french').translate_words(words)
# print('Linguee: ',translated_linguee)
