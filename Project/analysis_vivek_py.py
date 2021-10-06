from spacy import displacy


def firstLineQuote(text):
    
    pattern_qp = r'\"(.*?)\"[@\-\w\s]+'
    pattern_q = r'\"(.*?)\"'
    r_qp = re.fullmatch(pattern_qp,text)
    r_q = re.fullmatch(pattern_q, text)
    if r_qp != None:
        return 'personal_quote'
    if r_q != None:
        return 'quote'
    return ''


def generateNER(text):
    doc = NLP(text)
    
    ners = []
    for word in doc.ents:
        if word.label_.lower() in ['event', 'fac', 'gpe', 'law', 'loc', 'money', 'norp', 'org', 'person', 'product', 'work_of_art']:
            ners.append(word.label_.lower())
    
    return ','.join(ners)

def analyzeText(text):
    
    print("----------- Actual Text ------------")
    print(text)
    doc = NLP(text)
    
    print("----------- Spacy render -----------")
    displacy.render(doc,style="ent",jupyter=True)
    
    print("----------- Spacy NERs -------------")
    for word in doc.ents:
        print(word.text,word.label_)
    
    print("----------- POS tag ----------------")
    # Token and Tag
    for token in doc:
        print(token, token.pos_)        



nike_data['caption_cleaned'] = nike_data['caption'].apply(lambda x : wrangle(x))

nike_data['NER'] = nike_data['caption_cleaned'].apply(lambda x : generateNER(x))

nike_data['firstLineQuote'] = nike_data['caption_cleaned'].apply(lambda x : firstLineQuote(x.strip().split('\n')[0].strip())) 




