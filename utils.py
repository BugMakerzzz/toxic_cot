from allennlp.predictors.predictor import Predictor
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import nltk
import re, string
# nltk.download()

predictor = Predictor.from_path(
    "./model/structured-prediction-srl-bert.2020.12.15.tar.gz",
    # overrides={
    #     "dataset_reader.token_indexers.tokens.model_name": local_config_path,
    #     "validation_dataset_reader.token_indexers.tokens.model_name": local_config_path,
    #     "model.text_field_embedder.tokens.model_name": local_config_path,
    # }
    )
# nlp = spacy.load("en_core_web_sm")

def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)


def extract_frame(tags, words, desc):
    prev = 'O'
    start, end = None, None
    if len(set(tags)) == 1: return ''
    tags = [t if 'C-ARG' not in t else 'O' for t in tags] #check if the modifier is a verb phrase
    for w in range(len(words)):
        if 'B-' in tags[w] and start is None: start = w
        if tags[len(words) - w -1]!='O' and end is None: end = len(words) - w -1 
    
    if end is None: end = start
    # print(words)
    sent = detokenize(words[start: end + 1]).rstrip('.')
    return sent

def remove_modifiers(sent, modifiers):
    if not len(modifiers): return sent
    for mod in modifiers:
        sent = sent.replace(mod, "")
        sent = re.sub(' +', ' ', sent) # remove any double spaces
        sent = sent.strip(string.punctuation + ' ') # remove stray punctuations
    return sent

def verb_modifiers(desc):
    filtered_mods = []
    mods = re.findall(r"\[ARGM.*?\]", desc)
    if not len(mods): return filtered_mods
    for mod in mods:
        phrase = mod.split(': ')[1].rstrip(']')
        verb_match = ['VB' in k[1] for k in nltk.pos_tag(word_tokenize(phrase))]
        if sum(verb_match) and len(phrase.split()) > 2: filtered_mods.append(phrase) # put in a length criteria
    return filtered_mods


def get_phrases(sent):
    # Simple RCU extractor without conjunction check for premises
    phrases = []
    history = ''
    srl_out = predictor.predict(sent) 
    words = srl_out['words']  
    frames = [s['tags'] for s in srl_out['verbs']]
    descs = [s['description'] for s in srl_out['verbs']]
    mod_sent = detokenize(words).rstrip('.')
    for frame, desc in zip(frames, descs):
        phrase = extract_frame(frame, words, desc)
        if phrase == mod_sent: phrase = remove_modifiers(phrase, verb_modifiers(desc))
        phrases.append(phrase)
 
    phrases.sort(key=lambda s: len(s), reverse=True)
    filtered_phrases = []
    for p in phrases: 
        if p not in history:  
            history += ' ' + p
            filtered_phrases.append(p)
   
    if len(filtered_phrases): 
        filtered_phrases.sort(key=lambda s: mod_sent.find(s))
        left = mod_sent
        mod_filt = False
        for fp in filtered_phrases: left = left.replace(fp, '#').strip(string.punctuation + ' ')
        for l in left.split('#'): 
            l = l.strip(string.punctuation + ' ')
            if len(l.split()) >=4 and l not in " ".join(filtered_phrases): 
                verb_match = ['VB' in k[1] for k in nltk.pos_tag(word_tokenize(l))]
                if sum(verb_match):
                    filtered_phrases.append(l)
                    mod_filt = True
        if mod_filt: filtered_phrases.sort(key=lambda s: mod_sent.find(s))
        return filtered_phrases
    else: return [sent.rstrip('.')]