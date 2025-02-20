import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import string
from lexicalrichness import LexicalRichness
from .get_emotion import get_emotion



def mtld_LD(word_list):
    def mtld_calc(words, ttr_threshold=0.72):
        count, total = 0, 0
        types = set()
        for word in words:
            types.add(word)
            ttr = len(types) / (total + 1)
            if ttr <= ttr_threshold:
                count += 1
                types = set()
                total = 0
            else:
                total += 1
        if total > 0: count += total / len(words)  # Fractional part
        return count
    
    length_fwd = mtld_calc(word_list)
    length_bwd = mtld_calc(word_list[::-1])
    
    return (length_fwd + length_bwd) / 2

# Modified analyze_text_llm function
def analyze_text_llm(text, client):
    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    emotion = get_emotion(text, client)
    # Punctuation analysis
    punctuation_counts = {key: text.count(key) for key in string.punctuation}
    common_punctuation = ', '.join([p for p, count in punctuation_counts.items() if count > 0])

    # Active vs. Passive Voice
    def is_passive(sentence):
        tagged = pos_tag(word_tokenize(sentence))
        passive = False
        for i in range(len(tagged) - 1):
            if tagged[i][1] in ['was', 'were'] and tagged[i+1][1] == 'VBN':
                passive = True
        return passive

    passive_sentences = sum(is_passive(sentence) for sentence in sentences)
    passive_voice_usage = "none" if passive_sentences == 0 else "some"

    # Sentence length variability
    sentence_lengths = [len(s.split()) for s in sentences]
    min_length = min(sentence_lengths)
    max_length = max(sentence_lengths)
    average_length = sum(sentence_lengths) / len(sentence_lengths)

    # Lexical diversity
    lex = LexicalRichness(text)
    lex = lex.mtld(threshold=0.72)


    results = {
        "top emotions": emotion,
        "punctuation_usage": f"moderate, with {common_punctuation} being most frequent",
        "passive_voice_usage": passive_voice_usage,
        "sentence_length_variation": f"ranging from short ({min_length} words) to long ({max_length} words) with an average of {average_length:.1f} words per sentence",
        "lexical_diversity": f"{lex:.2f} (MTLD)"
    }

    return results