# the script for labeling a post as durg-related or not
import time
from openai import OpenAI
from together import Together

def initialize(api_key=None, client=None, client_type="openai"):
    """
    Initialize and return a client for the Reframe library.
    
    Args:
        api_key (str, optional): API key for the language model service
        client (object, optional): Pre-configured client instance
        client_type (str, optional): Type of client ("openai" or "together")
        
    Returns:
        object: Configured client instance
        
    Raises:
        ValueError: If neither api_key nor client is provided, or if client_type is unsupported
    """
    if client:
        return client, client_type
    elif api_key:
        if client_type.lower() == "openai":
            return OpenAI(api_key=api_key), client_type
        elif client_type.lower() == "together":
            return Together(api_key=api_key), client_type
        else:
            raise ValueError(f"Unsupported client type: {client_type}")
    else:
        raise ValueError("Either api_key or client must be provided")
    
    
retry_wait_time = 5  # seconds between retries

def create_completion(client, messages, model=None, temperature=0, client_type="together"):
    """
    Create a chat completion using the specified client and model.
    
    Args:
        client: The client instance (OpenAI or Together)
        messages: List of message dictionaries
        model: Model identifier
        temperature: Sampling temperature
        client_type: Type of client ("openai" or "together")
        
    Returns:
        str: The generated response content
    """
    try:
        # Set default model based on client type
        if not model:
            model = "gpt-4o" if client_type == "openai" else "meta-llama/Meta Llama 3.1 70B Instruct Turbo"
            
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error creating completion: {str(e)}")

def classify_if_drug(post, retries = 2, model = None, client=None, client_type="openai"):
    """
    Classify if a post contains drug-related content.

    Args:
        post (str): The text content to classify
        retries (int, optional): Number of API call retries. Defaults to 2.
        model (str, optional): OpenAI model to use. Defaults to None.
        client: Client instance. Defaults to None.
        client_type (str, optional): Type of client ("openai" or "together"). Defaults to "openai".

    Returns:
        str: 'D' for drug-related, 'ND' for non-drug-related, 'skipped' on error

    Raises:
        ValueError: If client is None
    """
    prompt = f"""
    *Instructions for Labeling Drug References in Social Media Posts*

    1. **Objective**: Identify references to drugs or people who use drugs in each post.

    2. **Include**:
    - Illicit Drugs: All controlled substances with no legal usage (e.g. cannabis, heroin, cocaine, methamphetamine)
    - Prescription Drugs: drugs that are often abused even if they have legitimate medical uses (e.g., opioids, benzodiazepines).
    - Other Drugs: drugs that are non-prescription and known to be abused (e.g., inhalants, k2, bath salts).
    - Explicit mentions of drug use, abuse, or addiction related terms (e.g., "getting high", "stoned").

    3. **Exclude**:
    - Tobacco, nicotine, or alcohol unless explicitly linked to drug use.
    - Do not include medical or psychological discussions unless there is a direct and clear mention of drug use or abuse.

    4. **Clarifications**:
    - Mental health discussions should not be labeled as 'D' unless there is an explicit mention of drugs as defined above.
    - Use 'ND' for posts that discuss health or psychological issues without specific drug references.

    5. **Language Cues**:
    - Focus on clear drug-related terminology (e.g., "junkie", "addict") and slang.
    - If a post is ambiguous and does not clearly fit the drug reference criteria, label as 'ND'.

    6. **Response Requirement**:
    - Respond with either 'D' (Drug) or 'ND' (Non-Drug) based on these guidelines. No additional commentary is needed.
    """

    example1 = "I'm so high right now, I can't even feel my face. This is the best weed I've ever smoked."
    answer1 = "D"
    example2 = "I hope my junkie sister OD's or disappears out of our lives My sister is an alcoholic junkie who has 2 DUIs under her belt as well as loves taking Xanax and alcohol together and wreaking havoc for our family and even strangers."
    answer2 = "D"
    example3 = "My mom is going to kick me out. She graciously gave me the choice of getting dropped off in a shelter in either San Diego or the desert area (Palm Springs and surrounding areas). I would choose the desert because that is one of my old stomping grounds. The dope is phenomenal and cheap (3g's for $100) and the homeless population is a majority young people. I can also hustle up $350 and rent a room at a buddy's place. I have a few options that I can look at but I have to figure it out soon."
    answer3 = "D"
    example4 = "I cut tonight I cut myself tonight after I've been clean for a few months now and I'm honestly just broken inside. I had a huge fight with my girlfriend, I'm worried we might break up. I just kinda lost control, grabbed my razorblade and cut. I feel like crap and am honestly just lost right now."
    answer4 = "ND"
    example5 = "My drinking problem is way worse than anyone knows. I'm an alcoholic and the people around me know that, because I told them. But if I hadn't, they wouldn't have guessed, judging by their reactions of shock and disbelief and the actual ‚are you sure you're not exaggerating, that I got when I came clean a couple years ago."
    answer5 = "ND"
    example6 = "My mother died two years ago from cancer of basically all of her internal organs starting in the colon and then liver and spleen and just everywhere. I hadn't seen her in years because she struggled with drug addiction since I was about four or five. We had years together but not enough. I was there the day she died and I wouldn't wish anyone to go through what she did or what our family did when we saw her in that state. My father also had an addiction to pain pills and alcohol. My last few years of high school were ruined because of that. I watched him overdoes on thanksgiving. I watched him try to kill himself with his own hand gun more than once. I watched him aim it at my stepmom when she tried to take it from him. I heard him yell at me while he kicked me and told me that he didn't want me anymore and tried to drag me up the stairs. Maybe these are some of the reasons I drink. Maybe because I barely have a relationship with my dad now and he still hasn't said anything to me today."
    answer6 = "D"
    example7 = "Hate my new job I just got hired at Taco Bell a few weeks ago and I really hate it. The only thing that's gets me through it is smoking weed. I wonder if i should just quit or no call no show?"
    answer7 = "D"
    example8 = "I'm trying not to get emotionally invested again in you because I know all to well how you are. You're an alcoholic and I know there is nothing I can do to save you. It kills me to see you drink like you do but I know how it is, you have to want to stop. As much as I love fucking you and being around you, It's not a good idea to continue being around you. "
    answer8 = "ND"
    example9 = "Recently I took a psychological exam for work. To know if I'm fit to work.The doctor looked at my paper and asked 'Do your have urges to hurt/killpeople?' I laughed in my head and lied.."
    answer9 = "ND"


    while retries > 0:
        try:
            label = create_completion(
                client,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": example1},
                    {"role": "system", "content": answer1},
                    {"role": "user", "content": example2},
                    {"role": "system", "content": answer2},
                    {"role": "user", "content": example3},
                    {"role": "system", "content": answer3},
                    {"role": "user", "content": example4},
                    {"role": "system", "content": answer4},
                    {"role": "user", "content": example5},
                    {"role": "system", "content": answer5},
                    {"role": "user", "content": example6},
                    {"role": "system", "content": answer6},
                    {"role": "user", "content": example7},
                    {"role": "system", "content": answer7},
                    {"role": "user", "content": example8},
                    {"role": "system", "content": answer8},
                    {"role": "user", "content": example9},
                    {"role": "system", "content": answer9},
                    {"role": "user", "content": post}
                ],
                model=model,
                client_type=client_type
            )
            return label.lower().strip()
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            retries -= 1
            time.sleep(retry_wait_time)

    print("Retrying...")
    return "skipped"

# the script for labeling a post as containing stigmatizing language or not
import time


retry_wait_time = 5  # seconds between retries



def classify_if_stigma(post, retries = 2, model = None, client=None, client_type="openai"):
    prompt = f"""
    **Instructions:**
    You are an expert in identifying stigma related to drug use and addiction in social media posts. Your task is to analyze each post and determine if it contains stigmatizing language. 

    1. **Label:**
        * If the post contains stigmatizing language, label it as **"S" (Stigma)**.
        * If the post does not contain stigmatizing language, label it as **"NS" (Non-Stigma)**.

    2. **Explanation:**
        * **For "S" (Stigma) labels:**
            * If the post demonstrates **stigmatizing language (directed at others)**, analyze it based on the following four attributes and provide an explanation for each attribute:
                * **Labeling:** Does the post use derogatory or othering language related to drug use/addiction?
                * **Stereotyping:** Does the post reinforce negative generalizations about people who use drugs?
                * **Separation:** Does the post create a divide between people who use drugs and those who don't?
                * **Discrimination:** Does the post imply or suggest unfair treatment based on drug use?
            * If the post demonstrates **self-stigma or structural stigma**, provide a brief, one-sentence explanation of the type of stigma present.

        * **For "NS" (Non-Stigma) labels:** No explanation is needed.

    **Definitions:**

    * **Stigma:** Negative attitudes or beliefs directed at individuals based on their drug use.
    * **Self-Stigma:** Internalization of negative societal attitudes about drug use.
    * **Structural Stigma:** Societal systems, policies, or practices that disadvantage people who use drugs.

    **Additional Considerations:**

    * Consider the context of the post and the author's intent.
    * Focus on language that is harmful, discriminatory, or perpetuates negative stereotypes.

    **Task:**
    Read the following post, label it as "S" or "NS", and provide the appropriate explanation:
    """
    example1 = "Once an addict, always an addict. They'll never change."
    answer1 = "S, Labeling: 'addict', Stereotyping: Assumes that people with addiction are incapable of recovery or personal growth, Separation: Creates a sense of hopelessness and perpetuates the idea that people with addiction are fundamentally different, Discrimination: Discourages offering support or opportunities to individuals in recovery due to a belief they'll relapse."

    example2 = "He's just a junkie looking for his next fix."
    answer2 = "S, Labeling: 'junkie', a derogatory term, Stereotyping: Reduces the person's identity to their substance use, ignoring other aspects of their life, Separation: Creates a distance between 'us' (non-users) and 'them' (people who use substances), Discrimination: Implies the person is solely motivated by drug use and lacks other values or goals."

    example3 = "She's constantly nodding off at work. Probably using again."
    answer3 = "S, Labeling: Implies the person has a substance use disorder by suggesting they are 'using again,' Stereotyping: Generalizes that the person's behavior (nodding off) is solely due to drug use, Separation: Creates a distinction between the person and their colleagues, implying they are unreliable or irresponsible due to assumed substance use, Discrimination: Could lead to unfair treatment at work based on an assumption of relapse."

    example4 = "If you hang out with those druggies, you'll end up like them."
    answer4 = "S, Labeling: 'druggies', Stereotyping: Generalizes that all people who use drugs are bad influences and will lead others to the same behavior, Separation: Creates a clear division between 'us' (those who don't use drugs) and 'them' (those who do), Discrimination: Promotes exclusion and fear of people who use drugs, potentially leading to social isolation."

    # Non-Stigmatizing Examples

    example5 = "My friend is in recovery from opioid use disorder and is doing incredibly well."
    answer5 = "NS"

    example6 = "Addiction is a complex disease, but with the right support, people can and do recover."
    answer6 = "NS"

    example7 = "I'm concerned about my brother's substance use. I'm learning about available resources to help him."
    answer7 = "NS"

    # self and systemic
    example8 = "I always hide my medication because I feel ashamed of my condition. I'm just an addict, and that's all people will see if they know."
    answer8 = "S, The author internalizes the negative stereotype associated with addiction, labeling themselves pejoratively as 'just an addict.'"
    example9 = "Our town's only rehab facility refuses to accept people who have been arrested for drug offenses. They say it's a policy to maintain safety, but it just prevents those who need help the most from getting it."
    answer9 = "S, This post describes a systemic policy that discriminates against individuals with drug offenses, reinforcing stereotypes that they are dangerous. The policy creates a separation by denying these individuals access to rehabilitation."
    example10 = "Crack is a better alternative to coffee All coffee does is make you feel less drowsy. When I smoke a bowl of crack in the morning, shit gets me gooooinn, makes me go absolutely bonkers, crazy ridiculous out of the world bonkers ! I'm getting the jitters just thinkin about it"
    answer10 = "NS"



    while retries > 0:
        try:
            label = create_completion(
                client,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": example1},
                    {"role": "system", "content": answer1},
                    {"role": "user", "content": example2},
                    {"role": "system", "content": answer2},
                    {"role": "user", "content": example3},
                    {"role": "system", "content": answer3},
                    {"role": "user", "content": example4},
                    {"role": "system", "content": answer4},
                    {"role": "user", "content": example5},
                    {"role": "system", "content": answer5},
                    {"role": "user", "content": example6},
                    {"role": "system", "content": answer6},
                    {"role": "user", "content": example7},
                    {"role": "system", "content": answer7},
                    {"role": "user", "content": example8},
                    {"role": "system", "content": answer8},
                    {"role": "user", "content": example9},
                    {"role": "system", "content": answer9},
                    {"role": "user", "content": example10},
                    {"role": "system", "content": answer10},
                    {"role": "user", "content": post}
                ],
                model=model,
                client_type=client_type
            )
            return label.lower().strip()
        
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            retries -= 1
            time.sleep(retry_wait_time)
    print("Retrying...")
    return "skipped"

# the script for labeling a post as containing stigmatizing language or not
import time

retry_wait_time = 5  # seconds between retries


def rewrite_to_destigma(post, explanation, style_instruct, step, model=None, retries=2, client=None, client_type="openai"):
    # explanation in the form of Labeling: Uses the term 'junkies,' a derogatory label, Stereotyping: Portrays people who use drugs as irresponsible and a burden on society, Separation: Explicitly states that people with addiction do not belong in society, Discrimination: Advocates for stripping rights and marking individuals with addiction, suggesting they should be treated differently and excluded from societal opportunities.

    if step == 1:
        instruction = "Rewrite this post to remove any and all labeling."
        definition = "Labeling includes the use of derogatory or othering language related to drug use/addiction."
        explanation = explanation.split('Labeling: ')[1].split(',')[0]
    else:
        instruction = "Rewrite this post to remove any all instances of stereotyping, insinuations of separation, and/or discriminatory language."
        definition = "Stereotyping reinforces negative generalizations about people who use drugs. Separation creates a divide between people who use drugs and those who don't. Discrimination implies or suggests unfair treatment based on drug use."
        stereotype = explanation.split('Stereotyping: ')[1].split(',')[0]
        separation = explanation.split('Separation: ')[1].split(',')[0]
        discrimination = explanation.split('Discrimination: ')[1].split(',')[0]
        explanation = stereotype + ';' + separation + ';' + discrimination

    prompt = f"""
    {instruction}; 
    {definition};
    Only rewrite the relevant parts of the post, do not rewrite the whole post. Do not change the meaning of the post or add any new information.
    Also, match the output to the given stylistic profile.
    Example:
    post: "My mom is an addict."; This post uses the term 'addict'; [('tone': 'negative'),('punctuation_usage': 'moderate, with . being most frequent'),('passive_voice_usage': 'none'),('sentence_length_variation': 'ranging from short (5 words) to long (5 words) with an average of 5.0 words per sentence'),('lexical_diversity': 'moderately high')]
    rewrite: "My mom has a substance use disorder."

    Do not include "Here is the rewritten post:" in your response. Just return the rewritten post. Nothing more.
    """
    ex = f"This post {explanation}"
    # example1 = "my mom is an addict."
    # answer1 = "my mom has a substance use disorder."

   

    while retries > 0:
        try:
            label = create_completion(
                client,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": post + ";" + ex + ";" + style_instruct}
                ],
                model=model,
                client_type=client_type
            )
            return label.lower().strip()
        
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            retries -= 1
            time.sleep(retry_wait_time)
    print("Retrying...")
    return "skipped"
    
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import string
from lexicalrichness import LexicalRichness

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

import pandas as pd
import numpy as np
import openai
import json

prompt = """
Please play the role of an emotion recognition expert. Pleae provide the most likely emotion that the following text conveys.
Only one emotion should be provided.

"""
def get_emotion(text, client, model=None, temperature=0, retries=2, client_type="openai"):
    while retries > 0:
        try:
            label = create_completion(
                client,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ],
                model=model,
                temperature=temperature,
                client_type=client_type
            )
            return label.lower().strip()
        except Exception as e:
            retries -= 1
            print(f"Error: {e}")
            print(f"Retries left: {retries}")
            print("Retrying...")
            return "skipped"
