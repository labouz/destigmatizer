# the script for labeling a post as containing stigmatizing language or not
import time

retry_wait_time = 5  # seconds between retries


def get_destigma_style(post, explanation, style_instruct, step, retries = 2, model = "gpt-4-turbo-2024-04-09", openai_client=None):
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
            response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": post + ";" + ex + ";" + style_instruct
                }
            ],
            model=model,
            temperature=0
        )

            label = response.choices[0].message.content.lower().strip()
            return label
        
        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            retries -= 1
            time.sleep(retry_wait_time)
    print("Retrying...")
    return "skipped"
    