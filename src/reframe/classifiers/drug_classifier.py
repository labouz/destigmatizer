# the script for labeling a post as durg-related or not
import time

retry_wait_time = 5  # seconds between retries


def classify_if_drug(post, retries = 2, model = None, openai_client=None):
    """
    Classify if a post contains drug-related content.

    Args:
        post (str): The text content to classify
        retries (int, optional): Number of API call retries. Defaults to 2.
        model (str, optional): OpenAI model to use. Defaults to None.
        openai_client: OpenAI client instance. Defaults to None.

    Returns:
        str: 'D' for drug-related, 'ND' for non-drug-related, 'skipped' on error

    Raises:
        ValueError: If openai_client is None
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
            response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": example1

                },
                {
                    "role": "system",
                    "content": answer1
                },
                {
                    "role": "user",
                    "content": example2
                },
                {
                    "role": "system",
                    "content": answer2
                },
                {
                    "role": "user",
                    "content": example3
                },
                {
                    "role": "system",
                    "content": answer3
                },
                {
                    "role": "user",
                    "content": example4
                },
                {
                    "role": "system",
                    "content": answer4
                },
                {
                    "role": "user",
                    "content": example5
                },
                {
                    "role": "system",
                    "content": answer5
                },
                {
                    "role": "user",
                    "content": example6
                },
                {
                    "role": "system",
                    "content": answer6
                },
                {
                    "role": "user",
                    "content": example7
                },
                {
                    "role": "system",
                    "content": answer7
                },
                {
                    "role": "user",
                    "content": example8
                },
                {
                    "role": "system",
                    "content": answer8
                },
                {
                    "role": "user",
                    "content": example9
                },
                {
                    "role": "system",
                    "content": answer9
                },
                {
                    "role": "user",
                    "content": post
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

