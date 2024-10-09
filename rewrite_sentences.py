from openai import OpenAI
import pandas as pd


client = OpenAI()
prompts = pd.read_csv("./prompts/memorized_laion_prompts.csv", sep=';')

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Write a haiku about recursion in programming."
        }
    ]
)

print(completion.choices[0].message)