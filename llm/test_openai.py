from openai import OpenAI

# ✅ Client will auto-read OPENAI_API_KEY from your environment
client = OpenAI()

# ✅ Simple test
response = client.chat.completions.create(
    model="gpt-4o-mini",  # Or "gpt-3.5-turbo"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ]
)

print(response.choices[0].message.content)
