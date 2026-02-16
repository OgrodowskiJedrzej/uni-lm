from pydantic import BaseModel
import litellm

def select_model(intent: str):
    mapping = {
        "coding": "gpt-4o-mini"
    }
    return mapping[intent]


async def get_final_anwer(intent: str, user_query: str, image: str | None = None):
    model = select_model(intent)
    prompt = "Code should be clean, simple and without any comments."

    user_content = [{"type": "text", "text": user_query}]
    if image and intent == "tests":
        user_content.append(
            {"type": "image_url", "image_url": {"url": image}})

    response = litellm.completion(
        model = model,
        messages=[
            { "content": prompt, "role": "system" },
            { "content": user_content, "role": "user" }
        ],
        temperature=0 if intent == "tests" else 0.7,
        stream=True
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content: yield content