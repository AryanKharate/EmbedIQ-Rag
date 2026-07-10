from google import genai
from google.genai import types
from django.conf import settings

from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from google.genai.errors import ServerError

# Create a dedicated client instance for HyDE to keep the module standalone
_client = genai.Client(api_key=settings.GEMINI_API_KEY)

@retry(
    wait=wait_random_exponential(multiplier=1, max=15),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(ServerError),
)
def generate_hypothetical_answer(query: str) -> str:
    """
    Generates a plausible, hypothetical answer to the query as if it were 
    pulled from a reference document. We embed this answer instead of the query 
    so the embedding matches the structure and style of the target documents.
    """
    prompt = (
        "Write a short, plausible answer to this question, as if it appeared "
        "in a reference document. Do not hedge or say you're unsure — just write "
        f"the answer directly.\n\nQuestion: {query}"
    )
    
    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3, 
            max_output_tokens=200
        ),
    )
    
    return response.text.strip()
