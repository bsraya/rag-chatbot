import base64
from ollama import chat
from schemas import ImageMetadata

def generate_image_metadata(image_path: str, llm: str = "gemma3:12b", ) -> ImageMetadata:
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    system_prompt = """
    You are an expert AI specializing in analyzing and describing diagrams, charts, and technical figures for a multimodal Retrieval-Augmented Generation (RAG) system.

    Analyze the following image and provide a detailed description in a structured JSON format. The JSON object should contain the following keys:
    - "title" (string): A concise, descriptive title for the image.
    - "summary" (string): A brief, one-paragraph summary of the image's content and purpose.
    - "key_objects" (array of strings): A list of the main components, labels, or objects visible in the image (e.g., "Encoder block", "Attention mechanism", "Input Embedding").
    - "text_in_image" (array of strings): A list of all transcribed text found in the image.
    - "contextual_description" (string): A detailed, step-by-step explanation of the diagram, describing how the components are connected and what process they illustrate.
    - "tags" (array of strings): A list of relevant keywords or tags for easy searching (e.g., "transformer", "neural network", "NLP", "self-attention").
    """

    response = chat(
        llm,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content":"Describe the following image in details, following the system prompt's JSON instructions.",
                "images": [image_base64]
            }
        ],
        format="json"
    )

    response_content = response['message']['content']
    image_metadata = ImageMetadata.model_validate_json(response_content)
    return image_metadata


if __name__ == "__main__":
    image_url = "./images/a-survey-to-transformers/image_1.png"

    image_metadata = generate_image_metadata(image_url)
    print(f"Image summary: {image_metadata.summary}")

    print(image_metadata.model_dump_json(indent=2))