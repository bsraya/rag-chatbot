from base64 import b64encode
from ollama import chat
from schemas import ImageMetadata, TextMetadata


def generate_text_metadata(
    chunk_text: str,
    section_context: str,
    llm: str = "ollama/gemma3:12b",
) -> TextMetadata:
    system_prompt = """
    You are an expert AI system 🤖 specializing in technical document analysis and metadata extraction for a Retrieval-Augmented Generation (RAG) system.

    Your task is to analyze a provided text chunk and its surrounding section context. Based on this information, you must generate a single, valid JSON object that strictly conforms to the specified schema. Do not add any conversational text or explanations outside of the JSON object.

    The user will provide the input in the following format:
    ---
    [SECTION CONTEXT]:
    <The full text of the section where the chunk is located>
    ---
    [TEXT CHUNK TO ANALYZE]:
    <The specific text chunk to be processed>
    ---

    Based on the input, generate a JSON object with the following keys:

    - "summary" (string): A brief, one-sentence summary of the core message of the **[TEXT CHUNK TO ANALYZE]**.
    - "keywords" (array of strings): A list of the most important multi-word technical terms and concepts from the text (e.g., "natural language processing", "masked language model").
    - "entities" (array of strings): A list of specific named entities, such as model names (e.g., "BERT", "GPT-3"), libraries (e.g., "TensorFlow"), organizations, or people mentioned.
    - "key_objects" (array of strings): A list of the primary subjects or components being described in the text. Think of these as the main "nouns" of the text, like "Encoder Block", "Attention Mechanism", or "Positional Encoding".
    - "tags" (array of strings): A list of general, single-word or short-phrase categorical tags suitable for filtering (e.g., "NLP", "deep-learning", "architecture", "evaluation").
    - "contextual_text" (string): Rewrite the **[TEXT CHUNK TO ANALYZE]** to be more self-contained. Subtly integrate key information from the **[SECTION CONTEXT]** so that a reader could understand the chunk's main point without having read the rest of the document. Keep it concise and clear.
    - "hypothetical_questions" (array of strings): A list of 3-5 insightful questions that a reader might have after reading the chunk, focusing on deeper understanding or implications of the content.
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
                "content": f"""
                [SECTION CONTEXT]:
                {section_context}
                ---
                [TEXT CHUNK TO ANALYZE]:
                {chunk_text}
                ---
                """
            }
        ],
        format="json"
    )

    text_metadata = TextMetadata.model_validate_json(response['message']['content'])
    return text_metadata


def generate_image_metadata(image_path: str, llm: str = "gemma3:12b", ) -> ImageMetadata:
    with open(image_path, "rb") as image_file:
        image_base64 = b64encode(image_file.read()).decode('utf-8')

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

    image_metadata = ImageMetadata.model_validate_json(response['message']['content'])
    return image_metadata
