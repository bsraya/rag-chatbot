from base64 import b64encode
from ollama import chat
from schemas import ImageMetadata, TextMetadata
from loguru import logger

def generate_text_metadata(
    chunk_text: str,
    section_context: str,
    llm: str = "gemma3:12b",
) -> TextMetadata:
    """
    Extract metadata from a piece of text chunk and its surrounding section context.

    Args:
        chunk_text (str): The text content of a single chunk.
        section_context (str): The surrounding context of the text chunk.
        llm (str): The LLM model to use for generating metadata. Defaults to "ollama/gemma3:12b".
    
    Returns:
        TextMetadata: An instance of TextMetadata containing metadata extracted from the text chunk and its surrounding context.
    """

    system_prompt = """
    You are an expert AI system 🤖 specializing in technical document analysis and metadata extraction for a Retrieval-Augmented Generation (RAG) system.

    **Your task**  
    1. Analyze the provided text chunk and its surrounding section context.  
    2. Return *exactly* one JSON object that matches the following schema.  
    3. **Ensure all extracted information is directly supported by the provided texts.**


    **Here is the template of the input:**
    ---
    [SECTION CONTEXT]:
    <The full text of the section where the chunk is located>
    ---
    [TEXT CHUNK TO ANALYZE]:
    <The specific text chunk to be processed>
    ---

    **Here is an example of the desired output format:**
    {
        "summary": "Positional encodings match the embedding dimension to allow for summation and can be either learned or fixed.",
        "keywords": ["positional encodings", "input embeddings"],
        "entities": ["Transformer"],
        "key_objects": ["Positional Encodings", "Embeddings"],
        "tags": ["architecture", "NLP", "transformer"],
        "contextual_text": "In the Transformer model, positional encodings are vectors that have the same dimension as the input embeddings, which allows them to be summed together to provide the model with information about the sequence order of the tokens. These positional encodings can be based on either learned parameters or fixed mathematical functions.",
        "hypothetical_questions": [
            "Why do positional encodings need to be the same size as embeddings?",
            "What is the difference between learned and fixed positional encodings?",
            "How does adding positional encodings help the Transformer model understand sequence order?"
        ]
    }

    **Output Schema**:
    - "summary" (str): A brief, one-sentence summary of the core message of the **[TEXT CHUNK TO ANALYZE]**.
    - "keywords" (list(str)): A list of the most important multi-word technical terms and concepts from the text (e.g., "natural language processing", "masked language model").
    - "entities" (list(str)): A list of specific named entities, such as model names (e.g., "BERT", "GPT-3"), libraries (e.g., "TensorFlow"), organizations, or people mentioned.
    - "key_objects" (list(str)): A list of the primary subjects or components being described in the text. Think of these as the main "nouns" of the text, like "Encoder Block", "Attention Mechanism", or "Positional Encoding".
    - "tags" (list(str)): A list of general, single-word or short-phrase categorical tags suitable for filtering (e.g., "NLP", "deep-learning", "architecture", "evaluation").
    - "contextual_text" (str): Rewrite the **[TEXT CHUNK TO ANALYZE]** to be more self-contained. Subtly integrate key information from the **[SECTION CONTEXT]** so that a reader could understand the chunk's main point without having read the rest of the document. Keep it concise and clear.
    - "hypothetical_questions" (list(str)): A list of 3-5 insightful questions that a reader might have after reading the chunk, focusing on deeper understanding or implications of the content.
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
                "content": (
                    "[SECTION CONTEXT]:"
                    f"{section_context}"
                    "---"
                    "[TEXT CHUNK TO ANALYZE]:"
                    f"{chunk_text}"
                    "---"
                )
            }
        ],
        format=TextMetadata.model_json_schema()
    )
    logger.debug(f"Response: {response["message"]["content"]}")

    text_metadata = TextMetadata.model_validate_json(response["message"]["content"])
    return text_metadata


def generate_image_metadata(
    image_path: str,
    llm: str = "gemma3:12b"
) -> ImageMetadata:
    """
    Extract metadata from an image.

    Args:
        image_path (str): Path to the image file.
        llm (str): Language model to use for analysis.
    
    Returns:
        ImageMetadata: An instance of ImageMetadata containing metadata extracted from an image.
    """

    with open(image_path, "rb") as image_file:
        image_base64 = b64encode(image_file.read()).decode('utf-8')

    system_prompt = """
    You are an expert AI system 🤖 specializing in analyzing and describing technical diagrams, charts, and figures for a multimodal Retrieval-Augmented Generation (RAG) system.

    **Your task:**
    1. Analyze the provided image in detail.
    2. Return *exactly* one JSON object that matches the required schema.
    3. **Ensure all extracted information is directly supported by the visual content of the image.** Do not infer or add outside knowledge.

    **Here is an example of the desired output:**
    {
        "title": "The Transformer Model Architecture",
        "summary": "This diagram illustrates the encoder-decoder architecture of the Transformer model. It shows how input embeddings, augmented with positional encoding, are processed through a stack of identical encoder layers on the left, and how the decoder, using the encoder's output, generates the final output probabilities on the right.",
        "key_objects": [
            "Encoder Stack (Nx)",
            "Decoder Stack (Nx)",
            "Multi-Head Attention",
            "Masked Multi-Head Attention",
            "Feed-Forward Network",
            "Positional Encoding",
            "Input Embedding",
            "Output Embedding",
            "Linear Layer",
            "Softmax Layer"
        ],
        "text_in_image": [
            "Inputs",
            "Input Embedding",
            "Positional Encoding",
            "Multi-Head Attention",
            "Add & Norm",
            "Feed Forward",
            "Outputs",
            "Output Embedding",
            "Masked Multi-Head Attention",
            "Linear",
            "Softmax",
            "Output Probabilities",
            "Nx"
        ],
        "contextual_description": "The diagram shows a data flow starting from the bottom. The 'Inputs' pass through an 'Input Embedding' layer and are combined with 'Positional Encoding'. This result is fed into the encoder stack on the left, which consists of 'N' identical layers, each with a 'Multi-Head Attention' sub-layer followed by a 'Feed-Forward' network. The output of the encoder stack is then used as the key and value for the attention mechanism in each layer of the decoder stack on the right. The decoder takes the 'Outputs' (target sequence) as input, processes them through a 'Masked Multi-Head Attention' layer, and then a standard 'Multi-Head Attention' layer that incorporates the encoder's output. Finally, the result passes through another 'Feed-Forward' network, a 'Linear' layer, and a 'Softmax' layer to produce the 'Output Probabilities'.",
        "tags": [
            "transformer",
            "neural-network",
            "NLP",
            "self-attention",
            "architecture",
            "encoder-decoder"
        ]
    }

    **Output Schema**:
    - "title" (str): A concise, descriptive title for the image.
    - "summary" (str): A brief, one-paragraph summary explaining the image's core content and purpose.
    - "key_objects" (list(str)): A list of the main components, labels, or distinct objects visible in the image.
    - "text_in_image" (list(str)): An exact transcription of all text found within the image, in order of appearance if possible.
    - "contextual_description" (str): A detailed, step-by-step explanation of the diagram. Describe the data flow, how components are connected, and what process they illustrate.
    - "tags" (list(str)): A list of general, lowercase, single-word or short-phrase categorical tags suitable for filtering.
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
        format=ImageMetadata.model_json_schema()
    )

    logger.debug(f"Response: {response['message']['content']}")

    image_metadata = ImageMetadata.model_validate_json(response["message"]["content"])
    return image_metadata
