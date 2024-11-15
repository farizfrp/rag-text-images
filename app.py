# app.py
from flask import Flask, render_template, request, jsonify
from extract import extract_text_and_images
from indexer import index_documents
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import load_prompt

app = Flask(__name__)

# Global variables
vectorstore = None
retriever = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data['question']
    
    # Add maximum question length restriction
    MAX_QUESTION_LENGTH = 500
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({
            'error': f'Question too long. Please limit to {MAX_QUESTION_LENGTH} characters.',
            'images': []
        }), 400

    # Check if question is empty
    if not question.strip():
        return jsonify({
            'error': 'Please enter a valid question.',
            'images': []
        }), 400

    # Use vectorstore's similarity search to get documents with scores
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=5)
    
    # Debug: Print scores
    print("Similarity scores:", [score for _, score in docs_and_scores])
    
    MIN_SIMILARITY = 0.4
    if not docs_and_scores or all(score > MIN_SIMILARITY for _, score in docs_and_scores):
        return jsonify({
            'answer': 'Mohon maaf, saya hanya dapat membantu untuk pertanyaan seputar Cahaya Benteng Mas. Silakan ajukan pertanyaan yang berkaitan dengan produk, layanan, atau informasi tentang perusahaan kami.',
            'images': []
        }), 200

    # Load a QA prompt from LangChain's hub
    qa_prompt = hub.pull("rlm/rag-prompt")
    # Or alternatively, you could use other popular prompts like:
    # qa_prompt = load_prompt("hwchase17/qa-with-sources")
    # qa_prompt = load_prompt("rlm/rag-prompt-llama")

    # Use LangChain's RetrievalQA chain with the loaded prompt
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", api_key="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"),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": qa_prompt,
        }
    )

    answer = qa_chain.run(question)

    # Collect images from the retrieved documents if confidence is high enough
    CONFIDENCE_THRESHOLD = 0
    MAX_IMAGES = 1 # Add maximum image limit
    image_data = []
    
    for doc, score in docs_and_scores:
        # Break if we've reached the maximum number of images
        if len(image_data) >= MAX_IMAGES:
            break
            
        similarity_score = 1 - score
        
        if similarity_score >= CONFIDENCE_THRESHOLD:
            images = doc.metadata.get('image_filenames', [])
            # Only add images up to the maximum limit
            remaining_slots = MAX_IMAGES - len(image_data)
            image_data.extend([
                {'filename': img, 'confidence': similarity_score} 
                for img in images[:remaining_slots]
            ])

    # Return the answer and image data
    return jsonify({'answer': answer, 'images': image_data})

# Add initialization code at the module level
pdf_path = 'knowledge.pdf'
documents = extract_text_and_images(pdf_path)
vectorstore = index_documents(documents)
retriever = vectorstore.as_retriever()