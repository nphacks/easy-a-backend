from transformers import pipeline

pipe = pipeline("text2text-generation", model="google/flan-t5-base")     

def generate_answer(question: str, context: str):
    print('Reached generate answer')
    # Combine question and context
    input_text = f"Take the context given below and Answer the question asked.\n\nContext: {context}\n\nQuestion: {question}"
    
    # Generate answer using the pipeline
    output = pipe(input_text, max_new_tokens=200, num_return_sequences=1)
    generated_text = output[0]["generated_text"]
    # Remove the input text from the output to get only the answer
    answer = generated_text.replace(input_text, "").strip()
    return answer

larger_pipe = pipeline("text2text-generation", model="google/flan-t5-large")

def generate_questions_for_concept(text_chunks: list):
    context = " ".join(text_chunks)
    
    # Prompt to generate questions
    input_text = f"Generate a distinct short questions for students based on the following notes. Separate questions with a newline:\n\nNotes: {context}"
    
    output = pipe(input_text, max_new_tokens=200, num_return_sequences=1)
    generated_text = output[0]["generated_text"].strip()
    
    # Split the generated text into individual questions
    questions = [q.strip() for q in generated_text.split("\n") if q.strip()]
    
    return questions