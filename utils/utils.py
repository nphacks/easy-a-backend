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

"""
Model citation:
@misc{https://doi.org/10.48550/arxiv.2210.11416,
  doi = {10.48550/ARXIV.2210.11416},
  url = {https://arxiv.org/abs/2210.11416},
  author = {Chung, Hyung Won and Hou, Le and Longpre, Shayne and Zoph, Barret and Tay, Yi and Fedus, William and Li, Eric and Wang, Xuezhi and Dehghani, Mostafa and Brahma, Siddhartha and Webson, Albert and Gu, Shixiang Shane and Dai, Zhuyun and Suzgun, Mirac and Chen, Xinyun and Chowdhery, Aakanksha and Narang, Sharan and Mishra, Gaurav and Yu, Adams and Zhao, Vincent and Huang, Yanping and Dai, Andrew and Yu, Hongkun and Petrov, Slav and Chi, Ed H. and Dean, Jeff and Devlin, Jacob and Roberts, Adam and Zhou, Denny and Le, Quoc V. and Wei, Jason},
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Scaling Instruction-Finetuned Language Models},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
"""
