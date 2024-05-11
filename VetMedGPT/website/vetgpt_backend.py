from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "azsxscdvfb/VetMedGPT-chat-V0.1"
model = AutoModelForCausalLM.from_pretrained(model_id)
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.use_default_system_prompt = False


from flask import Flask, request, jsonify

app = Flask(__name__)

def chat_with_tinyllama(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.to('cuda')
    output = model.generate(input_ids, max_new_tokens=128,do_sample=True,eos_token_id=tokenizer.eos_token_id)
    response = tokenizer.batch_decode(output)
    return response

def generate_answer_with_tinyllama(question):
    prompt = f"You are an professional veterinarian, you will have a question and write your answer follow by answer: ###input: question: {question}\n\n### answer:"
    response = chat_with_tinyllama(prompt)
    answer_start_idx = response[0].rfind('answer:') + len('answer:')
    try:
        answer = response[0][answer_start_idx:].strip().lower()
    except:
        return None
    return answer,response[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    output,_ = generate_answer_with_tinyllama(text)
    
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port = 8000)
