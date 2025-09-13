import os
import sys
import torch
import json
import argparse
from pathlib import Path
import re
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Import technical definitions and identity answers
from src.interface.definitions import TECHNICAL_DEFINITIONS, IDENTITY_ANSWERS, HALLUCINATION_PRONE_TOPICS, SAFETY_RESPONSES

# Add project root to path to import model
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_root)
from src.model.theta_model import ThetaModel

class ThetaInterface:
    """Interface for the Theta AI model with improved retrieval and validation."""
    
    def __init__(self, model_path="models/theta_final", model_type="gpt2", model_name="gpt2-medium", dataset_path=None):
        """Initialize the Theta AI interface."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        print(f"Loading model from {model_path}...")
        try:
            self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to base GPT2-medium model...")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(self.device)
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
        
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load knowledge base for retrieval
        self.knowledge_base = self.load_knowledge_base()
        
        # Initialize TF-IDF vectorizer for better retrieval
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.kb_questions = [qa['question'] for qa in self.knowledge_base]
        self.kb_answers = [qa['answer'] for qa in self.knowledge_base]
        
        # Create vectorized representation of questions
        if self.kb_questions:
            self.question_vectors = self.vectorizer.fit_transform(self.kb_questions)
        
        # Set generation parameters
        self.max_length = 250  # Increased for more detailed responses
        self.temperature = 0.7
        self.top_p = 0.9
        self.repetition_penalty = 1.2
        
        # System prompts and validation parameters
        self.system_prompt = (
            "You are Theta AI, a helpful assistant created by Frostline Solutions. "
            "Frostline Solutions was founded by Dakota Fryberger (CEO) and Devin Fox (Co-CEO). "
            "Answer questions accurately and professionally based on your knowledge. "
            "If you're unsure about something, acknowledge the limitations of your training. "
            "Provide concise, clear responses that directly address the user's question. "
            "For greetings, respond briefly and professionally without unnecessary verbosity. "
            "You can have general conversations with users, but keep responses focused and relevant. "
            "Remember to abide by ethics and laws when teaching users about cybersecurity and software development for educational purposes only."
        )
        
        # Response quality thresholds
        self.min_response_length = 30
        self.max_repetition_ratio = 0.3
        
    def load_knowledge_base(self):
        """Load knowledge base from disk with priority for conversational and Theta-specific data."""
        knowledge_base = []
        
        # Get project root
        project_root = Path(__file__).resolve().parent.parent.parent
        
        # Priority files to load first (ensure these are loaded first for retrieval priority)
        priority_files = ["conversational_examples.json", "theta_info.json"]
        
        # Add knowledge from priority JSON files first
        datasets_dir = project_root / "Datasets"
        if datasets_dir.exists():
            for priority_file in priority_files:
                file_path = datasets_dir / priority_file
                if file_path.exists():
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                print(f"Loading priority knowledge from {priority_file}...")
                                for item in data:
                                    if isinstance(item, dict) and "question" in item and "answer" in item:
                                        knowledge_base.append(item)
                    except Exception as e:
                        print(f"Error loading knowledge base from {file_path}: {e}")
            
            # Then load all other JSON files
            for json_file in datasets_dir.glob("*.json"):
                if json_file.name not in priority_files:
                    try:
                        with open(json_file, "r") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and "question" in item and "answer" in item:
                                        knowledge_base.append(item)
                    except Exception as e:
                        print(f"Error loading knowledge base from {json_file}: {e}")
        
        print(f"Loaded {len(knowledge_base)} QA pairs into knowledge base.")
        return knowledge_base
    
    def find_relevant_information(self, query):
        """Find relevant information from the knowledge base using TF-IDF similarity."""
        if not self.kb_questions:
            return ""
        
        # Clean and prepare the query
        query = re.sub(r'[^\w\s]', '', query.lower())
        
        # Check for direct conversational queries - improve greeting detection
        greeting_patterns = ["hi", "hello", "hey", "hey theta", "hi theta", "hello theta", "whats up", "how are you"]
        if any(query == pattern or query.startswith(pattern + " ") for pattern in greeting_patterns):
            # Find greeting responses in knowledge base
            for qa_pair in self.knowledge_base:
                if qa_pair['question'].lower() in ["hi", "hello", "hey theta", "hey", "hello", "how are you"]:
                    return f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}\n\n"
        
        # Check explicitly for identity/creator questions to ensure correct attribution
        identity_patterns = ["who created you", "who made you", "who built you", "who developed you", 
                           "who are you", "what are you", "tell me about yourself", "who created theta",
                           "who made theta", "who is your creator"]
        
        if any(pattern in query for pattern in identity_patterns):
            # Prioritize the correct creator information from theta_info.json
            for qa_pair in self.knowledge_base:
                if qa_pair['question'].lower() in ["who created you?", "who developed theta ai?", "who created you", "who developed theta ai"]:
                    return f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}\n\n"
        
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(query_vector, self.question_vectors).flatten()
            
            # Get top 3 most similar questions
            top_indices = similarity_scores.argsort()[-3:][::-1]
            
            # Only consider relevant matches (similarity > 0.2)
            relevant_info = ""
            for idx in top_indices:
                if similarity_scores[idx] > 0.2:  # Minimum similarity threshold
                    relevant_info += f"Question: {self.kb_questions[idx]}\nAnswer: {self.kb_answers[idx]}\n\n"
            
            # Additional check for Theta AI specific queries
            if any(term in query for term in ['theta', 'you', 'yourself', 'your', 'created', 'made', 'built', 'developed']):
                for qa_pair in self.knowledge_base:
                    if any(term in qa_pair['question'].lower() for term in ['theta ai', 'who created', 'developed', 'what are you']):
                        # Check if this question/answer is already included
                        question_already_included = False
                        for line in relevant_info.split('\n'):
                            if qa_pair['question'] in line:
                                question_already_included = True
                                break
                                
                        if not question_already_included:
                            relevant_info += f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}\n\n"
            
            return relevant_info
            
        except Exception as e:
            print(f"Error in retrieval mechanism: {e}")
            # Fallback to keyword matching
            query_keywords = set(re.findall(r'\b\w+\b', query.lower()))
            
            relevant_info = ""
            for qa_pair in self.knowledge_base:
                question = qa_pair['question'].lower()
                answer = qa_pair['answer']
                
                # Check for keyword overlap
                question_keywords = set(re.findall(r'\b\w+\b', question))
                overlap = query_keywords.intersection(question_keywords)
                
                if len(overlap) >= 2 or any(kw in question for kw in query_keywords):  
                    relevant_info += f"Question: {qa_pair['question']}\nAnswer: {answer}\n\n"
            
            return relevant_info
    
    def run_interactive_mode(self):
        """
        Run Theta in interactive mode where user can ask questions.
        """
        print("\n" + "="*50)
        print("  THETA AI ASSISTANT - FROSTLINE SOLUTIONS")
        print("="*50)
        print("Type your questions below. Type 'exit' to quit.")
        print("Example questions:")
        print("- What is Frostline?")
        print("- What is defense in depth?")
        print("- Where is Frostline headquartered?")
        print("="*50 + "\n")
        
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for using Theta AI. Goodbye!")
                break
                
            # Find relevant information
            relevant_info = self.find_relevant_information(user_input)
            
            if relevant_info:
                print("\nTheta: ")
                print(relevant_info)
            else:
                # Generate response
                print("\nTheta: ", end="")
                
                try:
                    # Use the hybrid retrieval + generative approach
                    response = self.generate_response(user_input)
                    print(response)
                except Exception as e:
                    print(f"Sorry, I encountered an error: {str(e)}")
                
            print()  # Extra line for readability
    
    def answer_question(self, query):
        """Answer a question using the Theta AI model with strong hallucination prevention."""
        # Find relevant information from the knowledge base
        relevant_info = self.find_relevant_information(query)
        
        # Detect potentially problematic queries that might lead to hallucinations
        query_lower = query.lower()
        
        # Categorize the question type for specialized handling
        identity_related = any(term in query_lower for term in [
            "who created", "who made", "who built", "founder", "ceo", "owner", 
            "who are you", "who is theta", "what are you", "tell me about yourself",
            "developed by", "made by", "built by", "creator", "developers"
        ])
        
        tech_definition = any(term in query_lower for term in [
            "what is", "define", "explain", "how does", "tell me about", "describe"
        ])
        
        # Add strong safety guardrails based on question type
        safety_prompt = ""
        
        # For identity questions, provide explicit facts to prevent hallucination
        if identity_related:
            safety_prompt = (
                "IMPORTANT: Theta AI was created by Frostline Solutions, founded by Dakota Fryberger (CEO) "
                "and Devin Fox (Co-CEO). The company is headquartered in Copperas Cove, Texas. "
                "Do not attribute Theta's creation to anyone else or make up any other details. "
                "If uncertain about any details, ONLY state the facts explicitly mentioned here. "
                "DO NOT mention or include any other names, companies, or institutions in your response. "
                "DO NOT make up background stories, educational history, or company details."
            )
        
        # For technical definitions, prefer knowledge base over generation
        elif tech_definition:
            safety_prompt = (
                "IMPORTANT: Provide accurate technical information based ONLY on your knowledge base. "
                "If the information isn't in your knowledge base, state that you don't have specific information "
                "rather than generating a potentially inaccurate response. Keep definitions concise and accurate. "
                "DO NOT make up technical specifications, version numbers, or compatibility information."
            )
        
        # For all other questions, add general hallucination prevention
        else:
            safety_prompt = (
                "IMPORTANT: Answer based only on known information. If you're unsure, acknowledge "
                "the limitations of your knowledge rather than making up information."
            )
            
        # If relevant info found, use it to guide the model's response
        if relevant_info:
            prompt = f"{self.system_prompt}\n\n{safety_prompt}\n\nThe following information might help answer the question:\n\n{relevant_info}\n\nQuestion: {query}\nAnswer:"
        else:
            if identity_related or tech_definition:
                # For high-risk questions without relevant info, provide a safe response
                generated = self.generate_safe_response(query, identity_related, tech_definition)
                return generated
            else:
                prompt = f"{self.system_prompt}\n\n{safety_prompt}\n\nQuestion: {query}\nAnswer:"
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=self.max_length + len(inputs["input_ids"][0]),
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode output
        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the answer part
        answer = generated.split("Answer:")[-1].strip()
        
        # Validate response
        if len(answer) < self.min_response_length:
            print("Response too short, retrying...")
            # Retry with different temperature
            temp_backup = self.temperature
            self.temperature = 0.9
            result = self.answer_question(query)
            self.temperature = temp_backup
            return result
        
        if self.calculate_repetition_ratio(answer) > self.max_repetition_ratio:
            print("Response too repetitive, retrying...")
            # Retry with higher repetition penalty
            rep_backup = self.repetition_penalty
            self.repetition_penalty = 1.5
            result = self.answer_question(query)
            self.repetition_penalty = rep_backup
            return result
            
        return answer
    
    def generate_response(self, question):
        """
        Generate a response to a question with improved controls and validations.
        
        Args:
            question: The question to answer
            
        Returns:
            Theta's response
        """
        try:
            # Normalize the question for lookup and analysis
            question_lower = question.lower().strip()
            question_without_punct = re.sub(r'[^\w\s]', '', question_lower)
            
            # Check for exact identity questions in our templates
            for key, response in IDENTITY_ANSWERS.items():
                if question_lower == key or question_lower.startswith(key + "?"):
                    return response
            
            # Check for technical definitions based on common patterns
            is_definition_question = any(pattern in question_lower for pattern in [
                "what is", "what's", "what are", "define", "explain", "tell me about", 
                "what does", "meaning of", "definition of", "stands for"
            ])
            
            # Extract the term being defined
            term = None
            if is_definition_question:
                # Try to extract the term after common patterns
                for pattern in ["what is", "what's", "what does", "define", "explain", "tell me about"]:
                    if pattern in question_lower:
                        # Get everything after the pattern
                        term_part = question_lower.split(pattern, 1)[1].strip().rstrip('?')
                        # If it contains "mean" or "stand for", handle specially
                        if "mean" in term_part:
                            term = term_part.split("mean")[0].strip()
                        elif "stand for" in term_part:
                            term = term_part.split("stand for")[0].strip()
                        else:
                            term = term_part
                        break
                
                # If we extracted a term, check if we have a definition
                if term:
                    # Clean up term (remove articles like "a", "an", "the")
                    term = re.sub(r'^(a|an|the)\s+', '', term).strip()
                    # Check in our technical definitions
                    if term in TECHNICAL_DEFINITIONS:
                        return TECHNICAL_DEFINITIONS[term]
            
            # Set dynamic max_length based on question type
            dynamic_max_length = self.max_length
            if is_definition_question:
                # Shorter responses for definitions
                dynamic_max_length = min(150, self.max_length)
            elif any(topic in question_lower for topic in HALLUCINATION_PRONE_TOPICS):
                # Very cautious with hallucination-prone topics
                dynamic_max_length = min(120, self.max_length)
                # For these topics, sometimes directly return a safety response
                if random.random() < 0.7:  # 70% chance to use safety response
                    safety_keys = list(SAFETY_RESPONSES.keys())
                    return SAFETY_RESPONSES[random.choice(safety_keys)]
            
            # Prepare the input prompt
            input_prompt = self.system_prompt + "\n\n" + question
            
            # Tokenize the input prompt
            input_ids = self.tokenizer.encode(input_prompt, return_tensors='pt').to(self.device)
            
            # Generate the response with dynamic max length
            output = self.model.generate(input_ids, max_length=dynamic_max_length, 
                                         temperature=self.temperature, top_p=self.top_p, 
                                         repetition_penalty=self.repetition_penalty)
            
            # Convert the response to text
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Validate the response
            if len(response) < self.min_response_length:
                return "Sorry, I couldn't find a good answer to your question."
            elif self.calculate_repetition_ratio(response) > self.max_repetition_ratio:
                return "Sorry, I'm not sure I understand your question."
            
            # Post-process the response to remove the input prompt if it's included
            if input_prompt in response:
                response = response.replace(input_prompt, "").strip()
                
            return response
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_safe_response(self, query, is_identity_related, is_tech_definition):
        """
        Generate a safe response for high-risk questions when no relevant information is found.
        
        Args:
            query: The user's question
            is_identity_related: Whether the question is about identity
            is_tech_definition: Whether the question is a technical definition request
            
        Returns:
            A safe response that won't contain hallucinations
        """
        if is_identity_related:
            return (
                "I'm Theta AI, an assistant developed by Frostline Solutions. Frostline Solutions was founded by "
                "Dakota Fryberger (CEO) and Devin Fox (Co-CEO) and is headquartered in Copperas Cove, Texas. "
                "I'm designed to provide information about cybersecurity, software development, and other technical topics. "
                "I don't have information about other details or background that wasn't part of my training data."
            )
        elif is_tech_definition:
            return (
                f"I don't have specific information about '{query.strip()}' in my knowledge base. "
                f"I'm trained on specific cybersecurity, software development, and IT concepts, but this particular topic "
                f"may not be covered in my training data or may require more recent information than I have available. "
                f"To get accurate information on this topic, I'd recommend consulting official documentation or trusted sources."
            )
        else:
            return "I don't have enough information to answer this question accurately. I'm designed to provide reliable information about cybersecurity, software development, and IT topics based on my training data."
    
    def calculate_repetition_ratio(self, text):
        """
        Calculate the repetition ratio of a given text.
        
        Args:
            text: The text to calculate the repetition ratio for
            
        Returns:
            The repetition ratio
        """
        words = text.split()
        unique_words = set(words)
        repetition_ratio = 1 - len(unique_words) / len(words)
        return repetition_ratio


def main():
    parser = argparse.ArgumentParser(description="Run the Theta AI interface")
    
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the trained model directory")
    parser.add_argument("--model_type", type=str, default="gpt2",
                        help="Model type (gpt2, bert-qa)")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Specific model name/version")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to the dataset for retrieval-based answers")
    
    args = parser.parse_args()
    
    # Initialize and run the interface
    interface = ThetaInterface(
        model_path=args.model_path,
        model_type=args.model_type,
        model_name=args.model_name,
        dataset_path=args.dataset_path
    )
    
    interface.run_interactive_mode()

if __name__ == "__main__":
    main()
