from Levenshtein import matching_blocks
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz
import requests
from bs4 import BeautifulSoup
from Levenshtein import distance as levenshtein_distance
from fuzzywuzzy import process
import os


app = Flask(__name__)
# CORS(app)
# CORS(app, origins=["https://chat-vit-us-frontend.vercel.app"])
# CORS(app, resources={r"/*": {"origins": "*"}})
CORS(app, resources={r"/*": {
    "origins": "https://chat-vit-us-frontend.vercel.app",
    "methods": ["POST"],  # Or ["GET", "POST"] if needed
    "headers": ["Content-Type"] # If you're sending Content-Type header
}})

# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

campus_finding_intents = [
    "which campus offers 'x' course", "which campus offers b.tech 'x'", "which campus offers b.tech 'x' engineering", "in which campus 'x' course is available",
    "in which campus b.tech 'x' is available", "in which campus b.tech 'x' engineering is available", "campus that offers b.tech 'x'", "campus that offers'x' course", "campus that offers b.tech 'x' engineering",
    "which campus has b.tech 'x", "which campus has b.tech 'x' engineering", "which campus has 'x' course"
]

campus_finding_embeddings = model.encode(campus_finding_intents, convert_to_tensor=True)

course_list_intents = [
    "list all the courses", "list all the courses in VIT", "name all the courses", "name all the courses in vit", "give all the courses in vit", "what are all the courses available in VIT",
    "courses in vit", "what are the courses offered by vit", "courses offered by vit", "all the courses offeredd by vit", "list all the courses available in vit", "list all the courses offered by vit",
    "b.tech courses available in vit 'x'", "b.tech courses available in 'x'"
]

course_list_embedding = model.encode(course_list_intents, convert_to_tensor=True)

def scrape_programs_offered(course_type=None):
    url = 'https://vit.ac.in/admissions/programmes-offered'
    response = requests.get(url)
    
    if(response.status_code == 200):
        soup = BeautifulSoup(response.text, 'html.parser')
        
        course_elements = soup.find_all('h2', class_='elementor-heading-title')
        
        exclude_list = [
            "Announcements", "Career Development Centre", "Recruiting Companies", "International Admission", "UG Programmes,PG Programmes,Integrated Programmes,Research Programmes", "Research Organisation","Students Welfare", "Overview,Newsletter,Students Club,Students Chapter,Campus Events,Counselling Division", 
            "Programmes Offered", "Home,Admissions,Programmes Offered", "Programmes Offered 2025-26", "VIT @ Connect", "IQAC,NATS,Industrial Visit Form,Student's Code Of Conduct,Refund Policy,Intranet,Events Portal,Student Login,Parent Login,Peopleorbit,VIT Gmail", "Other Links", "COVID 19 - Initiatives,Site Map",
            "VISITORS", "Careers,Alumni,Contact Us,Guest House,Hotels in Vellore", "Committees @ VIT", "IPR and Technology Transfer Cell,Internal Complaints Committee (ICC),Student Grievance Redressal", "Don't Trust Fake Website/ Page / Channels", "VIT Campus Tour", "BEWARE OF ILLEGAL/FAKE WEBSITES", "Follow Us:", "Last Updated : February 2025", "Beware of VITEEE fake websites", "Admission details not found."
        ]
        
        course_campus_mapping = []
        
        for course in course_elements:
            course_text = course.get_text().strip()
            if(course_text not in exclude_list):
                campus_section = course.find_next('ul', class_='elementor-icon-list-items')
                if(campus_section):
                    campuses = campus_section.find_all('a')
                    campus_names = [campus.get_text().strip() for campus in campuses if campus.get_text()]
                    campus_info = ','.join(campus_names) if campus_names else "Campus not found"
                else:
                    campus_info = "Campus not listed"
                
                if(course_type):
                    if(course_type.lower() in course_text.lower()):
                        course_campus_mapping.append((course_text, campus_info))
                else:
                    course_campus_mapping.append((course_text, campus_info))
        
        admission_section = soup.find('div', {'class': 'admission-info'})
        if admission_section:
            admission_info = admission_section.text.strip()
        else:
            admission_info = "Admission details not found."
        
        return course_campus_mapping, admission_info
    else:
        return [], "Error: Could not fetch data from the website"

def correct_spelling(user_query, intents):
    best_match = min(intents, key=lambda intent: levenshtein_distance(user_query, intent))
    if levenshtein_distance(user_query, best_match) <= 2: 
        return best_match
    return user_query

def load_responses_from_file(file_path):
    responses = {}
    with open(file_path, "r") as file:
        for line in file:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                responses[key.lower()] = value
    return responses

custom_responses = load_responses_from_file("custom_responses.txt")
stored_prompts = list(custom_responses.keys())
stored_prompt_embeddings = model.encode(stored_prompts, convert_to_tensor=True)

@app.route('/ask', methods=['POST','OPTIONS'])
def ask():
    if request.method == 'OPTIONS':  # Handle preflight request
        # No need to return any data, just the headers
        response = jsonify({'message': 'Preflight request successful'})  # Or an empty response
        return response, 200
        
    user_query = request.json.get('query').lower()
    user_query = correct_spelling(user_query, campus_finding_intents + course_list_intents)
    user_id = request.json.get('user_id', None) 

    print(f"User Query: {user_query}")
    
    #Campus finding score
    campus_finding_similarity_score = util.cos_sim(model.encode(user_query, convert_to_tensor=True), campus_finding_embeddings)
    best_match_idx_campus = campus_finding_similarity_score.argmax().item()
    best_match_score_campus = campus_finding_similarity_score[0][best_match_idx_campus].item()
    best_match_intent_campus = campus_finding_intents[best_match_idx_campus]
    
    print(f"Best Match (Campus Finding): {best_match_intent_campus} | Score: {best_match_score_campus}")
    
    #Course list score
    course_list_similarity_score = util.cos_sim(model.encode(user_query, convert_to_tensor=True), course_list_embedding)
    best_match_idx_course = course_list_similarity_score.argmax().item()
    best_match_score_course = course_list_similarity_score[0][best_match_idx_course].item()
    best_match_intent_course = course_list_intents[best_match_idx_course]
    
    print(f"Best Match (Courses list): {best_match_intent_course} | Score: {best_match_score_course}")
    
    fuzzy_match, fuzzy_score = process.extractOne(user_query, stored_prompts)
    print(f"Fuzzy match: {fuzzy_match}, Score: {fuzzy_score}")
    
    if(best_match_score_campus > 0.60 and best_match_score_campus>best_match_score_course):
        course_name = user_query.split("which campus has")[-1].strip()
        course_name = course_name.replace("b.tech", "B.Tech").strip()
        course_name = course_name.replace("m.tech", "M.Tech").strip()
        course_name = course_name.replace("ph.d", "Ph.D").strip()
        
        course_info, _ = scrape_programs_offered()
        
        course_name_lower = course_name.lower()
        matching_courses = [course for course, _ in course_info if fuzz.partial_ratio(course_name_lower, course.lower()) > 90]
        
        if(matching_courses):
            campus_details = []
            for course in matching_courses:
                campuses = [campus for course_item, campus in course_info if fuzz.partial_ratio(course.lower(), course_item.lower()) > 90]
                unique_campuses = sorted(set(','.join(campuses).split(',')))
                campus_details.append(f"{course} is available at {','.join(unique_campuses)}")
            
            response = "\n".join(campus_details)
        else:
            response = f"Sorry, I coudn't find the exact matches for the course '{course_name}'.\nTry rephrasing or checking similar courses"
    
    elif(best_match_score_course > 0.77):
        
        if "vellore" in user_query.lower():
            campus_filter = "Vellore"
        elif "chennai" in user_query.lower():
            campus_filter = "Chennai"
        elif "bhopal" in user_query.lower():
            campus_filter = "VIT-Bhopal"
        elif "amaravati" in user_query.lower():
            campus_filter = "VIT-AP"
        else:
            campus_filter = None
        
        if("b.tech" in user_query.lower()):
            course_info, admission_info = scrape_programs_offered("B.Tech")
        elif("m.tech" in user_query.lower()):
            course_info, admission_info = scrape_programs_offered("M.Tech")
        elif("integrated" in user_query.lower()):
            course_info, admission_info = scrape_programs_offered("Integrated")
        else:
            course_info, admission_info = scrape_programs_offered()
        
        if campus_filter:
            course_info = [(course, campus) for course, campus in course_info if campus_filter in campus]
        
        if course_info:
            response = "Here are the Courses offered by VIT University:\n"
            for course, campus in course_info:
                response += f"{course} - Offered at: {campus}\n"
            
        # response += f"\n{admission_info}"
        
    elif(fuzzy_score >75):
        response = custom_responses[fuzzy_match]
    
    else:
        user_embedding = model.encode(user_query, convert_to_tensor=True)
        similarity_scores = util.cos_sim(user_embedding, stored_prompt_embeddings)
        best_match_idx = similarity_scores.argmax().item()
        best_match_score = similarity_scores[0][best_match_idx].item()

        if best_match_score > 0.7: 
            best_match = stored_prompts[best_match_idx]
            response = custom_responses[best_match]
        else:
            response = "Sorry, I didn't understand that."
    
    # mongo_backend_url = "http://localhost:4000/chats/storeMessage"
    mongo_backend_url = "https://chatvitus-backend.onrender.com/chats/storeMessage"
    chat_data = {
        "user_id": user_id, 
        "messages": [
            {"sender": "user", "text": user_query},
            {"sender": "bot", "text": response}
        ]
    }
    try:
        requests.post(mongo_backend_url, json=chat_data)
    except requests.exceptions.RequestException as e:
        print(f"Error sending chat to DB: {e}")
    
    return jsonify({'response': response})

if(__name__== '__main__'):
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
