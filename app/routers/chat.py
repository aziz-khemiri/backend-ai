from fastapi import APIRouter
from pydantic import BaseModel
import re

from app.services.deepseek_client import get_response
from app.core.logger import logger

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: str
    prompt: str
    chat_type: str  # one of "symptom", "food", "explore"

class ChatResponse(BaseModel):
    response: str

# Configuration des validations par type de chat
CHAT_VALIDATIONS = {
    "symptom": {
        "min_words": 3,
        "error_message": "Please provide at least 3 words describing your symptoms"
    },
    "explore": {
        "min_length": 5,
        "error_message": "Please provide at least 5 characters for your query"
    },
    "food": {
        "min_words": 1,
        "error_message": "Please enter a food name"
    }
}

# Liste de domaines médicaux (français et anglais)
MEDICAL_DOMAINS = [
    r"médic", r"health", r"santé", r"illness", r"maladie", r"condition",
    r"symptom", r"symptôme", r"douleur", r"pain", r"fièvre", r"fever",
    r"toux", r"cough", r"nausée", r"nausea", r"vertige", r"dizziness",
    r"fatigue", r"tiredness", r"infection", r"inflammation", r"blessure",
    r"injury", r"fracture", r"brûlure", r"burn", r"allergie", r"allergy",
    r"éruption", r"rash", r"vomit", r"vomissement", r"cardiaque", r"cardiac",
    r"respiratoire", r"respiratory", r"digestif", r"digestive", r"nerveux",
    r"nervous", r"musculo", r"muscular", r"squelettique", r"skeletal",
    r"endocrinien", r"endocrine", r"immunitaire", r"immune", r"reproducteur",
    r"reproductive", r"blood", r"sang", r"médicament", r"drug", r"traitement",
    r"treatment", r"thérapie", r"therapy", r"vaccin", r"vaccine", r"chirurgie",
    r"surgery", r"diagnostic", r"prévention", r"prevention", r"rétablissement",
    r"recovery", r"réhabilitation", r"rehab", r"tête", r"head", r"coeur", r"heart",
    r"poumon", r"lung", r"estomac", r"stomach", r"foie", r"liver", r"rein", r"kidney",
    r"muscle", r"os", r"bone", r"peau", r"skin", r"yeux", r"eyes", r"oreille", r"ear",
    r"nez", r"nose", r"gorge", r"throat", r"dent", r"tooth", r"brain", r"cerveau",
    r"spine", r"colonne", r"médecin", r"doctor", r"infirmier", r"nurse", r"hôpital",
    r"hospital", r"clinique", r"clinic", r"urgence", r"emergency", r"généraliste",
    r"GP", r"spécialiste", r"specialist", r"dermatologue", r"dermatologist",
    r"cardiologue", r"cardiologist", r"neurologue", r"neurologist", r"pédiatre",
    r"pediatrician", r"gynécologue", r"gynecologist", r"psychiatre", r"psychiatrist",
    r"cancer", r"diabète", r"diabetes", r"asthme", r"asthma", r"hypertension",
    r"cholestérol", r"cholesterol", r"depression", r"dépression", r"anxiété", r"anxiety",
    r"arthrite", r"arthritis", r"alzheimer", r"parkinson", r"epilepsy", r"épilepsie"
]

# Termes alimentaires (français et anglais)
FOOD_DOMAINS = [
    r"food", r"aliment", r"nourriture", r"nutrition", r"nutritif", r"meal", r"repas",
    r"fruit", r"fruits", r"légume", r"vegetable", r"viande", r"meat", r"poisson", r"fish",
    r"produit laitier", r"dairy", r"céréale", r"grain", r"légumineuse", r"legume",
    r"noix", r"nut", r"graine", r"seed", r"épice", r"spice", r"herbe", r"herb",
    r"boisson", r"drink", r"recette", r"recipe", r"cuisine", r"cooking", r"calorie",
    r"protéine", r"protein", r"glucide", r"carb", r"carbohydrate", r"lipide", r"fat",
    r"vitamine", r"vitamin", r"minéral", r"mineral", r"fibre", r"fiber", r"régime",
    r"diet", r"allergie alimentaire", r"food allergy", r"intolérance", r"intolerance",
    r"valeur nutritive", r"nutritional value", r"composition", r"ingrédient", r"ingredient",
    r"bienfait", r"benefit", r"healthy", r"sain", r"manger", r"eat", r"consommer", r"consume",
    r"apple", r"pomme", r"banana", r"banane", r"orange", r"tomato", r"tomate", r"potato",
    r"patate", r"riz", r"rice", r"pâtes", r"pasta", r"pain", r"bread", r"fromage", r"cheese",
    r"lait", r"milk", r"œuf", r"egg", r"sucre", r"sugar", r"sel", r"salt", r"huile", r"oil",
    r"beurre", r"butter", r"yaourt", r"yogurt", r"café", r"coffee", r"thé", r"tea", r"eau", r"water"
]

def is_medical_question(prompt: str) -> bool:
    """Vérifie si la question concerne le domaine médical"""
    prompt_lower = prompt.lower()
    return any(re.search(domain, prompt_lower) for domain in MEDICAL_DOMAINS)

def is_food_related(prompt: str) -> bool:
    """Vérifie si la question concerne l'alimentation"""
    prompt_lower = prompt.lower()
    return any(re.search(domain, prompt_lower) for domain in FOOD_DOMAINS)

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the medical chat bot",
)
async def chat_handler(payload: ChatRequest) -> ChatResponse:
    # Vérification du type de chat
    if payload.chat_type not in CHAT_VALIDATIONS:
        return ChatResponse(response="Invalid chat type")

    validation = CHAT_VALIDATIONS[payload.chat_type]
    
    # Validation de la longueur minimale
    if "min_length" in validation:
        if len(payload.prompt.strip()) < validation["min_length"]:
            return ChatResponse(response=validation["error_message"])
    
    # Validation du nombre de mots
    if "min_words" in validation:
        word_count = len(re.findall(r'\w+', payload.prompt))
        if word_count < validation["min_words"]:
            return ChatResponse(response=validation["error_message"])
    
    # Vérification du domaine selon le type de chat
    if payload.chat_type in ["symptom", "explore"]:
        if not is_medical_question(payload.prompt):
            return ChatResponse(response="Please ask a health or medical-related question")
    
    elif payload.chat_type == "food":
        if not is_food_related(payload.prompt):
            return ChatResponse(response="Please ask a food or nutrition-related question")

    # Traitement de la requête
    try:
        logger.info(
            f"[Chat][{payload.session_id}] {payload.chat_type} → {payload.prompt}"
        )
        
        reply = await get_response(
            session_id=payload.session_id,
            user_input=payload.prompt,
            chat_type=payload.chat_type,
        )
        
        # Vérification de la réponse vide
        if not reply or not reply.strip():
            logger.warning(f"Empty response for session: {payload.session_id}")
            return ChatResponse(response="⚠️ I couldn't generate a response. Please try again with more details.")
        
        # Vérification de la pertinence de la réponse
        if payload.chat_type in ["symptom", "explore"]:
            if not is_medical_question(reply):
                logger.warning(f"Non-medical response detected: {reply[:50]}...")
                return ChatResponse(response="⚠️ I can only provide medical information. Please ask a health-related question.")
        
        elif payload.chat_type == "food":
            if not is_food_related(reply):
                logger.warning(f"Non-food response detected: {reply[:50]}...")
                return ChatResponse(response="⚠️ I can only provide food and nutrition information. Please ask a food-related question.")
        
        return ChatResponse(response=reply)

    except Exception as e:
        logger.exception(f"Error in chat_handler: {str(e)}")
        
        # Messages d'erreur spécifiques
        if payload.chat_type == "symptom":
            return ChatResponse(response="⚠️ Unable to analyze symptoms. Please try again with more details.")
        elif payload.chat_type == "explore":
            return ChatResponse(response="⚠️ Could not find information. Please try a different query.")
        elif payload.chat_type == "food":
            return ChatResponse(response="⚠️ Could not find nutritional information. Please try a different food.")
        
        return ChatResponse(response="Internal chat error")