from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, constr
import re
from typing import Literal

from app.services.deepseek_client import get_response
from app.core.logger import logger

router = APIRouter()

# === MODELS ===
class ChatRequest(BaseModel):
    session_id: constr(min_length=1, max_length=64)
    prompt: constr(min_length=1, max_length=500)
    chat_type: Literal["symptom", "qa", "food", "explore"]

class ChatResponse(BaseModel):
    response: str

# === VALIDATION RULES ===
CHAT_VALIDATIONS = {
    "symptom": {
        "min_words": 3,
        "error": "❌ Please provide at least 3 words describing your symptoms"
    },
    "explore": {
        "min_length": 3,
        "error": "🔍 Please provide at least 5 characters for your query"
    },
    "food": {
        "min_words": 1,
        "error": "🍎 Please enter a food name"
    },
    "qa": {
        "min_length": 5,
        "error": "❓ Please provide at least 5 characters for your question"
    }
}

# === PRE-COMPILED REGEX PATTERNS ===
MEDICAL_PATTERNS = [
    r"\bmédic\w*", r"\bmedical\b", r"\bhealth\b", r"\bsanté\b", r"\billness\b", 
    r"\bmaladie\b", r"\bcondition\b", r"\bsymptom\b", r"\bsymptôme\b", r"\bdouleur\b",
    r"\bpain\b", r"\bfièvre\b", r"\bfever\b", r"\btoux\b", r"\bcough\b", r"\bnausée\b",
    r"\bnausea\b", r"\bvertige\b", r"\bdizziness\b", r"\bfatigue\b", r"\btiredness\b",
    r"\bvomissement\b", r"\bvomit\b", r"\bcardiaque\b", r"\bcardiac\b", r"\bcœur\b",
    r"\bheart\b", r"\brespiratoire\b", r"\brespiratory\b", r"\bpoumon\b", r"\blung\b",
    r"\bdigestif\b", r"\bdigestive\b", r"\bestomac\b", r"\bstomach\b", r"\bfoie\b",
    r"\bliver\b", r"\brein\b", r"\bkidney\b", r"\bmuscle\b", r"\bmuscular\b", r"\bos\b",
    r"\bbone\b", r"\bpeau\b", r"\bskin\b", r"\byeux\b", r"\beyes\b", r"\boreille\b",
    r"\bear\b", r"\bnez\b", r"\bnose\b", r"\bgorge\b", r"\bthroat\b", r"\bdent\b",
    r"\btooth\b", r"\bcerveau\b", r"\bbrain\b", r"\bcolonne\b", r"\bspine\b", r"\btête\b",
    r"\bhead\b", r"\bcancer\b", r"\bdiabète\b", r"\bdiabetes\b", r"\basthme\b",
    r"\basthma\b", r"\bhypertension\b", r"\bcholestérol\b", r"\bcholesterol\b",
    r"\bdépression\b", r"\bdepression\b", r"\banxiété\b", r"\banxiety\b", r"\barthrite\b",
    r"\barthritis\b", r"\balzheimer\b", r"\bparkinson\b", r"\bépilepsie\b", r"\bepilepsy\b",
    r"\binfection\b", r"\binflammation\b", r"\bblessure\b", r"\binjury\b", r"\bfracture\b",
    r"\bbrûlure\b", r"\bburn\b", r"\ballergie\b", r"\ballergy\b", r"\béruption\b",
    r"\brush\b", r"\bmédicament\b", r"\bdrug\b", r"\btraitement\b", r"\btreatment\b",
    r"\bthérapie\b", r"\btherapy\b", r"\bvaccin\b", r"\bvaccine\b", r"\bchirurgie\b",
    r"\bsurgery\b", r"\bdiagnostic\b", r"\bprévention\b", r"\bprevention\b",
    r"\brétablissement\b", r"\brecovery\b", r"\bréhabilitation\b", r"\brehab\b",
    r"\bmédecin\b", r"\bdoctor\b", r"\binfirmier\b", r"\bnurse\b", r"\bhôpital\b",
    r"\bhospital\b", r"\bclinique\b", r"\bclinic\b", r"\burgence\b", r"\bemergency\b",
    r"\bgénéraliste\b", r"\bgp\b", r"\bspécialiste\b", r"\bspecialist\b", r"\bdermatologue\b",
    r"\bdermatologist\b", r"\bcardiologue\b", r"\bcardiologist\b", r"\bneurologue\b",
    r"\bneurologist\b", r"\bpédiatre\b", r"\bpediatrician\b", r"\bgynécologue\b",
    r"\bgynecologist\b", r"\bpsychiatre\b", r"\bpsychiatrist\b"
]

FOOD_PATTERNS = [
    r"\bfood\b", r"\baliment\b", r"\bnourriture\b", r"\bnutrition\b", r"\bnutritif\b",
    r"\bnutritional\b", r"\bvaleur nutritive\b", r"\bcomposition\b", r"\bingrédient\b",
    r"\bingredient\b", r"\brégime\b", r"\bdiet\b", r"\bcalorie\b", r"\bmeal\b", r"\brepas\b",
    r"\bcuisine\b", r"\bcooking\b", r"\brecette\b", r"\brecipe\b", r"\bmanger\b", r"\beat\b",
    r"\bconsommer\b", r"\bconsume\b", r"\bfruit\b", r"\bfruits\b", r"\blégume\b",
    r"\bvegetable\b", r"\bviande\b", r"\bmeat\b", r"\bpoisson\b", r"\bfish\b", r"\bproduit laitier\b",
    r"\bdairy\b", r"\bcéréale\b", r"\bgrain\b", r"\blégumineuse\b", r"\blegume\b", r"\bnoix\b",
    r"\bnut\b", r"\bgraine\b", r"\bseed\b", r"\bprotéine\b", r"\bprotein\b", r"\bglucide\b",
    r"\bcarb\b", r"\bcarbohydrate\b", r"\blipide\b", r"\bfat\b", r"\bvitamine\b", r"\bvitamin\b",
    r"\bminéral\b", r"\bmineral\b", r"\bfibre\b", r"\bfiber\b", r"\bépice\b", r"\bspice\b",
    r"\bherbe\b", r"\bherb\b", r"\bsucre\b", r"\bsugar\b", r"\bsel\b", r"\bsalt\b", r"\bhuile\b",
    r"\boil\b", r"\bbeurre\b", r"\bbutter\b", r"\bapple\b", r"\bpomme\b", r"\bbanana\b", r"\bbanane\b",
    r"\borange\b", r"\btomato\b", r"\btomate\b", r"\bpotato\b", r"\bpatate\b", r"\briz\b", r"\brice\b",
    r"\bpâtes\b", r"\bpasta\b", r"\bpain\b", r"\bbread\b", r"\bfromage\b", r"\bcheese\b", r"\blait\b",
    r"\bmilk\b", r"\bœuf\b", r"\begg\b", r"\byaourt\b", r"\byogurt\b", r"\bboisson\b", r"\bdrink\b",
    r"\beau\b", r"\bwater\b", r"\bcafé\b", r"\bcoffee\b", r"\bthé\b", r"\btea\b", r"\ballergie alimentaire\b",
    r"\bfood allergy\b", r"\bintolérance\b", r"\bintolerance\b", r"\bbienfait\b", r"\bbenefit\b", r"\bsain\b",
    r"\bhealthy\b"
]

# Ajout de patterns génériques pour capturer plus de noms d'aliments
GENERIC_FOOD_PATTERNS = [
    r"\b\w+food\b",  # capture "seafood", "superfood", etc.
    r"\b\w+nut\b",   # capture "chestnut", "coconut", etc.
    r"\b\w+seed\b",  # capture "sunflowerseed", "pumpkinseed", etc.
    r"\b\w+fish\b",  # capture "shellfish", "goldfish", etc.
    r"\b\w+fruit\b", # capture "dragonfruit", "passionfruit", etc.
    r"\b\w+berry\b", # capture "strawberry", "blueberry", etc.
]

# === SYMPTOM DOMAIN PATTERNS ===
SYMPTOM_WORDS = [
    # English symptoms
    "headache", "migraine", "dizziness", "nausea", "vomit", "fatigue", "fever", 
    "chills", "cough", "pain", "cramps", "rash", "itch", "swelling", "numbness",
    "weakness", "tiredness", "sore", "stiff", "shortness", "breath", "wheezing",
    "diarrhea", "constipation", "bleeding", "discharge", "palpitations", "anxiety",
    "depression", "insomnia", "drowsiness", "shivering", "shaking", "tremors",
    "congestion", "sneezing", "runny nose", "sore throat", "heartburn", "indigestion",
    "bloating", "gas", "urination", "dehydration", "appetite", "hunger", "thirst",
    "weight", "vision", "hearing", "taste", "smell", "balance", "coordination",
    "memory", "concentration", "confusion", "mood", "irritability", "agitation",
    
    # French symptoms
    "migraine", "vertige", "nausée", "vomissement", "fatigue", "fièvre", "frissons",
    "toux", "douleur", "crampe", "éruption", "démangeaison", "gonflement", "engourdissement",
    "faiblesse", "raideur", "essoufflement", "respiration", "sifflement", "diarrhée",
    "constipation", "saignement", "écoulement", "palpitation", "anxiété", "dépression",
    "insomnie", "somnolence", "tremblement", "frisson", "congestion", "éternuement",
    "nez qui coule", "mal de gorge", "brûlure d'estomac", "indigestion", "ballonnement",
    "gaz", "urination", "déshydratation", "appétit", "faim", "soif", "poids", "vision",
    "audition", "goût", "odorat", "équilibre", "coordination", "mémoire", "concentration",
    "confusion", "humeur", "irritabilité", "agitation"
]

GENERIC_SYMPTOM_PATTERNS = [
    r"\bache\b", r"\bpain\b", r"\bsick\b", r"\bnauseous\b", r"\bthrobbing\b", 
    r"\bsharp\b", r"\bdull\b", r"\burning\b", r"\btenderness\b", r"\bsensitivity\b",
    r"\bdiscomfort\b", r"\bpressure\b", r"\btingling\b", r"\bpins and needles\b",
    r"\bhot flash\b", r"\bcold sweat\b", r"\bnight sweat\b", r"\bloss of appetite\b",
    r"\bincreased thirst\b", r"\bfrequent urination\b", r"\bdifficulty swallowing\b",
    r"\bjoint stiffness\b", r"\blimited mobility\b", r"\bdifficulty breathing\b",
    r"\bchest tightness\b", r"\bracing pain\b", r"\bstabbing pain\b", r"\baching pain\b",
    r"\bpersistent cough\b", r"\bblood in stool\b", r"\bblood in urine\b"
]

CONTEXTUAL_PHRASES = [
    "I feel", "I am experiencing", "I have been having", "suffering from",
    "bothering me", "woke up with", "started feeling", "having trouble with",
    "my symptoms are", "I've noticed", "recently developed", "for the past",
    "je ressens", "j'ai des", "je souffre de", "je me plains de", "depuis",
    "mes symptômes", "j'ai remarqué", "j'ai développé", "au cours des",
    "j'ai mal au", "j'ai mal à la", "j'ai des difficultés à", "je ne peux plus"
]

BODY_PARTS = [
    "head", "neck", "back", "chest", "stomach", "abdomen", "arm", "leg", "hand", "foot",
    "eye", "ear", "nose", "throat", "skin", "joint", "muscle", "bone", "heart", "lung",
    "liver", "kidney", "bladder", "thigh", "knee", "ankle", "shoulder", "elbow", "wrist",
    "finger", "toe", "jaw", "hip", "spine", "pelvis", "groin", "buttocks", "calf", "shin",
    "forearm", "bicep", "tricep", "palm", "sole", "heel", "toenail", "fingernail", "scalp",
    "face", "forehead", "temple", "cheek", "chin", "lip", "tongue", "gum", "tooth", "uvula",
    "tonsil", "esophagus", "diaphragm", "rib", "collarbone", "shoulder blade", "tailbone",
    "vein", "artery", "nerve", "tendon", "ligament",
    # French translations
    "tête", "cou", "dos", "poitrine", "ventre", "abdomen", "bras", "jambe", "main", "pied",
    "œil", "oreille", "nez", "gorge", "peau", "articulation", "muscle", "os", "cœur", "poumon",
    "foie", "rein", "vessie", "cuisse", "genou", "cheville", "épaule", "coude", "poignet",
    "doigt", "orteil", "mâchoire", "hanche", "colonne vertébrale", "bassin", "aine", "fesse",
    "mollet", "tibia", "avant-bras", "biceps", "triceps", "paume", "plante du pied", "talon",
    "ongle d'orteil", "ongle", "cuir chevelu", "visage", "front", "tempe", "joue", "menton",
    "lèvre", "langue", "gencive", "dent", "luette", "amygdale", "œsophage", "diaphragme",
    "côte", "clavicule", "omoplate", "coccyx", "veine", "artère", "nerf", "tendon", "ligament"
]

# Combiner tous les patterns
ALL_FOOD_PATTERNS = FOOD_PATTERNS + GENERIC_FOOD_PATTERNS
ALL_SYMPTOM_PATTERNS = MEDICAL_PATTERNS + GENERIC_SYMPTOM_PATTERNS

# Compiler les regex
MEDICAL_REGEX = re.compile("|".join(ALL_SYMPTOM_PATTERNS), re.IGNORECASE)
FOOD_REGEX = re.compile("|".join(ALL_FOOD_PATTERNS), re.IGNORECASE)

# Liste de noms d'aliments spécifiques
FOOD_NAME_LIST = [
    "scallop", "pétoncle", "almond", "amande", "quinoa", "avocado", "avocat",
    "salmon", "saumon", "tuna", "thon", "shrimp", "crevette", "lobster", "homard",
    "oyster", "huître", "mussel", "moule", "octopus", "poulpe", "squid", "calamar",
    "beef", "bœuf", "chicken", "poulet", "pork", "porc", "lamb", "agneau", "duck", "canard",
    "turkey", "dinde", "veal", "veau", "bacon", "ham", "jambon", "sausage", "saucisse",
    "rice", "riz", "pasta", "pâtes", "noodle", "nouille", "bread", "pain", "baguette",
    "croissant", "cheese", "fromage", "yogurt", "yaourt", "milk", "lait", "cream", "crème",
    "butter", "beurre", "egg", "œuf", "tofu", "tempeh", "seitan", "wheat", "blé", "rye", "seigle",
    "oat", "avoine", "barley", "orge", "corn", "maïs", "lentil", "lentille", "bean", "haricot",
    "pea", "pois", "chickpea", "pois chiche", "soy", "soja", "nut", "noix", "peanut", "cacahuète",
    "walnut", "noix", "hazelnut", "noisette", "pistachio", "pistache", "cashew", "noix de cajou",
    "pecan", "pécan", "macadamia", "coconut", "noix de coco", "pineapple", "ananas", "mango", "mangue",
    "papaya", "papaye", "kiwi", "melon", "watermelon", "pastèque", "grape", "raisin", "lemon", "citron",
    "lime", "citron vert", "orange", "tangerine", "mandarine", "peach", "pêche", "plum", "prune",
    "apricot", "abricot", "cherry", "cerise", "strawberry", "fraise", "raspberry", "framboise",
    "blueberry", "myrtille", "blackberry", "mûre", "cranberry", "canneberge", "pomegranate", "grenade",
    "fig", "figue", "date", "datte", "olive", "artichoke", "artichaut", "asparagus", "asperge",
    "broccoli", "brocoli", "cabbage", "chou", "carrot", "carotte", "celery", "céleri", "cucumber", "concombre",
    "eggplant", "aubergine", "garlic", "ail", "ginger", "gingembre", "onion", "oignon", "pepper", "poivron",
    "potato", "pomme de terre", "pumpkin", "citrouille", "spinach", "épinard", "tomato", "tomate", "zucchini",
    "courgette", "lettuce", "laitue", "mushroom", "champignon", "truffle", "truffe", "basil", "basilic",
    "thyme", "thym", "rosemary", "romarin", "parsley", "persil", "coriander", "coriandre", "mint", "menthe",
    "oregano", "origan", "sage", "sauge", "dill", "aneth", "chive", "ciboulette", "vanilla", "vanille",
    "cinnamon", "cannelle", "nutmeg", "noix de muscade", "clove", "clou de girofle", "saffron", "safran",
    "turmeric", "curcuma", "cumin", "paprika", "chili", "piment", "honey", "miel", "maple", "érable",
    "sugar", "sucre", "salt", "sel", "pepper", "poivre", "vinegar", "vinaigre", "oil", "huile",
    "soy sauce", "sauce soja", "mustard", "moutarde", "ketchup", "mayonnaise", "salsa", "guacamole"
]

# === HELPERS ===
def validate_prompt(prompt: str, rules: dict) -> bool:
    """Validate prompt against specified rules"""
    if "min_length" in rules and len(prompt) < rules["min_length"]:
        return False
    if "min_words" in rules and len(re.findall(r'\w+', prompt)) < rules["min_words"]:
        return False
    return True

def contains_food_domain(prompt: str) -> bool:
    """Flexible food domain validation"""
    # Check against regex patterns
    if FOOD_REGEX.search(prompt):
        return True
        
    # Check against food name list
    prompt_lower = prompt.lower()
    if any(food in prompt_lower for food in FOOD_NAME_LIST):
        return True
        
    # Heuristic: Single word prompts are likely food items
    if len(prompt.split()) == 1:
        return True
        
    return False

def contains_symptom_domain(prompt: str) -> bool:
    """Robust symptom detection with multi-layered validation"""
    prompt_lower = prompt.lower()
    
    # 1. Check against medical regex patterns
    if MEDICAL_REGEX.search(prompt_lower):
        return True
        
    # 2. Check for symptom words with whole-word matching
    word_count = 0
    for word in SYMPTOM_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", prompt_lower):
            word_count += 1
            if word_count >= 2:  # Require at least 2 distinct symptoms
                return True
    
    # 3. Check for contextual phrases
    if any(phrase in prompt_lower for phrase in CONTEXTUAL_PHRASES):
        return True
        
    # 4. Body part + sensation heuristic
    if (any(phrase in prompt_lower for phrase in ["my", "I've", "j'ai mal au", "ma"]) and
        any(part in prompt_lower for part in BODY_PARTS)):
        return True
        
    return False

# === MAIN ROUTE ===
@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a message to the medical chat bot",
)
async def chat_handler(payload: ChatRequest) -> ChatResponse:
    """
    Receives a chat request, validates input, ensures domain relevance,
    dispatches to the deepseek_client, and validates the response.
    """
    # Input validation
    chat_type = payload.chat_type
    rules = CHAT_VALIDATIONS.get(chat_type)
    
    if not rules:
        raise HTTPException(
            status_code=400,
            detail="Invalid chat type. Must be 'symptom', 'qa', 'food', or 'explore'"
        )

    # Trim and validate prompt
    prompt = payload.prompt.strip()
    if not validate_prompt(prompt, rules):
        return ChatResponse(response=rules["error"])

    # Domain validation
    if chat_type == "explore":
        if not MEDICAL_REGEX.search(prompt):
            return ChatResponse(response="🩺 Please ask a health-related question")
    
    elif chat_type == "symptom":
        if not contains_symptom_domain(prompt):
            return ChatResponse(
                response="🤒 Please describe your symptoms in more detail. "
                "Examples: 'headache and fever', 'dizziness after standing up'"
            )
    
    elif chat_type == "food":
        if not contains_food_domain(prompt):
            return ChatResponse(response="🍏 Please ask about food or nutrition")
    
    # QA type doesn't have strict domain validation

    try:
        # Log and process request
        logger.info(
            f"Chat [{chat_type}] - Session: {payload.session_id}", 
            extra={"prompt": prompt}
        )
        
        reply = await get_response(
            session_id=payload.session_id,
            user_input=prompt,
            chat_type=chat_type,
        )

        # Validate response
        if not reply or not reply.strip():
            logger.warning(f"Empty response - Session: {payload.session_id}")
            return ChatResponse(
                response="⚠️ I couldn't generate a response. Please try again with more details."
            )

        # Ensure response stays in domain
        if chat_type == "explore":
            if not MEDICAL_REGEX.search(reply):
                logger.warning(f"Non-medical response - Session: {payload.session_id}")
                return ChatResponse(
                    response="⚠️ I only provide medical information. Please rephrase your question."
                )
        
        elif chat_type == "symptom":
            # Relaxed validation for symptom responses
            if not contains_symptom_domain(reply):
                # Allow response if it contains advice keywords
                if any(keyword in reply.lower() for keyword in ["advice", "recommend", "suggest", "doctor", "medical attention"]):
                    return ChatResponse(response=reply)
                    
                logger.warning(f"Non-symptom response - Session: {payload.session_id}")
                return ChatResponse(
                    response="⚠️ I only provide health information. Please ask about symptoms."
                )
        
        elif chat_type == "food":
            # Relaxed validation for food responses
            if not contains_food_domain(reply):
                # Allow response if it's short and contains nutritional info
                if any(keyword in reply.lower() for keyword in ["nutrition", "calories", "vitamin", "mineral", "protein"]):
                    return ChatResponse(response=reply)
                    
                logger.warning(f"Non-food response - Session: {payload.session_id}")
                return ChatResponse(
                    response="⚠️ I only provide food and nutrition information. Please ask about food items."
                )

        return ChatResponse(response=reply)

    except Exception as e:
        logger.error(f"Chat error - Session: {payload.session_id}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing your request"
        )