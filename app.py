from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import time
from typing import List, Optional
import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Menu Scanner Backend", version="1.0.0")

# CORS middleware for your iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your app's domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Environment variables (secure API key storage)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BACKEND_API_SECRET = os.getenv("BACKEND_API_SECRET", "your-secret-key")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Data models with detailed field descriptions
class MenuAnalysisRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image data")
    target_language: str = Field(default="english", description="Target language for translation")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    app_version: Optional[str] = Field(None, description="App version for analytics")

class MenuItem(BaseModel):
    originalName: str = Field(
        ..., 
        description="Exact name from menu preserving original language and characters"
    )
    translatedName: str = Field(
        ..., 
        description="Translation of the dish name in the target language"
    )
    price: str = Field(
        ..., 
        description="Price as shown on menu with currency symbol (e.g., '$12.99', '¥50', '€15.50')"
    )
    description: str = Field(
        ..., 
        description="Original description from menu, or '[AI Generated] factual description' if not available"
    )
    translatedDescription: str = Field(
        ..., 
        description="Translation of description in target language, prefix with '[AI生成] ' if AI generated"
    )
    estimatedIngredients: List[str] = Field(
        ..., 
        description="List of main ingredients, translated to target language"
    )
    estimatedAllergens: List[str] = Field(
        ..., 
        description="Common allergens like nuts, dairy, gluten, shellfish, etc."
    )
    cookingMethod: str = Field(
        ..., 
        description="Primary cooking method: grilled, fried, steamed, baked, raw, etc."
    )
    dietaryLabels: List[str] = Field(
        ..., 
        description="Dietary tags: vegetarian, vegan, gluten-free, spicy, halal, etc."
    )
    regionalCuisine: str = Field(
        ..., 
        description="Cuisine type: Chinese, Italian, Mexican, Japanese, etc."
    )
    category: str = Field(
        ..., 
        description="Menu category: appetizer, main, dessert, drink, side, etc."
    )
    isEstimated: bool = Field(
        ..., 
        description="True if any information (except name and price) was estimated by AI"
    )

class MenuResponse(BaseModel):
    items: List[MenuItem] = Field(
        ..., 
        description="Complete list of all menu items found in the image"
    )

class MenuAnalysisResponse(BaseModel):
    items: List[MenuItem]
    processing_time: float
    tokens_used: Optional[int] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str

# Direct mapping from iOS TranslationLanguage rawValue to target language
def parse_ios_language(ios_language_raw: str) -> str:
    """Parse iOS TranslationLanguage rawValue and return appropriate target language"""
    
    # Create mapping from iOS rawValue to target language
    language_mapping = {
        # East Asian Languages
        "Chinese (简体中文)": "Simplified Chinese (简体中文)",
        "Traditional Chinese (繁體中文)": "Traditional Chinese (繁體中文)", 
        "Japanese (日本語)": "Japanese (日本語)",
        "Korean (한국어)": "Korean (한국어)",
        
        # Southeast Asian Languages
        "Vietnamese (Tiếng Việt)": "Vietnamese (Tiếng Việt) with proper diacritical marks",
        "Thai (ภาษาไทย)": "Thai (ภาษาไทย) with proper Thai script",
        "Indonesian (Bahasa Indonesia)": "Indonesian (Bahasa Indonesia)",
        "Malay (Bahasa Melayu)": "Malay (Bahasa Melayu)",
        
        # South Asian Languages
        "Hindi (हिन्दी)": "Hindi (हिन्दी) - provide both Devanagari script and transliteration",
        "Bengali (বাংলা)": "Bengali (বাংলা) - provide both Bengali script and transliteration", 
        "Urdu (اردو)": "Urdu (اردو) - provide both Urdu script and transliteration",
        
        # European Languages
        "English": "English",
        "Spanish (Español)": "Spanish (Español) with proper accents",
        "French (Français)": "French (Français) with proper accents",
        "German (Deutsch)": "German (Deutsch) with proper umlauts",
        "Italian (Italiano)": "Italian (Italiano) with proper accents",
        "Portuguese (Português)": "Portuguese (Português) with proper accents",
        "Russian (Русский)": "Russian (Русский) - provide both Cyrillic and transliteration",
        "Dutch (Nederlands)": "Dutch (Nederlands) with proper diacritics",
        
        # Middle Eastern Languages
        "Arabic (العربية)": "Arabic (العربية) - provide both Arabic script and transliteration",
        "Turkish (Türkçe)": "Turkish (Türkçe) with proper Turkish characters"
    }
    
    # Try exact match first
    if ios_language_raw in language_mapping:
        return language_mapping[ios_language_raw]
    
    # Fallback: try to match by extracting language name
    lower_lang = ios_language_raw.lower()
    
    if "chinese" in lower_lang and "traditional" in lower_lang:
        return "Traditional Chinese (繁體中文)"
    elif "chinese" in lower_lang:
        return "Simplified Chinese (简体中文)"
    elif "japanese" in lower_lang:
        return "Japanese (日本語)"
    elif "korean" in lower_lang:
        return "Korean (한국어)"
    elif "vietnamese" in lower_lang:
        return "Vietnamese (Tiếng Việt) with proper diacritical marks"
    elif "thai" in lower_lang:
        return "Thai (ภาษาไทย) with proper Thai script"
    elif "indonesian" in lower_lang:
        return "Indonesian (Bahasa Indonesia)"
    elif "malay" in lower_lang:
        return "Malay (Bahasa Melayu)"
    elif "hindi" in lower_lang:
        return "Hindi (हिन्दी) - provide both Devanagari script and transliteration"
    elif "bengali" in lower_lang:
        return "Bengali (বাংলা) - provide both Bengali script and transliteration"
    elif "urdu" in lower_lang:
        return "Urdu (اردو) - provide both Urdu script and transliteration"
    elif "spanish" in lower_lang:
        return "Spanish (Español) with proper accents"
    elif "french" in lower_lang:
        return "French (Français) with proper accents"
    elif "german" in lower_lang:
        return "German (Deutsch) with proper umlauts"
    elif "italian" in lower_lang:
        return "Italian (Italiano) with proper accents"
    elif "portuguese" in lower_lang:
        return "Portuguese (Português) with proper accents"
    elif "russian" in lower_lang:
        return "Russian (Русский) - provide both Cyrillic and transliteration"
    elif "dutch" in lower_lang:
        return "Dutch (Nederlands) with proper diacritics"
    elif "arabic" in lower_lang:
        return "Arabic (العربية) - provide both Arabic script and transliteration"
    elif "turkish" in lower_lang:
        return "Turkish (Türkçe) with proper Turkish characters"
    else:
        # Default to English
        return "English"

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Simple token verification - enhance this for production"""
    if credentials.credentials != BACKEND_API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0"
    )

# Main analysis endpoint using OpenAI SDK with structured output
@app.post("/analyze-menu", response_model=MenuAnalysisResponse)
async def analyze_menu(
    request: MenuAnalysisRequest,
    token: str = Depends(verify_token)
):
    """Analyze menu image directly using GPT-4o Vision with structured output"""
    start_time = time.time()
    
    try:
        logger.info(f"Starting menu analysis for user: {request.user_id}")
        logger.info(f"Target language (raw): {request.target_language}")
        
        # Parse iOS language to target language
        target_lang = parse_ios_language(request.target_language)
        
        logger.info(f"Using target language: {target_lang}")
        
        # Enhanced system prompt with detailed instructions
        system_prompt = f"""
        You are a professional menu analyzer with expertise in multiple cuisines and languages. 
        
        TASK: Analyze the provided menu image and extract ALL visible menu items with complete, accurate details.
        
        INSTRUCTIONS:
        1. COMPLETENESS: Extract every dish, drink, dessert, side, and extra item visible on the menu
        2. ACCURACY: Keep original names exactly as shown, preserving all characters and formatting
        3. DESCRIPTIONS: For items without descriptions, create factual descriptions prefixed with '[AI Generated]'
        4. TRANSLATIONS: Provide accurate translations in {target_lang}
        5. THOROUGHNESS: Include comprehensive details for each item
        6. FORMATTING: Follow the exact field requirements specified in the schema
        
        LANGUAGE REQUIREMENTS:
        - Target language: {target_lang}
        - Preserve original script/characters in originalName
        - Use proper script and formatting for translations
        - For languages with both script and transliteration, provide the script version
        
        FIELD REQUIREMENTS:
        - originalName: Exact text from menu (no modifications)
        - price: Include currency symbol as shown (e.g., '$12.99', '¥50', '€15.50')
        - description: Original text or '[AI Generated] factual description'
        - translatedDescription: Prefix AI descriptions with '[AI生成] ' in target language
        - estimatedIngredients: Main ingredients translated to target language
        - estimatedAllergens: Common allergens (nuts, dairy, gluten, shellfish, etc.)
        - cookingMethod: Primary method (grilled, fried, steamed, baked, raw, etc.)
        - dietaryLabels: Tags like vegetarian, vegan, gluten-free, spicy, halal
        - regionalCuisine: Cuisine type (Chinese, Italian, Mexican, Japanese, etc.)
        - category: Menu section (appetizer, main, dessert, drink, side)
        - isEstimated: true if any info beyond name/price was estimated
        
        QUALITY STANDARDS:
        - Be thorough - don't miss any menu items
        - Ensure all translations are culturally appropriate
        - Use proper formatting for the target language
        - Include all visible price information
        """
        
        user_prompt = f"""
        Analyze this menu image comprehensively and extract all menu items.
        
        Target language for translations: {target_lang}
        
        Requirements:
        - Find every single menu item visible in the image
        - Preserve original names exactly as written
        - Provide accurate translations in {target_lang}
        - Include complete details for each item
        - Use proper formatting and scripts for the target language
        """
        
        # Use OpenAI SDK with structured output
        logger.info("Starting GPT-4o Vision analysis with structured output")
        
        response = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{request.image}"}
                        }
                    ]
                }
            ],
            response_format=MenuResponse,
            max_tokens=10000,
            temperature=0.1
        )
        
        # Extract structured data
        menu_data = response.choices[0].message.parsed
        processing_time = time.time() - start_time
        tokens_used = response.usage.total_tokens if response.usage else None
        
        logger.info(f"Analysis completed successfully in {processing_time:.2f}s")
        logger.info(f"Found {len(menu_data.items)} menu items")
        logger.info(f"Tokens used: {tokens_used}")
        
        # Log sample items for debugging
        if menu_data.items:
            logger.info(f"Sample item: {menu_data.items[0].originalName} -> {menu_data.items[0].translatedName}")
        
        return MenuAnalysisResponse(
            items=menu_data.items,
            processing_time=processing_time,
            tokens_used=tokens_used
        )
        
    except Exception as e:
        logger.error(f"Error during menu analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Menu analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 