from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re
import spacy
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Embedding model for semantic similarity
sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FastAPI app (equivalent to your Flask app)
app = FastAPI(
    title="Resume Enhancement API",
    description="API for resume enhancement and keyword extraction",
    version="1.0.0"
)

# Add CORS middleware (equivalent to CORS(app) in Flask)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates (equivalent to Flask's render_template)
templates = Jinja2Templates(directory="templates")

# Mount static files (equivalent to Flask's static folder)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load necessary NLP models (exactly the same as your Flask version)
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully")
except OSError:
    print("📥 Downloading spaCy model...")
    subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded after download")

try:
    # sentence_transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_transformer_model = SentenceTransformer('all-mpnet-base-v2')
    print("✅ Sentence transformer model loaded successfully")
except Exception as e:
    print(f"❌ Error loading sentence transformer: {e}")
    sentence_transformer_model = None

# Pydantic models for JSON API endpoints
class ResumeEnhancementRequest(BaseModel):
    resume_text: str
    job_description: str

class KeywordExtractionRequest(BaseModel):
    job_description: str

class ResumeEnhancementResponse(BaseModel):
    enhanced_resume: str
    ats_score: float
    top_missing_keywords: List[str]

class KeywordExtractionResponse(BaseModel):
    top_keywords: List[str]

# Helper functions (exactly the same as your Flask version)
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text).lower()
    return text

def extract_keywords(resume_text, job_description):
    preprocessed_resume = preprocess_text(resume_text)
    preprocessed_job_description = preprocess_text(job_description)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_resume, preprocessed_job_description])

    embeddings = sentence_transformer_model.encode([preprocessed_resume, preprocessed_job_description])

    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return tfidf_vectorizer.get_feature_names_out(), similarity

# Improved enhance_resume function with semantic matching
# def enhance_resume(resume_text, job_description):
#     preprocessed_resume = preprocess_text(resume_text)
#     preprocessed_job_description = preprocess_text(job_description)

#     # --- Extract keywords with TF-IDF ---
#     tfidf_vectorizer_resume = TfidfVectorizer(stop_words='english')
#     resume_keywords = tfidf_vectorizer_resume.fit([preprocessed_resume]).get_feature_names_out()

#     tfidf_vectorizer_job = TfidfVectorizer(stop_words='english')
#     job_keywords = tfidf_vectorizer_job.fit([preprocessed_job_description]).get_feature_names_out()

#     # --- Embed keywords for semantic matching ---
#     job_embeddings = sentence_transformer_model.encode(job_keywords)
#     resume_embeddings = sentence_transformer_model.encode(resume_keywords)

#     missing_keywords = []
#     for idx, job_kw in enumerate(job_keywords):
#         similarities = cosine_similarity([job_embeddings[idx]], resume_embeddings)[0]
#         max_sim = max(similarities) if len(similarities) > 0 else 0

#         # Only consider keyword missing if no semantic match found
#         if max_sim < 0.75:
#             missing_keywords.append(job_kw)

#     # Sort missing keywords by TF-IDF importance (descending)
#     tfidf_matrix_job = tfidf_vectorizer_job.fit_transform([preprocessed_job_description])
#     job_keyword_scores = dict(zip(job_keywords, tfidf_matrix_job.sum(axis=0).A1))
#     missing_keyword_scores = {kw: job_keyword_scores.get(kw, 0) for kw in missing_keywords}
#     sorted_missing_keywords = sorted(missing_keyword_scores.items(), key=lambda item: item[1], reverse=True)

#     top_10_missing_keywords = [keyword for keyword, score in sorted_missing_keywords[:10]]

#      # --- Simple Enhancement: Append missing keywords to summary section ---
#     enhanced_resume = resume_text + "\n\n# Added Keywords:\n" + ", ".join(top_10_missing_keywords)

#     embeddings = sentence_transformer_model.encode([resume_text, preprocessed_job_description])
#     similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
#     ats_score = similarity * 100

#     # --- Document-level similarity for ATS score ---
#     embeddings_enhanced = sentence_transformer_model.encode([enhanced_resume, preprocessed_job_description])
#     similarity_enhanced = cosine_similarity([embeddings_enhanced[0]], [embeddings_enhanced[1]])[0][0]
#     ats_score_enhanced = similarity_enhanced * 100

   
#     return enhanced_resume, ats_score, ats_score_enhanced, top_10_missing_keywords

def enhance_resume(resume_text: str, job_description: str):
    preprocessed_resume = preprocess_text(resume_text)
    preprocessed_job_description = preprocess_text(job_description)

    # --- Extract candidate skill phrases from JD using spaCy ---
    doc_jd = nlp(job_description)
    candidate_phrases = set()

    # Noun chunks (multi-word phrases)
    for chunk in doc_jd.noun_chunks:
        phrase = chunk.text.lower().strip()
        if len(phrase) > 2:
            candidate_phrases.add(phrase)

    # Proper nouns and nouns (technologies, tools, etc.)
    for token in doc_jd:
        if token.pos_ in {"PROPN", "NOUN"} and len(token.text) > 2:
            candidate_phrases.add(token.text.lower().strip())

    # --- TF-IDF with n-grams for job description ---
    tfidf_vectorizer_job = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
    tfidf_vectorizer_job.fit([preprocessed_job_description])

    job_keywords = [
        kw for kw in tfidf_vectorizer_job.get_feature_names_out()
        if kw in candidate_phrases  # filter only meaningful phrases
    ]

    # --- Resume keywords (same filtering logic) ---
    tfidf_vectorizer_resume = TfidfVectorizer(stop_words="english", ngram_range=(1, 3))
    tfidf_vectorizer_resume.fit([preprocessed_resume])

    resume_keywords = [
        kw for kw in tfidf_vectorizer_resume.get_feature_names_out()
    ]

    # --- Embed keywords for semantic matching (MPNet) ---
    job_embeddings = sentence_transformer_model.encode(job_keywords)
    resume_embeddings = sentence_transformer_model.encode(resume_keywords)

    missing_keywords = []
    for idx, job_kw in enumerate(job_keywords):
        similarities = cosine_similarity([job_embeddings[idx]], resume_embeddings)[0]
        max_sim = max(similarities) if len(similarities) > 0 else 0

        # Only consider keyword missing if no semantic match found
        if max_sim < 0.75:
            missing_keywords.append(job_kw)

    # --- Rank missing keywords by TF-IDF importance ---
    tfidf_matrix_job = tfidf_vectorizer_job.transform([preprocessed_job_description])
    job_keyword_scores = dict(zip(
        tfidf_vectorizer_job.get_feature_names_out(),
        tfidf_matrix_job.sum(axis=0).A1
    ))

    missing_keyword_scores = {kw: job_keyword_scores.get(kw, 0) for kw in missing_keywords}
    sorted_missing_keywords = sorted(missing_keyword_scores.items(), key=lambda item: item[1], reverse=True)

    top_10_missing_keywords = [keyword for keyword, _ in sorted_missing_keywords[:10]]

    # --- Simple Enhancement: Append missing keywords to resume ---
    enhanced_resume = resume_text + "\n\n# Added Keywords:\n" + ", ".join(top_10_missing_keywords)

    # --- Document-level similarity for ATS score ---
    embeddings_original = sentence_transformer_model.encode([resume_text, preprocessed_job_description])
    similarity_original = cosine_similarity([embeddings_original[0]], [embeddings_original[1]])[0][0]
    ats_score = similarity_original * 100

    embeddings_enhanced = sentence_transformer_model.encode([enhanced_resume, preprocessed_job_description])
    similarity_enhanced = cosine_similarity([embeddings_enhanced[0]], [embeddings_enhanced[1]])[0][0]
    ats_score_enhanced = similarity_enhanced * 100

    return enhanced_resume, ats_score, ats_score_enhanced, top_10_missing_keywords


def extract_job_keywords(job_description):
    if not job_description:
        return []

    preprocessed_job_description = preprocess_text(job_description)

    # --- Extract noun phrases with spaCy ---
    doc = nlp(job_description)
    noun_phrases = list(set([chunk.text.lower() for chunk in doc.noun_chunks]))

    # --- TF-IDF with n-grams (unigrams + bigrams + trigrams) ---
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocessed_job_description])

    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    keyword_scores = dict(zip(feature_names, tfidf_scores))

    # Keep only phrases that are either in noun_phrases OR have high TF-IDF scores
    combined_keywords = set(noun_phrases).union(set(feature_names))

    # Rank combined keywords by TF-IDF score
    sorted_keywords = sorted(
        [(kw, keyword_scores.get(kw, 0)) for kw in combined_keywords],
        key=lambda item: item[1],
        reverse=True
    )

    # Top 10 keywords/phrases
    top_keywords = [keyword for keyword, score in sorted_keywords[:10]]

    return top_keywords

# HTML Routes (converted from your Flask routes)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page - equivalent to your Flask home() function"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/enhance_resume", response_class=HTMLResponse)
async def enhance_resume_page_get(request: Request):
    """GET route for resume enhancement page"""
    return templates.TemplateResponse("enhance_resume.html", {"request": request})

@app.post("/enhance_resume", response_class=HTMLResponse)
async def enhance_resume_page_post(
    request: Request,
    resume_text: str = Form(...),
    job_description: str = Form(...)
):
    """POST route for resume enhancement page - equivalent to your Flask enhance_resume_page()"""
    if not resume_text or not job_description:
        return templates.TemplateResponse(
            "enhance_resume.html", 
            {
                "request": request,
                "error": "Please provide both resume and job description."
            }
        )

    try:
        enhanced_resume, ats_score, ats_score_enhanced, top_missing_keywords = enhance_resume(resume_text, job_description)

        return templates.TemplateResponse(
            "enhance_resume.html",
            {
                "request": request,
                "enhanced_resume": enhanced_resume,
                "ats_score": ats_score,
                "ats_score_enhanced": ats_score_enhanced,
                "top_missing_keywords": top_missing_keywords
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "enhance_resume.html",
            {
                "request": request,
                "error": f"Error processing resume: {str(e)}"
            }
        )

@app.get("/extract_keywords", response_class=HTMLResponse)
async def extract_keywords_page_get(request: Request):
    """GET route for keyword extraction page"""
    return templates.TemplateResponse("extract_keywords.html", {"request": request})

@app.post("/extract_keywords", response_class=HTMLResponse)
async def extract_keywords_page_post(
    request: Request,
    job_description: str = Form(...)
):
    """POST route for keyword extraction page - equivalent to your Flask extract_keywords_page()"""
    if not job_description:
        return templates.TemplateResponse(
            "extract_keywords.html",
            {
                "request": request,
                "error": "Please provide a job description."
            }
        )

    try:
        top_keywords = extract_job_keywords(job_description)

        return templates.TemplateResponse(
            "extract_keywords.html",
            {
                "request": request,
                "top_keywords": top_keywords
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "extract_keywords.html",
            {
                "request": request,
                "error": f"Error extracting keywords: {str(e)}"
            }
        )

# Additional JSON API Routes (bonus functionality)
@app.post("/api/enhance-resume", response_model=ResumeEnhancementResponse)
async def api_enhance_resume(request: ResumeEnhancementRequest):
    """JSON API endpoint for resume enhancement"""
    try:
        if not request.resume_text or not request.job_description:
            raise HTTPException(status_code=400, detail="Both resume_text and job_description are required")
        
        enhanced_resume, ats_score, ats_score_enhanced, top_missing_keywords = enhance_resume(
            request.resume_text, 
            request.job_description
        )
        
        return ResumeEnhancementResponse(
            enhanced_resume=enhanced_resume,
            ats_score=float(ats_score),
            ats_score_enhanced=float(ats_score_enhanced),
            top_missing_keywords=top_missing_keywords
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@app.post("/api/extract-keywords", response_model=KeywordExtractionResponse)
async def api_extract_keywords(request: KeywordExtractionRequest):
    """JSON API endpoint for keyword extraction"""
    try:
        if not request.job_description:
            raise HTTPException(status_code=400, detail="job_description is required")
        
        top_keywords = extract_job_keywords(request.job_description)
        
        return KeywordExtractionResponse(top_keywords=top_keywords)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting keywords: {str(e)}")

# Health check endpoint (bonus)
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "spacy": nlp is not None,
            "sentence_transformer": sentence_transformer_model is not None
        }
    }

# Run the application (equivalent to your Flask app.run())
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)