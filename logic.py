from ctransformers import AutoModelForCausalLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob

MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
KNOWLEDGE_DIR = "knowledge"


# ==========================================
# PDF LOADING + TEXT CHUNKING
# ==========================================
def load_pdfs_from_folder(folder_path):
    """Load all PDFs from a folder, extract text, and chunk them."""
    chunks = []

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created '{folder_path}/' folder. Drop your interior design PDFs there and restart.")
        return chunks

    pdf_files = glob.glob(os.path.join(folder_path, '*.pdf'))

    if not pdf_files:
        print(f"No PDFs found in '{folder_path}/'. Using built-in rules only.")
        return chunks

    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("PyPDF2 not installed. Run: pip install PyPDF2")
        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(pdf_path)
            pdf_name = os.path.basename(pdf_path)
            full_text = ""

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

            if full_text.strip():
                pdf_chunks = splitter.split_text(full_text)
                for chunk in pdf_chunks:
                    chunks.append(Document(
                        page_content=chunk,
                        metadata={"source": pdf_name}
                    ))
                print(f"  Loaded '{pdf_name}': {len(pdf_chunks)} chunks from {len(reader.pages)} pages")
            else:
                print(f"  '{pdf_name}': No extractable text (might be scanned/image-based)")

        except Exception as e:
            print(f"  Error reading '{os.path.basename(pdf_path)}': {e}")

    return chunks


# ==========================================
# RAG KNOWLEDGE BASE SETUP
# ==========================================
def load_knowledge_base():
    """Build FAISS vector database from PDFs + built-in architectural rules."""

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    builtin_rules = [
        "Beds should be placed against walls or in corners, never in the center of the room.",
        "Desks require at least 2 feet of clearance behind them to allow pulling out the chair.",
        "Do not block pathways. Keep the center of the room clear of heavy furniture.",
        "If furniture pieces are touching or blocking each other, suggest separating them by at least 1 foot.",
        "Maximize natural light by keeping large items away from windows and doors.",
        "If a room is highly cluttered, suggest pushing all tables to the edges of the room.",
        "Bookshelves and tall furniture should be placed against walls to avoid tipping hazards.",
        "Maintain at least 3 feet of clearance around the bed for easy movement.",
        "Position the desk near a window for natural light, but avoid glare on screens.",
        "In small rooms, use vertical storage and wall-mounted shelves to save floor space."
    ]
    builtin_docs = [Document(page_content=rule, metadata={"source": "built-in"}) for rule in builtin_rules]

    print("Scanning for PDF knowledge files...")
    pdf_docs = load_pdfs_from_folder(KNOWLEDGE_DIR)

    all_docs = builtin_docs + pdf_docs
    pdf_count = len(glob.glob(os.path.join(KNOWLEDGE_DIR, '*.pdf'))) if os.path.exists(KNOWLEDGE_DIR) else 0

    print(f"Knowledge base: {len(builtin_rules)} built-in rules + {len(pdf_docs)} PDF chunks from {pdf_count} file(s)")

    vector_db = FAISS.from_documents(all_docs, embeddings)
    return vector_db


# Initialize Vector DB when the script loads
print("")
print("=" * 50)
print("Loading RAG Knowledge Base...")
print("=" * 50)
db = load_knowledge_base()
print("=" * 50)
print("Knowledge base ready!")
print("")


# ==========================================
# SPACE EFFICIENCY SCORING
# ==========================================
def calculate_space_score(metrics):
    """Calculate a 0-100 Space Efficiency Score from spatial metrics."""
    score = 100

    # Coverage penalty (softer curve)
    coverage = metrics.get("total_coverage_pct", 0)
    if coverage > 70:
        score -= (coverage - 70) * 1.0   # Heavy penalty only above 70%
    elif coverage > 50:
        score -= (coverage - 50) * 0.4   # Mild penalty 50-70%

    # Collision penalty: 8 pts each, capped at 30 total
    # (Multi-image can inflate collision counts, so we cap it)
    collisions = metrics.get("collision_count", 0)
    collision_penalty = min(collisions * 8, 30)
    score -= collision_penalty

    # Center zone crowding
    zone_densities = metrics.get("zone_densities", {})
    center_density = zone_densities.get("Center-Middle", 0)
    if center_density > 30:
        score -= (center_density - 30) * 0.3

    # Penalize extremely overcrowded zones
    for zone, density in zone_densities.items():
        if density > 80:
            score -= 3

    # Reward open walkway space
    open_zones = sum(1 for d in zone_densities.values() if d < 10)
    score += open_zones * 3

    # Bonus for good coverage range (30-50% is ideal)
    if 30 <= coverage <= 50:
        score += 5

    return max(0, min(100, round(score)))


# ==========================================
# GENERATIVE AI CORE
# ==========================================
def get_design_suggestions(spatial_sentences, item_list, metrics=None, preferences=None):
    """
    Generate design suggestions using RAG + structured spatial data.

    Args:
        spatial_sentences: List of position/relationship descriptions
        item_list: List of detected furniture names
        metrics: Dict with coverage, densities, collisions
        preferences: Dict with room_type, style, priority
    """
    if not spatial_sentences:
        return "No layout data available.", None

    if not os.path.exists(MODEL_FILE):
        return "Error: Model file not found: " + MODEL_FILE, None

    try:
        llm = AutoModelForCausalLM.from_pretrained(
            os.path.abspath(MODEL_FILE),
            model_type="llama",
            gpu_layers=0,
            context_length=2048
        )
    except Exception as e:
        return "Error loading local model: " + str(e), None

    # Compact spatial sentences to avoid token overflow
    # Keep max 8 most important observations (collisions first, then positions)
    collision_sentences = [s for s in spatial_sentences if "COLLISION" in s or "blocking" in s.lower()]
    position_sentences = [s for s in spatial_sentences if s not in collision_sentences]
    compacted = collision_sentences + position_sentences[:max(2, 8 - len(collision_sentences))]
    room_description = " ".join(compacted)
    # Hard cap at 600 chars to leave room for other prompt sections
    if len(room_description) > 600:
        room_description = room_description[:597] + "..."
    items_str = ", ".join(set(item_list))

    # RAG Retrieval - get 2 most relevant knowledge chunks (compact)
    retrieved_docs = db.similarity_search(room_description, k=2)
    expert_rules = "\n".join(["- " + doc.page_content[:200] for doc in retrieved_docs])

    sources = set(doc.metadata.get("source", "unknown") for doc in retrieved_docs)
    print("RAG sources used: " + ", ".join(sources))

    # Build structured metrics string
    metrics_str = ""
    space_score = None
    if metrics:
        space_score = calculate_space_score(metrics)
        metrics_str = "\nQuantitative Analysis:\n"
        metrics_str += "- Total furniture coverage: {:.1f}% of room area\n".format(metrics.get("total_coverage_pct", 0))
        metrics_str += "- Furniture pieces: {}\n".format(metrics.get("furniture_count", 0))
        metrics_str += "- Collisions: {}\n".format(metrics.get("collision_count", 0))
        metrics_str += "- Space Efficiency Score: {}/100\n".format(space_score)

        zone_densities = metrics.get("zone_densities", {})
        if zone_densities:
            crowded = ["{} ({:.0f}%)".format(z, d) for z, d in zone_densities.items() if d > 30]
            empty = [z for z, d in zone_densities.items() if d < 5]
            if crowded:
                metrics_str += "- Overcrowded zones: {}\n".format(", ".join(crowded))
            if empty:
                metrics_str += "- Empty zones: {}\n".format(", ".join(empty))

        collision_details = metrics.get("collision_details", [])
        if collision_details:
            metrics_str += "- Blocking pairs: {}\n".format("; ".join(collision_details))

    # User preferences string
    pref_str = ""
    if preferences:
        pref_str = "\nUser Requirements:\n"
        pref_str += "- Room type: {}\n".format(preferences.get("room_type", "General"))
        pref_str += "- Desired style: {}\n".format(preferences.get("style", "Functional"))
        pref_str += "- Top priority: {}\n".format(preferences.get("priority", "General Flow"))

    # Build prompt with TinyLlama chat format tags
    SYS = chr(60) + "|system|" + chr(62)
    USR = chr(60) + "|user|" + chr(62)
    AST = chr(60) + "|assistant|" + chr(62)

    prompt = SYS + "\n"
    prompt += "You are an expert Interior Architect analyzing a real room scan.\n"
    prompt += "Use the quantitative data and rules below for precise, actionable advice.\n\n"
    prompt += "Expert Knowledge:\n" + expert_rules + "\n"
    if metrics_str:
        prompt += metrics_str
    if pref_str:
        prompt += pref_str
    prompt += "\nSpatial Observations: " + room_description + "\n"
    prompt += "Furniture Detected: " + items_str + "\n\n"
    prompt += USR + "\n"
    prompt += "Based on all data above, give 3 specific, actionable layout improvements. Reference actual numbers and zones. Do NOT suggest buying new furniture.\n"
    prompt += AST

    print("AI is generating response locally with RAG...")
    response = llm(prompt, max_new_tokens=256, temperature=0.6, repetition_penalty=1.1)

    return response, space_score
