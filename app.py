from datetime import datetime
from uuid import uuid4

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from src.helper import download_hugging_face_embeddings 
from langchain_pinecone import PineconeVectorStore 
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 
from dotenv import load_dotenv 
from pymongo import ASCENDING, MongoClient
from werkzeug.security import check_password_hash, generate_password_hash
from src.prompt import * 
from src.seed_data import DOCTOR_SEED, MEDICINE_SEED
import os

app = Flask(__name__) 
load_dotenv() 
app.secret_key = os.getenv("FLASK_SECRET_KEY", "carebridge-dev-secret")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "carebridge_hospital")
mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = mongo_client[MONGO_DB_NAME]

def get_session_id():
    if "session_id" not in session:
        session["session_id"] = uuid4().hex
    return session["session_id"]

def public_docs(cursor):
    docs = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        docs.append(doc)
    return docs

def seed_database():
    mongo_client.admin.command("ping")
    db.users.create_index([("email", ASCENDING)], unique=True)
    db.doctors.create_index([("id", ASCENDING)], unique=True)
    db.doctors.create_index([("specialty", ASCENDING)])
    db.medicines.create_index([("id", ASCENDING)], unique=True)
    db.medicines.create_index([("category", ASCENDING)])
    db.appointments.create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])
    db.cart_items.create_index([("session_id", ASCENDING), ("medicine_id", ASCENDING)], unique=True)
    db.prescriptions.create_index([("user_id", ASCENDING)])
    db.medical_history.create_index([("user_id", ASCENDING)])

    if db.doctors.count_documents({}) == 0:
        db.doctors.insert_many(DOCTOR_SEED)
    if db.medicines.count_documents({}) == 0:
        db.medicines.insert_many(MEDICINE_SEED)

def ensure_patient_records(user_id):
    if db.prescriptions.count_documents({"user_id": user_id}) == 0:
        db.prescriptions.insert_many([
            {"user_id": user_id, "medicine": "Vitamin D3 Tablets", "dose": "1 tablet daily", "doctor": "Dr. Aditya Menon", "created_at": datetime.utcnow()},
            {"user_id": user_id, "medicine": "Paracetamol 500mg", "dose": "As needed after meals", "doctor": "Dr. Aanya Rao", "created_at": datetime.utcnow()},
        ])
    if db.medical_history.count_documents({"user_id": user_id}) == 0:
        db.medical_history.insert_many([
            {"user_id": user_id, "date": "2026-05-12", "title": "Annual wellness check", "notes": "Vitals normal, routine labs requested."},
            {"user_id": user_id, "date": "2026-03-28", "title": "Orthopedic consultation", "notes": "Mild knee strain, physiotherapy advised."},
        ])

seed_database()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY 
os.environ["GROQ_API_KEY"] = GROQ_API_KEY  
 
embeddings = download_hugging_face_embeddings() 
index_name = "medicalbot" 

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
) 

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3}) 

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.4,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)  

@app.route("/")
def index():
    doctors = public_docs(db.doctors.find().sort("rating", -1).limit(3))
    return render_template("index.html", doctors=doctors)

@app.route("/chat")
def chat_page():
    return render_template("chat.html") 

@app.route("/pharmacy")
def pharmacy():
    medicines = public_docs(db.medicines.find().sort("name", 1))
    categories = sorted(db.medicines.distinct("category"))
    return render_template("pharmacy.html", medicines=medicines, categories=categories)

@app.route("/appointment", methods=["GET", "POST"])
def appointment():
    if request.method == "POST":
        doctor_id = int(request.form.get("doctor_id", 0))
        doctor = db.doctors.find_one({"id": doctor_id}) or db.doctors.find_one()
        appointment_data = {
            "user_id": session.get("user", {}).get("id"),
            "user_email": session.get("user", {}).get("email"),
            "patient_name": request.form.get("patient_name", "").strip(),
            "age": request.form.get("age", "").strip(),
            "specialty": request.form.get("specialty", "").strip(),
            "doctor": doctor["name"],
            "doctor_id": doctor["id"],
            "date": request.form.get("date", "").strip(),
            "time": request.form.get("time", "").strip(),
            "symptoms": request.form.get("symptoms", "").strip(),
            "status": "Confirmed",
            "created_at": datetime.utcnow(),
        }
        db.appointments.insert_one(appointment_data)
        flash(f"Appointment confirmed with {doctor['name']} on {appointment_data['date']} at {appointment_data['time']}.", "success")
        return redirect(url_for("appointment"))
    doctors = public_docs(db.doctors.find().sort("specialty", 1))
    specialties = sorted(db.doctors.distinct("specialty"))
    return render_template("appointment.html", doctors=doctors, specialties=specialties)

@app.route("/doctors")
def doctors():
    doctors = public_docs(db.doctors.find().sort("specialty", 1))
    departments = sorted(db.doctors.distinct("specialty"))
    return render_template("doctors.html", doctors=doctors, departments=departments)

@app.route("/dashboard")
def dashboard():
    if not session.get("user"):
        flash("Please log in to view your patient dashboard.", "info")
        return redirect(url_for("auth"))
    user_id = session["user"]["id"]
    appointments = public_docs(db.appointments.find({"user_id": user_id}).sort("created_at", -1))
    prescriptions = public_docs(db.prescriptions.find({"user_id": user_id}).sort("created_at", -1))
    history = public_docs(db.medical_history.find({"user_id": user_id}).sort("date", -1))
    return render_template("dashboard.html", appointments=appointments, prescriptions=prescriptions, history=history)

@app.route("/auth")
def auth():
    return render_template("auth.html")

@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    user = db.users.find_one({"email": email})
    if not user or not check_password_hash(user["password_hash"], password):
        flash("Invalid email or password.", "danger")
        return redirect(url_for("auth"))
    session["user"] = {"id": str(user["_id"]), "name": user["name"], "email": user["email"]}
    db.cart_items.update_many({"session_id": get_session_id()}, {"$set": {"user_id": str(user["_id"])}})
    flash("Welcome back. Your dashboard is ready.", "success")
    return redirect(url_for("dashboard"))

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name", "Patient").strip() or "Patient"
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    if db.users.find_one({"email": email}):
        flash("An account already exists with this email. Please log in.", "warning")
        return redirect(url_for("auth"))
    result = db.users.insert_one({
        "name": name,
        "email": email,
        "password_hash": generate_password_hash(password),
        "created_at": datetime.utcnow(),
    })
    user_id = str(result.inserted_id)
    ensure_patient_records(user_id)
    session["user"] = {"id": user_id, "name": name, "email": email}
    db.cart_items.update_many({"session_id": get_session_id()}, {"$set": {"user_id": user_id}})
    flash("Registration complete. You are now signed in.", "success")
    return redirect(url_for("dashboard"))

@app.route("/cart/add", methods=["POST"])
def add_to_cart():
    data = request.get_json(silent=True) or request.form
    medicine_id = int(data.get("medicine_id", 0))
    medicine = db.medicines.find_one({"id": medicine_id})
    if not medicine:
        return jsonify({"error": "Medicine not found"}), 404
    db.cart_items.update_one(
        {"session_id": get_session_id(), "medicine_id": medicine_id},
        {
            "$inc": {"quantity": 1},
            "$set": {
                "user_id": session.get("user", {}).get("id"),
                "updated_at": datetime.utcnow(),
            },
            "$setOnInsert": {"created_at": datetime.utcnow()},
        },
        upsert=True,
    )
    return jsonify(build_cart_response())

@app.route("/cart")
def cart():
    return jsonify(build_cart_response())

def build_cart_response():
    items = []
    total = 0
    count = 0
    for cart_item in db.cart_items.find({"session_id": get_session_id()}):
        medicine = db.medicines.find_one({"id": cart_item["medicine_id"]})
        if medicine:
            quantity = cart_item.get("quantity", 1)
            subtotal = medicine["price"] * quantity
            total += subtotal
            count += quantity
            medicine.pop("_id", None)
            items.append({**medicine, "quantity": quantity, "subtotal": round(subtotal, 2)})
    return {"items": items, "total": round(total, 2), "count": count}

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.json.get("msg")
        print(f"User message: {msg}")
        response = rag_chain.invoke({"input": msg})
        print("Response:", response["answer"])
        return jsonify({"response": response["answer"]})
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        error_msg = "Sorry, something went wrong on the server."
        if "rate_limit" in str(e).lower() or "429" in str(e):
            error_msg = "Rate limit exceeded: Too many requests. Please wait a moment and try again."
        elif "api_key" in str(e).lower() or "authentication" in str(e).lower():
            error_msg = "API Key Error: Please check if your GROQ_API_KEY is correctly set in the .env file."
        return jsonify({"response": error_msg}), 200

if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=8080, debug=True)
