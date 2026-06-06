from datetime import datetime, timedelta
from email.message import EmailMessage
from uuid import uuid4
import smtplib
import threading

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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "ArogyaPlus-dev-secret")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ArogyaPlus_hospital")
mongo_client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000,
    tls=True,
    tlsAllowInvalidCertificates=True
)
db = mongo_client[MONGO_DB_NAME]

DOCTOR_IMAGE_URLS = {
    1: "https://images.unsplash.com/photo-1559839734-2b71ea197ec2?auto=format&fit=crop&w=700&q=80",
    2: "https://images.unsplash.com/photo-1622253692010-333f2da6031d?auto=format&fit=crop&w=700&q=80",
    3: "https://images.unsplash.com/photo-1587884964288-62e6c7731fc0?auto=format&fit=crop&w=700&q=80",
    4: "https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?auto=format&fit=crop&w=700&q=80",
    5: "https://images.unsplash.com/photo-1594824476967-48c8b964273f?auto=format&fit=crop&w=700&q=80",
    6: "https://images.unsplash.com/photo-1537368910025-700350fe46c7?auto=format&fit=crop&w=700&q=80",
    7: "https://images.unsplash.com/photo-1551601651-2a8555f1a136?auto=format&fit=crop&w=700&q=80",
    8: "https://images.unsplash.com/photo-1538108149393-fbbd81895907?auto=format&fit=crop&w=700&q=80",
    9: "https://images.unsplash.com/photo-1651008376811-b90baee60c1f?auto=format&fit=crop&w=700&q=80",
    10: "https://images.unsplash.com/photo-1605684954998-685c79d6a018?auto=format&fit=crop&w=700&q=80",
    11: "https://images.unsplash.com/photo-1666887360742-974c8fce8e6b?auto=format&fit=crop&w=700&q=80",
    12: "https://images.unsplash.com/photo-1598256989800-fe5f95da978b?auto=format&fit=crop&w=700&q=80",
    13: "https://images.unsplash.com/photo-1612531386530-97286d97c2d2?auto=format&fit=crop&w=700&q=80",
    14: "https://images.unsplash.com/photo-1581056771107-24ca5f033842?auto=format&fit=crop&w=700&q=80",
    15: "https://images.unsplash.com/photo-1638202993928-7267aad84c31?auto=format&fit=crop&w=700&q=80",
    16: "https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=700&q=80",
    17: "https://images.unsplash.com/photo-1587653263995-422546a7a569?auto=format&fit=crop&w=700&q=80",
    18: "https://images.unsplash.com/photo-1527613426441-4da17471b66d?auto=format&fit=crop&w=700&q=80",
}

MEDICINE_UPDATES = {
    101: {"price": 35, "image_url": "https://images.unsplash.com/photo-1584100936595-c0654b55a2e6?auto=format&fit=crop&w=700&q=80"},
    102: {"price": 52, "image_url": "https://images.unsplash.com/photo-1584100936595-c0654b55a2e6?auto=format&fit=crop&w=700&q=80"},
    103: {"price": 48, "image_url": "https://images.unsplash.com/photo-1587737119753-09663860d6ee?auto=format&fit=crop&w=700&q=80"},
    104: {"price": 42, "image_url": "https://images.unsplash.com/photo-1577003811926-53b288a6e5d9?auto=format&fit=crop&w=700&q=80"},
    105: {"price": 46, "image_url": "https://images.unsplash.com/photo-1577003811926-53b288a6e5d9?auto=format&fit=crop&w=700&q=80"},
    106: {"price": 95, "image_url": "https://images.unsplash.com/photo-1631549916768-4119b2e5f926?auto=format&fit=crop&w=700&q=80"},
    107: {"price": 115, "image_url": "https://images.unsplash.com/photo-1631549916768-4119b2e5f926?auto=format&fit=crop&w=700&q=80"},
    108: {"price": 135, "image_url": "https://images.unsplash.com/photo-1550572017-edd951b55104?auto=format&fit=crop&w=700&q=80"},
    109: {"price": 160, "image_url": "https://images.unsplash.com/photo-1550572017-edd951b55104?auto=format&fit=crop&w=700&q=80"},
    110: {"price": 220, "image_url": "https://images.unsplash.com/photo-1550572017-edd951b55104?auto=format&fit=crop&w=700&q=80"},
    111: {"price": 88, "image_url": "https://images.unsplash.com/photo-1550572017-edd951b55104?auto=format&fit=crop&w=700&q=80"},
    112: {"price": 145, "image_url": "https://images.unsplash.com/photo-1550572017-edd951b55104?auto=format&fit=crop&w=700&q=80"},
    113: {"price": 70, "image_url": "https://images.unsplash.com/photo-1587737119753-09663860d6ee?auto=format&fit=crop&w=700&q=80"},
    114: {"price": 28, "image_url": "https://images.unsplash.com/photo-1628771065518-0d82f1938462?auto=format&fit=crop&w=700&q=80"},
    115: {"price": 310, "image_url": "https://images.unsplash.com/photo-1584100936595-c0654b55a2e6?auto=format&fit=crop&w=700&q=80"},
    116: {"price": 92, "image_url": "https://images.unsplash.com/photo-1584100936595-c0654b55a2e6?auto=format&fit=crop&w=700&q=80"},
    117: {"price": 249, "image_url": "https://images.unsplash.com/photo-1581360742512-021d5b2157d8?auto=format&fit=crop&w=700&q=80"},
    118: {"price": 899, "image_url": "https://images.unsplash.com/photo-1612277795421-9bc7706a4a34?auto=format&fit=crop&w=700&q=80"},
    119: {"price": 1499, "image_url": "https://images.unsplash.com/photo-1631217872822-f403d29c3bb9?auto=format&fit=crop&w=700&q=80"},
    120: {"price": 1199, "image_url": "https://images.unsplash.com/photo-1579154204601-01588f351e67?auto=format&fit=crop&w=700&q=80"},
    121: {"price": 625, "image_url": "https://images.unsplash.com/photo-1579154204601-01588f351e67?auto=format&fit=crop&w=700&q=80"},
    122: {"price": 85, "image_url": "https://images.unsplash.com/photo-1582152629442-4a864303fb96?auto=format&fit=crop&w=700&q=80"},
    123: {"price": 399, "image_url": "https://images.unsplash.com/photo-1603398938378-e54eab446dde?auto=format&fit=crop&w=700&q=80"},
    124: {"price": 65, "image_url": "https://images.unsplash.com/photo-1603398938378-e54eab446dde?auto=format&fit=crop&w=700&q=80"},
    125: {"price": 55, "image_url": "https://images.unsplash.com/photo-1603398938378-e54eab446dde?auto=format&fit=crop&w=700&q=80"},
    126: {"price": 115, "image_url": "https://images.unsplash.com/photo-1612817288484-6f916006741a?auto=format&fit=crop&w=700&q=80"},
    127: {"price": 105, "image_url": "https://images.unsplash.com/photo-1612817288484-6f916006741a?auto=format&fit=crop&w=700&q=80"},
    128: {"price": 240, "image_url": "https://images.unsplash.com/photo-1612817288484-6f916006741a?auto=format&fit=crop&w=700&q=80"},
    129: {"price": 399, "image_url": "https://images.unsplash.com/photo-1556228578-8c89e6adf883?auto=format&fit=crop&w=700&q=80"},
    130: {"price": 165, "image_url": "https://images.unsplash.com/photo-1612776572997-76cc42e058c3?auto=format&fit=crop&w=700&q=80"},
    131: {"price": 45, "image_url": "https://images.unsplash.com/photo-1628771065518-0d82f1938462?auto=format&fit=crop&w=700&q=80"},
    132: {"price": 85, "image_url": "https://images.unsplash.com/photo-1550572017-edd951b55104?auto=format&fit=crop&w=700&q=80"},
    133: {"price": 180, "image_url": "https://images.unsplash.com/photo-1628771065518-0d82f1938462?auto=format&fit=crop&w=700&q=80"},
    134: {"price": 260, "image_url": "https://images.unsplash.com/photo-1584100936595-c0654b55a2e6?auto=format&fit=crop&w=700&q=80"},
    135: {"price": 75, "image_url": "https://images.unsplash.com/photo-1583947215259-38e31be8751f?auto=format&fit=crop&w=700&q=80"},
    136: {"price": 120, "image_url": "https://images.unsplash.com/photo-1584727638096-042c45049ebe?auto=format&fit=crop&w=700&q=80"},
}

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

@app.context_processor
def inject_current_user():
    return {"current_user": session.get("user")}

def _send_email_worker(to_email, subject, body):
    mail_host = os.getenv("MAIL_HOST", "smtp.gmail.com")
    mail_port = int(os.getenv("MAIL_PORT", "587"))
    mail_user = os.getenv("MAIL_USERNAME")
    mail_password = os.getenv("MAIL_PASSWORD")
    mail_from = os.getenv("MAIL_FROM", mail_user or "care@arogyaplus.example")

    if not mail_user or not mail_password:
        print(f"\n--- Email preview ---\nTo: {to_email}\nSubject: {subject}\n{body}\n---\n")
        return

    message = EmailMessage()
    message["From"] = mail_from
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(body)

    try:
        with smtplib.SMTP(mail_host, mail_port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(mail_user, mail_password)
            smtp.send_message(message)
        print(f"Email sent to {to_email}")
    except Exception as exc:
        print(f"Email delivery failed for {to_email}: {exc}")

def send_email(to_email, subject, body):
    if not to_email:
        return
    # Send email in background thread so it never blocks the response
    thread = threading.Thread(target=_send_email_worker, args=(to_email, subject, body))
    thread.daemon = True
    thread.start()

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
    db.purchases.create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])

    if db.doctors.count_documents({}) == 0:
        db.doctors.insert_many(DOCTOR_SEED)
    if db.medicines.count_documents({}) == 0:
        db.medicines.insert_many(MEDICINE_SEED)
    for doctor_id, image_url in DOCTOR_IMAGE_URLS.items():
        db.doctors.update_one({"id": doctor_id}, {"$set": {"image_url": image_url}})
    for medicine_id, update in MEDICINE_UPDATES.items():
        db.medicines.update_one({"id": medicine_id}, {"$set": update})

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
    if request.method == "POST" and not session.get("user"):
        flash("Please log in before booking an appointment.", "info")
        return redirect(url_for("auth"))
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
        body = (
            f"Hello {appointment_data['patient_name']},\n\n"
            f"Your appointment is confirmed with {doctor['name']} "
            f"({appointment_data['specialty']}) on {appointment_data['date']} at {appointment_data['time']}.\n\n"
            "ArogyaPlus"
        )
        send_email(appointment_data["user_email"], "ArogyaPlus appointment confirmation", body)
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
    purchases = public_docs(db.purchases.find({"user_id": user_id}).sort("created_at", -1).limit(5))
    return render_template("dashboard.html", appointments=appointments, prescriptions=prescriptions, history=history, purchases=purchases)

@app.route("/auth")
def auth():
    return render_template("auth.html")

@app.route("/login", methods=["POST"])
def login():
    previous_session_id = session.get("session_id")
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    user = db.users.find_one({"email": email})
    if not user or not check_password_hash(user["password_hash"], password):
        flash("Invalid email or password.", "danger")
        return redirect(url_for("auth"))
    session.clear()
    if previous_session_id:
        session["session_id"] = previous_session_id
    session["user"] = {"id": str(user["_id"]), "name": user["name"], "email": user["email"]}
    db.cart_items.update_many({"session_id": get_session_id()}, {"$set": {"user_id": str(user["_id"])}})
    flash("Welcome back. Your dashboard is ready.", "success")
    return redirect(url_for("dashboard"))

@app.route("/register", methods=["POST"])
def register():
    previous_session_id = session.get("session_id")
    name = request.form.get("name", "Patient").strip() or "Patient"
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    if not email or not password:
        flash("Please enter a valid email and password.", "danger")
        return redirect(url_for("auth"))
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
    session.clear()
    if previous_session_id:
        session["session_id"] = previous_session_id
    session["user"] = {"id": user_id, "name": name, "email": email}
    db.cart_items.update_many({"session_id": get_session_id()}, {"$set": {"user_id": user_id}})
    flash("Registration complete. You are now signed in.", "success")
    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth"))

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

@app.route("/cart/checkout", methods=["POST"])
def checkout():
    if not session.get("user"):
        return jsonify({"error": "Please log in before checkout.", "login_url": url_for("auth")}), 401
    cart = build_cart_response()
    if not cart["items"]:
        return jsonify({"error": "Your cart is empty."}), 400

    data = request.get_json(silent=True) or request.form
    delivery_details = {
        "name": data.get("name", "").strip(),
        "phone": data.get("phone", "").strip(),
        "address": data.get("address", "").strip(),
        "city": data.get("city", "").strip(),
        "state": data.get("state", "").strip(),
        "pincode": data.get("pincode", "").strip(),
        "notes": data.get("notes", "").strip(),
    }
    required_fields = ["name", "phone", "address", "city", "state", "pincode"]
    missing_fields = [field for field in required_fields if not delivery_details[field]]
    if missing_fields:
        return jsonify({"error": "Please complete all required delivery details."}), 400

    delivery_date = (datetime.utcnow() + timedelta(days=5)).date().isoformat()
    user = session["user"]
    purchase = {
        "user_id": user["id"],
        "user_email": user["email"],
        "patient_name": delivery_details["name"],
        "delivery_details": delivery_details,
        "payment_method": "Cash on Delivery",
        "payment_status": "Pay on delivery",
        "items": cart["items"],
        "total": cart["total"],
        "status": "Confirmed",
        "delivery_date": delivery_date,
        "created_at": datetime.utcnow(),
    }
    db.purchases.insert_one(purchase)
    db.cart_items.delete_many({"session_id": get_session_id()})

    item_lines = "\n".join(
        f"- {item['name']} x {item['quantity']} (Rs. {item['subtotal']:.2f})"
        for item in cart["items"]
    )
    body = (
        f"Hello {user['name']},\n\n"
        "Your ArogyaPlus pharmacy order was successful.\n\n"
        f"{item_lines}\n\n"
        f"Total: Rs. {cart['total']:.2f}\n"
        "Payment mode: Cash on Delivery\n"
        f"Expected delivery date: {delivery_date}\n\n"
        "Delivery address:\n"
        f"{delivery_details['name']}\n"
        f"{delivery_details['phone']}\n"
        f"{delivery_details['address']}\n"
        f"{delivery_details['city']}, {delivery_details['state']} - {delivery_details['pincode']}\n\n"
        "ArogyaPlus"
    )
    send_email(user["email"], "ArogyaPlus pharmacy purchase confirmation", body)

    return jsonify({
        "message": f"Purchase successful. Confirmation sent to {user['email']}. Expected delivery: {delivery_date}.",
        "delivery_date": delivery_date,
        "cart": build_cart_response(),
    })

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
    app.run(host="0.0.0.0", port=7860, debug=True)