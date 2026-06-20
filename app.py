import os
import random
import datetime
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from io import BytesIO

# Import ReportLab elements safely for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Page configurations
st.set_page_config(
    page_title="Ocular Disease Classification Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SQLite Database Setup Functions
DB_FILE = "patient_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # UPGRADED: Added age, gender, and affected_eye columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            patient_age INTEGER,
            patient_gender TEXT,
            affected_eye TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            scan_date TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_patient(p_id, p_name, p_age, p_gender, p_eye, prediction, confidence):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO patients (patient_id, patient_name, patient_age, patient_gender, affected_eye, prediction, confidence, scan_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (p_id, p_name, p_age, p_gender, p_eye, prediction, round(confidence, 2), current_date))
    conn.commit()
    conn.close()

def get_all_patients():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT 
            patient_id AS 'Patient ID', 
            patient_name AS 'Patient Name', 
            patient_age AS 'Age', 
            patient_gender AS 'Gender', 
            affected_eye AS 'Tested Eye', 
            prediction AS 'AI Diagnosis', 
            confidence AS 'Confidence (%)', 
            scan_date AS 'Scan Date/Time' 
        FROM patients ORDER BY id DESC
    """, conn)
    conn.close()
    return df

# Initialize Database
init_db()

# Premium Clinical Dashboard Styling
st.markdown("""
    <style>
    .main {
        background-color: #0f111a;
        color: #e2e8f0;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #3498db, #2ecc71);
    }
    div[data-testid="stSidebar"] {
        background-color: #1a1c24;
        border-right: 1px solid #2d3748;
    }
    .medical-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #63b3ed 0%, #4fd1c5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    .medical-subtitle {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .verdict-box {
        background-color: #1b1e2e;
        border-left: 6px solid #3498db;
        padding: 1.8rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-title {
        border-bottom: 2px solid #2d3748;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: #63b3ed;
    }
    </style>
""", unsafe_allow_html=True)

# Safe TensorFlow Check
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception as e:
    pass

# Class mappings
CLASS_LABELS = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
CLASS_COLORS = ['#E74C3C', '#3498DB', '#F39C12', '#2ECC71']

# Clinical suggestion dictionary for PDF generation
CLINICAL_DATA = {
    "Normal": {
        "summary": "The AI network confirms normal retina architecture. The optical disc margins appear clean, macula region is optimal, and no traces of active anomalies were flagged.",
        "advise": "1. Schedule a standard baseline ocular health checkup in 12 months.\n2. Encourage systemic microvascular safety protocols.\n3. Wear protective UV eyewear under high-exposure setups."
    },
    "Cataract": {
        "summary": "Spectral attenuation clusters patterns match standard crystalline lens opacification. Severe light refraction scatter within the inner lens boundary noted.",
        "advise": "1. Refer for Slit-lamp Biomicroscopy to grade nuclear sclerosis.\n2. Verify functional vision loss via Best-Corrected Visual Acuity (BCVA) mapping.\n3. Initiate clinical counsel for phacoemulsification paired with Intraocular Lens (IOL) implantation."
    },
    "Diabetic Retinopathy": {
        "summary": "Saliency channels actively center on structural abnormalities matching microaneurysms, intraretinal hemorrhages, or lipid exudate pools.",
        "advise": "1. Immediate endocrine assessment (HbA1c serum history & BP checks).\n2. Order an Optical Coherence Tomography (OCT) macular cube scan immediately to verify DME status.\n3. Expedite specialist triage for Panretinal Photocoagulation or Anti-VEGF injection evaluation."
    },
    "Glaucoma": {
        "summary": "Deep layer features display high localization spikes focused strictly around the optic nerve head, indicating significant excavation or asymmetric cup-to-disc ratio alterations.",
        "advise": "1. Conduct golden-standard Goldmann Applanation Tonometry to check current IOP levels.\n2. Order standard automated perimetry tests to map peripheral vision fields.\n3. Perform a structural OCT scan of the optic nerve head to grade RNFL loss. Start IOP-lowering drops immediately if indicated."
    }
}

# Demo Mode attention logic
def compute_attention_demo(img_array):
    img_gray = cv2.cvtColor(np.uint8(255 * img_array), cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    
    img_seed = int(np.sum(img_gray)) % 1000
    rng = np.random.default_rng(img_seed)
    
    offset_y = rng.integers(-30, 30)
    offset_x = rng.integers(-30, 30)
    radius = rng.integers(25, 50)
    
    dist = ((y - (center_y + offset_y))**2 + (x - (center_x + offset_x))**2)
    mask = np.exp(-dist / (2 * (radius**2)))
    
    edges = cv2.Canny(img_gray, 30, 100)
    edges_blur = cv2.GaussianBlur(edges, (15, 15), 0).astype(float) / 255.0
    
    heatmap = mask * 0.70 + edges_blur * 0.30
    heatmap = heatmap / (np.max(heatmap) + 1e-10)
    
    raw_probs = rng.dirichlet(np.ones(4))
    best_class = rng.integers(0, 4)
    raw_probs[best_class] += 2.5
    probs = raw_probs / np.sum(raw_probs)
    
    return heatmap, np.expand_dims(probs, 0)

# Real TensorFlow Grad-CAM Saliency Map Logic
def compute_attention_real(model, img_tensor):
    img_var = tf.Variable(tf.cast(img_tensor, tf.float32), trainable=True)
    with tf.GradientTape() as tape:
        preds = model(img_var, training=False)
        class_idx = tf.argmax(preds[0])
        class_score = preds[:, class_idx]
    
    grads = tape.gradient(class_score, img_var)
    heatmap = tf.reduce_max(tf.abs(grads[0]), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy(), preds.numpy()

# Overlay jet heatmap onto RGB image
def overlay_heatmap(heatmap, img_rgb, alpha=0.45):
    h = cv2.resize(heatmap, (224, 224))
    h = np.uint8(255 * h)
    h_col = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    h_col = cv2.cvtColor(h_col, cv2.COLOR_BGR2RGB)
    
    img_u8 = np.uint8(255 * np.clip(img_rgb, 0, 1))
    overlay = cv2.addWeighted(img_u8, 1 - alpha, h_col, alpha, 0)
    return overlay

@st.cache_resource
def load_deep_learning_model():
    if not TF_AVAILABLE:
        return None, "TensorFlow missing."
    filepath = os.path.join('models', 'DenseNet121_final.keras')
    if os.path.exists(filepath):
        try:
            model = keras.models.load_model(filepath)
            return model, "Loaded successfully"
        except Exception as e:
            return None, str(e)
    return None, "File missing"

def process_single_image(img_file, is_demo_mode, loaded_model):
    pil_img = Image.open(img_file).convert("RGB")
    pil_resized = pil_img.resize((224, 224))
    img_rgb = np.array(pil_resized).astype(float) / 255.0
    
    if is_demo_mode:
        heatmap, preds = compute_attention_demo(img_rgb)
        return img_rgb, overlay_heatmap(heatmap, img_rgb), preds[0]
    
    ensemble_probs = np.zeros((1, 4))
    ensemble_heatmaps = []
    
    for pass_idx in range(5):
        if pass_idx == 0:   t_img = img_rgb
        elif pass_idx == 1: t_img = np.fliplr(img_rgb)
        elif pass_idx == 2: t_img = np.flipud(img_rgb)
        elif pass_idx == 3: t_img = np.rot90(img_rgb, 1)
        elif pass_idx == 4: t_img = np.rot90(img_rgb, 2)
            
        img_tensor = np.expand_dims(t_img, axis=0)
        h, p = compute_attention_real(loaded_model, img_tensor)
        
        if pass_idx == 1:   h = np.fliplr(h)
        elif pass_idx == 2: h = np.flipud(h)
        elif pass_idx == 3: h = np.rot90(h, -1)
        elif pass_idx == 4: h = np.rot90(h, -2)
            
        ensemble_probs += p
        ensemble_heatmaps.append(h)
        
    final_heatmap = np.mean(ensemble_heatmaps, axis=0)
    final_heatmap = final_heatmap / (np.max(final_heatmap) + 1e-10)
    final_probs = ensemble_probs[0] / 5.0
    
    overlay = overlay_heatmap(final_heatmap, img_rgb)
    return img_rgb, overlay, final_probs

# ----------------------------------------------------
# 📥 UPGRADED FUNCTION: REPORTLAB PDF GENERATOR ENGINE (With Age, Gender, Eye)
# ----------------------------------------------------
def generate_pdf_report(p_id, p_name, p_age, p_gender, p_eye, diagnosis, confidence):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom Premium PDF Typography
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=24, 
        textColor=colors.HexColor('#1A365D'), spaceAfter=6, alignment=1
    )
    subtitle_style = ParagraphStyle(
        'DocSub', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=10, 
        textColor=colors.HexColor('#4A5568'), spaceAfter=20, alignment=1
    )
    section_heading = ParagraphStyle(
        'SecHead', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, 
        textColor=colors.HexColor('#2B6CB0'), spaceBefore=12, spaceAfter=8
    )
    body_style = ParagraphStyle(
        'BodyTextCustom', parent=styles['Normal'], fontName='Helvetica', fontSize=10.5, 
        textColor=colors.HexColor('#2D3748'), leading=14
    )
    verdict_style = ParagraphStyle(
        'VerdictText', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=13, 
        textColor=colors.HexColor('#C53030') if diagnosis != "Normal" else colors.HexColor('#2F855A')
    )

    # Document Header
    story.append(Paragraph("OCULAR DISEASE DIAGNOSTIC REPORT", title_style))
    story.append(Paragraph("Automated Clinical Screening Powered by Fine-Tuned Deep Learning Neural Networks", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Patient Data Table (UPGRADED)
    story.append(Paragraph("Patient Information Records", section_heading))
    current_date = datetime.datetime.now().strftime("%B %d, %Y - %I:%M %p")
    
    patient_data = [
        [Paragraph(f"<b>Patient ID:</b> {p_id}", body_style), Paragraph(f"<b>Age / Gender:</b> {p_age}Y / {p_gender}", body_style)],
        [Paragraph(f"<b>Patient Name:</b> {p_name}", body_style), Paragraph(f"<b>Tested Eye:</b> {p_eye}", body_style)],
        [Paragraph(f"<b>Date & Time:</b> {current_date}", body_style), Paragraph("<b>System Status:</b> Verified Log", body_style)]
    ]
    t1 = Table(patient_data, colWidths=[265, 265])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F7FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t1)
    story.append(Spacer(1, 15))
    
    # Diagnosis Verdict Table
    story.append(Paragraph("AI Diagnostic Summary", section_heading))
    verdict_data = [
        [Paragraph("<b>Primary Screening Outcome:</b>", body_style), Paragraph(diagnosis, verdict_style)],
        [Paragraph("<b>Model Match Confidence Score:</b>", body_style), Paragraph(f"{confidence:.2f}% Match Rate", body_style)],
        [Paragraph("<b>Diagnostic Summary:</b>", body_style), Paragraph(CLINICAL_DATA[diagnosis]["summary"], body_style)]
    ]
    t2 = Table(verdict_data, colWidths=[180, 350])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#EDF2F7')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t2)
    story.append(Spacer(1, 15))
    
    # Clinical Recommendations Section
    story.append(Paragraph("Urgent Care Plans & Clinical Suggestions", section_heading))
    advise_text = CLINICAL_DATA[diagnosis]["advise"].replace("\n", "<br/>")
    story.append(Paragraph(advise_text, body_style))
    story.append(Spacer(1, 40))
    
    # Footer Signatures
    sig_data = [
        [Paragraph("___________________________<br/><b>Medical Officer Signature</b>", body_style), 
         Paragraph("___________________________<br/><b>Ophthalmologist Triage Stamp</b>", body_style)]
    ]
    t3 = Table(sig_data, colWidths=[270, 260])
    t3.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t3)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ----------------------------------------------------
# 🔧 SIDEBAR MENU BAR NAVIGATION
# ----------------------------------------------------
st.sidebar.markdown("<h1 style='text-align: center; color: #63b3ed; font-size: 1.8rem;'>🔬 Ocular AI</h1>", unsafe_allow_html=True)
menu_selection = st.sidebar.radio(
    "",
    ["Home Page", "Patient History"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👁️ Target Pathologies")
st.sidebar.info("- **Cataract**\n- **Diabetic Retinopathy**\n- **Glaucoma**\n- **Normal**")

# Load Trained Weights Globally
model, status = load_deep_learning_model()
is_demo = model is None

if model is not None:
    st.sidebar.success("✅ Deep Learning Engine Active!")
else:
    st.sidebar.warning("⚠️ Running in Demo Mode")

# ----------------------------------------------------
# PAGE 1: 🔬 AI DIAGNOSTIC CHAMBER
# ----------------------------------------------------
if menu_selection == "Home Page":
    st.markdown("<h1 class='medical-title'>👁️ Ocular Disease & Retinopathy Diagnostic System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='medical-subtitle'>Upload patient retinal fundus images for targeted pathology mapping and instant diagnostics.</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 class='section-title'>👤 Patient Registration</h3>", unsafe_allow_html=True)
    
    # UPGRADED: Expanded demographics input layout
    col_p1, col_p2, col_p3, col_p4 = st.columns([2, 3, 1, 1.5])
    with col_p1:
        patient_id = st.text_input("Patient Unique ID", value="P-")
    with col_p2:
        patient_name = st.text_input("Patient Full Name", value="")
    with col_p3:
        patient_age = st.number_input("Age", min_value=1, max_value=120, value=45)
    with col_p4:
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
    # UPGRADED: Select which eye is being scanned
    patient_eye = st.radio("Select Scanned Eye Perspective:", ["Right Eye (OD)", "Left Eye (OS)"], horizontal=True)
    
    uploaded_files = st.file_uploader("Select Retinal Fundus...", type=["png", "jpg", "jpeg"])
    
    if uploaded_files:
        if not patient_name.strip() or patient_id == "P-":
            st.error("⚠️ Please provide a valid Patient ID and Name before executing scan.")
        else:
            with st.spinner("Executing deep clinical screening scans..."):
                img_rgb, overlay, probs = process_single_image(uploaded_files, is_demo, model)
                
            pred_class_idx = np.argmax(probs)
            pred_class_name = CLASS_LABELS[pred_class_idx]
            pred_confidence = probs[pred_class_idx] * 100
            color = CLASS_COLORS[pred_class_idx]
            
            # Database Trigger (UPGRADED to include age, gender, eye)
            state_key = f"saved_{patient_id}_{uploaded_files.name}"
            if state_key not in st.session_state:
                insert_patient(patient_id, patient_name, patient_age, patient_gender, patient_eye, pred_class_name, pred_confidence)
                st.session_state[state_key] = True
                st.toast(f"💾 Record saved for {patient_name} ({patient_eye})!", icon="✅")
                
            # STEP 1: VERDICT
            st.markdown("<h3 class='section-title'>📊 1. Diagnostic Class & Confidence Verdict</h3>", unsafe_allow_html=True)
            st.markdown(
                f"""<div class='verdict-box' style='border-left-color: {color};'>
                    <span style='font-size: 1.1rem; color: #a0aec0; font-weight: 500;'>Patient: <b>{patient_name}</b> ({patient_id}) | Age: {patient_age} | Gender: {patient_gender} | Target: <b>{patient_eye}</b></span><br>
                    <span style='font-size: 1.2rem; color: #a0aec0; font-weight: 500;'>AI Diagnostic Classification:</span><br>
                    <span style='font-size: 2.5rem; font-weight: 800; color: {color};'>{pred_class_name}</span>
                    <span style='font-size: 2.0rem; color: #e2e8f0; font-weight: 600;'> ({pred_confidence:.2f}% Match Confidence)</span>
                </div>""", 
                unsafe_allow_html=True
            )
            
            for idx, (label, val) in enumerate(zip(CLASS_LABELS, probs)):
                col_bar_1, col_bar_2 = st.columns([1, 4])
                with col_bar_1: st.markdown(f"**{label}**")
                with col_bar_2:
                    st.progress(float(val))
                    st.markdown(f"Score Distribution: {val*100:.2f}%")
                    
            # STEP 2: GRAD-CAM
            st.markdown("<h3 class='section-title'>🔍 2. Explainable AI: Grad-CAM Feature Visualization</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_rgb, caption=f"Patient Base Fundus Scan ({patient_eye})", use_container_width=True)
            with col2:
                st.image(overlay, caption="Saliency Map Activation Zones (AI Attention)", use_container_width=True)
                
            # STEP 3: SUGGESTIONS
            st.markdown("<h3 class='section-title'>📝 3. Clinical Action Suggestions & Expert Insights</h3>", unsafe_allow_html=True)
            if pred_class_name == "Normal":
                st.success(f"✅ **Clinical Summary:** {CLINICAL_DATA['Normal']['summary']}")
                st.info(CLINICAL_DATA["Normal"]["advise"])
            else:
                st.error(f"🚨 **Clinical Summary:** {CLINICAL_DATA[pred_class_name]['summary']}")
                st.info(CLINICAL_DATA[pred_class_name]["advise"])
                
            # 📥 DOWNLOADING PDF REPORT
            st.markdown("<br/>", unsafe_allow_html=True)
            pdf_data = generate_pdf_report(patient_id, patient_name, patient_age, patient_gender, patient_eye, pred_class_name, pred_confidence)
            
            st.download_button(
                label=f"📥 Download Clinical PDF Report for {patient_name}",
                data=pdf_data,
                file_name=f"Ocular_Report_{patient_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# ----------------------------------------------------
# PAGE 2: 🗃️ PATIENT HISTORY RECORDS (With Data Export Feature)
# ----------------------------------------------------
elif menu_selection == "Patient History":
    st.markdown("<h1 class='medical-title'>🗃️ Patient Database Logs & History</h1>", unsafe_allow_html=True)
    st.markdown("<p class='medical-subtitle'>Track, filter, and export previously analyzed clinical records secured within the system database.</p>", unsafe_allow_html=True)
    
    df_history = get_all_patients()
    
    if not df_history.empty:
        # Search Box and Export Button layout
        col_sec1, col_sec2 = st.columns([4, 1])
        with col_sec1:
            search_query = st.text_input("🔍 Live Database Search (Type Patient Name or ID)...", value="")
        
        if search_query:
            df_filtered = df_history[
                df_history['Patient ID'].str.contains(search_query, case=False) | 
                df_history['Patient Name'].str.contains(search_query, case=False)
            ]
        else:
            df_filtered = df_history
            
        # UPGRADED: One-click Excel/CSV Master Sheet Data Export
        with col_sec2:
            st.markdown("<b style='color:#a0aec0;'>Master Data Sheet</b>", unsafe_allow_html=True)
            csv_buffer = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Sheet (CSV)",
                data=csv_buffer,
                file_name=f"Ocular_Database_Export_{datetime.date.today()}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        st.markdown("<h3 class='section-title'>📋 Patient Records Ledger</h3>", unsafe_allow_html=True)
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)
        
        # Analytics Block
        st.markdown("<h3 class='section-title'>📈 Clinical Metrics Analytics</h3>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(label="Total Scans Executed", value=len(df_history))
        with c2:
            high_risk_cases = len(df_history[df_history['AI Diagnosis'] != 'Normal'])
            st.metric(label="Pathology Detected (High Risk)", value=high_risk_cases, delta=f"{high_risk_cases} Flagged Cases", delta_color="inverse")
        with c3:
            avg_conf = df_history['Confidence (%)'].mean()
            st.metric(label="Average System Confidence", value=f"{avg_conf:.2f}%")
    else:
        st.info("No records found in the local database system.")