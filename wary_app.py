!pip install gradio fpdf2 google-generativeai -q
import os, re, tempfile, uuid
import gradio as gr
from fpdf import FPDF
from datetime import datetime
from PIL import Image as PILImage
import google.generativeai as genai

# ============ GEMINI SETUP (key code mein NAHI, secrets mein) ============
# Colab: left sidebar -> key icon (Secrets) -> Add new secret -> Name: GEMINI_API_KEY -> Value: nayi key -> Notebook access ON
# Hugging Face Spaces: Settings -> Variables and secrets -> New secret -> GEMINI_API_KEY
try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get("GEMINI_API_KEY")
    IN_COLAB = True
except Exception:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    IN_COLAB = False

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
gemini_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)

FALLBACK_TEXT = "A detailed clinical explanation will be provided by your doctor during review."

def get_friendly_explanation(condition, confidence, module_name):
    if gemini_model is None:
        return FALLBACK_TEXT
    try:
        prompt = f"""You are a calm, compassionate medical communication assistant.
A patient just received this AI analysis:
Module: {module_name}
Result: {condition}
Model score: {confidence}

Write a short, warm, 2-3 sentence explanation in simple language. Do NOT diagnose or give
medical advice. Reassure them a doctor will review this. Avoid alarming language."""
        return gemini_model.generate_content(prompt).text.strip()
    except Exception as e:
        print("Gemini error:", e)
        return FALLBACK_TEXT

# ============ DECISION BANDS (chest X-ray) ============
REVIEW_LOW = 0.50

# ============ CSS: BLUE + BLACK ============
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Noto+Naskh+Arabic:wght@500&display=swap');
:root { --pine:#0f2e2a; --pine-2:#17433c; --emerald:#0f8b6d; --emerald-dark:#0b6b54; --mint:#e3f4ee; --sand:#f7f3ea; --card:#fffdf8;
        --line:#e4dccb; --ink:#14201d; --muted:#6a6f68; --risk:#d9482b; --risk-soft:#fde8e1; --warn:#c07a0a; --warn-soft:#fbf0d6; }
.gradio-container { background:var(--sand) !important; font-family:'Plus Jakarta Sans',system-ui,sans-serif !important; color:var(--ink); max-width:1180px !important; }

#hero { background:radial-gradient(1200px 300px at 85% -20%, #1f6b5c 0%, transparent 60%), var(--pine); color:#fff; border-radius:22px; padding:34px 38px 26px; margin-bottom:18px; }
#hero .hero-row { display:flex; gap:20px; align-items:center; }
#hero .hero-logo { width:64px; height:64px; flex:none; }
#hero h1 { margin:0; font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:48px; line-height:1; letter-spacing:-.5px; color:#fff; display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }
#hero h1 .ur { font-family:'Noto Naskh Arabic',serif; font-size:30px; color:#7fe0bf; font-weight:500; }
#hero .hero-tag { margin:10px 0 0; font-size:16px; color:#bfe3d6; max-width:62ch; line-height:1.5; }
#hero .hero-flow { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:22px; font-size:13px; }
#hero .hero-flow span { border:1px solid #2f6b5e; background:#12403a; color:#d6efe6; border-radius:999px; padding:6px 14px; font-weight:600; }
#hero .hero-flow span.hot { background:#d9482b; border-color:#d9482b; color:#fff; }
#hero .hero-flow i { color:#5fd3ae; font-style:normal; }
@media (max-width:700px){ #hero h1{font-size:36px;} #hero{padding:26px 22px 20px;} }

.eh-card { background:var(--card) !important; border:1px solid var(--line) !important; border-radius:16px !important; padding:20px !important; margin-bottom:16px; box-shadow:none !important; }
.eh-card h3, .eh-card h4 { font-family:'Fraunces',Georgia,serif; font-weight:500; color:var(--ink); margin:0 0 6px; }
.section-hint { color:var(--muted); font-size:14px; margin:0 0 12px; line-height:1.55; }

button[role="tab"] { font-size:15px !important; font-weight:600 !important; padding:8px 18px !important; color:var(--muted) !important; border-radius:999px !important; border:1px solid var(--line) !important; background:var(--card) !important; margin-right:8px !important; }
button[role="tab"][aria-selected="true"] { background:var(--pine) !important; color:#fff !important; border-color:var(--pine) !important; }

button.primary { background:var(--emerald) !important; border:none !important; border-radius:12px !important; color:#fff !important; font-weight:700 !important; font-size:15px !important; }
button.primary:hover { background:var(--emerald-dark) !important; }
button.secondary { background:var(--pine) !important; color:#fff !important; border:none !important; border-radius:12px !important; font-weight:700 !important; }
button:focus-visible,input:focus-visible,textarea:focus-visible { outline:3px solid #7fe0bf !important; outline-offset:2px; }

.result-empty { border:1.5px dashed var(--line); border-radius:14px; padding:30px 20px; color:var(--muted); text-align:center; font-size:15px; }
.result-badge { border-radius:14px; padding:16px 18px; display:flex; align-items:center; gap:12px; border:1.5px solid; }
.result-badge h3 { margin:0; font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:25px; line-height:1.1; }
.result-badge .dot { width:14px; height:14px; border-radius:50%; flex:none; }
.badge-risk { background:var(--risk-soft); border-color:var(--risk); color:var(--risk); } .badge-risk .dot { background:var(--risk); }
.badge-ok { background:var(--mint); border-color:var(--emerald); color:var(--emerald-dark); } .badge-ok .dot { background:var(--emerald); }
.badge-review { background:var(--warn-soft); border-color:var(--warn); color:var(--warn); } .badge-review .dot { background:var(--warn); }

.meter-wrap { margin-top:18px; }
.meter-head { display:flex; justify-content:space-between; align-items:baseline; gap:10px; }
.meter-label { font-weight:700; font-size:15px; }
.meter-value { font-family:'Fraunces',Georgia,serif; font-size:32px; line-height:1; }
.meter-track { position:relative; height:18px; border-radius:9px; margin-top:8px; overflow:hidden; border:1px solid var(--line);
               background:linear-gradient(90deg,var(--warn-soft) 0 50%,#eef2ea 50% 80%,var(--mint) 80% 100%); }
.meter-fill { height:100%; border-radius:9px 0 0 9px; transition:width .7s cubic-bezier(.2,.8,.2,1); }
.fill-low { background:var(--warn); } .fill-mid { background:#5aa58f; } .fill-high { background:var(--emerald); } .fill-risk { background:var(--risk); }
.meter-marker { position:absolute; top:-2px; bottom:-2px; width:3px; background:var(--ink); }
.meter-scale { display:flex; justify-content:space-between; font-size:12px; color:var(--muted); margin-top:4px; }
.meter-level { display:inline-block; margin-top:10px; padding:4px 12px; border-radius:999px; font-size:13px; font-weight:700; }
.lvl-low { background:var(--warn-soft); color:var(--warn); } .lvl-mid { background:#e6efe9; color:#3d6b5c; } .lvl-high { background:var(--mint); color:var(--emerald-dark); }
.meter-note { font-size:12.5px; color:var(--muted); margin:10px 0 0; line-height:1.5; }
.explanation-box { background:#f3efe4; border-left:4px solid var(--emerald); border-radius:10px; padding:14px 16px; margin-top:16px; font-size:14.5px; line-height:1.65; color:var(--ink); }
.review-note { font-size:12.5px; color:var(--muted); margin-top:12px; }
.reason-list { margin:12px 0 0; padding-left:18px; font-size:14px; line-height:1.65; }

.pipe { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-top:4px; }
.pipe-card { background:var(--card); border:1px solid var(--line); border-top:4px solid var(--emerald); border-radius:14px; padding:14px 14px 12px; }
.pipe-card.pipe-idle { border-top-color:var(--line); opacity:.85; }
.pipe-card.pipe-warn { border-top-color:var(--warn); background:var(--warn-soft); }
.pipe-card.pipe-risk { border-top-color:var(--risk); background:var(--risk-soft); }
.pipe-top { display:flex; justify-content:space-between; align-items:center; }
.pipe-num { width:26px; height:26px; border-radius:50%; background:var(--pine); color:#fff; font-size:13px; font-weight:700; display:inline-flex; align-items:center; justify-content:center; }
.pipe-ms { font-family:'JetBrains Mono',monospace; font-size:11.5px; color:var(--muted); }
.pipe-title { font-family:'Fraunces',Georgia,serif; font-size:18px; font-weight:600; margin-top:10px; }
.pipe-pill { display:inline-block; margin-top:6px; font-size:11.5px; font-weight:700; letter-spacing:.4px; padding:3px 10px; border-radius:999px; background:var(--pine); color:#fff; }
.pipe-body { font-size:13px; line-height:1.55; margin:8px 0 0; color:#2b3a36; }
@media (max-width:900px){ .pipe{grid-template-columns:repeat(2,1fr);} }
@media (max-width:520px){ .pipe{grid-template-columns:1fr;} }

table.selftest { border-collapse:collapse; width:100%; margin-top:10px; font-size:14px; }
table.selftest th, table.selftest td { border:1px solid var(--line); padding:8px 10px; text-align:center; }
table.selftest th { background:var(--mint); }
#footer { text-align:center; color:var(--muted); font-size:13px; margin-top:24px; padding:18px; border-top:1px solid var(--line); line-height:1.7; }
"""

# ============ HTML HELPERS ============
EMPTY_HTML = "<div class='result-empty'>Results will appear here after you run an analysis.</div>"

def validate_patient_info(name, age, phone):
    if not name or name.strip() == "":
        return False, "⚠️ Patient name is required."
    if not age or not str(age).strip().isdigit():
        return False, "⚠️ Please enter a valid numeric age."
    if not phone or len(str(phone).strip()) < 10:
        return False, "⚠️ Please enter a valid contact number (at least 10 digits)."
    return True, ""

def confidence_level(pct):
    if pct >= 80: return "lvl-high", "fill-high", "High model confidence"
    if pct >= 60: return "lvl-mid", "fill-mid", "Moderate model confidence"
    return "lvl-low", "fill-low", "Low confidence: treat as uncertain"

def confidence_bar_html(pct, label="Model confidence", marker=None, show_level=True, risk=False):
    lvl_cls, fill_cls, lvl_text = confidence_level(pct)
    if risk: fill_cls = "fill-risk"
    marker_html = f"<div class='meter-marker' style='left:{marker:.1f}%;' title='Decision threshold'></div>" if marker is not None else ""
    level_html = f"<span class='meter-level {lvl_cls}'>{lvl_text}</span>" if show_level else ""
    note = (f"Decision threshold: {marker:.0f}% (dark line). " if marker is not None else "") + \
           "This is the model's raw score, not a calibrated probability."
    return f"""<div class='meter-wrap'>
      <div class='meter-head'><span class='meter-label'>{label}</span><span class='meter-value'>{pct:.1f}%</span></div>
      <div class='meter-track'><div class='meter-fill {fill_cls}' style='width:{pct:.1f}%;'></div>{marker_html}</div>
      <div class='meter-scale'><span>0%</span><span>50%</span><span>100%</span></div>{level_html}
      <p class='meter-note'>{note}</p></div>"""

def result_badge_html(text, state):  # state: "risk" | "ok" | "review"
    return f"<div class='result-badge badge-{state}'><span class='dot'></span><h3>{text}</h3></div>"

def explanation_html(text):
    return (f"<div class='explanation-box'>💬 {text}</div>"
            "<p class='review-note'>AI-assisted analysis. Requires clinical review.</p>")

def build_overlay(image, heatmap):
    h = cv2.resize(heatmap, (224, 224))
    h = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    base = np.array(image.convert("RGB").resize((224, 224)))
    return cv2.addWeighted(base, 0.6, h, 0.4, 0)

def save_overlay(overlay):
    path = os.path.join(tempfile.gettempdir(), f"cm_{uuid.uuid4().hex[:8]}.png")
    PILImage.fromarray(overlay).save(path)
    return path

def add_log(log, entry):
    return (list(log or []) + [entry])[-12:]

# ============ FULL SESSION PDF (blue + black) ============
BLACK, BLUE, BLUE_SOFT, GREY, RISK, AMBER = (15, 46, 42), (15, 139, 109), (227, 244, 238), (106, 111, 104), (217, 72, 43), (192, 122, 10)

def pdf_safe(text):
    # Helvetica sirf latin-1 support karta hai; Gemini ke curly quotes / Urdu text se crash na ho.
    return str(text).encode("latin-1", "replace").decode("latin-1")

class ReportPDF(FPDF):
    def header(self):
        self.set_fill_color(*BLACK); self.rect(0, 0, 210, 30, "F")
        self.set_fill_color(*BLUE);  self.rect(0, 30, 210, 2, "F")
        self.set_text_color(255, 255, 255); self.set_font("Helvetica", "B", 20)
        self.set_xy(12, 8);  self.cell(0, 9, "Wary")
        self.set_font("Helvetica", "", 9.5); self.set_text_color(190, 230, 215)
        self.set_xy(12, 18); self.cell(0, 6, "Safety-Audited Triage Report")
        self.set_y(38)
    def footer(self):
        self.set_y(-14); self.set_font("Helvetica", "", 8); self.set_text_color(*GREY)
        self.cell(0, 6, f"Wary | AI-assisted screening, not a diagnosis | Page {self.page_no()}", align="C")

def make_full_report(name, age, phone, log):
    pdf = ReportPDF(); pdf.set_auto_page_break(True, 18); pdf.add_page()
    report_id = uuid.uuid4().hex[:8].upper()
    pdf.set_font("Helvetica", "", 9); pdf.set_text_color(*GREY)
    pdf.cell(0, 6, pdf_safe(f"Report ID: CM-{report_id}   |   Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}"), ln=True)
    pdf.ln(2)
    # patient card
    y = pdf.get_y(); pdf.set_fill_color(*BLUE_SOFT); pdf.rect(10, y, 190, 26, "F")
    pdf.set_fill_color(*BLUE); pdf.rect(10, y, 2, 26, "F")
    pdf.set_xy(16, y + 3); pdf.set_font("Helvetica", "B", 12); pdf.set_text_color(*BLACK); pdf.cell(0, 6, "Patient information", ln=True)
    pdf.set_font("Helvetica", "", 10.5); pdf.set_x(16)
    pdf.cell(80, 7, pdf_safe(f"Name: {name}")); pdf.cell(40, 7, pdf_safe(f"Age: {age}")); pdf.cell(0, 7, pdf_safe(f"Contact: {phone}"), ln=True)
    pdf.set_y(y + 32)
    # summary table
    pdf.set_font("Helvetica", "B", 13); pdf.set_text_color(*BLACK); pdf.cell(0, 8, "Summary of analyses explored", ln=True)
    pdf.set_fill_color(*BLACK); pdf.set_text_color(255, 255, 255); pdf.set_font("Helvetica", "B", 9.5)
    for w, t in [(52, "Module"), (58, "Result"), (30, "Model score"), (50, "Status")]:
        pdf.cell(w, 8, "  " + t, fill=True)
    pdf.ln(8); pdf.set_font("Helvetica", "", 9.5)
    for i, e in enumerate(log):
        pdf.set_fill_color(*(BLUE_SOFT if i % 2 == 0 else (255, 255, 255))); pdf.set_text_color(*BLACK)
        pdf.cell(52, 8, pdf_safe("  " + e["module"]), fill=True); pdf.cell(58, 8, pdf_safe("  " + e["result"][:30]), fill=True)
        pdf.cell(30, 8, pdf_safe("  " + e["score"]), fill=True)
        col = RISK if e["state"] == "risk" else (AMBER if e["state"] == "review" else BLUE)
        pdf.set_text_color(*col); pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(50, 8, pdf_safe("  " + e["status"]), fill=True); pdf.set_font("Helvetica", "", 9.5); pdf.ln(8)
    pdf.ln(4)
    # detail per module
    for e in log:
        if pdf.get_y() > 200: pdf.add_page()
        pdf.set_font("Helvetica", "B", 12); pdf.set_text_color(*BLUE); pdf.cell(0, 8, pdf_safe(e["module"]), ln=True)
        pdf.set_draw_color(*BLUE); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(3)
        y0 = pdf.get_y(); text_x = 10
        if e.get("image") and os.path.exists(e["image"]):
            pdf.image(e["image"], x=10, y=y0, w=52); text_x = 68
        pdf.set_xy(text_x, y0); pdf.set_text_color(*BLACK); pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 6, pdf_safe(f"Result: {e['result']}"), ln=True)
        pdf.set_x(text_x); pdf.set_font("Helvetica", "", 10); pdf.cell(0, 6, pdf_safe(f"Model score: {e['score']}   ({e['status']})"), ln=True)
        if e.get("detail"):
            pdf.set_x(text_x); pdf.set_font("Helvetica", "I", 9); pdf.set_text_color(*GREY)
            pdf.multi_cell(190 - (text_x - 10), 5, pdf_safe(e["detail"]))
        pdf.set_x(text_x); pdf.set_font("Helvetica", "", 10); pdf.set_text_color(*BLACK)
        pdf.multi_cell(190 - (text_x - 10), 5.5, pdf_safe(e["explanation"]))
        pdf.set_y(max(pdf.get_y(), y0 + (40 if text_x > 10 else 0)) + 6)
    # next steps + disclaimer
    if pdf.get_y() > 235: pdf.add_page()
    any_flag = any(e["state"] in ("risk", "review") for e in log)
    pdf.set_font("Helvetica", "B", 12); pdf.set_text_color(*BLACK); pdf.cell(0, 8, "Recommended next steps", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, "One or more results need clinical attention. Please consult a licensed physician and share this report. "
                           "Seek emergency care if you have severe symptoms." if any_flag else
                           "No high-risk indicators were flagged by the models. Routine follow-up with your physician is still recommended.")
    pdf.ln(5); pdf.set_font("Helvetica", "I", 8.5); pdf.set_text_color(*GREY)
    pdf.multi_cell(0, 4.5, "Disclaimer: Wary (built on the ClarityMed models) is a research prototype for decision support. Model scores are raw outputs and are not "
                           "calibrated probabilities. This report does not replace professional medical diagnosis.")
    out = os.path.join(tempfile.gettempdir(), f"Wary_Report_{re.sub(r'[^A-Za-z0-9_-]', '_', name.strip())}_{report_id}.pdf")
    pdf.output(out)
    return out

def generate_report(name, age, phone, log):
    valid, msg = validate_patient_info(name, age, phone)
    if not valid:
        return None, msg
    if not log:
        return None, "⚠️ Run at least one analysis first, then generate the report."
    return make_full_report(name, age, phone, log), "✅ Report ready. Download it below."

# ============ INFERENCE (model logic unchanged) ============
def chest_xray_interface(image, name, age, phone, log):
    valid, msg = validate_patient_info(name, age, phone)
    if not valid: return None, msg, log
    if image is None: return None, "⚠️ Please upload a chest X-ray image.", log
    x = val_transforms(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(chest_model_v2(x), dim=1)
    p = probs[0, 1].item()
    if p >= saved_threshold:   label, state, status = "PNEUMONIA", "risk", "Flagged"
    elif p >= REVIEW_LOW:      label, state, status = "NEEDS REVIEW", "review", "Uncertain: manual review"
    else:                      label, state, status = "NORMAL", "ok", "No flag"
    heatmap, _, _ = gradcam.generate(x)
    overlay = build_overlay(image, heatmap)
    expl = get_friendly_explanation(label, f"{p*100:.1f}% pneumonia probability", "Chest X-ray")
    html = (result_badge_html(label, state) +
            confidence_bar_html(p * 100, "Pneumonia probability", marker=saved_threshold * 100, show_level=False, risk=(state == "risk")) +
            explanation_html(expl))
    entry = dict(module="Chest X-ray", result=label, score=f"{p*100:.1f}% pneumonia prob.", state=state, status=status,
                 image=save_overlay(overlay), explanation=expl,
                 detail=f"Grad-CAM overlay shown. Decision threshold {saved_threshold*100:.0f}%; review band {REVIEW_LOW*100:.0f}-{saved_threshold*100:.0f}%.")
    return overlay, html, add_log(log, entry)

def brain_mri_interface(image, name, age, phone, log):
    valid, msg = validate_patient_info(name, age, phone)
    if not valid: return None, msg, log
    if image is None: return None, "⚠️ Please upload a brain MRI image.", log
    x = val_transforms(image.convert("RGB")).unsqueeze(0).to(device)
    heatmap, idx, conf = gradcam_brain.generate(x)
    label = full_brain_train.classes[idx]
    overlay = build_overlay(image, heatmap)
    low = conf < 0.60
    state = "review" if low else ("risk" if label != "notumor" else "ok")
    status = "Low confidence: manual review" if low else ("Flagged" if label != "notumor" else "No flag")
    expl = get_friendly_explanation(label, f"{conf*100:.1f}%", "Brain MRI")
    html = (result_badge_html(label.upper(), state) + confidence_bar_html(conf * 100) + explanation_html(expl))
    entry = dict(module="Brain MRI", result=label.upper(), score=f"{conf*100:.1f}%", state=state, status=status,
                 image=save_overlay(overlay), explanation=expl, detail="Grad-CAM overlay shown (inspection aid, not proof of medical reasoning).")
    return overlay, html, add_log(log, entry)

def symptom_checker_interface(symptom_text, name, age, phone, log):
    valid, msg = validate_patient_info(name, age, phone)
    if not valid: return msg, log
    if not symptom_text or symptom_text.strip() == "": return "⚠️ Please describe your symptoms.", log
    vec = vectorizer.transform([symptom_text])
    pred = text_model.predict(vec)[0]
    conf = max(text_model.predict_proba(vec)[0]) * 100
    low = conf < 60
    state, status = ("review", "Low confidence: see a doctor") if low else ("ok", "Screening result")
    expl = get_friendly_explanation(pred, f"{conf:.1f}%", "Symptom Checker")
    html = (result_badge_html(pred, state) + confidence_bar_html(conf) + explanation_html(expl))
    entry = dict(module="Symptom Checker", result=str(pred), score=f"{conf:.1f}%", state=state, status=status,
                 image=None, explanation=expl, detail=f"Patient described: {symptom_text.strip()[:160]}")
    return html, add_log(log, entry)

# ============ ERROR HANDLING ============
ERROR_MSG = "⚠️ Something went wrong while analysing this input. Please try a different file or text."

def safe(fn, n_outputs):
    """Turn any unexpected exception into a friendly message instead of a raw Gradio error."""
    def wrapper(*args):
        try:
            return fn(*args)
        except Exception as e:
            print("Error in", fn.__name__, ":", repr(e))
            log = args[-1]
            if n_outputs == 3:
                return None, ERROR_MSG, log
            return ERROR_MSG, log
    wrapper.__name__ = fn.__name__
    return wrapper

chest_xray_interface = safe(chest_xray_interface, 3)
brain_mri_interface = safe(brain_mri_interface, 3)
symptom_checker_interface = safe(symptom_checker_interface, 2)

# ============ WARY: AGENT CONSOLE + SAFETY LAB ============
# Cell A (run_triage, audit, run_audit_selftest, AUDIT_CFG) pehle run hona zaroori hai.
import html as _html, json
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

AGENT_EXAMPLES = [
    ["mujhe 3 din se tez bukhar hai, gala kharab hai aur jism mein dard hai"],
    ["seene mein dard hai aur saans phool rahi hai"],
    ["pait mein dard aur ulti kal se, kamzori bhi hai"],
    ["thori si tabiyat kharab hai"],
    ["mere walid ka chehra tirha ho gaya aur zubaan lar kharane lagi"],
]
_BADGE = {"EMERGENCY": ("EMERGENCY: ABHI HOSPITAL JAYEN", "risk"),
          "ABSTAIN": ("NO ANSWER: DOCTOR SE MILEIN", "review")}
_STATE = {"EMERGENCY": "risk", "ABSTAIN": "review", "PRIORITY_REVIEW": "review", "ROUTINE_REVIEW": "ok"}
PIPE_STEPS = [("Intake", "Roman Urdu, Urdu ya English ko structured symptoms mein badalta hai."),
              ("Specialist", "Symptom model top-3 andaze deta hai (raw scores)."),
              ("Safety Auditor", "Red flags, kam confidence, qareebi muqable aur umar ke rules check karta hai."),
              ("Report", "Mareez ke liye paighaam aur doctor ke liye handoff note.")]

def pipeline_idle_html():
    cards = "".join(f"<div class='pipe-card pipe-idle'><div class='pipe-top'><span class='pipe-num'>{i+1}</span>"
                    f"<span class='pipe-ms'>waiting</span></div><div class='pipe-title'>{n}</div>"
                    f"<p class='pipe-body'>{d}</p></div>" for i, (n, d) in enumerate(PIPE_STEPS))
    return f"<div class='pipe'>{cards}</div>"

def trace_html(trace, decision=None):
    cards = ""
    for i, t in enumerate(trace):
        name = t["agent"].split(". ", 1)[-1]
        cls, pill = "", ""
        if "Auditor" in t["agent"] and decision:
            cls = {"EMERGENCY": " pipe-risk", "ABSTAIN": " pipe-warn", "PRIORITY_REVIEW": " pipe-warn"}.get(decision, "")
            pill = f"<span class='pipe-pill'>{_html.escape(decision.replace('_', ' '))}</span>"
        cards += (f"<div class='pipe-card{cls}'><div class='pipe-top'><span class='pipe-num'>{i+1}</span>"
                  f"<span class='pipe-ms'>{t['ms']} ms</span></div><div class='pipe-title'>{_html.escape(name)}</div>{pill}"
                  f"<p class='pipe-body'>{_html.escape(t['summary'])}</p></div>")
    return f"<div class='pipe'>{cards}</div>"

def decision_html(tri):
    d, spec = tri["decision"], tri["spec"]
    label, conf = spec["top"][0]
    if d["decision"] in _BADGE:
        text, state = _BADGE[d["decision"]]
    elif d["decision"] == "PRIORITY_REVIEW":
        text, state = f"POSSIBLE: {label.upper()} (PRIORITY REVIEW)", "review"
    else:
        text, state = f"POSSIBLE: {label.upper()}", "ok"
    out = result_badge_html(_html.escape(text), state)
    if d["show_prediction"]:
        out += confidence_bar_html(conf * 100, "Model score (raw)", marker=d["need_conf"] * 100, show_level=False)
    if d["reasons"]:
        out += "<ul class='reason-list'>" + "".join(f"<li>{_html.escape(r)}</li>" for _, r in d["reasons"]) + "</ul>"
    out += f"<div class='explanation-box'>💬 {_html.escape(tri['patient_message'])}</div>"
    out += "<p class='review-note'>AI-assisted screening. A doctor must sign off below before this is final.</p>"
    return out

def _audit_file(tri, raw_text):
    path = os.path.join(tempfile.gettempdir(), f"wary_audit_{tri['id']}.json")
    d = tri["decision"]
    payload = dict(id=tri["id"], time=datetime.now().isoformat(timespec="seconds"), complaint=str(raw_text)[:500],
                   config=dict(AUDIT_CFG), intake=tri["intake"], specialist=tri["spec"],
                   auditor={k: d[k] for k in ("decision", "reasons", "flags", "conf", "margin", "need_conf", "vulnerable")},
                   trace=tri["trace"], doctor_review="pending")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path

def agent_triage_interface(text, name, age, phone, log, triage):
    try:
        valid, msg = validate_patient_info(name, age, phone)
        if not valid:
            return msg, pipeline_idle_html(), "", log, triage, None
        if not text or not text.strip():
            return "⚠️ Please describe the symptoms first.", pipeline_idle_html(), "", log, triage, None
        tri = run_triage(text, name, age, log, vectorizer, text_model, gemini=gemini_model)
        d = tri["decision"]
        top = tri["spec"]["top"]
        entry = dict(module="Agent Triage",
                     result={"EMERGENCY": "EMERGENCY", "ABSTAIN": "ABSTAINED"}.get(d["decision"], top[0][0].upper())[:24],
                     score=(f"{top[0][1]*100:.1f}%" if d["show_prediction"] else "withheld"),
                     state=_STATE[d["decision"]], status="Pending doctor review", image=None,
                     explanation=tri["patient_message"], tid=tri["id"],
                     detail=("Agents: Intake > Specialist > Safety Auditor > Report. Decision: " + d["decision"] + ". " +
                             " | ".join(r for _, r in d["reasons"]))[:420])
        return (decision_html(tri), trace_html(tri["trace"], d["decision"]), tri["doctor_note"],
                add_log(log, entry), tri, _audit_file(tri, text))
    except Exception as e:
        print("Error in agent_triage_interface:", repr(e))
        return ERROR_MSG, pipeline_idle_html(), "", log, triage, None

def doctor_signoff_interface(verdict, notes, triage, log):
    try:
        if not triage or "id" not in triage:
            return "⚠️ Pehle 'Run agents' chalayen, phir sign-off karein."
        log = list(log or [])
        for e in reversed(log):
            if e.get("tid") == triage["id"]:
                e["status"] = f"Doctor: {verdict}"
                if notes and notes.strip():
                    e["detail"] = (e.get("detail", "") + " Doctor notes: " + notes.strip())[:560]
                break
        return f"✅ Doctor sign-off recorded: <b>{_html.escape(verdict)}</b>. Ab 'Session report' tab se PDF mein aa jayega."
    except Exception as e:
        print("Error in doctor_signoff_interface:", repr(e))
        return ERROR_MSG

# ---------- Safety lab ----------
def set_strictness(v):
    v = float(v)
    AUDIT_CFG["min_conf"] = v
    AUDIT_CFG["vulnerable_conf"] = min(0.95, v + 0.15)
    return (f"<p class='meter-note'>Auditor abstains below <b>{v*100:.0f}%</b> model score "
            f"(<b>{AUDIT_CFG['vulnerable_conf']*100:.0f}%</b> for age under 5 and 65 and above). Applies from the next run.</p>")

def selftest_interface():
    r = run_audit_selftest()
    bad = "".join(f"<li>{_html.escape(t)}</li>" for t in r["misses"] + r["false_alarms"])
    return (f"<table class='selftest'><tr><th>Cases</th><th>Emergency recall</th><th>Precision</th><th>Missed</th><th>False alarms</th></tr>"
            f"<tr><td>{r['n']}</td><td>{r['recall']*100:.0f}%</td><td>{r['precision']*100:.0f}%</td><td>{r['fn']}</td><td>{r['fp']}</td></tr></table>"
            + (f"<ul class='reason-list'>{bad}</ul>" if bad else "")
            + "<p class='meter-note'>30 hand-written demo cases (Roman Urdu, Urdu, English, with negations). Replace them with your own benchmark queries for a real number.</p>")

def riskcov_interface():
    try:
        Xv, yt = globals().get("X_test_vec"), globals().get("y_test")
        if Xv is None or yt is None:
            return None, "<p class='meter-note'>X_test_vec / y_test nahi mile. Symptom model wale cells pehle run karein.</p>"
        proba = text_model.predict_proba(Xv)
        pred = np.asarray(text_model.classes_)[proba.argmax(1)]
        conf, y = proba.max(1), np.asarray(yt)
        grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
        cov, acc = [], []
        for t in grid:
            m = conf >= t
            cov.append(m.mean() * 100)
            acc.append((pred[m] == y[m]).mean() * 100 if m.any() else np.nan)
        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig = Figure(figsize=(6.4, 3.9))
            ax = fig.subplots()
            ax.plot(cov, acc, marker="o", color="#0f8b6d", linewidth=2)
            cur = AUDIT_CFG["min_conf"]
            i = int(np.argmin(np.abs(grid - cur)))
            ax.scatter([cov[i]], [acc[i]], s=130, color="#d9482b", zorder=5, label=f"current threshold {cur*100:.0f}%")
            ax.set_xlabel("Coverage: % of queries answered")
            ax.set_ylabel("Accuracy on answered (%)")
            ax.set_title("Risk-coverage: answering less makes the answers safer")
            ax.legend(loc="lower left")
            fig.tight_layout()
        rows = "".join(f"<tr><td>{t*100:.0f}%</td><td>{c:.1f}%</td><td>{a:.1f}%</td></tr>" for t, c, a in zip(grid, cov, acc) if abs(t*100 % 10) < 1e-6)
        table = ("<table class='selftest'><tr><th>Threshold</th><th>Answered</th><th>Accuracy on answered</th></tr>" + rows +
                 "</table><p class='meter-note'>Computed on your held-out symptom test set. Raw model scores, not calibrated probabilities.</p>")
        return fig, table
    except Exception as e:
        print("Error in riskcov_interface:", repr(e))
        return None, f"<p class='meter-note'>{_html.escape(ERROR_MSG)}</p>"

# ============ UI ============
HEADER_HTML = """<div id='hero'>
  <div class='hero-row'>
    <svg class='hero-logo' viewBox='0 0 64 64' aria-hidden='true'>
      <path d='M32 4 L56 13 V32 C56 46 46 56 32 60 C18 56 8 46 8 32 V13 Z' fill='none' stroke='#5fd3ae' stroke-width='3' stroke-linejoin='round'/>
      <path d='M16 33 H25 L29 22 L35 43 L39 33 H48' fill='none' stroke='#5fd3ae' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/></svg>
    <div>
      <h1>Wary</h1>
      <p class='hero-tag'>Safety-first health triage in Roman Urdu. It audits itself before it answers, and stays silent when it is not sure.</p>
    </div>
  </div>
  <div class='hero-flow'><span>Intake</span><i>→</i><span>Specialist</span><i>→</i><span class='hot'>Safety Auditor</span><i>→</i><span>Report</span><i>→</i><span>Doctor sign-off</span></div>
</div>"""

with gr.Blocks(title="Wary: Safety-First Health Triage") as demo:
    log_state = gr.State([])
    triage_state = gr.State({})
    gr.HTML(HEADER_HTML)

    with gr.Column(elem_classes="eh-card"):
        gr.HTML("<p class='section-hint'><b>Patient details.</b> Ek baar bharein, har analysis aur PDF report mein use hote hain.</p>")
        with gr.Row():
            name_in = gr.Textbox(label="Patient name", placeholder="e.g. Ayesha Khan")
            age_in = gr.Textbox(label="Age", placeholder="e.g. 34")
            phone_in = gr.Textbox(label="Contact number", placeholder="e.g. 03001234567")

    with gr.Tabs():
        # ---------------- Triage console ----------------
        with gr.Tab("Triage console"):
            with gr.Row():
                with gr.Column(scale=5, elem_classes="eh-card"):
                    gr.HTML("<p class='section-hint'>Roman Urdu, Urdu ya English mein likhein. Pehle <b>Safety Auditor</b> emergency check karta hai, "
                            "aur shak ho to Wary jawab nahi deta.</p>")
                    agent_input = gr.Textbox(label="Apni takleef likhein", lines=5,
                                             placeholder="e.g. mujhe 3 din se tez bukhar hai, gala kharab hai aur jism mein dard hai")
                    agent_btn = gr.Button("Run agents", variant="primary")
                    gr.Examples(examples=AGENT_EXAMPLES, inputs=[agent_input], label="Try an example")
                with gr.Column(scale=6, elem_classes="eh-card"):
                    agent_decision = gr.HTML(value=EMPTY_HTML)
            with gr.Column(elem_classes="eh-card"):
                gr.Markdown("#### Agent pipeline")
                agent_trace = gr.HTML(value=pipeline_idle_html())
            with gr.Row():
                with gr.Column(scale=6, elem_classes="eh-card"):
                    doctor_note_box = gr.Textbox(label="Doctor handoff note (auto-generated)", lines=11, interactive=False)
                    audit_file = gr.File(label="Audit trail (JSON)")
                with gr.Column(scale=4, elem_classes="eh-card"):
                    gr.Markdown("#### Doctor sign-off (human in the loop)")
                    doc_verdict = gr.Radio(["Approved", "Modified", "Rejected"], value="Approved", label="Doctor decision")
                    doc_notes = gr.Textbox(label="Doctor notes (optional)", lines=3)
                    doc_btn = gr.Button("Sign off", variant="secondary")
                    doc_out = gr.HTML()
            agent_btn.click(agent_triage_interface, [agent_input, name_in, age_in, phone_in, log_state, triage_state],
                            [agent_decision, agent_trace, doctor_note_box, log_state, triage_state, audit_file])
            doc_btn.click(doctor_signoff_interface, [doc_verdict, doc_notes, triage_state, log_state], [doc_out])

        # ---------------- Evidence modules ----------------
        with gr.Tab("Evidence modules"):
            gr.HTML("<p class='section-hint'>Imaging aur symptom classifier yahan alag se chalte hain. Agar X-ray ya MRI flag ho jaye, to agla triage "
                    "run usay dekh kar case ko <b>priority review</b> mein daal deta hai.</p>")
            with gr.Tabs():
                with gr.Tab("Chest X-ray"):
                    with gr.Row():
                        with gr.Column(elem_classes="eh-card"):
                            xray_input = gr.Image(type="pil", label="Upload chest X-ray")
                            xray_btn = gr.Button("Analyze X-ray", variant="primary")
                        with gr.Column(elem_classes="eh-card"):
                            xray_img = gr.Image(label="Where the model looked (Grad-CAM)")
                            xray_html = gr.HTML(value=EMPTY_HTML)
                    xray_btn.click(chest_xray_interface, [xray_input, name_in, age_in, phone_in, log_state], [xray_img, xray_html, log_state])
                with gr.Tab("Brain MRI"):
                    with gr.Row():
                        with gr.Column(elem_classes="eh-card"):
                            mri_input = gr.Image(type="pil", label="Upload brain MRI")
                            mri_btn = gr.Button("Analyze MRI", variant="primary")
                        with gr.Column(elem_classes="eh-card"):
                            mri_img = gr.Image(label="Where the model looked (Grad-CAM)")
                            mri_html = gr.HTML(value=EMPTY_HTML)
                    mri_btn.click(brain_mri_interface, [mri_input, name_in, age_in, phone_in, log_state], [mri_img, mri_html, log_state])
                with gr.Tab("Symptom classifier"):
                    with gr.Row():
                        with gr.Column(elem_classes="eh-card"):
                            symptom_input = gr.Textbox(label="Describe your symptoms (English)", lines=5,
                                                       placeholder="e.g. I have joint pain and a skin rash since last week...")
                            symptom_btn = gr.Button("Check symptoms", variant="primary")
                        with gr.Column(elem_classes="eh-card"):
                            symptom_output = gr.HTML(value=EMPTY_HTML)
                    symptom_btn.click(symptom_checker_interface, [symptom_input, name_in, age_in, phone_in, log_state], [symptom_output, log_state])

        # ---------------- Safety lab ----------------
        with gr.Tab("Safety lab"):
            with gr.Row():
                with gr.Column(elem_classes="eh-card"):
                    gr.Markdown("#### Auditor strictness")
                    gr.HTML("<p class='section-hint'>Kitne model score se neeche Wary jawab dene se inkar kare. Zyada strict = kam jawab, lekin zyada mehfooz.</p>")
                    strict = gr.Slider(0.20, 0.90, value=AUDIT_CFG["min_conf"], step=0.05, label="Minimum model score to answer")
                    strict_out = gr.HTML(value=set_strictness(AUDIT_CFG["min_conf"]))
                    strict.change(set_strictness, [strict], [strict_out])
                with gr.Column(elem_classes="eh-card"):
                    gr.Markdown("#### Red-flag self-test")
                    gr.HTML("<p class='section-hint'>Roman Urdu, Urdu aur English emergency cases par Auditor ka recall aur precision.</p>")
                    st_btn = gr.Button("Run safety self-test", variant="secondary")
                    st_out = gr.HTML()
                    st_btn.click(selftest_interface, None, [st_out])
            with gr.Column(elem_classes="eh-card"):
                gr.Markdown("#### Risk-coverage curve")
                gr.HTML("<p class='section-hint'>Apne test set par: model jitne queries ka jawab de (coverage), un mein kitne sahi hain. Isi se threshold chuna jata hai.</p>")
                rc_btn = gr.Button("Plot risk-coverage", variant="secondary")
                with gr.Row():
                    rc_plot = gr.Plot(label="Risk-coverage")
                    rc_table = gr.HTML()
                rc_btn.click(riskcov_interface, None, [rc_plot, rc_table])

        # ---------------- Session report ----------------
        with gr.Tab("Session report"):
            with gr.Column(elem_classes="eh-card"):
                gr.Markdown("#### Session report")
                gr.HTML("<p class='section-hint'>Ek PDF jis mein patient details, triage decision, doctor sign-off aur is session ke saare analyses hote hain.</p>")
                report_btn = gr.Button("Generate full report (PDF)", variant="secondary")
                report_msg = gr.Markdown()
                report_file = gr.File(label="📄 Download report")
                report_btn.click(generate_report, [name_in, age_in, phone_in, log_state], [report_file, report_msg])

    gr.HTML("""<div id='footer'>AI-assisted screening. Not a diagnosis and not a replacement for a doctor.<br>
        Wary &nbsp;|&nbsp; built on the ClarityMed models &nbsp;|&nbsp; Developed by Aila Nasir</div>""")

clean_theme = gr.themes.Base(primary_hue="emerald", neutral_hue="stone").set(
    body_background_fill="#f7f3ea",
    background_fill_primary="#fffdf8", background_fill_secondary="#fffdf8",
    block_background_fill="#fffdf8", block_border_width="0px", block_shadow="none",
    block_label_background_fill="transparent", block_label_border_width="0px",
    block_label_text_color="#14201d", block_label_text_weight="700",
    block_title_background_fill="transparent", block_title_text_color="#14201d",
    panel_background_fill="#fffdf8", panel_border_width="0px",
    input_background_fill="#ffffff", input_border_color="#e4dccb", input_border_width="1px",
    button_primary_background_fill="#0f8b6d", button_primary_background_fill_hover="#0b6b54",
    button_primary_text_color="#ffffff",
)
demo.launch(share=IN_COLAB, debug=IN_COLAB, css=custom_css, theme=clean_theme)
