# ==========================================================================
# CELL A: Wary Agents  (NAYA CELL: Gradio wale cell se PEHLE run karo)
# Is cell ko chalane ke liye sirf ye chahiye: vectorizer + text_model (symptom model)
# jo tumhare notebook mein pehle se bane hue hain. Koi nayi library nahi.
# ==========================================================================
import re, time, json, uuid
import numpy as np

# ---------- Safety Auditor ki settings (inhe risk_coverage() se tune karo, neeche dekho) ----------
AUDIT_CFG = dict(
    min_conf=0.50,          # is se kam model score = abstain
    min_margin=0.10,        # top-1 aur top-2 ka farq is se kam = abstain (confusing case)
    vulnerable_conf=0.65,   # bachon (<5) aur buzurgon (>=65) ke liye zyada strict
    vulnerable_young=5,
    vulnerable_old=65,
    min_info_nnz=3,         # model ko kitne known words mile (bohat kam = maloomat kam)
)

# ---------- Red flags: deterministic rules, LLM par depend NAHI karte ----------
# (key, English label, Roman Urdu label, [regex patterns on lowercase text])
RED_FLAGS = [
    ("chest_pain", "chest pain or pressure", "seene mein dard ya dabao", [
        r"\b(seene|seena|sine|sina|chhati|chati|dil)\b.{0,25}\b(dard|dabao|bhari|bhaari|bojh|pain)\b",
        r"\bchest\b.{0,25}\b(pain|pressure|tight\w*|crush\w*)\b", r"\bheart attack\b",
        r"سینے.{0,15}(درد|دباؤ)", r"دل کا دورہ"]),
    ("breathing", "severe difficulty breathing", "saans lene mein shadeed mushkil", [
        r"\b(saans|sans)\b.{0,25}\b(phool\w*|ruk\w*|band|mushkil|takleef|tang)\b",
        r"\b(saans|sans)\s+(nahi|nahin|nhi)\s+(aa|aati|aata|le|li)\w*",
        r"\b(short(ness)? of breath|can'?t breathe|cannot breathe|difficulty breathing|trouble breathing|breathless\w*|gasping)\b",
        r"سانس.{0,15}(نہیں|پھول|مشکل|تکلیف)"]),
    ("stroke", "possible stroke signs", "stroke ki alamat (chehra tirha, zubaan larkharana, ek taraf kamzori)", [
        r"\b(chehra|chehre|munh|muh|face)\b.{0,25}\b(tirha|terha|tedha|latk\w*|drooping|droop\w*|numb|sun)\b",
        r"\b(zubaan|zuban|bolne)\b.{0,25}\b(lar\w*|ruk\w*|nahi|slur\w*|mushkil)\b",
        r"\b(ek taraf|one side)\b.{0,25}\b(kamzori|kamzor|sun|numb|weak\w*|paralys\w*|falij)\b",
        r"\b(stroke|falij|laqwa|lakwa|paralys\w*|slurred speech)\b",
        r"(چہرہ|منہ).{0,12}(ٹیڑھا|ٹیڑھ)", r"زبان.{0,12}(لڑکھڑا|بند)"]),
    ("bleeding", "heavy or unusual bleeding", "bohat ya ghair mamooli khoon behna", [
        r"\b(khoon|blood)\b.{0,25}\b(ulti|ultiyan|vomit\w*|nahi ruk\w*|bohat|bahut|zyada|heavy|profuse)\b",
        r"\b(bohat|bahut|zyada|heavy)\b.{0,15}\b(khoon|blood)\b", r"\bkhoon\b.{0,15}\b(beh|bah|nikal)\w*",
        r"\b(vomit\w* blood|khooni ulti|blood in (stool|vomit)|kala pakhana|black stool)\b",
        r"خون.{0,12}(الٹی|بہہ|بہ )"]),
    ("unconscious", "loss of consciousness", "behoshi", [
        r"\b(behosh|bayhosh|hosh\s*(nahi|nahin|kho)\w*|unconscious|fainted|passed out|collapsed|unresponsive)\b",
        r"بے ?ہوش"]),
    ("seizure", "seizure or convulsions", "daura ya jhatke", [
        r"\b(mirgi|seizure|convuls\w*|jhatke|jhatka|daura par\w*|dora par\w*)\b", r"(مرگی|جھٹکے)"]),
    ("allergy", "severe allergic reaction", "shadeed allergy (zubaan/hont/gale mein sujan)", [
        r"\b(allergic reaction|anaphyla\w*|throat (swelling|closing)|zubaan (sooj|suj)\w*|hont (sooj|suj)\w*)\b"]),
    ("poisoning", "poisoning or overdose", "zehar ya zyada dawa kha lena", [
        r"\b(zehr|zeher|zahar|poison\w*|overdose|kerosene)\b", r"زہر"]),
    ("pregnancy", "pregnancy with bleeding or severe pain", "hamal mein khoon ya shadeed dard", [
        r"\b(hamal|hamla|haamla|pregnan\w*)\b.{0,40}\b(khoon|bleed\w*|bohat dard|shadeed dard|severe pain)\b",
        r"\b(khoon|bleed\w*)\b.{0,40}\b(hamal|hamla|haamla|pregnan\w*)\b"]),
    ("self_harm", "thoughts of self-harm", "khud ko nuqsan pohanchane ke khayalat", [
        r"\b(khudkushi|khud kushi|suicide|suicidal|jaan dena|apni jaan|marna chahta|marna chahti|mar jana chahta|mar jana chahti|kill myself|end my life|want to die)\b",
        r"خودکشی"]),
]
_RF_COMPILED = [(k, en, ur, [re.compile(p, re.I) for p in pats]) for k, en, ur, pats in RED_FLAGS]

def _norm(text):
    t = str(text or "").lower()
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()

_NEG_AFTER = re.compile(r"^\s*(hai\s+|hain\s+)?(nahi|nahin|nhi)\b")
_NEG_BEFORE = re.compile(r"\b(no|not|without|never|nahi|nahin|nhi)\s+(\w+\s+)?$")

def detect_red_flags(text):
    """Emergency alamat dhoondta hai. Simple negation ("dard nahi hai", "no chest pain") ko ignore karta hai,
    lekin shak ho to ESCALATE karta hai (false alarm theek hai, missed emergency nahi)."""
    t = _norm(text)
    found, seen = [], set()
    for key, en, ur, pats in _RF_COMPILED:
        for p in pats:
            m = p.search(t)
            if not m or key in seen:
                continue
            after, before = t[m.end():m.end() + 14], t[max(0, m.start() - 14):m.start()]
            if _NEG_AFTER.match(after) or _NEG_BEFORE.search(before):
                continue
            seen.add(key)
            found.append(dict(key=key, label=en, label_ur=ur, snippet=m.group(0)[:60]))
    return found

# ---------- Roman Urdu -> English symptom lexicon (symptom model English par train hua hai) ----------
LEXICON = [
    (r"\b(sar|sir)\s*(mein|me|main|ma)?\s*dard\b|\b(sar|sir)dard\b|\bheadache\b", "headache"),
    (r"\b(bukhar|bukhaar|bokhar|fever|temperature)\b", "fever"),
    (r"\b(khansi|khaansi|khasi|cough\w*)\b", "cough"),
    (r"\b(zukam|zukaam|nazla|nazle|runny nose|cold)\b", "cold and runny nose"),
    (r"\b(gala|gale|galay)\s*(mein|me|main|ma)?\s*(kharab|dard|kharash|kharaash|sooj\w*)\b|\bsore throat\b|\bthroat pain\b", "sore throat"),
    (r"\b(pait|pet|peit)\s*(mein|me|main|ma)?\s*(dard|marror|maror)\b|\b(stomach|abdominal|belly)\s*(pain|ache)\b|\bstomach ache\b", "stomach pain"),
    (r"\b(ulti|ultiyan|ultee|qay|vomit\w*)\b", "vomiting"),
    (r"\b(matli|mitli|nausea|nauseous)\b", "nausea"),
    (r"\b(dast|patle pakhane|loose motions?|diarrh?oea|diarrhea)\b", "diarrhea"),
    (r"\b(qabz|kabz|constipation)\b", "constipation"),
    (r"\b(jism|badan|body)\s*(mein|me|main|ma)?\s*(dard|pain|ache\w*)\b|\bbody ache\w*\b", "body aches"),
    (r"\b(jod|jodon|jodo|joron|joro|joints?)\s*(mein|me|main|ma|ka|ke|ki)?\s*(dard|pain|akar\w*|stiff\w*|swelling)\b|\bjoint pain\b", "joint pain"),
    (r"\b(kamar|kamr|back)\s*(mein|me|main|ma)?\s*(dard|pain|ache)\b", "back pain"),
    (r"\b(kamzori|kamzor|weakness|weak)\b", "weakness"),
    (r"\b(thakan|thakawat|thaka|fatigue|tired\w*|exhaust\w*)\b", "fatigue"),
    (r"\b(chakkar|chakar|dizz\w*|vertigo)\b", "dizziness"),
    (r"\b(khujli|kharish|kharash|itch\w*)\b", "itching"),
    (r"\b(daane|dane|dhabbe|dhabe|rash\w*)\b", "skin rash"),
    (r"\b(peshab|pishab)\s*(mein|me|main|ma)?\s*(jalan|dard|takleef)\b|\bburning urination\b|\bpainful urination\b", "burning urination"),
    (r"\b(bar bar|baar baar|frequent)\s*(peshab|pishab|urinat\w*)\b", "frequent urination"),
    (r"\b(pyaas|pyas|excessive thirst)\b", "excessive thirst"),
    (r"\b(bhook|bhuk)\s*(nahi|nahin|kam|nhi)\b|\bloss of appetite\b", "loss of appetite"),
    (r"\b(peeli|peela|pili|pila)\s*(aankh\w*|ankh\w*|rang|jild|skin)\b|\byellow(ish)? (eyes|skin)\b|\bjaundice\b", "yellow skin and eyes"),
    (r"\b(seene|sine|seena)\s*(mein|me|main|ma)?\s*jalan\b|\b(khatti dakar|khatti dakaar|dakar|acidity|heartburn|acid reflux)\b", "heartburn and acid reflux"),
    (r"\b(kapkapi|kampkampi|kanpna|chills?|shiver\w*)\b", "chills"),
    (r"\b(pasina|pasine|sweating|sweats?)\b", "sweating"),
    (r"\b(kaan|kan)\s*(mein|me|main|ma)?\s*dard\b|\bear ?ache\b", "ear pain"),
    (r"\b(daant|dant)\s*(mein|me|main|ma)?\s*dard\b|\btooth ?ache\b", "toothache"),
    (r"\b(naak|naq)\s*(band|behna|beh)\w*|\bnasal congestion\b|\bblocked nose\b", "nasal congestion"),
    (r"\b(seene|sine|seena|chest)\s*(mein|me|main|ma)?\s*(dard|pain|tight\w*)\b", "chest pain"),
    (r"\b(saans|sans)\s*(phool\w*|ki takleef|lene mein)\b|\bshort(ness)? of breath\b|\bbreathless\w*\b|\bwheez\w*\b", "shortness of breath"),
    (r"\b(muhasay|muhase|pimples?|acne)\b", "acne"),
    (r"\b(sujan|sooj\w*|swell\w*)\b", "swelling"),
    (r"\b(nind nahi|neend nahi|insomnia)\b", "insomnia"),
]
_LEX_COMPILED = [(re.compile(p, re.I), en) for p, en in LEXICON]
_UR_MARKERS = {"mein", "mujhe", "hai", "hain", "raha", "rahi", "bohat", "bahut", "aur", "nahi", "kal", "din", "mera", "meri",
               "ho", "gaya", "gayi", "se", "hua", "hui", "thora", "thori", "bhi"}
_DUR = re.compile(r"(\d+)\s*(din|days?|hafta|haftay|hafte|weeks?|mahina|mahine|months?|ghante|hours?)", re.I)

def _rule_intake(text):
    t = _norm(text)
    phrases = []
    for rx, en in _LEX_COMPILED:
        if rx.search(t) and en not in phrases:
            phrases.append(en)
    toks = t.split()
    if re.search(r"[\u0600-\u06FF]", str(text or "")):
        lang = "urdu"
    elif sum(1 for w in toks if w in _UR_MARKERS) >= 2:
        lang = "roman_urdu"
    else:
        lang = "english"
    dur = _DUR.search(t)
    return dict(symptoms=phrases, lang=lang, duration=(f"{dur.group(1)} {dur.group(2)}" if dur else None),
                english_text=(t + " " + ", ".join(phrases)).strip(), source="rules")

def _parse_json(txt):
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    return json.loads(txt)

def intake_agent(text, gemini=None):
    """Agent 1: Roman Urdu / Urdu / English complaint ko structured symptoms mein badalta hai.
    Gemini ho to usse bhi use karta hai, warna rules. Red flags yahan se NAHI nikalte (Auditor raw text dekhta hai)."""
    base = _rule_intake(text)
    if gemini is None:
        return base
    try:
        prompt = ("Convert this patient message (Roman Urdu, Urdu, English or mixed) to JSON ONLY, no markdown, with keys: "
                  "english_text (plain English description of the symptoms only, no diagnosis), "
                  "symptoms (list of short English symptom phrases), duration (string or null). "
                  "Do not diagnose or add symptoms that are not mentioned.\nMessage: " + str(text)[:1000])
        data = _parse_json(gemini.generate_content(prompt).text)
        eng = str(data.get("english_text", ""))[:500].strip()
        syms = [str(s).strip().lower() for s in (data.get("symptoms") or []) if str(s).strip()][:15]
        if eng:
            merged = list(dict.fromkeys(base["symptoms"] + syms))
            base.update(english_text=(eng + " " + ", ".join(merged)).strip(), symptoms=merged,
                        duration=data.get("duration") or base["duration"], source="gemini+rules")
    except Exception as e:
        print("Intake Gemini fallback to rules:", type(e).__name__)
    return base

def specialist_agent(english_text, vectorizer, text_model, k=3):
    """Agent 2: tumhara trained symptom model. Top-k predictions + raw scores."""
    vec = vectorizer.transform([english_text])
    proba = text_model.predict_proba(vec)[0]
    order = np.argsort(proba)[::-1][:k]
    return dict(top=[(str(text_model.classes_[i]), float(proba[i])) for i in order], nnz=int(vec.nnz))

def audit(raw_text, intake, spec, age, imaging_log=None, cfg=None):
    """Agent 3: SAFETY AUDITOR. Faisla karta hai ke jawab dena hai, abstain karna hai ya emergency escalate karni hai."""
    cfg = {**AUDIT_CFG, **(cfg or {})}
    flags = detect_red_flags(raw_text)
    reasons = []
    top = spec["top"]
    conf = top[0][1]
    margin = conf - top[1][1] if len(top) > 1 else conf
    age_n = int(str(age).strip()) if str(age).strip().isdigit() else None
    vulnerable = age_n is not None and (age_n < cfg["vulnerable_young"] or age_n >= cfg["vulnerable_old"])
    need = cfg["vulnerable_conf"] if vulnerable else cfg["min_conf"]
    imaging_risk = [e for e in (imaging_log or []) if e.get("module") in ("Chest X-ray", "Brain MRI") and e.get("state") == "risk"]

    if flags:
        for f in flags:
            reasons.append(("red_flag", f"Red flag: {f['label']} (matched '{f['snippet']}')"))
        decision = "EMERGENCY"
    else:
        if spec["nnz"] < cfg["min_info_nnz"] and len(intake["symptoms"]) < 2:
            reasons.append(("low_info", "Too little symptom information to reason safely"))
        if conf < need:
            code = "vulnerable_age" if (vulnerable and conf >= cfg["min_conf"]) else "low_conf"
            msg = (f"Model score {conf*100:.0f}% is below the required {need*100:.0f}%" +
                   (" (stricter threshold for age under 5 or 65 and above)" if vulnerable else ""))
            reasons.append((code, msg))
        if margin < cfg["min_margin"]:
            reasons.append(("low_margin", f"Top-2 conditions are too close ({top[0][0]} vs {top[1][0]}, gap {margin*100:.0f} points)"))
        if imaging_risk:
            reasons.append(("imaging", "Imaging flagged in this session: " + ", ".join(f"{e['module']}: {e['result']}" for e in imaging_risk)))
        blocking = [r for r in reasons if r[0] in ("low_info", "low_conf", "vulnerable_age", "low_margin")]
        if blocking:
            decision = "ABSTAIN"
        elif imaging_risk:
            decision = "PRIORITY_REVIEW"
        else:
            decision = "ROUTINE_REVIEW"
    return dict(decision=decision, flags=flags, reasons=reasons, conf=conf, margin=margin, need_conf=need,
                vulnerable=vulnerable, show_prediction=decision in ("ROUTINE_REVIEW", "PRIORITY_REVIEW"))

_REASON_UR = {
    "low_info": "maloomat kafi nahi hain",
    "low_conf": "model apne andaze par pur-yaqeen nahi hai",
    "vulnerable_age": "is umar mein hum zyada ehtiyat rakhte hain",
    "low_margin": "do ya zyada bimariyan ek jaisi lag rahi hain",
}

def report_agent(decision, spec, intake, raw_text, name, age):
    """Agent 4: mareez ke liye Roman Urdu paighaam + doctor ke liye handoff note.
    Emergency/abstain wala text template hai (LLM se nahi), taake safety message kabhi 'behek' na jaye."""
    d = decision["decision"]
    top = spec["top"]
    if d == "EMERGENCY":
        labels = ", ".join(f["label_ur"] for f in decision["flags"])
        msg = (f"Aap ki batayi hui alamat ({labels}) emergency ho sakti hain. Model ka jawab ya report ka intezar na karein. "
               "Abhi kisi ko saath lekar qareebi hospital ke emergency mein jayen, ya Rescue 1122 par call karein (agar aap ke ilaqe mein available ho).")
        if all(f["key"] == "self_harm" for f in decision["flags"]):
            msg = ("Aap ne jo likha woh bohat ahem hai, aur aap akele nahi hain. Abhi kisi bharosemand insaan ko apne paas bula lein aur "
                   "qareebi hospital ya doctor se foran rabta karein. Agar aap ko lagta hai ke aap khud ko nuqsan pohancha sakte hain, "
                   "to Rescue 1122 par call karein ya emergency mein jayen (agar aap ke ilaqe mein available ho).")
        elif any(f["key"] == "self_harm" for f in decision["flags"]):
            msg += (" Aap ne khud ko nuqsan pohanchane ka bhi zikr kiya hai, aur ye baat bohat ahem hai. Aap akele nahi hain: "
                    "abhi kisi bharosemand insaan ko apne paas bula lein.")
    elif d == "ABSTAIN":
        why = "; ".join(dict.fromkeys(_REASON_UR[c] for c, _ in decision["reasons"] if c in _REASON_UR))
        msg = (f"Wary is waqt andaza nahi de raha, kyunke {why}. Behtar hai ke doctor se mashwara karein. "
               "Agar aap chahein to kab se hai, kitna bukhar hai, umar aur doosri alamat likh kar dobara try karein.")
    else:
        label, sc = top[0]
        extra = " Imaging ke natayej ki wajah se is case ko doctor ki priority review mein rakha gaya hai." if d == "PRIORITY_REVIEW" else ""
        msg = (f"Aap ki batayi hui alamat ke hisaab se model ka mushtabah andaza: {label} (raw score {sc*100:.0f}%). "
               f"Ye tashkhees nahi hai, sirf screening hai. Final raay doctor ke review ke baad hi hogi.{extra}")
    shown = ("; ".join(f"{l} {s*100:.0f}%" for l, s in top) if decision["show_prediction"]
             else "WITHHELD from patient (internal only): " + "; ".join(f"{l} {s*100:.0f}%" for l, s in top))
    note = "\n".join([
        "WARY AGENT HANDOFF NOTE",
        f"Patient: {name or 'n/a'} | Age: {age or 'n/a'}",
        f"Complaint (verbatim): {str(raw_text).strip()[:400]}",
        f"Language: {intake['lang']} | Duration: {intake['duration'] or 'not stated'}",
        f"Extracted symptoms: {', '.join(intake['symptoms']) or 'none recognised'} (intake: {intake['source']})",
        f"Specialist model top-3 (raw scores, not calibrated): {shown}",
        f"Safety Auditor decision: {d}",
        "Auditor reasons: " + (" | ".join(r for _, r in decision["reasons"]) or "none"),
        "Doctor review: PENDING. A licensed clinician must confirm, modify or reject before any action.",
    ])
    return msg, note

def run_triage(raw_text, name, age, imaging_log, vectorizer, text_model, gemini=None, cfg=None):
    """Orchestrator: 4 agents ek ke baad ek, har step ka time aur output trace mein."""
    trace = []
    def step(agent, fn, summarize):
        t0 = time.perf_counter()
        out = fn()
        trace.append(dict(agent=agent, ms=int((time.perf_counter() - t0) * 1000), summary=summarize(out)))
        return out
    intake = step("1. Intake agent", lambda: intake_agent(raw_text, gemini),
                  lambda o: f"Language: {o['lang']}. Symptoms: {', '.join(o['symptoms']) or 'none recognised'}. Duration: {o['duration'] or 'not stated'}. Source: {o['source']}.")
    spec = step("2. Specialist agent", lambda: specialist_agent(intake["english_text"], vectorizer, text_model),
                lambda o: "Top-3 (raw scores): " + "; ".join(f"{l} {s*100:.0f}%" for l, s in o["top"]))
    dec = step("3. Safety Auditor", lambda: audit(raw_text, intake, spec, age, imaging_log, cfg),
               lambda o: f"Decision: {o['decision']}. " + (" | ".join(r for _, r in o["reasons"]) or "All safety checks passed; doctor sign-off still required."))
    msg, note = step("4. Report agent", lambda: report_agent(dec, spec, intake, raw_text, name, age),
                     lambda o: "Patient message (Roman Urdu) and doctor handoff note generated. Waiting for doctor sign-off.")
    return dict(id=uuid.uuid4().hex[:8], decision=dec, intake=intake, spec=spec, trace=trace, patient_message=msg, doctor_note=note)

# ---------- SELF-TEST: Safety Auditor ka red-flag recall/precision (hand-written demo cases) ----------
# NOTE: ye 30 cases demo ke liye mene khud likhe hain. Asli number ke liye apne Roman Urdu benchmark ki
# Emergency_RedFlag category (aur baaqi categories as negatives) yahan plug karo.
AUDIT_TESTS = [
    ("seene mein dard hai aur saans phool rahi hai", True),
    ("I have crushing chest pain spreading to my arm", True),
    ("meri maa behosh ho gayi hain", True),
    ("mere walid ka chehra tirha ho gaya aur zubaan lar kharane lagi", True),
    ("bacha ko daura para aur jhatke aa rahe hain", True),
    ("khoon ki ulti ho rahi hai", True),
    ("bohat khoon beh raha hai", True),
    ("hamal mein khoon aa raha hai aur bohat dard hai", True),
    ("bache ne ghalti se zehr pi liya", True),
    ("I can't breathe properly", True),
    ("sans nahi aa rahi", True),
    ("mujhe lagta hai mein khudkushi kar loon", True),
    ("sudden weakness on one side of the body and slurred speech", True),
    ("سینے میں درد ہے", True),
    ("mujhe halka bukhar aur khansi hai", False),
    ("sar dard aur chakkar kal se", False),
    ("pait dard aur ulti 2 din se", False),
    ("jodon mein dard subah ke waqt", False),
    ("gala kharab aur zukam hai", False),
    ("kamzori aur thakan mehsoos hoti hai", False),
    ("badan par daane aur khujli", False),
    ("peshab mein jalan hai", False),
    ("I have a mild headache and a runny nose", False),
    ("seene mein jalan aur khatti dakar", False),
    ("mujhe saans ki koi takleef nahi hai, bas zukam hai", False),
    ("no chest pain, only a sore throat", False),
    ("mujhe seene mein dard nahi hai bas khansi hai", False),
    ("pet mein dard hai aur dast", False),
    ("skin par rash aur itching", False),
    ("neend nahi aati aur sar dard", False),
]

def run_audit_selftest():
    tp = fp = fn = tn = 0
    misses, false_alarms = [], []
    for text, expect in AUDIT_TESTS:
        got = bool(detect_red_flags(text))
        if expect and got: tp += 1
        elif expect and not got: fn += 1; misses.append(text)
        elif not expect and got: fp += 1; false_alarms.append(text)
        else: tn += 1
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    return dict(n=len(AUDIT_TESTS), tp=tp, fp=fp, fn=fn, tn=tn, recall=recall, precision=precision,
                misses=misses, false_alarms=false_alarms)

# ---------- THRESHOLD TUNING: risk-coverage table (apne test set par, ek line se) ----------
def risk_coverage(text_model, X_vec, y_true, grid=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    """Har threshold par: model kitne % queries ka jawab deta hai (coverage) aur un jawabon mein kitne sahi hain.
    Isi se AUDIT_CFG['min_conf'] chuno aur CV/LinkedIn ke liye ye table use karo."""
    proba = text_model.predict_proba(X_vec)
    pred = np.asarray(text_model.classes_)[proba.argmax(1)]
    conf = proba.max(1)
    y = np.asarray(y_true)
    print(f"{'threshold':>9} | {'answered':>8} | {'accuracy on answered':>20}")
    rows = []
    for t in grid:
        m = conf >= t
        cov = float(m.mean())
        acc = float((pred[m] == y[m]).mean()) if m.any() else float("nan")
        rows.append((t, cov, acc))
        print(f"{t:>9.2f} | {cov*100:>7.1f}% | {acc*100:>19.1f}%")
    return rows

print("Cell A ready: run_triage, audit, run_audit_selftest, risk_coverage")
