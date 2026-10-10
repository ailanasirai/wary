<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F2E2A,100:0F8B6D&height=210&section=header&text=Wary&fontSize=68&fontColor=ffffff&fontAlignY=38&animation=fadeIn&desc=Roman%20Urdu%20health%20triage%20agents%20that%20know%20when%20not%20to%20answer&descSize=19&descAlignY=60" alt="Wary: Roman Urdu health triage agents that know when not to answer" width="100%">

<br>

![Hackathon](https://img.shields.io/badge/Pak%20Angels%20GenAI%20%26%20Agentic%20AI-Cohort%2011-0F2E2A?style=flat-square)
![Category](https://img.shields.io/badge/Category-Health%20Care-0F8B6D?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research%20prototype-C07A0A?style=flat-square)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-F97316?style=flat-square&logo=gradio&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-4285F4?style=flat-square&logo=googlegemini&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-0F8B6D?style=flat-square)

<br>

[![Watch the demo](https://img.shields.io/badge/Watch%20the%20demo-0F8B6D?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1hN9hPrZRxvbvVBp9uQU2xCWI99vnqXfg/view?usp=sharing)
[![Read the PRD](https://img.shields.io/badge/Read%20the%20PRD-0F2E2A?style=for-the-badge&logo=readthedocs&logoColor=white)](PRD.pdf)
[![Screenshots](https://img.shields.io/badge/Screenshots-0F2E2A?style=for-the-badge&logo=googlephotos&logoColor=white)](#screenshots)
[![Run it](https://img.shields.io/badge/Run%20it-0F2E2A?style=for-the-badge&logo=googlecolab&logoColor=white)](#run-it)

<sub>
<a href="#problem">Problem</a> &nbsp;·&nbsp;
<a href="#why-wary-is-different">Why it is different</a> &nbsp;·&nbsp;
<a href="#how-it-works">How it works</a> &nbsp;·&nbsp;
<a href="#safety-design">Safety design</a> &nbsp;·&nbsp;
<a href="#safety-lab">Safety Lab</a> &nbsp;·&nbsp;
<a href="#screenshots">Screenshots</a> &nbsp;·&nbsp;
<a href="#limitations">Limitations</a>
</sub>

</div>

<br>

Wary is a multi-agent health triage prototype built for the Pak Angels GenAI & Agentic AI Hackathon (Cohort 11, Health Care). It reads a complaint written in Roman Urdu, Urdu or English, runs it through four agents, and returns a cautious screening result, an abstention, or an emergency escalation. A doctor signs off on every case.

> [!WARNING]
> Wary is a research prototype and a screening aid. It is not a diagnosis tool and must not be used for real medical decisions.

## Demo

<p align="center">
  <a href="https://drive.google.com/file/d/1hN9hPrZRxvbvVBp9uQU2xCWI99vnqXfg/view?usp=sharing">
    <img src="assets/Thumbwary.jpg" alt="Watch the Wary demo video" width="90%">
  </a>
  <br>
  <sub>Click the image to watch the demo video (opens Google Drive).</sub>
</p>

## At a glance

| | |
|---|---|
| **Input** | Free-text complaint in Roman Urdu, Urdu or English |
| **Pipeline** | Intake, Specialist, Safety Auditor and Report agents |
| **Outcomes** | EMERGENCY, ABSTAIN, PRIORITY REVIEW or ROUTINE REVIEW |
| **Human in the loop** | Doctor sign-off on every case: Approved, Modified or Rejected |
| **Outputs** | PDF report and JSON audit trail |
| **Event** | Pak Angels GenAI & Agentic AI Hackathon, Cohort 11, Health Care |

## Problem

- Health chatbots can sound confident when they are wrong. An independent 2026 evaluation reported in Nature Medicine found that ChatGPT Health under-triaged more than half of the cases it was given (as reported by The Guardian, 26 Feb 2026).
- Most tools are built for English, while many people in Pakistan write symptoms in Roman Urdu or mixed Urdu and English.
- Most symptom checkers leave no audit trail, so a clinician cannot see why a decision was made.

## Solution

Wary puts a Safety Auditor between the model and the patient. The auditor is allowed to say "I should not answer this". Emergencies are caught by deterministic rules, uncertain cases are not guessed, and every case ends with a doctor sign-off and an audit trail.

Wary evolved from my earlier project ClarityMed (chest X-ray, brain MRI and symptom models). Those classifiers are reused. The agent layer, Safety Auditor, Safety Lab, audit trail and the interface are new work for this hackathon.

## Why Wary is different

- **It is allowed to say no.** The Safety Auditor can abstain instead of guessing.
- **Emergencies never depend on an LLM.** Red flags are detected by rules.
- **Honest about uncertainty.** Scores are shown as raw model scores, not calibrated probabilities, and the predicted condition is hidden from the patient when the decision is EMERGENCY or ABSTAIN.
- **Extra care for higher-risk ages.** Stricter confidence threshold for age under 5 and 65 and above.
- **Built for how people write.** Roman Urdu, Urdu and English input.
- **Always reviewable.** Every case ends with a doctor sign-off and an audit trail.

## How it works

```mermaid
flowchart LR
    A["Patient complaint<br/>Roman Urdu, Urdu or English"] --> B["1. Intake agent"]
    B --> C["2. Specialist agent<br/>top 3 conditions with raw scores"]
    X["Optional evidence<br/>chest X-ray, brain MRI"] -.-> C
    B --> D{"3. Safety Auditor"}
    C --> D
    D -->|red flag| E["EMERGENCY"]
    D -->|unsure| F["ABSTAIN"]
    D -->|imaging flagged| G["PRIORITY REVIEW"]
    D -->|checks pass| H["ROUTINE REVIEW"]
    E --> I["4. Report agent"]
    F --> I
    G --> I
    H --> I
    I --> J["Doctor sign-off<br/>Approved, Modified or Rejected"]
    J --> K["PDF report and JSON audit trail"]

    classDef io fill:#FFFDF8,stroke:#6A6F68,color:#14201D
    classDef agent fill:#E3F4EE,stroke:#0F8B6D,color:#0F2E2A,stroke-width:1.5px
    classDef gate fill:#0F2E2A,stroke:#0F2E2A,color:#FFFFFF
    classDef risk fill:#FDE8E1,stroke:#D9482B,color:#8A2A15
    classDef warn fill:#FBF0D6,stroke:#C07A0A,color:#6B4306
    classDef ok fill:#E3F4EE,stroke:#0F8B6D,color:#0F2E2A
    class A,X,J,K io
    class B,C,I agent
    class D gate
    class E risk
    class F,G warn
    class H ok
```

### The four agents

| # | Agent | What it does |
|:-:|---|---|
| 1 | **Intake** | Roman Urdu, Urdu or English text to structured symptoms (rule lexicon, optional Gemini refinement) |
| 2 | **Specialist** | Symptom model returns the top 3 conditions with raw scores. X-ray and MRI modules (Grad-CAM) add optional evidence |
| 3 | **Safety Auditor** | Deterministic checks: emergency red flags, low confidence, close calls, thin information, age risk, flagged imaging. Output: EMERGENCY, ABSTAIN, PRIORITY REVIEW or ROUTINE REVIEW |
| 4 | **Report** | Roman Urdu patient message and English doctor handoff note |

## Safety design

- Red flags are detected by rules, never by an LLM.
- Emergency and abstain messages are fixed templates.
- When the decision is EMERGENCY or ABSTAIN, the predicted condition is hidden from the patient.
- Stricter confidence threshold for age under 5 and 65 and above.
- Scores are shown as raw model scores, not calibrated probabilities.

## Safety Lab

- Adjustable abstention threshold
- Red-flag self-test: 30 hand-written Roman Urdu, Urdu and English cases, with negations. This is a demo check, not a clinical validation.
- Risk-coverage curve on the held-out symptom test set

## Screenshots

### Triage console

<table>
  <tr>
    <td width="50%"><img src="assets/1.png" alt="Home screen with patient details"><br><sub><b>1.</b> Home screen with patient details and the agent flow.</sub></td>
    <td width="50%"><img src="assets/2.png" alt="A no-answer case"><br><sub><b>2.</b> A no-answer case: the Safety Auditor refuses to guess and sends the patient to a doctor.</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="assets/3.png" alt="Agent pipeline"><br><sub><b>3.</b> The four agents with timing and a short summary of what each one produced.</sub></td>
    <td width="50%"><img src="assets/4.png" alt="Doctor sign-off and audit trail"><br><sub><b>4.</b> Doctor handoff note, doctor sign-off and the JSON audit trail.</sub></td>
  </tr>
</table>

<details>
<summary><b>Evidence modules</b> (chest X-ray, brain MRI, symptom classifier)</summary>
<br>
<table>
  <tr>
    <td width="50%"><img src="assets/5.png" alt="Chest X-ray with Grad-CAM"><br><sub><b>5.</b> Chest X-ray module with a Grad-CAM overlay.</sub></td>
    <td width="50%"><img src="assets/6.png" alt="Brain MRI module"><br><sub><b>6.</b> Brain MRI module.</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="assets/7.png" alt="Symptom classifier"><br><sub><b>7.</b> Symptom classifier carried over from ClarityMed.</sub></td>
    <td width="50%"></td>
  </tr>
</table>
</details>

<details>
<summary><b>Safety Lab and report</b></summary>
<br>
<table>
  <tr>
    <td width="50%"><img src="assets/8.png" alt="Safety Lab"><br><sub><b>8.</b> Safety Lab: abstention threshold and red-flag self-test.</sub></td>
    <td width="50%"><img src="assets/9.png" alt="Risk-coverage curve"><br><sub><b>9.</b> Risk-coverage curve on the held-out symptom test set.</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="assets/10.png" alt="PDF session report"><br><sub><b>10.</b> Session report exported as a PDF.</sub></td>
    <td width="50%"></td>
  </tr>
</table>
</details>

## Run it

1. Open `Wary.ipynb` in Google Colab (GPU recommended for the imaging models).
2. Run the cells that train or load the symptom, chest X-ray and brain MRI models.
3. Run `wary_agents.py` as one cell, then `wary_app.py` as the next cell. This launches the Gradio app.
4. Optional: add `GEMINI_API_KEY` in Colab Secrets. Never put keys in code. Without it Wary falls back to rules.

<details>
<summary><b>Repository layout</b></summary>

```
wary/
  README.md
  LICENSE
  requirements.txt
  Wary.ipynb         training and evaluation notebook
  wary_agents.py     Intake, Specialist, Safety Auditor, Report, self-test
  wary_app.py        Gradio interface
  PRD.pdf            product requirements document
  assets/       Thumbwary.jpg and 1.png to 10.png
```
</details>

## Limitations

- Prototype, not validated on real patient data.
- The symptom model is trained on an English public dataset, so Roman Urdu support depends on a limited lexicon.
- Red-flag rules are keyword based and can miss unusual phrasing.
- Model scores are raw, not calibrated.

<details>
<summary><b>Roadmap</b></summary>
<br>

1. Evaluate on a larger Roman Urdu health-query benchmark with an emergency category
2. Doctor-reviewed red-flag lexicon
3. Calibrated scores and age-specific thresholds
4. Hugging Face Spaces deployment
5. Voice input
</details>

## Author

**Aila Nasir.** Built with Python, scikit-learn, PyTorch, Gemini and Gradio.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-aila--nasir-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aila-nasir)
[![GitHub](https://img.shields.io/badge/GitHub-ailanasirai-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/ailanasirai)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F8B6D,100:0F2E2A&height=90&section=footer" alt="" width="100%">
