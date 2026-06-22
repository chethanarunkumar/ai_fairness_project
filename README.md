# AI for Detecting Model Bias & Fairness Auditing
**Project Batch No: 56 | Dept. of CSE, MSRUAS, Bengaluru**

## Team
| Name | USN |
|------|-----|
| Chandu A | 22ETIS411001 |
| Chethan B A | 22ETIS411301 |
| Manju V Gowda | 22ETIS411302 |
| Sudeep Kumar S N | 22ETCS002106 |

**Mentor:** Praneetha G N, Asst. Professor, Dept. of CSE

---

## Project Structure
```
ai_fairness_project/
├── app.py                  # Flask application
├── requirements.txt
├── static/
│   ├── css/style.css       # Main stylesheet
│   └── js/main.js          # JavaScript
└── templates/
    ├── base.html           # Shared layout
    ├── index.html          # Home page
    ├── introduction.html
    ├── literature.html     # Literature survey
    ├── problem.html        # Problem statement
    ├── objectives.html
    ├── methodology.html
    ├── architecture.html
    ├── datasets.html
    ├── outcomes.html
    ├── sdg.html            # SDG goals
    ├── timeline.html       # Gantt chart
    ├── conclusion.html
    ├── references.html
    └── dashboard.html      # Live fairness audit demo
```

## Setup & Run
```bash
pip install -r requirements.txt
python app.py
```
Open **http://localhost:5000** in your browser.

## Pages
| Route | Description |
|-------|-------------|
| `/` | Home — project overview, team, module summary |
| `/introduction` | Domain background and motivation |
| `/literature` | Literature survey tables + research gaps |
| `/problem` | Problem statement |
| `/objectives` | Project objectives |
| `/methodology` | 6-module pipeline + fairness metric formulas |
| `/architecture` | System architecture layers |
| `/datasets` | Dataset descriptions |
| `/outcomes` | Expected outcomes |
| `/sdg` | UN Sustainable Development Goals alignment |
| `/timeline` | 16-week Gantt chart |
| `/conclusion` | Project conclusion |
| `/references` | All 10 references in IEEE format |
| `/dashboard` | **Live fairness auditing demo** with charts |

## Live Demo Features (Dashboard)
- Select dataset (Adult Income, COMPAS, German Credit)
- Select sensitive attribute (Gender, Race, Age)
- Choose bias mitigation method
- View: Accuracy, Demographic Parity, Equalized Odds, Disparate Impact, Fairness Score
- Group-level accuracy and FP/FN rate charts
- SHAP feature importance visualization
- Before vs. After mitigation comparison chart and table
