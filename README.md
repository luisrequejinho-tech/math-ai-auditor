# 🧮 Math AI Auditor

An AI-powered tool that generates, evaluates, and visualizes step-by-step solutions to math problems using Claude.

## What It Does

1. **Generates** a step-by-step solution to any math problem
2. **Audits** the solution against a structured rubric (correctness, logical rigor, completeness, clarity)
3. **Visualizes** the mathematics with plots tailored to the problem type

## Supported Problem Types

- Probability & statistics
- Integration
- Differentiation
- Linear algebra
- Geometry
- Series & sequences
- Optimization

## How to Run Locally

```bash
git clone https://github.com/yourname/math-ai-auditor
cd math-ai-auditor
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860` in your browser.

## How to Use

1. Enter your [Anthropic API key](https://console.anthropic.com/settings/keys)
2. Paste a math problem
3. Click **Run Auditor**

## Files

| File | Description |
|---|---|
| `app.py` | Gradio web app |
| `math_auditor.ipynb` | Original Jupyter notebook |
| `requirements.txt` | Python dependencies |

## Built With

- [Claude](https://anthropic.com) — solution generation, auditing, and visualization code
- [Gradio](https://gradio.app) — web interface
- [Matplotlib](https://matplotlib.org) — visualizations
- [NumPy](https://numpy.org) — numerical computation

## Live Demo

[Hugging Face Space](https://huggingface.co/spaces/yourname/math-ai-auditor)
