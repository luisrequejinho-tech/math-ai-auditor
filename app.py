import os
import re
import json
import anthropic
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr

# ── Global client (set once via UI) ──────────────────────────────────────────
client = None

PROBLEM_TYPES = [
    "probability", "integration", "differentiation",
    "linear_algebra", "geometry", "series", "optimization", "other"
]

# ── Core functions (ported directly from notebook) ────────────────────────────

def make_client(api_key: str):
    return anthropic.Anthropic(api_key=api_key.strip())


def generate_solution(client, problem: str) -> str:
    system_prompt = (
        "You are a mathematics tutor. When given a math problem, produce a clear, "
        "step-by-step solution. Show all integration steps explicitly. Use plain text "
        "notation for math (e.g. x^2 for x squared, integral from a to b). Be thorough and precise."
    )
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Solve this problem step by step:\n{problem}"}]
    )
    return message.content[0].text


def audit_solution(client, problem: str, solution: str) -> dict:
    audit_prompt = f"""You are an expert mathematics evaluator. Audit the AI-generated solution below.

RUBRIC (score each 1-5):
1. CORRECTNESS — Are all calculations and final answers mathematically correct?
2. LOGICAL_RIGOR — Are all reasoning steps justified? Any logical gaps?
3. COMPLETENESS — Does the solution address every part of the problem?
4. CLARITY — Are the steps clear enough for a student to follow?

PROBLEM:
{problem}

SOLUTION TO AUDIT:
{solution}

Respond ONLY with a JSON object in exactly this format (no extra text):
{{
  "scores": {{
    "correctness": <1-5>,
    "logical_rigor": <1-5>,
    "completeness": <1-5>,
    "clarity": <1-5>
  }},
  "overall": <average score rounded to 1 decimal>,
  "feedback": {{
    "correctness": "<specific feedback>",
    "logical_rigor": "<specific feedback>",
    "completeness": "<specific feedback>",
    "clarity": "<specific feedback>"
  }},
  "errors_found": ["<list any specific errors, or empty list if none>"],
  "summary": "<2-3 sentence overall assessment>"
}}"""
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": audit_prompt}]
    )
    raw = message.content[0].text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def classify_problem(client, problem: str) -> str:
    classify_prompt = f"""You are a mathematics classifier. Given a math problem, return ONLY one of these labels:
probability, integration, differentiation, linear_algebra, geometry, series, optimization, other

Rules:
- Return the single most appropriate label
- No explanation, no punctuation, just the label
- If the problem involves a PDF/CDF or expected value, return: probability
- If the problem involves finding areas or integrals (no probability), return: integration
- If the problem involves derivatives or rates of change, return: differentiation
- If the problem involves matrices, vectors, or systems of equations, return: linear_algebra
- If the problem involves shapes, angles, or coordinates, return: geometry
- If the problem involves sums, sequences, or convergence, return: series
- If the problem involves maximizing or minimizing a function, return: optimization
- Otherwise return: other

Problem:
{problem}"""
    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": classify_prompt}]
    )
    label = message.content[0].text.strip().lower()
    return label if label in PROBLEM_TYPES else "other"


def generate_visualization_code(client, problem: str, solution: str, problem_type: str) -> str:
    viz_prompt = f"""You are a Python data visualization expert. Generate matplotlib code to illustrate a math problem and its solution.

Problem type: {problem_type}

Problem:
{problem}

Solution:
{solution}

Instructions:
- Generate code for exactly 2 matplotlib subplots: ax1 and ax2
- The variables `fig` and `axes` already exist: `fig, axes = plt.subplots(1, 3, figsize=(16, 5))`
- Use `ax1 = axes[0]` and `ax2 = axes[1]` — do NOT create a new figure
- ax1: illustrate the core mathematical concept (e.g. the function, PDF, shape, vectors)
- ax2: show a numerical verification or second perspective (e.g. Monte Carlo, Riemann sum, convergence)
- Use colors #2563eb (blue) and #dc2626 (red) for consistency
- Add clear titles, axis labels, legends, and grid lines
- Use numpy (imported as np) for all calculations
- Do NOT call plt.show() or plt.savefig()
- Do NOT import anything — numpy as np and matplotlib.pyplot as plt are already imported
- Return ONLY the Python code, no explanation, no markdown fences

Problem type guidance:
- probability: ax1=PDF/CDF plot with mean marked, ax2=Monte Carlo histogram vs theoretical
- integration: ax1=function with shaded area, ax2=Riemann sum approximation converging
- differentiation: ax1=function + derivative, ax2=tangent line at a point
- linear_algebra: ax1=vector plot or transformation, ax2=before/after transformation
- geometry: ax1=shape diagram with labels, ax2=decomposition or proof illustration
- series: ax1=partial sums converging, ax2=terms of the series
- optimization: ax1=objective function with minimum/maximum marked, ax2=gradient descent path
- other: ax1=primary concept illustration, ax2=numerical verification"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": viz_prompt}]
    )
    code = message.content[0].text.strip()
    code = re.sub(r"^```python\s*", "", code)
    code = re.sub(r"^```\s*", "", code)
    code = re.sub(r"\s*```$", "", code)
    return code


def plot_audit_scores(ax, audit: dict):
    categories = ["Correctness", "Logical\nRigor", "Completeness", "Clarity"]
    keys = ["correctness", "logical_rigor", "completeness", "clarity"]
    scores = [audit["scores"][k] for k in keys]
    colors = ["#16a34a" if s >= 4 else "#ca8a04" if s == 3 else "#dc2626" for s in scores]

    bars = ax.bar(categories, scores, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Score (1–5)", fontsize=11)
    ax.set_title(f"Audit Scores\nOverall: {audit['overall']}/5", fontsize=12, fontweight="bold")
    ax.axhline(5, color="#ddd", linestyle="-", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.1,
                str(score), ha="center", va="bottom", fontweight="bold", fontsize=12)
    legend_patches = [
        mpatches.Patch(color="#16a34a", label="Strong (4–5)"),
        mpatches.Patch(color="#ca8a04", label="Adequate (3)"),
        mpatches.Patch(color="#dc2626", label="Weak (1–2)"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="lower right")


def build_figure(client, problem: str, solution: str, problem_type: str, audit: dict):
    viz_code = generate_visualization_code(client, problem, solution, problem_type)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"AI Math Solution Auditor — {problem_type.replace('_', ' ').title()}",
                 fontsize=14, fontweight="bold", y=1.02)
    try:
        exec(viz_code, {"np": np, "plt": plt, "fig": fig, "axes": axes})
    except Exception as e:
        axes[0].set_title("Visualization Error")
        axes[0].text(0.5, 0.5, str(e), ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=9, wrap=True)
        axes[1].set_visible(False)

    plot_audit_scores(axes[2], audit)
    plt.tight_layout()
    return fig


def format_audit_markdown(audit: dict) -> str:
    errors = ", ".join(audit["errors_found"]) if audit["errors_found"] else "None"
    rows = "\n".join([
        f"| Correctness | {audit['scores']['correctness']}/5 | {audit['feedback']['correctness']} |",
        f"| Logical Rigor | {audit['scores']['logical_rigor']}/5 | {audit['feedback']['logical_rigor']} |",
        f"| Completeness | {audit['scores']['completeness']}/5 | {audit['feedback']['completeness']} |",
        f"| Clarity | {audit['scores']['clarity']}/5 | {audit['feedback']['clarity']} |",
        f"| **Overall** | **{audit['overall']}/5** | |",
    ])
    return f"""## Audit Report

| Category | Score | Feedback |
|---|---|---|
{rows}

**Errors Found:** {errors}

**Summary:** {audit['summary']}"""


# ── Gradio pipeline ───────────────────────────────────────────────────────────

def run_pipeline(api_key: str, problem: str):
    if not api_key.strip():
        yield "❌ Please enter your Anthropic API key.", "", "", None
        return
    if not problem.strip():
        yield "❌ Please enter a math problem.", "", "", None
        return

    try:
        c = make_client(api_key)
    except Exception as e:
        yield f"❌ Invalid API key: {e}", "", "", None
        return

    # Step 1 — solution
    yield "⏳ Generating solution...", "", "", None
    try:
        solution = generate_solution(c, problem)
    except Exception as e:
        yield f"❌ Error generating solution: {e}", "", "", None
        return

    # Step 2 — audit
    yield "⏳ Auditing solution...", solution, "", None
    try:
        audit = audit_solution(c, problem, solution)
    except Exception as e:
        yield f"❌ Error during audit: {e}", solution, "", None
        return

    audit_md = format_audit_markdown(audit)

    # Step 3 — classify
    yield "⏳ Classifying problem and generating visualization...", solution, audit_md, None
    try:
        problem_type = classify_problem(c, problem)
        fig = build_figure(c, problem, solution, problem_type, audit)
    except Exception as e:
        yield f"❌ Error generating visualization: {e}", solution, audit_md, None
        return

    yield f"✅ Done! Problem type: **{problem_type}**", solution, audit_md, fig


# ── UI ────────────────────────────────────────────────────────────────────────

EXAMPLE_PROBLEM = """Let X be a continuous random variable with probability density function:

    f(x) = 3x²  for x in [0, 1]
    f(x) = 0    otherwise

1. Verify that f(x) is a valid PDF.
2. Find the expected value E[X] using integration.
3. Find the variance Var(X) using the formula Var(X) = E[X²] - (E[X])².
4. Verify E[X] numerically using a Monte Carlo simulation with 100,000 samples.

Show all integration steps clearly."""

with gr.Blocks(title="AI Math Solution Auditor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# 🧮 AI Math Solution Auditor
Paste any math problem below. Claude will generate a step-by-step solution, audit it on a structured rubric, and produce a tailored visualization.""")

    with gr.Row():
        api_key = gr.Textbox(
            label="Anthropic API Key",
            placeholder="sk-ant-...",
            type="password",
            scale=1
        )

    problem_input = gr.Textbox(
        label="Math Problem",
        placeholder="Enter your math problem here...",
        lines=8,
        value=EXAMPLE_PROBLEM
    )

    run_btn = gr.Button("▶ Run Auditor", variant="primary")

    status_out = gr.Markdown(label="Status")

    with gr.Tabs():
        with gr.TabItem("📝 Solution"):
            solution_out = gr.Markdown(label="Generated Solution")
        with gr.TabItem("📊 Audit Report"):
            audit_out = gr.Markdown(label="Audit Report")
        with gr.TabItem("📈 Visualization"):
            plot_out = gr.Plot(label="Visualization")

    run_btn.click(
        fn=run_pipeline,
        inputs=[api_key, problem_input],
        outputs=[status_out, solution_out, audit_out, plot_out],
    )

if __name__ == "__main__":
    demo.launch()
