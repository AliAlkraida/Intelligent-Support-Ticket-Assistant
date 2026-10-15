"""Gradio interface comparing the three approaches side by side."""
import html
import json
from functools import lru_cache
import gradio as gr
from src.pipeline import build_assistant

CSS = """
.gradio-container {max-width: 1440px !important; margin: auto !important;}
#hero {padding: 27px 30px; border-radius: 22px; background: linear-gradient(115deg,#c74608,#ee650b 65%,#ff983b); color: white; margin-bottom: 18px; box-shadow: 0 12px 30px rgba(179,73,9,.20);}
#hero h1 {font-size: 2.05rem; margin: 0; color: white; letter-spacing: -.035em;}
#hero p {color:#fff3e9; margin: 7px 0 0;}
#ticket-input textarea {font-size: 1.08rem; line-height: 1.7;}
.model-card {border: 1px solid var(--border-color-primary); border-top: 3px solid #ed6a18; border-radius: 16px; padding: 17px; background: var(--background-fill-secondary); min-height: 230px;}
.model-card h3 {margin: 0 0 10px; font-size: 1.16rem;}
.pill {display:inline-block; border-radius:99px; background:#fff0e3; color:#99400b; padding:4px 9px; margin:2px 5px 5px 0; font-size:.79rem; font-weight:600;}
.reply {white-space:pre-wrap; line-height:1.75; margin: 15px 0; unicode-bidi:plaintext;}
.meta {font-size:.84rem; opacity:.8; margin-top: 10px;}
.err {color:#ba3333; white-space:pre-wrap;}
#analyze-btn {background: linear-gradient(95deg,#e85a0a,#ff871f) !important; border-color:#e85a0a !important; color:white !important; font-weight:700 !important;}
#analyze-btn:hover {filter:brightness(1.07);}
"""

@lru_cache(maxsize=3)
def assistant(name):
    # Cache the local embedding model and index between user requests.
    return build_assistant(name)

def render_result(result, title):
    if result.get("error"):
        return f'<div class="model-card"><h3>{html.escape(title)}</h3><p class="err">{html.escape(result["error"])}</p></div>'
    category = html.escape(str(result.get("category", "unknown")).title())
    priority = html.escape(str(result.get("priority", "normal")).title())
    reply = html.escape(str(result.get("suggested_reply", "")))
    evidence = ", ".join(html.escape(str(x)) for x in result.get("evidence", [])) or "—"
    latency = float(result.get("latency_ms", 0)) / 1000
    escalation = "Human review recommended" if result.get("escalate") else "Standard handling"
    return (f'<div class="model-card"><h3>{html.escape(title)}</h3>'
            f'<span class="pill">{category}</span><span class="pill">{priority} priority</span>'
            f'<p class="reply" dir="auto">{reply}</p><div class="meta">{escalation}</div>'
            f'<div class="meta">Evidence: {evidence}</div><div class="meta">Response time: {latency:.2f}s</div></div>')

def compare(ticket, show_details):
    if not ticket or not ticket.strip():
        msg = '<div class="model-card">Enter a support request to get started.</div>'
        return msg, msg, msg, {}
    results = []
    # Independent failures: one provider error never hides the other approaches.
    for name in ("knowledge_based", "rag", "llm_prompting"):
        try:
            results.append(assistant(name).answer(ticket).to_dict())
        except Exception as exc:
            results.append({"model": name, "error": str(exc)})
    return (render_result(results[0], "Knowledge-based"),
            render_result(results[1], "Retrieval-augmented generation"),
            render_result(results[2], "LLM prompting"),
            {r["model"]: r for r in results} if show_details else {})

with gr.Blocks(title="Support Intelligence | Ticket Assistant") as demo:
    gr.HTML('<div id="hero"><h1>Intelligent Support Ticket Assistant</h1><p>Knowledge-based vs RAG vs LLM prompting · Arabic + English</p></div>')
    with gr.Row():
        with gr.Column(scale=3):
            ticket = gr.Textbox(label="Customer request", lines=4, elem_id="ticket-input", placeholder="Describe the customer issue here... / اكتب استفسار العميل هنا...")
        with gr.Column(scale=1):
            gr.Markdown("**Quick examples**")
            gr.Examples(examples=[
                ["طلبت استرجاع المبلغ من أسبوع وما وصلني للحين، وش الحل؟"],
                ["I was charged twice for the same order. What should I do?"],
                ["وين طلبي؟ الشحنة تأخرت وما وصلت."],
            ], inputs=ticket, label="")
    run = gr.Button("Compare all three approaches", variant="primary", size="lg", elem_id="analyze-btn")
    gr.Markdown("### Response comparison")
    with gr.Row(equal_height=True):
        with gr.Column():
            kb = gr.HTML()
        with gr.Column():
            rag = gr.HTML()
        with gr.Column():
            llm = gr.HTML()
    with gr.Accordion("Advanced details · Technical results", open=False):
        show_details = gr.Checkbox(label="Include structured results", value=True)
        details = gr.JSON(label="Structured output and diagnostics")
    run.click(compare, inputs=[ticket, show_details], outputs=[kb, rag, llm, details], show_progress="full")

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=CSS)
