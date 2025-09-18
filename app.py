
# DreamRewrite Pro (LLM) — Arabic Streamlit app for Dream Interpretation Articles
# - Rule-based analysis + optional OpenAI LLM analysis
# - Rule-based rewrite + optional OpenAI LLM rewrite
#
# Run locally:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Deploy on Streamlit Cloud:
#   Push to GitHub, then New app → pick repo → app.py

import re
import json
import textwrap
from datetime import datetime
from collections import Counter, defaultdict

import streamlit as st

# ============ Optional: OpenAI LLM ============
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI
except Exception:
    OPENAI_AVAILABLE = False

# -----------------------
# Helpers (Arabic-aware)
# -----------------------
AR_SENT_SPLIT = re.compile(r"[\.!؟\?\n]+")
AR_WS = re.compile(r"\s+")
AR_PUNCT = re.compile(r"[\,\.;:!؟\-—\(\)\[\]\{\}\"\'«»…]")

def normalize_text(s: str) -> str:
    s = s.replace("\u200f", "").replace("\u200e", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_sentences(s: str):
    s = normalize_text(s)
    parts = [p.strip() for p in AR_SENT_SPLIT.split(s) if p.strip()]
    return parts

def tokenize(s: str):
    s = AR_PUNCT.sub(" ", s)
    toks = [t for t in AR_WS.split(s) if t]
    return toks

# -----------------------
# Analysis (rule-based)
# -----------------------
def readability_metrics(text: str):
    sents = split_sentences(text)
    toks = tokenize(text)
    avg_sent_len = (sum(len(tokenize(s)) for s in sents) / max(1, len(sents))) if sents else 0
    long_sentences = [s for s in sents if len(tokenize(s)) > 30]
    return {
        "sentences": len(sents),
        "tokens": len(toks),
        "avg_sentence_tokens": round(avg_sent_len, 2),
        "long_sentence_count": len(long_sentences),
    }

def repetition_flags(text: str):
    sents = split_sentences(text)
    seen = Counter(sents)
    repeated_sents = [s for s, c in seen.items() if c > 1]
    words = tokenize(text)
    grams = [" ".join(words[i:i+5]) for i in range(max(0, len(words)-4))]
    gcount = Counter(grams)
    repeated_ngrams = [g for g, c in gcount.items() if c > 1 and len(g.split()) == 5]
    return {
        "repeated_sentence_samples": repeated_sents[:3],
        "repeated_5gram_samples": repeated_ngrams[:5],
    }

def heading_structure_flags(text: str):
    lines = [l.strip() for l in text.splitlines()]
    h2 = [l for l in lines if l.startswith("## ") or l.startswith("### ")]
    has_faq_section = any("FAQ" in l or "أسئلة" in l for l in lines)
    return {
        "heading_lines_found": len(h2),
        "has_faq_section": has_faq_section,
    }

def disclaimer_present(text: str):
    patterns = [r"اجتهاد", r"قد", r"يُحتمل", r"ليست بديلًا", r"تنبيه", r"ظروف", r"حسب السياق"]
    score = sum(bool(re.search(p, text)) for p in patterns)
    return {"has_disclaimer_signals": score >= 2, "signals_matched": score}

def keyword_stats(text: str, primary_kw: str, related_kws: list[str]):
    toks = tokenize(text)
    total = len(toks) or 1
    pk_count = sum(1 for t in toks if primary_kw and primary_kw in t)
    rel_counts = {rk: sum(1 for t in toks if rk and rk in t) for rk in related_kws}
    return {
        "total_tokens": total,
        "primary_kw_count": pk_count,
        "primary_kw_density": round(pk_count / total, 4),
        "related_counts": rel_counts,
    }

def analyze_text_rule(text: str, primary_kw: str, related_kws: list[str]):
    metrics = readability_metrics(text)
    reps = repetition_flags(text)
    heads = heading_structure_flags(text)
    disc = disclaimer_present(text)
    kws = keyword_stats(text, primary_kw, related_kws)

    issues = []
    if metrics["avg_sentence_tokens"] > 25:
        issues.append("جُمل طويلة؛ يفضّل تقصير الجُمل وجعلها مباشرة.")
    if metrics["long_sentence_count"] > 0:
        issues.append("هناك جُمل طويلة جدًا؛ قسّمها إلى جملتين أو ثلاث.")
    if reps["repeated_sentence_samples"]:
        issues.append("تكرار جُمل بعينها؛ أعد الصياغة وتخلّص من التكرار.")
    if reps["repeated_5gram_samples"]:
        issues.append("تكرار عبارات (n-grams) يوحي بحشو؛ قلّل التكرار.")
    if heads["heading_lines_found"] < 2:
        issues.append("هيكل ضعيف؛ أضف عناوين فرعية واضحة (H2/H3).")
    if not heads["has_faq_section"]:
        issues.append("لا توجد أسئلة شائعة حقيقية داخل الصفحة.")
    if not disc["has_disclaimer_signals"]:
        issues.append("أضف تنبيهًا مسؤولًا وصياغة احتمالية (سلامة الادعاءات).")
    if kws["primary_kw_density"] > 0.02:
        issues.append("كثافة الكلمة المفتاحية مرتفعة؛ خفّض التكرار لتجنّب الحشو.")

    report = {
        "readability": metrics,
        "repetition": reps,
        "structure": heads,
        "disclaimer": disc,
        "keywords": kws,
        "issues": issues,
        "score_people_first": _score_people_first(metrics, reps, heads, disc),
    }
    return report

def _score_people_first(metrics, reps, heads, disc):
    score = 0
    if metrics["avg_sentence_tokens"] <= 22: score += 6
    if metrics["long_sentence_count"] == 0: score += 2
    if not reps["repeated_sentence_samples"]: score += 4
    if heads["heading_lines_found"] >= 3: score += 4
    if disc["has_disclaimer_signals"]: score += 4
    return min(score, 20)

# -----------------------
# Rewrite (rule-based)
# -----------------------
def top_sentences(text: str, primary_kw: str, related_kws: list[str], k=3):
    sents = split_sentences(text)
    scores = []
    for s in sents:
        sc = 0
        if primary_kw:
            sc += s.count(primary_kw) * 3
        for rk in related_kws:
            sc += s.count(rk)
        sc += min(3, len(tokenize(s)) // 12)
        scores.append((sc, s))
    scores.sort(reverse=True, key=lambda x: x[0])
    picked = [s for _, s in scores[:k] if s]
    return picked

def soften_sentence(s: str):
    if re.search(r"(سيحدث|أكيد|حتماً|بالضرورة)", s):
        s = re.sub(r"(سيحدث|أكيد|حتماً|بالضرورة)", "قد", s)
    if not re.search(r"قد|يُحتمل|ربما|حسب", s):
        s = "قد " + s
    return s

def build_tldr(orig_text: str, primary_kw: str, related_kws: list[str]):
    picks = top_sentences(orig_text, primary_kw, related_kws, k=2)
    if not picks:
        picks = [f"{primary_kw} في الأحلام قد يدل على معانٍ مختلفة حسب سياق الرائي."] if primary_kw else ["الأحلام تُفسَّر وفق السياق الشخصي."]
    picks = [soften_sentence(normalize_text(p)) for p in picks]
    tldr = "\n".join([f"- {p.strip()}" for p in picks])
    if primary_kw:
        tldr += f"\n- يختلف معنى {primary_kw} باختلاف الحالة والمشاعر والمكان، والتفسير اجتهادي."
    return tldr

def plan_related_distribution(related_kws: list[str]):
    buckets = [
        "## المعنى الإجمالي",
        "## حسب حالة الرائي/الرائية",
        "## بحسب السياق (المشاعر/المكان/الفعل)",
        "## ما الذي لا يعنيه هذا الحلم؟",
        "## نصائح عملية عند القلق",
        "## أسئلة شائعة (FAQ)",
    ]
    plan = defaultdict(list)
    for i, rk in enumerate(related_kws):
        plan[buckets[i % len(buckets)]].append(rk)
    return plan

def rewrite_article_rule(orig_text: str, primary_kw: str, related_kws: list[str]):
    primary_kw = (primary_kw or "").strip()
    related_kws = [rk.strip() for rk in (related_kws or []) if rk.strip()]
    now = datetime.now().date().isoformat()
    rel_plan = plan_related_distribution(related_kws)
    tldr = build_tldr(orig_text or "", primary_kw, related_kws)

    h1 = f"تفسير {primary_kw} في المنام: الدلالات الأهم وكيف يختلف المعنى بالسياق" if primary_kw else "تفسير الأحلام: الدلالات الأهم وكيف يختلف المعنى بالسياق"

    parts = []
    parts.append(f"# {h1}")
    if primary_kw:
        parts.append(f"**TL;DR:**\n{tldr}")

    rel_here = ", ".join(rel_plan["## المعنى الإجمالي"]) if rel_plan else ""
    parts.append("## المعنى الإجمالي")
    p1 = (
        f"{primary_kw} قد يعبّر عن دلالاتٍ مختلفة تبعًا للسياق الشخصي للرائي، مثل ظروفه الحالية ومشاعره وقت الحلم. "
        f"يُحتمل أن ترتبط الدلالة برغبات أو مخاوف، ولا يُفهم المعنى على أنه نتيجة حتمية."
    ) if primary_kw else (
        "قد تعكس الرموز في الأحلام رغبات أو مخاوف أو أحداثًا قريبة من الوعي، وتتغيّر دلالتها حسب الشخص والسياق."
    )
    if rel_here:
        p1 += f" كما قد تتقاطع مع رموز مرتبطة مثل: {rel_here}."
    parts.append(p1)

    rel_here = ", ".join(rel_plan["## حسب حالة الرائي/الرائية"]) if rel_plan else ""
    parts.append("## حسب حالة الرائي/الرائية")
    parts.append(
        textwrap.dedent(
            f"""
            - **العزباء:** قد يشير ظهور {primary_kw or "الرمز"} إلى دلالات مرتبطة بالاستقلال أو الترقّب، ويختلف بحسب مشاعرها وظروفها.
            - **المتزوجة:** يُحتمل أن يرتبط بالسعي إلى التوازن والأمان، أو بإدارة المسؤوليات.
            - **الحامل:** قد يتصل بالطمأنينة أو القلق من التغيّر، وتفسيره متبدّل حسب الشعور المرافق.
            - **الرجل:** يُحتمل أن يعكس أهدافًا عملية أو مخاوفًا ظرفية.
            """
        ).strip()
    )
    if rel_here:
        parts.append(f"(رموز مرتبطة في هذا القسم: {rel_here}).")

    rel_here = ", ".join(rel_plan["## بحسب السياق (المشاعر/المكان/الفعل)"]) if rel_plan else ""
    parts.append("## بحسب السياق (المشاعر/المكان/الفعل)")
    parts.append(
        textwrap.dedent(
            f"""
            - **مشاعر إيجابية:** قد تدل على فرصٍ أو طمأنينة مؤقتة.
            - **مشاعر سلبية:** يُحتمل أن تعكس قلقًا يحتاج فهم أسبابه الواقعية.
            - **في البيت/العمل:** يتبدّل المعنى تبعًا لدلالات المكان وارتباطاته اليومية.
            - **يعطي/يأخذ:** قد يتغيّر التأويل وفق الفعل والعلاقة بين الأطراف.
            """
        ).strip()
    )
    if rel_here:
        parts.append(f"(رموز مرتبطة في هذا القسم: {rel_here}).")

    rel_here = ", ".join(rel_plan["## ما الذي لا يعنيه هذا الحلم؟"]) if rel_plan else ""
    parts.append("## ما الذي لا يعنيه هذا الحلم؟")
    parts.append(
        f"لا يعني ظهور {primary_kw or 'الرمز'} نتائج مستقبلية مؤكدة، ولا يُستخدم كأساس لقراراتٍ مصيرية. "
        f"التفسير احتمالي ويتبدّل حسب الشخص والظروف، والمعلومة هنا معرفية لا علاجية."
    )
    if rel_here:
        parts.append(f"(للاطلاع: رموز مرتبطة شائعة: {rel_here}).")

    rel_here = ", ".join(rel_plan["## نصائح عملية عند القلق"]) if rel_plan else ""
    parts.append("## نصائح عملية عند القلق")
    parts.append(
        "- دوِّن الحلم ومشاعرك فور الاستيقاظ.\n"
        "- اربط بين تفاصيل الحلم وظروفك الواقعية.\n"
        "- خفّف التفسير الحرفي، وفكّر في الدلالات الرمزية.\n"
        "- إن استمر القلق وظهر أثره على يومك، استشر مختصًا نفسيًا مؤهلًا."
    )
    if rel_here:
        parts.append(f"(قد تُفيدك مراجعة رموز: {rel_here}).")

    rel_here = ", ".join(rel_plan["## أسئلة شائعة (FAQ)"]) if rel_plan else ""
    parts.append("## أسئلة شائعة (FAQ)")
    faq = [
        f"**هل يختلف تفسير {primary_kw or 'الرمز'} حسب الحالة؟** — نعم، يختلف تبعًا لحالة الرائي/الرائية والمشاعر والسياق.",
        f"**هل لون/مكان ظهور {primary_kw or 'الرمز'} يغيّر المعنى؟** — قد يبدّل الدلالة تبعًا لارتباط اللون/المكان بخبرة الشخص.",
    ]
    parts.append("\n\n".join(faq))
    if rel_here:
        parts.append(f"(أسئلة مرتبطة: {rel_here}).")

    parts.append("## المراجع")
    parts.append("- [اكتب هنا الكتب/الطبعات/الصفحات إن وُجدت]\n- [ميّز رأي المحرر عن النقل]")

    parts.append("\n---\n**مربع الثقة:** الكاتب • المراجع (reviewedBy) • آخر تحديث: " + datetime.now().date().isoformat() + " • منهجية التفسير • سياسة التصحيح")

    parts.append(
        "> **تنبيه:** تفسير الأحلام مجال اجتهادي، تتغيّر دلالاته باختلاف الشخص والظروف."
        " المعلومات هنا لأغراض معرفية وليست بديلًا عن نصيحة دينية أو نفسية أو طبية."
    )

    doc = "\n\n".join(parts)

    stats = keyword_stats(doc, primary_kw, related_kws)
    if stats["primary_kw_density"] > 0.015:
        doc = doc.replace(primary_kw, "هذا الرمز", max(0, int(stats["primary_kw_count"] * 0.25)))

    return doc

# -----------------------
# OpenAI LLM helpers
# -----------------------
def get_openai_client(provided_key: str | None):
    if not OPENAI_AVAILABLE:
        raise RuntimeError("حزمة openai غير مثبّتة. تحقّق من requirements.txt")
    api_key = provided_key or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("لم يتم العثور على OPENAI_API_KEY في الأسرار أو الحقل.")
    client = OpenAI(api_key=api_key)
    return client

LLM_GUIDELINES = """
أنت محرر عربي يلتزم بـ People-first + E-E-A-T لمقالات تفسير الأحلام.
المطلوب: تحليل النص ثم إعادة كتابته وفق النقاط:
- مقدمة TL;DR (3–4 أسطر) تشرح المعنى الأشيع + متى يختلف + تنبيه للسياق.
- تغطية الحالات: (العزباء/المتزوجة/الحامل/الرجل).
- بحسب السياق: (مشاعر، مكان، فعل).
- قسم "ما الذي لا يعنيه الحلم؟" لتبديد المعتقدات الشائعة.
- نصائح عملية عند القلق + متى أستشير مختصًا.
- FAQ داخلي (أسئلة حقيقية متوقعة من القارئ).
- مراجع (إن وُجدت) + تمييز رأي المحرر عن النقل.
- مربع الثقة (الكاتب/المراجع/آخر تحديث/منهجية/سياسة التصحيح) + تنبيه مسؤول.
- صياغة احتمالية (قد/يُحتمل/بحسب السياق)، واحترام الحساسية الدينية والثقافية.
- لا ادعاءات قاطعة أو تشخيص أو تنبؤات.
- توزيع الكلمة المفتاحية والكلمات المرتبطة طبيعيًا (دون حشو)، واستهدف كثافة تقريبية 0.8%–1.5% للكلمة الأساسية، واذكر كل كلمة مرتبطة مرة على الأقل في قسم مناسب.
- عناوين H2/H3 واضحة، وجُمل قصيرة مفهومة.
- لا تُنشئ مراجع غير موجودة؛ اترك مكانًا للمصدر إن لم يتوفر.
أنتج الناتج بترميز Markdown فقط.
"""

def llm_analyze_text(client, model: str, orig: str, primary_kw: str, related_kws: list[str], temperature: float = 0.2):
    prompt = f"""
حلّل النص التالي وفق People-first/E-E-A-T، وارصد أكبر 10 مشكلات قابلة للإصلاح فورًا.
أعد النتيجة كقائمة عناصر مرقّمة (+ اقتراح إصلاح لكل مشكلة).
الكلمة المفتاحية: {primary_kw}
الكلمات المرتبطة: {", ".join(related_kws)}
النص:
\"\"\"
{orig}
\"\"\"
"""
    try:
        # Prefer Responses API
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=1200,
        )
        return resp.output_text
    except Exception:
        # Fallback: Chat Completions API
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "خبير تحرير عربي لمقالات تفسير الأحلام."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content

def llm_rewrite(client, model: str, orig: str, primary_kw: str, related_kws: list[str], temperature: float = 0.3):
    prompt = f"""{LLM_GUIDELINES}

أعد كتابة المقال التالي بالكامل بحيث يلتزم بكل المتطلبات أعلاه.
- الكلمة المفتاحية الأساسية: {primary_kw}
- الكلمات المرتبطة: {", ".join(related_kws)}
- أدرج TL;DR في الأعلى ثم بقية الأقسام بالترتيب المقترح.
- احرص على التضمين الطبيعي للكلمات دون حشو.
- أختم بتنبيه مسؤول.
النص الأصلي:
\"\"\"
{orig}
\"\"\"
"""
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=2200,
        )
        return resp.output_text
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": "محرر عربي يكتب بأسلوب واضح ويراعي E-E-A-T وسلامة الادعاءات."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content

# -----------------------
# JSON-LD generator
# -----------------------
def build_jsonld(url: str, site_name: str, primary_kw: str, author: str, reviewer: str, image_url: str = ""):
    today = datetime.now().date().isoformat()
    data = {
        "@context": "https://schema.org",
        "@type": "Article",
        "inLanguage": "ar",
        "headline": f"تفسير {primary_kw} في المنام: الدلالات الأهم وكيف يختلف المعنى بالسياق" if primary_kw else "تفسير الأحلام",
        "mainEntityOfPage": {"@type": "WebPage", "@id": url or ""},
        "author": {"@type": "Person", "name": author or ""},
        "reviewedBy": {"@type": "Person", "name": reviewer or ""},
        "publisher": {"@type": "Organization", "name": site_name or ""},
        "datePublished": today,
        "dateModified": today,
        "image": [image_url] if image_url else [],
        "articleSection": ["تفسير الأحلام"],
        "about": [{"@type": "Thing", "name": primary_kw}] if primary_kw else [],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="DreamRewrite Pro (LLM) — مُحلّل ومُعيد كتابة", page_icon="💤", layout="wide")
st.title("🛠️ DreamRewrite Pro (LLM) — مُحلّل ومُعيد كتابة مقالات تفسير الأحلام")
st.caption("أدخل النص + الكلمة المفتاحية + الكلمات المرتبطة → تحليل تلقائي + إعادة كتابة متوافقة مع People-first وE-E-A-T (مع خيار LLM).")

with st.sidebar:
    st.header("⚙️ الإعدادات")
    primary_kw = st.text_input("الكلمة المفتاحية الأساسية", value="المال في المنام")
    related_raw = st.text_area("كلمات مرتبطة (سطر لكل كلمة)", value="إعطاء المال\nتجميع المال\nالذهب\nالدَّين")
    related_kws = [r.strip() for r in related_raw.splitlines() if r.strip()]
    url = st.text_input("رابط الصفحة (اختياري للـ JSON-LD)", value="")
    site_name = st.text_input("اسم الموقع (لـ JSON-LD)", value="")
    author = st.text_input("اسم الكاتب", value="")
    reviewer = st.text_input("اسم المراجع (reviewedBy)", value="")
    image_url = st.text_input("رابط صورة المقال (اختياري)", value="")

    st.markdown("---")
    st.subheader("🧠 طبقة LLM (OpenAI)")
    llm_enabled = st.checkbox("تفعيل LLM", value=False)
    default_model = "gpt-4o-mini"
    model = st.text_input("اسم الموديل", value=default_model, help="مثال: gpt-4o-mini أو gpt-4.1-mini أو gpt-5-mini (إن كان متاحًا)")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    api_key_ui = st.text_input("OPENAI_API_KEY (اختياري هنا إذا لم تضعه في Secrets)", type="password")

col1, col2 = st.columns([1,1])

with col1:
    st.subheader("1) النص الأصلي")
    orig = st.text_area("الصق النص هنا", height=260, placeholder="ألصق المقال المراد تحليله وإعادة كتابته…")
    analyze_clicked = st.button("🔎 تحليل المشاكل (Rule)")
    analyze_llm_clicked = st.button("🧠 تحليل المشاكل (LLM)")

with col2:
    st.subheader("2) النتائج")
    if analyze_clicked and orig.strip():
        rep = analyze_text_rule(orig, primary_kw, related_kws)
        st.markdown("### تقرير Rule-based")
        st.write({
            "قراءة": rep["readability"],
            "هيكل": rep["structure"],
            "تكرار": rep["repetition"],
            "سلامة/تنبيه": rep["disclaimer"],
            "كلمات مفتاحية": rep["keywords"],
            "نقاط People-first (→/20)": rep["score_people_first"],
        })
        if rep["issues"]:
            st.warning("\n".join(f"- {i}" for i in rep["issues"]))
        else:
            st.success("لا مشاكل جوهرية مكتشفة بالتحليل القاعدي.")

    if analyze_llm_clicked and orig.strip():
        if not llm_enabled:
            st.error("فعّل LLM من الشريط الجانبي وأدخل مفتاح OPENAI_API_KEY.")
        else:
            try:
                client = get_openai_client(api_key_ui)
                diag = llm_analyze_text(client, model, orig, primary_kw, related_kws, temperature=temperature)
                st.markdown("### تقرير LLM")
                st.write(diag)
            except Exception as e:
                st.error(f"فشل استدعاء LLM: {e}")

st.markdown("---")

st.subheader("3) إعادة كتابة متوافقة مع المتطلبات")
colx, coly = st.columns([1,1])

with colx:
    if st.button("✍️ إعادة كتابة (Rule)"):
        rewritten = rewrite_article_rule(orig or "", primary_kw, related_kws)
        st.session_state["rewritten_doc"] = rewritten
        st.session_state["jsonld"] = build_jsonld(url, site_name, primary_kw, author, reviewer, image_url)

    if st.button("🧠 إعادة كتابة (LLM)"):
        if not llm_enabled:
            st.error("فعّل LLM من الشريط الجانبي وأدخل مفتاح OPENAI_API_KEY.")
        else:
            try:
                client = get_openai_client(api_key_ui)
                rewritten = llm_rewrite(client, model, orig or "", primary_kw, related_kws, temperature=temperature)
                st.session_state["rewritten_doc"] = rewritten
                st.session_state["jsonld"] = build_jsonld(url, site_name, primary_kw, author, reviewer, image_url)
            except Exception as e:
                st.error(f"فشل استدعاء LLM: {e}")

with coly:
    if "rewritten_doc" in st.session_state:
        st.download_button("⬇️ تنزيل النص كـ .md", data=st.session_state["rewritten_doc"], file_name="rewritten.md", mime="text/markdown")
        st.download_button("⬇️ تنزيل JSON-LD", data=st.session_state["jsonld"], file_name="article.schema.json", mime="application/json")

if "rewritten_doc" in st.session_state:
    st.markdown("### المعاينة النهائية (Markdown)")
    st.code(st.session_state["rewritten_doc"], language="markdown")
    st.markdown("### JSON-LD")
    st.code(st.session_state["jsonld"], language="json")

st.info(
    "ملاحظة: يمكنك استخدام الوضع القاعدي (Rule) بدون أي مفاتيح، أو تفعيل LLM (OpenAI) لتحسين التشخيص وإعادة الكتابة.\n"
    "ضع المفتاح في Secrets باسم OPENAI_API_KEY أو الصقه مؤقتًا في الحقل الجانبي."
)
