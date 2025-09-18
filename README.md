# DreamRewrite Pro (LLM, AR)
تحليل وإعادة كتابة مقالات تفسير الأحلام وفق People-first + E-E-A-T
- تحليل Rule-based + خيار تحليل LLM (OpenAI)
- إعادة كتابة Rule-based + خيار إعادة كتابة LLM

## التثبيت والتشغيل محليًا
```bash
pip install -r requirements.txt
streamlit run app.py
```

## إعداد مفتاح OpenAI
ضع المفتاح في أسرار ستريملت:
App → Settings → Secrets
```toml
OPENAI_API_KEY = "sk-..."
```
أو أدخله مؤقتًا في الحقل الجانبي داخل التطبيق.

## النشر على Streamlit Community Cloud
1) ارفع المشروع إلى GitHub (branch: main).
2) New app → اختر المستودع → ملف `app.py` → Deploy.
3) من Settings → Secrets: أضف `OPENAI_API_KEY` إن أردت استخدام LLM.

## ملاحظات
- النموذج الافتراضي: `gpt-4o-mini` — يمكنك تغييره من الشريط الجانبي.
- في حال عدم توفر واجهة Responses في حسابك، سيتحول التطبيق تلقائيًا إلى Chat Completions.
