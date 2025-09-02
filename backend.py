from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid, time, re, json, os, pathlib
from typing import Dict, List, Any
from datetime import datetime

# ==== Gemini ====
import google.generativeai as genai

# ---------- Config Loader ----------
CONFIG_DIR = pathlib.Path(os.environ.get("CONFIG_DIR", "config")).resolve()

def _read_json(path: pathlib.Path, default: Any):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _read_text(path: pathlib.Path, default: str):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default

def _template(s: str, ctx: dict) -> str:
    # templating sederhana {{a.b}}
    def repl(m):
        parts = m.group(1).split(".")
        val = ctx
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                return m.group(0)
        return str(val)
    return re.sub(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}", repl, s)

class Config:
    def __init__(self):
        self.mtimes = {}
        self.data = {}
        self.reload(force=True)

    def _load_all(self):
        settings = _read_json(CONFIG_DIR / "settings.json", {})
        clinic = settings.get("clinic", {})
        ctx = {"clinic": clinic, "settings": settings}

        crisis = _read_json(CONFIG_DIR / "crisis.json", {"keywords": [], "message": ""})
        crisis["message"] = _template(crisis.get("message",""), ctx)

        faq = _read_json(CONFIG_DIR / "faq.json", {})
        # render template di jawaban FAQ
        for k,v in list(faq.items()):
            faq[k] = _template(str(v), ctx)

        faq_keys = _read_json(CONFIG_DIR / "faq_keys.json", {})
        psy_words = _read_json(CONFIG_DIR / "psy_words.json", [])

        system_prompt = _read_text(CONFIG_DIR / "system_prompt.md", "").strip()
        system_prompt = _template(system_prompt, ctx)

        suggestions = _read_json(CONFIG_DIR / "suggestions.json", {})
        closing = _template(_read_text(CONFIG_DIR / "closing.txt", "").strip(), ctx)

        patterns = _read_json(CONFIG_DIR / "patterns.json", {})
        return {
            "settings": settings,
            "clinic": clinic,
            "crisis": crisis,
            "faq": faq,
            "faq_keys": faq_keys,
            "psy_words": psy_words,
            "system_prompt": system_prompt,
            "suggestions": suggestions,
            "closing": closing,
            "patterns": patterns,
        }

    def reload(self, force=False):
        changed = False
        for fn in ["settings.json","crisis.json","faq.json","faq_keys.json","psy_words.json",
                   "system_prompt.md","suggestions.json","closing.txt","patterns.json"]:
            p = CONFIG_DIR / fn
            mt = p.stat().st_mtime if p.exists() else -1
            if force or self.mtimes.get(fn) != mt:
                self.mtimes[fn] = mt
                changed = True
        if changed or force:
            self.data = self._load_all()

    def ctx(self):  # konteks untuk templating
        return {"clinic": self.data.get("clinic", {}), "settings": self.data.get("settings", {})}

CFG = Config()

# ---------- App ----------
app = FastAPI()

# CORS dari settings
cors = CFG.data["settings"].get("cors_origins", ["http://127.0.0.1:5500","http://localhost:5500"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors,
    allow_credentials=False,
    allow_methods=["GET","POST","PUT","OPTIONS"],
    allow_headers=["*"],
)

# ---------- Util bawaan ----------
def parse_json_obj(s: str) -> dict:
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: return {}
    try: return json.loads(m.group(0))
    except json.JSONDecodeError: return {}

def is_crisis(text: str) -> bool:
    t = (text or "").lower()
    kws = CFG.data["crisis"]["keywords"]
    return any(k in t for k in kws)

def crisis_message() -> str:
    return CFG.data["crisis"]["message"]

def match_faq(text: str):
    t = (text or "").lower()
    faq = CFG.data["faq"]; keys = CFG.data["faq_keys"]
    for k, kws in keys.items():
        if any(w in t for w in kws):
            return faq.get(k)
    return None

def is_psychology_domain(texts: List[str]) -> bool:
    t = " ".join(texts).lower()
    words = CFG.data["psy_words"]
    return any(w in t for w in words)

def _compile_patterns():
    pat = CFG.data["patterns"]
    def c(name):
        s = pat.get(name, "")
        return re.compile(s, re.I) if s else None
    return {
        "greet": c("greeting"),
        "thanks": c("thanks"),
        "ack": c("ack"),
        "who": c("who"),
        "can": c("can"),
        "about": c("about"),
    }
PATS = _compile_patterns()

def is_smalltalk(text: str) -> bool:
    t = (text or "").strip()
    if not t: return True
    return any(p and p.search(t) for p in [PATS["greet"], PATS["thanks"], PATS["ack"]])

def match_meta_query(text: str):
    t = (text or "").strip()
    if not t: return None
    if PATS["who"] and PATS["who"].search(t): return "who"
    if PATS["can"] and PATS["can"].search(t): return "can"
    if PATS["about"] and PATS["about"].search(t): return "about"
    if ("kamu" in t.lower() or "curhatkuy" in t.lower()) and any(w in t.lower() for w in ["siapa","apa","jelaskan","tentang","perkenalkan","kenalan"]):
        if any(w in t.lower() for w in ["bisa","fitur","kemampuan","fungsi"]): return "can"
        if "curhatkuy" in t.lower(): return "about"
        return "who"
    return None

# ---------- Sesi ----------
chat_sessions: Dict[str, dict] = {}
def get_sess(sid: str):
    now = time.time()
    s = chat_sessions.get(sid)
    if not s:
        s = chat_sessions[sid] = {"turns": 0, "texts": [], "ended": False, "ts": now}
    else:
        s["ts"] = now
    # TTL ringan
    if len(chat_sessions) % 100 == 0:
        cutoff = now - 60*60
        for k in list(chat_sessions.keys()):
            if chat_sessions[k]["ts"] < cutoff or chat_sessions[k]["ended"]:
                del chat_sessions[k]
    return s

# ---------- Gemini ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
def gemini_model():
    mdl = CFG.data["settings"].get("model") or os.environ.get("GEMINI_MODEL") or "gemini-1.5-flash"
    sys = CFG.data["system_prompt"]
    return genai.GenerativeModel(model_name=mdl, system_instruction=sys) if GEMINI_API_KEY else None

def gemini_generate(user_text: str) -> str:
    m = gemini_model()
    if not m:
        return "⚠️ Server belum dikonfigurasi dengan GEMINI_API_KEY. Hubungi admin."
    try:
        resp = m.generate_content(user_text)
        return (resp.text or "").strip()
    except Exception as e:
        return f"⚠️ Terjadi kendala saat memproses jawaban: {e}"

# ---------- Routes ----------
@app.get("/")
async def root():
    return {"message": f"{CFG.data['clinic'].get('name','CurhatKuy')} bot aktif.", "config_dir": str(CONFIG_DIR)}

@app.post("/status")
async def status(request: Request):
    CFG.reload()  # hot-reload ringan
    try:
        data = await request.json()
    except Exception:
        data = {}
    session_id = data.get("session_id") or str(uuid.uuid4())
    sess = get_sess(session_id)
    max_turns = int(CFG.data["settings"].get("max_turns", 5))
    remaining = 0 if sess["ended"] else max(0, max_turns - sess["turns"])
    return {"session_id": session_id, "remaining": remaining, "end": sess["ended"]}

@app.post("/chat")
async def chat(request: Request):
    CFG.reload()  # hot-reload ringan
    try:
        data = await request.json()
    except Exception:
        data = {}
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())
    sess = get_sess(session_id)

    max_turns = int(CFG.data["settings"].get("max_turns", 5))
    conf_thr  = float(CFG.data["settings"].get("classify_confidence", 0.4))
    suggestions = CFG.data["suggestions"]
    closing = CFG.data["closing"] or f"Sesi chat berakhir (kebijakan {max_turns} pesan). Kamu bisa mulai sesi baru kapan saja."
    fallback = CFG.data["settings"].get("reply_fallback","Baik, terima kasih sudah berbagi.")

    if sess["ended"]:
        return {"reply": "Sesi ini sudah berakhir. Klik 'Mulai Sesi Baru' untuk memulai lagi.", "end": True, "session_id": session_id, "remaining": 0}

    if is_crisis(user_message):
        sess["ended"] = True
        return {"reply": crisis_message(), "handoff": True, "end": True, "session_id": session_id, "remaining": 0}

    # FAQ (dihitung 1 turn — ubah sesuai kebijakan)
    faq_ans = match_faq(user_message)
    if faq_ans:
        sess["turns"] += 1; sess["texts"].append(user_message)
        remaining = max(0, max_turns - sess["turns"])
        return {"reply": faq_ans, "end": False, "session_id": session_id, "remaining": remaining}

    # Meta (who/can/about) → tidak mengurangi jatah
    meta = match_meta_query(user_message)
    if meta:
        if meta == "who":
            reply = f"### Aku asisten {CFG.data['clinic'].get('name','CurhatKuy')} 🤖\nAku membantu topik **psikologi** dan **FAQ** klinik."
        elif meta == "can":
            reply = "### Yang bisa kulakukan\n1. Menjawab pertanyaan psikologi & langkah awal yang aman.\n2. Menjawab **FAQ** (jam, lokasi, layanan, tarif, booking).\n3. Setelah **5 pesan**, mengklasifikasikan topik & menyarankan tipe psikolog."
        else:
            reply = f"### Tentang {CFG.data['clinic'].get('name','CurhatKuy')}\nLayanan klinik psikologi. Bot membantu menyaring kebutuhan sebelum buat janji."
        remaining = max(0, max_turns - sess["turns"])
        return {"reply": reply, "end": False, "session_id": session_id, "remaining": remaining}

    # Small talk → tidak mengurangi jatah
    if is_smalltalk(user_message):
        reply = ("Halo! 😊\n\nAku siap bantu seputar **psikologi** (kecemasan, tidur anak, hubungan, burnout) "
                 "atau **FAQ** klinik (jam, lokasi, layanan, tarif, booking). Ceritakan singkat yang ingin kamu bahas, ya.")
        remaining = max(0, max_turns - sess["turns"])
        return {"reply": reply, "end": False, "session_id": session_id, "remaining": remaining}

    # Pesan bermakna
    if user_message:
        sess["turns"] += 1
        sess["texts"].append(user_message)

    if not is_psychology_domain(sess["texts"]):
        reply = ("Maaf, aku fokus pada topik psikologi & FAQ klinik. Kalau ada kebutuhan lain, Admin bisa membantu. "
                 "Boleh ceritakan topik psikologi yang kamu pikirkan?")
    else:
        reply = gemini_generate(user_message) or fallback

    remaining = max(0, max_turns - sess["turns"])

    # Tutup sesi bila sudah mencapai max_turns → klasifikasi
    if sess["turns"] >= max_turns:
        combined = " ".join(sess["texts"][-max_turns:])
        cls_prompt = (
            "Klasifikasikan topik obrolan berikut ke salah satu label:\n"
            + ", ".join(list(suggestions.keys()) + ["non_psikologi"]) +
            ".\nBalas ONLY dalam JSON: {\"category\":\"<label>\", \"confidence\": <0..1>}\n\n"
            f"OBROLAN:\n{combined}"
        )
        cat_raw = gemini_generate(cls_prompt) or "{}"
        dataj = parse_json_obj(cat_raw)
        category = dataj.get("category","non_psikologi")
        confidence = dataj.get("confidence", None)
        try:
            confidence = float(confidence) if confidence is not None else None
        except Exception:
            confidence = None

        if category in suggestions and (confidence is None or confidence >= conf_thr):
            reply = f"{reply}\n\nDari obrolan kita, sepertinya kamu cocok berkonsultasi dengan **{suggestions[category]}**. Mau jadwalkan sesi?\n{closing}"
        else:
            reply = f"{reply}\n\n{closing}"

        sess["ended"] = True
        return {"reply": reply, "end": True, "category": category, "confidence": (round(confidence,3) if isinstance(confidence,(int,float)) else None), "session_id": session_id, "remaining": 0}

    return {"reply": reply, "end": False, "session_id": session_id, "remaining": remaining}

# ---------- Endpoint Admin (opsional sederhana; gunakan token) ----------
def _require_admin(req: Request):
    if not CFG.data["settings"].get("admin_enabled", False):
        raise HTTPException(403, "admin endpoints disabled")
    token = req.headers.get("X-Admin-Token") or ""
    want = CFG.data["settings"].get("admin_token","")
    if not want or token != want:
        raise HTTPException(401, "unauthorized")

ALLOWED_FILES = {
    "settings.json","crisis.json","faq.json","faq_keys.json","psy_words.json",
    "system_prompt.md","suggestions.json","closing.txt","patterns.json"
}

@app.get("/admin/config")
async def admin_get_config(request: Request):
    _require_admin(request)
    CFG.reload()
    return CFG.data

@app.post("/admin/reload")
async def admin_reload(request: Request):
    _require_admin(request)
    CFG.reload(force=True)
    return {"reloaded_at": datetime.utcnow().isoformat()+"Z"}

@app.put("/admin/config")
async def admin_put_config(request: Request):
    _require_admin(request)
    payload = await request.json()
    fname = payload.get("file"); data = payload.get("data")
    if fname not in ALLOWED_FILES:
        raise HTTPException(400, f"file not allowed: {fname}")
    path = CONFIG_DIR / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    if fname.endswith(".json"):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        # .md / .txt
        path.write_text(str(data or ""), encoding="utf-8")
    CFG.reload(force=True)
    return {"saved": fname, "mtime": path.stat().st_mtime}
