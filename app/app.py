import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

import faiss
import matplotlib.cm as cm

from openai import OpenAI

APP_TITLE = "Skin Lesion Screening with CNN (EfficientNet-B0)"

IMG_SIZE_DEFAULT = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CHECKPOINT_PATH = os.getenv("MODEL_CKPT", "checkpoints/model.pt")
CALIB_PATH = os.getenv("RECALL_CALIB", "assets/recall_calibration.json")
KB_DIR = os.getenv("KB_DIR", "kb")

DEVICE = "cpu"  # stable for HF Spaces 16GB

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def preprocess_pil(img: Image.Image, img_size: int) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    x = tfm(img.convert("RGB")).unsqueeze(0)  # (1,3,H,W)
    return x

# Load Model
def build_effb0_binary() -> nn.Module:
    m = efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, 1)
    return m


def load_checkpoint_b(model: nn.Module, path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Weights file missing at checkpoints/model.pt"
        )
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError("Checkpoint format mismatch. Expected dict with key 'model_state'.")
    model.load_state_dict(ckpt["model_state"], strict=True)
    return ckpt


@torch.inference_mode()
def predict_prob(model: nn.Module, img: Image.Image, img_size: int) -> float:
    x = preprocess_pil(img, img_size).to(DEVICE)
    logit = model(x).squeeze(1)
    return float(torch.sigmoid(logit).item())


# Target Recall calibration (optional)
@dataclass
class RecallCalibration:
    recall: np.ndarray
    threshold: np.ndarray

    @staticmethod
    def load(path: str) -> Optional["RecallCalibration"]:
        if not os.path.exists(path):
            return None
        obj = json.load(open(path, "r"))
        recall = np.array(obj["recall"], dtype=np.float32)
        thr = np.array(obj["threshold"], dtype=np.float32)
        if recall.ndim != 1 or thr.ndim != 1 or recall.shape[0] != thr.shape[0]:
            raise ValueError("Invalid recall_calibration.json: recall and threshold must be same-length 1D arrays.")
        return RecallCalibration(recall=recall, threshold=thr)

    def threshold_for_target_recall(self, target_recall: float) -> float:
        t = float(np.clip(target_recall, 0.0, 1.0))
        ok = np.where(self.recall >= t)[0]
        if ok.size == 0:
            return float(self.threshold.min())
        
        return float(self.threshold[ok].max())


# Grad-CAM 
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def fwd_hook(_, __, out):
            self.activations = out

        def bwd_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.h1 = target_layer.register_forward_hook(fwd_hook)
        self.h2 = target_layer.register_full_backward_hook(bwd_hook)

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def cam(self, x: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x).squeeze(1)
        score = logit[0]
        score.backward(retain_graph=False)

        acts = self.activations  # (1,C,h,w)
        grads = self.gradients   # (1,C,h,w)
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = grads.mean(dim=(2, 3), keepdim=True)          # (1,C,1,1)
        cam_map = (weights * acts).sum(dim=1, keepdim=False)    # (1,h,w)
        cam_map = torch.relu(cam_map)[0].detach().cpu().numpy()

        cam_map = cam_map - cam_map.min()
        cam_map = cam_map / (cam_map.max() + 1e-8)
        return cam_map


def overlay_heatmap(img: Image.Image, cam_map: np.ndarray, alpha: float) -> Image.Image:
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size

    cam_img = Image.fromarray((cam_map * 255).astype(np.uint8)).resize((w, h), resample=Image.BILINEAR)
    cam_arr = np.asarray(cam_img).astype(np.float32) / 255.0
    heat = cm.get_cmap("jet")(cam_arr)[:, :, :3]  

    base = np.asarray(img_rgb).astype(np.float32) / 255.0
    out = (1 - alpha) * base + alpha * heat
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# Lightweight RAG with simple FAISS vector database
def read_kb_texts(kb_dir: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(kb_dir):
        return []
    pairs = []
    for fn in sorted(os.listdir(kb_dir)):
        if fn.lower().endswith((".md", ".txt")):
            path = os.path.join(kb_dir, fn)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                pairs.append((fn, f.read()))
    return pairs


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text:
        return []
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(n, i + chunk_size)
        out.append(text[i:j])
        i = max(j - overlap, j)
    return out


@dataclass
class RAGIndex:
    chunks: List[str]
    metas: List[Dict[str, Any]]
    index: Any
    dim: int

    def search(self, q_emb: np.ndarray, k: int = 4) -> List[Tuple[str, Dict[str, Any]]]:
        q = q_emb.astype(np.float32)[None, :]
        _, idx = self.index.search(q, k)
        hits = []
        for i in idx[0]:
            if 0 <= i < len(self.chunks):
                hits.append((self.chunks[i], self.metas[i]))
        return hits


def build_rag_index(client: OpenAI, kb_dir: str) -> Optional[RAGIndex]:
    docs = read_kb_texts(kb_dir)
    if not docs:
        return None

    chunks, metas = [], []
    for fn, txt in docs:
        for c in chunk_text(txt):
            chunks.append(c)
            metas.append({"source": fn})

    vecs = []
    batch = 64
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=part)
        vecs.extend([d.embedding for d in resp.data])

    mat = np.array(vecs, dtype=np.float32)
    dim = mat.shape[1]
    faiss.normalize_L2(mat)

    index = faiss.IndexFlatIP(dim)
    index.add(mat)
    return RAGIndex(chunks=chunks, metas=metas, index=index, dim=dim)


def embed_query(client: OpenAI, text: str) -> np.ndarray:
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[text])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    faiss.normalize_L2(v.reshape(1, -1))
    return v


def format_sources(hits: List[Tuple[str, Dict[str, Any]]]) -> str:
    lines = []
    for chunk, meta in hits:
        src = meta.get("source", "kb")
        snippet = chunk.strip().replace("\n", " ")
        lines.append(f"- [{src}] {snippet[:260]}")
    return "\n".join(lines)


class AppState:
    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.cam: Optional[GradCAM] = None
        self.calib: Optional[RecallCalibration] = None
        self.oa: Optional[OpenAI] = None
        self.rag: Optional[RAGIndex] = None
        self.ckpt_meta: Dict[str, Any] = {}

    def init_once(self):
        if self.model is None:
            m = build_effb0_binary()
            ckpt = load_checkpoint_b(m, CHECKPOINT_PATH)
            m.eval().to(DEVICE)
            self.model = m
            self.ckpt_meta = ckpt if isinstance(ckpt, dict) else {}
            self.cam = GradCAM(m, m.features[-1])

        if self.calib is None:
            self.calib = RecallCalibration.load(CALIB_PATH)

        # Only create OpenAI client; DO NOT build embeddings/FAISS here
        if self.oa is None and OPENAI_API_KEY:
            self.oa = OpenAI(api_key=OPENAI_API_KEY)


STATE = AppState()


def smart_threshold(
    mode: str,
    manual_thr: float,
    target_recall: float,
    calib: Optional[RecallCalibration],
) -> Tuple[float, str]:
    manual_thr = float(manual_thr)
    target_recall = float(target_recall)

    if mode == "Target recall":
        if calib is None:
            return manual_thr, "Target recall requested, but calibration file is missing; used manual threshold."
        thr = calib.threshold_for_target_recall(target_recall)
        return float(thr), f"Target recall={target_recall:.2f} mapped to threshold={thr:.4f} using calibration."
    return manual_thr, "Manual threshold used."


def run_prediction(
    image: Image.Image,
    img_size: int,
    mode: str,
    manual_thr: float,
    target_recall: float,
    heat_alpha: float,
) -> Tuple[str, str, float, float, Optional[Image.Image]]:
    """
    Returns:
      - pred_card_md
      - debug_card_md
      - last_prob
      - last_thr
      - heatmap_overlay_img
    """
    try:
        if image is None:
            msg = "<div class='card'><h3>Prediction</h3><span class='err'>Upload an image first.</span></div>"
            dbg = "<div class='card'><h3>System</h3><span class='small'>No image provided.</span></div>"
            return msg, dbg, 0.0, float(manual_thr), None

        STATE.init_once()
        assert STATE.model is not None
        assert STATE.cam is not None

        img_size = int(img_size)
        heat_alpha = float(heat_alpha)

        prob = predict_prob(STATE.model, image, img_size)

        thr, thr_note = smart_threshold(mode, manual_thr, target_recall, STATE.calib)
        label = "malignant" if prob >= thr else "benign"

        # Grad-CAM
        x = preprocess_pil(image, img_size).to(DEVICE)
        x.requires_grad_(True)
        cam_map = STATE.cam.cam(x)
        overlay = overlay_heatmap(image, cam_map, alpha=heat_alpha)

        # Confidence pill (purely descriptive)
        if prob >= max(thr, 0.5):
            conf = "higher"
        elif prob >= thr:
            conf = "borderline"
        else:
            conf = "lower"

        pred_html = f"""
        <div class='card'>
          <h3>Prediction</h3>
          <div>
            <span class='pill'>label: <b>{label}</b></span>
            <span class='pill'>prob: <b>{prob:.4f}</b></span>
            <span class='pill'>threshold: <b>{thr:.4f}</b></span>
            <span class='pill'>confidence: <b>{conf}</b></span>
          </div>
          <div class='hr'></div>
          <div class='small'>{thr_note}</div>
          <div class='small warn'>Probability is a model estimate, not a diagnosis.</div>
        </div>
        """

        ckpt_img_size = STATE.ckpt_meta.get("img_size", None)
        sys_html = f"""
        <div class='card'>
          <h3>System</h3>
          <div class='small'>
            <span class='pill mono'>device={DEVICE}</span>
            <span class='pill mono'>img_size={img_size}</span>
            <span class='pill mono'>ckpt_img_size={ckpt_img_size}</span>
            <span class='pill mono'>calib={'yes' if STATE.calib is not None else 'no'}</span>
            <span class='pill mono'>kb_files={len(read_kb_texts(KB_DIR))}</span>
          </div>
          <div class='hr'></div>
          <div class='small mono'>ckpt_path: {CHECKPOINT_PATH}</div>
        </div>
        """

        return pred_html, sys_html, float(prob), float(thr), overlay

    except Exception as e:
        err_html = f"""
        <div class='card'>
          <h3>Prediction</h3>
          <div class='err'><b>ERROR during inference:</b> <span class='mono'>{type(e).__name__}: {e}</span></div>
          <div class='hr'></div>
          <div class='small'>Check Space logs for the full traceback.</div>
        </div>
        """
        sys_html = f"""
        <div class='card'>
          <h3>System</h3>
          <div class='small mono'>ckpt_path: {CHECKPOINT_PATH}</div>
          <div class='small mono'>ckpt_exists: {os.path.exists(CHECKPOINT_PATH)}</div>
          <div class='small mono'>calib_exists: {os.path.exists(CALIB_PATH)}</div>
          <div class='small mono'>kb_dir_exists: {os.path.isdir(KB_DIR)}</div>
        </div>
        """
        return err_html, sys_html, 0.0, float(manual_thr), None


def chat(
    user_msg: str,
    chat_history: List[Dict[str, str]],  # <-- CHANGED
    last_prob: float,
    last_thr: float,
) -> Tuple[List[Dict[str, str]], str]:
    STATE.init_once()

    msg = (user_msg or "").strip()
    if not msg:
        return chat_history, ""

    if STATE.oa is None:
        chat_history = (chat_history or []) + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": "Set OPENAI_API_KEY in Space Secrets to enable chat."},
        ]
        return chat_history, ""

    # Build RAG lonly when chat is used
    if STATE.rag is None:
        try:
            STATE.rag = build_rag_index(STATE.oa, KB_DIR)
        except Exception:
            STATE.rag = None

    hits = []
    if STATE.rag is not None:
        q = embed_query(STATE.oa, msg)
        hits = STATE.rag.search(q, k=4)

    sources = format_sources(hits)
    system = (
        "You are an educational medical-information assistant. "
        "You do not diagnose. You explain what the model output means and provide general guidance."
    )
    model_context = (
        "Model output for the uploaded image:\n"
        f"- malignant_probability: {float(last_prob):.4f}\n"
        f"- threshold_used: {float(last_thr):.4f}\n"
        f"- predicted_label: {'malignant' if float(last_prob) >= float(last_thr) else 'benign'}\n"
    )
    rag_context = f"Knowledge base excerpts:\n{sources}" if sources else "Knowledge base excerpts: (none)"

    messages = [
        {"role": "system", "content": system},
        {"role": "system", "content": model_context},
        {"role": "system", "content": rag_context},
    ]

    # include last few UI messages (already dicts)
    for m in (chat_history or [])[-12:]:
        if isinstance(m, dict) and "role" in m and "content" in m:
            messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": msg})

    resp = STATE.oa.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content.strip()

    chat_history = (chat_history or []) + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": answer},
    ]
    return chat_history, ""



# Build UI
with gr.Blocks(title=APP_TITLE) as demo:
    gr.HTML(
        f"""
        <div id="titlebar" class="card">
          <h1>{APP_TITLE}</h1>
          <div id="subtitle" class="small">
            Upload an image and get the probability of cancer, see Grad-CAM heatmap of image and ask questions using RAG Chatbot.
          </div>
          <div style="margin-top:10px;">
            <span class="pill">EfficientNet-B0</span>
            <span class="pill">2-stage fine-tune</span>
            <span class="pill">Albumentations train</span>
            <span class="pill">WRS imbalance</span>
            <span class="pill">Grad-CAM</span>
          </div>
        </div>
        """
    )

    with gr.Tab("Screening"):
        with gr.Row():
            with gr.Column(scale=5):
                #gr.HTML(
                #    "<div class='card'><h3>Input</h3><div class='small'>Upload a dermoscopic image. The app runs CPU-only for stability.</div></div>"
                #)
                image_in = gr.Image(type="pil", label="Upload lesion image", height=360)

                with gr.Row():
                    img_size = gr.Slider(128, 512, value=IMG_SIZE_DEFAULT, step=32, label="Image size")
                    heat_alpha = gr.Slider(0.10, 0.80, value=0.45, step=0.05, label="Heatmap overlay strength")

                with gr.Row():
                    mode = gr.Radio(
                        ["Manual threshold", "Target recall"],
                        value="Manual threshold",
                        label="Decision mode",
                    )
                    manual_thr = gr.Slider(0.0, 1.0, value=0.20, step=0.01, label="Manual threshold")

                target_recall = gr.Slider(
                    0.50, 0.99, value=0.85, step=0.01, label="Target recall (default 0.85; needs calibration JSON)"
                )

                run_btn = gr.Button("Run prediction", variant="primary")

            with gr.Column(scale=6):
                pred_card = gr.HTML(
                    "<div class='card'><h3>Prediction</h3><div class='small'></div></div>"
                )
                sys_card = gr.HTML(
                    "<div class='card'><h3>System</h3><div class='small'></div></div>"
                )

                with gr.Row():
                    orig_preview = gr.Image(type="pil", label="Original", height=320)
                    cam_preview = gr.Image(type="pil", label="Grad-CAM overlay", height=320)

                # hidden state for chat
                last_prob = gr.Number(value=0.0, visible=False)
                last_thr = gr.Number(value=0.20, visible=False)

        run_btn.click(
            fn=run_prediction,
            inputs=[image_in, img_size, mode, manual_thr, target_recall, heat_alpha],
            outputs=[pred_card, sys_card, last_prob, last_thr, cam_preview],
        )

        image_in.change(lambda x: x, inputs=[image_in], outputs=[orig_preview])

        '''
        gr.HTML(
            """
            <div class='card'>
              <h3>How Target Recall works</h3>
              <div class='small'>
                If <span class='mono'>assets/recall_calibration.json</span> exists, the app maps a target recall (default 0.85) to a threshold.
                If it doesn’t exist, the app uses the manual threshold and shows a note.
              </div>
            </div>
            """
        )
        '''

    with gr.Tab("Chat (RAG ChatBot)"):
        gr.HTML(
            """
            <div class='card'>
              <h3>Assistant</h3>
              <div class='small'>
                The assistant uses your <span class='mono'>kb/</span> files + the model output (probability/label) as context.
                Add/replace <span class='mono'>kb/*.md</span> or <span class='mono'>kb/*.txt</span> anytime and redeploy.
              </div>
            </div>
            """
        )

        chatbot = gr.Chatbot(label="Chat", height=420)
        with gr.Row():
            msg = gr.Textbox(label="Question", placeholder="e.g., What does a borderline score mean? What should I watch for?")
            send = gr.Button("Send", variant="primary")

        send.click(fn=chat, inputs=[msg, chatbot, last_prob, last_thr], outputs=[chatbot, msg])
'''
        gr.HTML(
            """
            <div class='card'>
              <h3>Space Setup</h3>
              <div class='small'>
                Set <span class='mono'>OPENAI_API_KEY</span> in Space Secrets to enable chat.
                Without it, screening + Grad-CAM still works.
              </div>
            </div>
            """
        )
'''

demo.queue().launch()


