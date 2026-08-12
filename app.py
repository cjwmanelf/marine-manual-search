# -*- coding: utf-8 -*-
"""
로컬 선박기기 매뉴얼 하이브리드 검색 파이프라인 (Gradio GUI)

- 외부 API / 인터넷 통신 없이 로컬에서만 동작 (모델 가중치는 최초 1회만 다운로드 후 오프라인)
- 실행 환경(CUDA / Apple MPS / CPU) 자동 감지
- 스캔본 / 텍스트본 PDF 모두 지원

하이브리드 검색 = 이미지 검색 + 텍스트 검색
  [이미지] PDF → 페이지 렌더(PyMuPDF)
              → DocLayout-YOLO 로 figure(그림·차트·도면)/table(표) 검출·크롭   [YOLO]
              → 크롭 이미지를 SigLIP2 로 임베딩·색인                            [SigLIP2]
          검색: 질의어 → SigLIP2 텍스트 임베딩 → 코사인 유사도
  [텍스트] PDF 페이지 → PyMuPDF get_text() (텍스트본) / OCR (스캔본, 엔진 있으면)
          검색: 질의어 → 키워드 매칭(문서 전체 Ctrl+F)

탭 구성
  1) 색인 생성(Indexing): PDF 드래그앤드롭 → 이미지+텍스트 동시 색인
  2) 실시간 검색(Search): 질의어로 도면·표(이미지) + 본문(텍스트) 동시 검색
"""

import os

# 콘솔 경고/잡음 억제 (transformers·huggingface_hub 로그) — 무거운 임포트 전에 설정해야 함
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# 단, 모델 다운로드 진행바(%)는 보이도록 강제로 켠다
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

import re
import shutil
import logging
import warnings
import importlib.util
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

print("[준비] 라이브러리를 불러오는 중입니다... (수 초 소요)", flush=True)

import pymupdf as fitz  # PyMuPDF (신 API 이름으로 임포트해 deprecation 경고 방지)
import numpy as np
import torch
import gradio as gr
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
# 다운로드 진행바 활성화 (경고 로그는 끄되, 진행률 표시는 유지)
try:
    hf_logging.enable_progress_bar()
    from huggingface_hub.utils import enable_progress_bars

    enable_progress_bars()
except Exception:  # noqa: BLE001
    pass
print("[준비] 라이브러리 로딩 완료.", flush=True)

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
MODEL_ID = "google/siglip2-base-patch16-224"

# 다국어 의미 검색용 텍스트 임베딩 모델 (bge-m3: 100여개 언어, 교차언어 검색 특화)
TEXT_MODEL_ID = "BAAI/bge-m3"
TEXT_MAXLEN = 512
TEXT_BATCH = 16
TEXT_THRESHOLD_DEFAULT = 0.5    # bge-m3 코사인 임계값. 교차언어(한글→영문) 상위는 대개 0.53~0.60대

# DocLayout-YOLO (문서 레이아웃 검출 사전학습 모델) — 학습 불필요
LAYOUT_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
LAYOUT_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"
LAYOUT_IMGSZ = 1024
# figure = 그림/차트/도면, table = 표. 이 두 종류만 크롭 대상으로 사용.
TARGET_CLASSES = {"figure": "그림/도면", "table": "표"}

MIN_TEXT_CHARS = 10   # 텍스트 레이어가 이보다 짧으면 스캔본으로 보고 OCR 시도
OCR_LANGS = ["ko", "en"]

BASE_DIR = Path(__file__).resolve().parent
# 색인 세트는 이름별 하위폴더로 저장: indexes/<세트이름>/{index.npz, crops/, pages/}
INDEXES_DIR = BASE_DIR / "indexes"
RENDER_DPI = 150                              # PDF 페이지 렌더 해상도
TOP_K = 5                                     # 검색 결과 최대 개수

INDEXES_DIR.mkdir(exist_ok=True)


def safe_set_name(name: str) -> str:
    """색인 세트 이름을 폴더로 쓸 수 있게 정리(한글 허용, 공백/기호는 _)."""
    name = (name or "").strip() or "default"
    return re.sub(r"[^\w\-]+", "_", name)[:60]


def set_paths(name: str):
    """세트 이름 → (세트폴더, index.npz, crops 폴더, pages 폴더)."""
    sd = INDEXES_DIR / safe_set_name(name)
    return sd, sd / "index.npz", sd / "crops", sd / "pages"


def list_index_sets():
    """색인이 완료된(=index.npz 존재) 세트 이름 목록."""
    if not INDEXES_DIR.exists():
        return []
    return sorted(
        d.name for d in INDEXES_DIR.iterdir() if d.is_dir() and (d / "index.npz").exists()
    )

LAYOUT_AVAILABLE = importlib.util.find_spec("doclayout_yolo") is not None
OCR_AVAILABLE = (
    importlib.util.find_spec("rapidocr_onnxruntime") is not None
    or importlib.util.find_spec("easyocr") is not None
)


# ---------------------------------------------------------------------------
# 실행 환경 자동 감지
# ---------------------------------------------------------------------------
def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = detect_device()
YOLO_DEVICE = "cuda" if DEVICE == "cuda" else "cpu"  # YOLO는 MPS 불안정 → CPU

# 모델은 최초 사용 시 1회만 로드 (지연 로딩)
_MODEL = None
_PROCESSOR = None
_YOLO = None
_OCR = None
_TEXT_TOK = None
_TEXT_MODEL = None


def get_model():
    """SigLIP2 모델/프로세서를 지연 로딩하고 캐시한다."""
    global _MODEL, _PROCESSOR
    if _MODEL is None:
        print("[로딩] 이미지 검색 모델(SigLIP2)... 처음엔 다운로드로 시간이 걸립니다.", flush=True)
        _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
        _MODEL = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
        print("[로딩] 이미지 검색 모델 준비 완료.", flush=True)
    return _MODEL, _PROCESSOR


def get_text_model():
    """다국어 텍스트 임베딩 모델(bge-m3)을 지연 로딩하고 캐시한다."""
    global _TEXT_TOK, _TEXT_MODEL
    if _TEXT_MODEL is None:
        print("[로딩] 다국어 텍스트 모델(bge-m3)... 처음엔 다운로드(약 2GB)로 오래 걸립니다.", flush=True)
        from transformers import AutoTokenizer

        _TEXT_TOK = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        _TEXT_MODEL = AutoModel.from_pretrained(TEXT_MODEL_ID).to(DEVICE).eval()
        print("[로딩] 다국어 텍스트 모델 준비 완료.", flush=True)
    return _TEXT_TOK, _TEXT_MODEL


def get_yolo():
    """DocLayout-YOLO 검출 모델을 지연 로딩한다. 사용 불가 시 None."""
    global _YOLO
    if _YOLO is None:
        if not LAYOUT_AVAILABLE:
            return None
        print("[로딩] 도면·표 검출 모델(DocLayout-YOLO)...", flush=True)
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(repo_id=LAYOUT_REPO, filename=LAYOUT_FILE)
        _YOLO = YOLOv10(weights)
        print("[로딩] 도면·표 검출 모델 준비 완료.", flush=True)
    return _YOLO


def get_ocr():
    """OCR 엔진을 지연 로딩한다(RapidOCR → EasyOCR 순). 없으면 None."""
    global _OCR
    if _OCR is None:
        _OCR = False
        # 1순위: RapidOCR (경량, 시스템 바이너리 불필요)
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR = ("rapidocr", RapidOCR())
        except Exception:  # noqa: BLE001
            # 2순위: EasyOCR
            try:
                import easyocr
                _OCR = ("easyocr", easyocr.Reader(OCR_LANGS, gpu=(DEVICE == "cuda")))
            except Exception:  # noqa: BLE001
                _OCR = False
    return _OCR or None


def ocr_text(engine, img: Image.Image, min_conf: float = 0.5) -> str:
    """OCR 엔진으로 이미지에서 텍스트를 추출한다.

    min_conf: 인식 신뢰도가 이 값 미만인 결과(깨진 글자 인식)는 버린다.
    """
    kind, eng = engine
    arr = np.array(img.convert("RGB"))
    try:
        if kind == "rapidocr":
            result, _ = eng(arr)
            if not result:
                return ""
            # result 각 항목: [box, text, score]
            return " ".join(line[1] for line in result if float(line[2]) >= min_conf)
        else:  # easyocr — detail=1 이면 (box, text, conf) 반환
            out = eng.readtext(arr, detail=1)
            return " ".join(t for (_box, t, c) in out if float(c) >= min_conf)
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# 임베딩 유틸
# ---------------------------------------------------------------------------
def _as_embedding(out) -> torch.Tensor:
    """get_image_features / get_text_features 반환값에서 임베딩 텐서를 추출한다."""
    if isinstance(out, torch.Tensor):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    return out.last_hidden_state[:, 0]


@torch.no_grad()
def embed_images(imgs) -> np.ndarray:
    """여러 이미지를 한 번에(batch) 임베딩하고 L2 정규화한다. 반환: (N, D)"""
    model, processor = get_model()
    inputs = processor(images=[im.convert("RGB") for im in imgs], return_tensors="pt").to(DEVICE)
    feats = _as_embedding(model.get_image_features(**inputs))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()


@torch.no_grad()
def embed_image(img: Image.Image) -> np.ndarray:
    """단일 이미지 임베딩 (배치 함수 래퍼)."""
    return embed_images([img])[0]


@torch.no_grad()
def embed_text(text: str) -> np.ndarray:
    """단일 텍스트 질의를 임베딩하고 L2 정규화한다."""
    model, processor = get_model()
    inputs = processor(
        text=[text], return_tensors="pt", padding="max_length", truncation=True,
    ).to(DEVICE)
    feats = _as_embedding(model.get_text_features(**inputs))
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy()[0]


# ---------------------------------------------------------------------------
# 레이아웃 검출 (그림/도면 · 표)
# ---------------------------------------------------------------------------
def detect_regions(img: Image.Image, conf: float):
    """페이지 이미지에서 figure/table 영역 검출. 반환 [(kind_ko,(x1,y1,x2,y2)),...]"""
    yolo = get_yolo()
    if yolo is None:
        return []
    res = yolo.predict(
        np.array(img.convert("RGB")),
        imgsz=LAYOUT_IMGSZ, conf=conf, device=YOLO_DEVICE, verbose=False,
    )[0]
    W, H = img.size
    regions = []
    for b in res.boxes:
        name = yolo.names[int(b.cls)]
        if name not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        pad = 0.01 * max(W, H)
        x1 = max(0, int(x1 - pad)); y1 = max(0, int(y1 - pad))
        x2 = min(W, int(x2 + pad)); y2 = min(H, int(y2 + pad))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        regions.append((TARGET_CLASSES[name], (x1, y1, x2, y2)))
    return regions


# ---------------------------------------------------------------------------
# 텍스트 검색 (키워드 매칭)
# ---------------------------------------------------------------------------
def _terms(query: str):
    return [t for t in re.split(r"\s+", query.lower().strip()) if t]


def text_search(query: str, texts, labels, pages):
    """질의어를 페이지 텍스트에 대해 키워드 매칭. 반환: [(label, snippet, score, page_path), ...]"""
    terms = _terms(query)
    if not terms or len(texts) == 0:
        return []
    hits = []
    for i, raw in enumerate(texts):
        t = str(raw)
        tl = t.lower()
        matched, freq = 0, 0
        for term in terms:
            c = tl.count(term)
            if c:
                matched += 1
                freq += c
        if matched == 0:
            continue
        # 점수: 매칭된 질의어 비율 + 빈도 보너스(약하게)
        score = matched / len(terms) + 0.02 * (freq - matched)
        hits.append((i, score, _snippet(t, terms)))
    hits.sort(key=lambda x: -x[1])
    out = []
    for i, score, snip in hits[:TOP_K]:
        out.append((str(labels[i]), snip, round(float(score), 3), str(pages[i])))
    return out


def _snippet(text: str, terms, width: int = 50) -> str:
    tl = text.lower()
    pos = -1
    for term in terms:
        p = tl.find(term)
        if p >= 0:
            pos = p
            break
    if pos < 0:
        return re.sub(r"\s+", " ", text[: 2 * width]).strip()
    a = max(0, pos - width); b = min(len(text), pos + width)
    seg = re.sub(r"\s+", " ", text[a:b]).strip()
    return ("…" if a > 0 else "") + seg + ("…" if b < len(text) else "")


# ---------------------------------------------------------------------------
# 다국어 의미 검색 (텍스트 임베딩)
# ---------------------------------------------------------------------------
@torch.no_grad()
def embed_texts_semantic(texts, is_query: bool = False) -> np.ndarray:
    """bge-m3 다국어 모델로 텍스트들을 임베딩(CLS pooling + L2 정규화). 반환 (N, D).

    bge-m3 는 질의/문서에 별도 접두사가 필요 없다(대칭). is_query 는 호환용 인자.
    """
    tok, model = get_text_model()
    out_vecs = []
    for i in range(0, len(texts), TEXT_BATCH):
        batch = [(str(t).strip() or " ") for t in texts[i : i + TEXT_BATCH]]
        enc = tok(
            batch, padding=True, truncation=True, max_length=TEXT_MAXLEN, return_tensors="pt",
        ).to(DEVICE)
        cls = model(**enc).last_hidden_state[:, 0]        # CLS pooling
        cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        out_vecs.append(cls.cpu().float().numpy())
    if not out_vecs:
        return np.zeros((0, 1024), dtype=np.float32)      # bge-m3 임베딩 차원 1024
    return np.concatenate(out_vecs, axis=0).astype(np.float32)


def _preview(text: str, query: str, width: int = 140) -> str:
    """검색어가 본문에 그대로 있으면 그 부분을, 없으면(교차언어) 앞부분을 미리보기로."""
    terms = _terms(query)
    tl = text.lower()
    for term in terms:
        p = tl.find(term)
        if p >= 0:
            a = max(0, p - 60); b = min(len(text), p + 80)
            seg = re.sub(r"\s+", " ", text[a:b]).strip()
            return ("…" if a > 0 else "") + seg + ("…" if b < len(text) else "")
    head = re.sub(r"\s+", " ", str(text)[:width]).strip()
    return head + ("…" if len(str(text)) > width else "")


def chunk_text(text: str, size: int = 320, overlap: int = 70):
    """페이지 텍스트를 검색·표시하기 좋은 청크로 쪼갠다.

    청크가 작을수록 한 주제만 담겨 특정 내용(경고문·수치 등) 검색 정밀도가 올라간다.
    긴 페이지도 전부 색인되도록 슬라이딩 윈도우(단어 경계 스냅 + 겹침)로 나눈다.
    """
    text = re.sub(r"[ \t]+", " ", str(text)).strip()
    text = re.sub(r"\n{2,}", "\n", text)
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # 단어/줄 중간에서 잘리지 않도록 뒤쪽 공백으로 스냅
            sp = text.rfind(" ", start + size - 70, end)
            if sp > start:
                end = sp
        seg = text[start:end].strip()
        if seg:
            chunks.append(seg)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def semantic_text_search(query, text_embeddings, texts, labels, pages, threshold):
    """질의 임베딩 vs 청크 임베딩 코사인 유사도 검색. 페이지당 최고 청크 1개만 남겨 다양성 확보.

    반환: [(label, chunk_text, score, page_path), ...]  (chunk_text = 매칭된 청크 전문)
    """
    if text_embeddings is None or len(text_embeddings) == 0:
        return []
    qv = embed_texts_semantic([query], is_query=True)[0]
    scores = text_embeddings @ qv
    order = np.argsort(-scores)
    out, seen_pages = [], set()
    for idx in order:
        s = float(scores[idx])
        if s < threshold:
            break
        label = str(labels[idx])
        if label in seen_pages:      # 같은 페이지의 다른 청크는 건너뜀(페이지당 1개)
            continue
        seen_pages.add(label)
        out.append((label, str(texts[idx]).strip(), round(s, 3), str(pages[idx])))
        if len(out) >= TOP_K:
            break
    return out


# ---------------------------------------------------------------------------
# 1) 색인 생성
# ---------------------------------------------------------------------------
def build_index(files, mode, conf, do_text, ocr_conf, set_name, append, progress=gr.Progress()):
    if not files:
        return "⚠️ 색인할 PDF 파일을 먼저 올려주세요."

    use_crop = mode.startswith("도면")
    if use_crop and get_yolo() is None:
        return "❌ DocLayout-YOLO 를 사용할 수 없습니다. '페이지 전체' 모드로 다시 시도하세요."

    # 색인 세트 경로 (indexes/<세트이름>/)
    set_name = safe_set_name(set_name)
    set_dir, index_path, crops_dir, pages_dir = set_paths(set_name)

    # 이미지 색인용
    embeddings, filenames, image_paths, kinds = [], [], [], []
    # 텍스트 색인용 (청크 단위)
    chunk_texts, chunk_labels, chunk_pages = [], [], []

    docs, total_pages = [], 0
    for f in files:
        path = f.name if hasattr(f, "name") else f
        try:
            doc = fitz.open(path)
        except Exception as e:  # noqa: BLE001
            return f"❌ '{Path(path).name}' 을(를) 열 수 없습니다: {e}"
        docs.append((Path(path).stem, doc))
        total_pages += doc.page_count

    if total_pages == 0:
        return "⚠️ 페이지가 있는 PDF가 없습니다."

    # 덮어쓰기 모드(append 아님)면 이 세트의 기존 파일(index.npz·크롭·페이지 찌꺼기)을 먼저 정리
    if not append and set_dir.exists():
        shutil.rmtree(set_dir, ignore_errors=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    n_fig = n_tab = n_page = n_textpage = n_ocr = n_chunk = 0
    done = 0
    for stem, doc in docs:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=RENDER_DPI)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pno = page_index + 1

            # --- 페이지 전체 이미지 저장 (텍스트 검색 결과 표시 & 페이지 모드 공용) ---
            page_safe = f"{stem}_p{pno}".replace(os.sep, "_").replace("/", "_")
            page_path = pages_dir / f"{page_safe}.png"
            img.save(page_path)

            # --- 텍스트 추출 (텍스트본 → get_text, 스캔본 → OCR) 후 청크 분할 ---
            if do_text:
                txt = page.get_text().strip()
                if len(txt) < MIN_TEXT_CHARS:
                    eng = get_ocr()
                    if eng is not None:
                        txt = ocr_text(eng, img, min_conf=ocr_conf)
                        if txt:
                            n_ocr += 1
                if txt:
                    n_textpage += 1
                    for chunk in chunk_text(txt):
                        chunk_texts.append(chunk)
                        chunk_labels.append(f"{stem} · p{pno}")
                        chunk_pages.append(str(page_path))
                        n_chunk += 1

            # --- 이미지 색인 대상(크롭 or 페이지 전체) ---
            items = []  # (kind_ko, image, save_path)
            if use_crop:
                for k, (kind, box) in enumerate(detect_regions(img, conf), start=1):
                    safe = f"{stem}_p{pno}_{kind}_{k}".replace(os.sep, "_").replace("/", "_")
                    cp = crops_dir / f"{safe}.png"
                    crop = img.crop(box)
                    crop.save(cp)
                    items.append((kind, crop, str(cp)))
                if not items:  # 검출 0건 → 페이지 전체로 폴백
                    items.append(("페이지", img, str(page_path)))
            else:
                items.append(("페이지", img, str(page_path)))

            page_vecs = embed_images([im for _, im, _ in items])  # 페이지 내 항목 일괄 임베딩
            for k, ((kind, im, sp), vec) in enumerate(zip(items, page_vecs), start=1):
                embeddings.append(vec)
                filenames.append(f"{stem} · p{pno} · {kind}#{k}")
                image_paths.append(sp)
                kinds.append(kind)
                if kind == "그림/도면":
                    n_fig += 1
                elif kind == "표":
                    n_tab += 1
                else:
                    n_page += 1

            done += 1
            progress(done / total_pages, desc=f"처리 중: {stem} p{pno} ({done}/{total_pages})")
        doc.close()

    if not embeddings:
        return "⚠️ 색인할 대상을 찾지 못했습니다."

    # 텍스트 청크 의미 임베딩 (다국어 bge-m3)
    if do_text and chunk_texts:
        progress(1.0, desc=f"텍스트 청크 {len(chunk_texts)}개 의미 임베딩 중 (다국어 모델)...")
        text_embeddings = embed_texts_semantic(chunk_texts, is_query=False)
    else:
        text_embeddings = np.zeros((0, 1024), dtype=np.float32)

    # 이번 실행에서 새로 만든 배열
    img_emb = np.stack(embeddings).astype(np.float32)
    img_fn = np.array(filenames); img_ip = np.array(image_paths); img_kd = np.array(kinds)
    ct = np.array(chunk_texts if do_text else [], dtype=object)
    cl = np.array(chunk_labels if do_text else [])
    cp = np.array(chunk_pages if do_text else [])

    # append 모드: 기존 세트 내용에 이어붙임 (같은 파일명=stem 은 갱신)
    appended_note = ""
    if append and index_path.exists():
        old = np.load(index_path, allow_pickle=True)
        new_stems = {stem for stem, _ in docs}
        stem_of = lambda lbl: str(lbl).split(" · ")[0]
        _get = lambda k, d: old[k] if k in old.files else d

        o_fn = _get("filenames", np.array([]))
        keep = [i for i, f in enumerate(o_fn) if stem_of(f) not in new_stems]
        img_emb = np.concatenate([_get("embeddings", np.zeros((0, 768), np.float32))[keep], img_emb])
        img_fn = np.concatenate([o_fn[keep], img_fn])
        img_ip = np.concatenate([_get("image_paths", np.array([]))[keep], img_ip])
        img_kd = np.concatenate([_get("kinds", np.array([]))[keep], img_kd])

        o_cl = _get("chunk_labels", np.array([]))
        tkeep = [i for i, l in enumerate(o_cl) if stem_of(l) not in new_stems]
        text_embeddings = np.concatenate([_get("text_embeddings", np.zeros((0, 1024), np.float32))[tkeep], text_embeddings])
        ct = np.concatenate([_get("chunk_texts", np.array([], dtype=object))[tkeep], ct])
        cl = np.concatenate([o_cl[tkeep], cl])
        cp = np.concatenate([_get("chunk_pages", np.array([]))[tkeep], cp])
        kept_docs = len({stem_of(f) for f in o_fn[keep]})
        appended_note = f" (기존 세트에 이어붙임: 유지 문서 {kept_docs}개 + 신규 {len(docs)}개)"

    np.savez(
        index_path,
        embeddings=img_emb,
        filenames=img_fn,
        image_paths=img_ip,
        kinds=img_kd,
        chunk_texts=ct,
        chunk_labels=cl,
        chunk_pages=cp,
        text_embeddings=text_embeddings,
    )

    lines = [
        "✅ 색인 생성 완료!",
        f"- 세트 이름: {set_name}" + ("  [추가 모드]" if append else ""),
        f"- 방식: {mode}",
        f"- 이번 처리: 문서 {len(docs)}개 / 페이지 {total_pages}개{appended_note}",
        f"- [이미지] 이번 {len(filenames)}개 → 세트 총 {len(img_fn)}개",
    ]
    if use_crop:
        lines.append(f"    · 그림/도면 {n_fig} · 표 {n_tab} · (검출없음)페이지 {n_page}")
    if do_text:
        ocr_note = f", OCR 처리 {n_ocr}p" if n_ocr else (" (OCR 엔진 없음 → 스캔본 텍스트 제외)" if not OCR_AVAILABLE else "")
        lines.append(f"- [텍스트] 이번 청크 {n_chunk}개 → 세트 총 {len(cl)}개{ocr_note}")
    lines += [f"- 저장 위치: indexes/{set_name}/", f"- 실행 환경: {DEVICE.upper()}"]
    lines.append("\n🔍 검색 탭의 '색인 세트' 드롭다운에서 이 세트를 선택해 검색하세요.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2) 검색 (하이브리드)
# ---------------------------------------------------------------------------
def search(query: str, threshold: float, text_threshold: float, set_name, progress=gr.Progress()):
    query = (query or "").strip()
    if not query:
        return [], [], "", "⚠️ 검색어를 입력해주세요."
    if not set_name:
        return [], [], "", "⚠️ 검색할 색인 세트를 선택하세요. (없으면 '색인 생성' 탭에서 먼저 만드세요)"
    _, index_path, _, _ = set_paths(set_name)
    if not index_path.exists():
        return [], [], "", f"⚠️ 색인 세트 '{set_name}' 를 찾을 수 없습니다. '색인 생성' 탭에서 먼저 만드세요."

    data = np.load(index_path, allow_pickle=True)
    embeddings = data["embeddings"]
    filenames = data["filenames"]
    image_paths = data["image_paths"]
    # 신버전: 청크 단위 / 구버전 폴백: 페이지 단위(doc_*)
    txt_texts = data["chunk_texts"] if "chunk_texts" in data else (
        data["doc_texts"] if "doc_texts" in data else np.array([], dtype=object))
    txt_labels = data["chunk_labels"] if "chunk_labels" in data else (
        data["doc_labels"] if "doc_labels" in data else np.array([]))
    txt_pages = data["chunk_pages"] if "chunk_pages" in data else (
        data["doc_pages"] if "doc_pages" in data else np.array([]))
    text_embeddings = data["text_embeddings"] if "text_embeddings" in data else None

    # ---- 이미지 검색 (SigLIP2 코사인) ----
    progress(0.4, desc="질의어 임베딩 중...")
    q = embed_text(query)
    scores = embeddings @ q
    order = np.argsort(-scores)
    img_gallery, kept = [], 0
    for idx in order:
        s = float(scores[idx])
        if s < threshold:
            break
        img_gallery.append((str(image_paths[idx]), f"{filenames[idx]}  |  유사도 {s:.3f}"))
        kept += 1
        if kept >= TOP_K:
            break

    # ---- 텍스트 검색 (다국어 의미, 청크 단위) ----
    progress(0.8, desc="텍스트 의미 검색 중...")
    if text_embeddings is not None and len(text_embeddings) > 0:
        thits = semantic_text_search(
            query, text_embeddings, txt_texts, txt_labels, txt_pages, text_threshold
        )
        text_mode = "의미"
    else:
        thits = text_search(query, txt_texts, txt_labels, txt_pages)  # 구버전 폴백
        text_mode = "키워드"

    # 결과 A: 페이지 이미지 갤러리 + 매칭 청크 전문(마크다운)
    text_gallery, md_parts = [], []
    for label, chunk, score, page_path in thits:
        text_gallery.append((page_path, f"{label} | 유사도 {score}"))
        excerpt = chunk if len(chunk) <= 500 else chunk[:500] + "…"
        md_parts.append(f"**📄 {label}**  ·  유사도 `{score}`\n\n> {excerpt}")
    text_md = "\n\n---\n\n".join(md_parts) if md_parts else "_텍스트 결과 없음_"

    # ---- 상태 메시지 ----
    img_msg = f"이미지 {kept}건" if kept else "이미지 0건"
    txt_msg = f"텍스트({text_mode}) {len(thits)}건" if len(txt_texts) else "텍스트(미색인)"
    if kept == 0 and not thits:
        status = f"🔍 찾는 내용이 문서에 없습니다. ({img_msg}·임계값 {threshold:.2f} / {txt_msg})"
    else:
        status = f"✅ 결과: {img_msg} · {txt_msg} (각 최대 {TOP_K}개)"
    return img_gallery, text_gallery, text_md, status


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def build_ui():
    layout_badge = "DocLayout-YOLO ✅" if LAYOUT_AVAILABLE else "DocLayout-YOLO ❌"
    ocr_badge = "OCR ✅" if OCR_AVAILABLE else "OCR ❌(스캔본 텍스트 제외)"
    with gr.Blocks(title="선박기기 매뉴얼 하이브리드 검색") as demo:
        gr.Markdown(
            f"# 🚢 선박기기 매뉴얼 하이브리드 검색 (이미지 + 다국어 텍스트)\n"
            f"로컬 전용 · 검출: `{layout_badge}` · OCR: `{ocr_badge}`\n\n"
            f"이미지: `{MODEL_ID}` · 텍스트(다국어): `{TEXT_MODEL_ID}` · 실행 환경: **{DEVICE.upper()}**"
        )

        with gr.Tab("📁 색인 생성"):
            gr.Markdown(
                "매뉴얼 PDF(스캔본/텍스트본 모두 가능)를 **드래그 앤 드롭**하세요.\n\n"
                "- **이미지 색인**: 도면·표를 잘라(YOLO) 또는 페이지 전체로 SigLIP2 임베딩\n"
                "- **텍스트 색인**: 본문 글자를 추출(텍스트본)/OCR(스캔본) 후 다국어 모델로 의미 임베딩\n"
                "  → 한글로 검색해도 영문 본문이 매칭됩니다 (예: `산소 분석기` → `oxygen analyzer`)"
            )
            pdf_input = gr.File(
                label="PDF 파일 (여러 개 가능)",
                file_count="multiple", file_types=[".pdf"], type="filepath",
            )
            with gr.Row():
                mode_radio = gr.Radio(
                    label="이미지 색인 방식",
                    choices=["도면·표 크롭 (YOLO)", "페이지 전체"],
                    value="도면·표 크롭 (YOLO)" if LAYOUT_AVAILABLE else "페이지 전체",
                )
                conf_slider = gr.Slider(
                    label="검출 신뢰도 (conf)", minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                )
            with gr.Row():
                text_check = gr.Checkbox(
                    label="텍스트 색인도 함께 (OCR)", value=True,
                )
                ocr_conf_slider = gr.Slider(
                    label="OCR 신뢰도 (스캔본 글자 인식 최소 신뢰도)",
                    minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                )
            with gr.Group():
                set_name_input = gr.Textbox(
                    label="색인 세트 이름",
                    value="default",
                    placeholder="예: 발전기_매뉴얼, 엔진_2호기 ...",
                    info="여러 매뉴얼 묶음을 이름별로 따로 저장해두고 검색 시 골라 쓸 수 있습니다.",
                )
                append_check = gr.Checkbox(
                    label="기존 세트에 이어붙이기 (append)",
                    value=False,
                    info="체크: 같은 이름 세트에 새 PDF를 더함(같은 파일명은 갱신) · 해제: 세트를 새로 덮어씀",
                )
            index_btn = gr.Button("색인 생성 시작", variant="primary")
            index_status = gr.Textbox(label="진행 상황 / 결과", lines=9, interactive=False)

        with gr.Tab("🔍 하이브리드 검색"):
            _sets = list_index_sets()
            with gr.Row():
                set_dropdown = gr.Dropdown(
                    label="색인 세트 선택",
                    choices=_sets,
                    value=(_sets[0] if _sets else None),
                    scale=4,
                )
                refresh_btn = gr.Button("🔄 목록 새로고침", scale=1)
            with gr.Row():
                query_input = gr.Textbox(
                    label="검색어",
                    placeholder="예: 냉각수 배관도 / cooling water pipe / 윤활유 교환 주기 ...",
                    scale=4,
                )
                search_btn = gr.Button("검색", variant="primary", scale=1)
            with gr.Row():
                threshold_slider = gr.Slider(
                    label="이미지 임계값 (SigLIP 코사인)",
                    minimum=0.0, maximum=1.0, value=0.1, step=0.01,
                )
                text_threshold_slider = gr.Slider(
                    label="텍스트 임계값 (다국어 bge-m3 코사인)",
                    minimum=0.3, maximum=0.9, value=TEXT_THRESHOLD_DEFAULT, step=0.01,
                )
            search_status = gr.Textbox(label="상태", interactive=False)
            gr.Markdown("### 🖼️ 이미지 결과 (도면·표)")
            results_gallery = gr.Gallery(
                label="파일·페이지·유형 | 유사도",
                columns=3, height="auto", object_fit="contain",
            )
            gr.Markdown("### 📝 텍스트 결과 (본문 · 다국어 의미 검색 · 청크 단위)")
            with gr.Row():
                text_gallery = gr.Gallery(
                    label="관련 페이지 (썸네일) | 유사도",
                    columns=3, height="auto", object_fit="contain", scale=1,
                )
                text_detail = gr.Markdown(
                    value="_검색하면 매칭된 본문 청크 전문이 여기에 표시됩니다._",
                )
            for trigger in (search_btn.click, query_input.submit):
                trigger(
                    fn=search,
                    inputs=[query_input, threshold_slider, text_threshold_slider, set_dropdown],
                    outputs=[results_gallery, text_gallery, text_detail, search_status],
                )

            # 세트 목록 새로고침
            refresh_btn.click(
                fn=lambda cur: gr.update(choices=list_index_sets(), value=cur),
                inputs=[set_dropdown],
                outputs=[set_dropdown],
            )

        # 색인 완료 시: 결과 메시지 출력 + 검색 탭 드롭다운 갱신(방금 만든 세트 선택)
        index_btn.click(
            fn=build_index,
            inputs=[pdf_input, mode_radio, conf_slider, text_check, ocr_conf_slider, set_name_input, append_check],
            outputs=[index_status],
        ).then(
            fn=lambda n: gr.update(choices=list_index_sets(), value=safe_set_name(n)),
            inputs=[set_name_input],
            outputs=[set_dropdown],
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="선박기기 매뉴얼 하이브리드 검색")
    parser.add_argument("--share", action="store_true",
                        help="Gradio 임시 공개 링크(72시간) 생성 — 발표·시연용 데모 배포")
    parser.add_argument("--host", default="127.0.0.1",
                        help="바인드 주소. 사내망(LAN) 공유는 0.0.0.0")
    parser.add_argument("--port", type=int, default=7860, help="포트 (기본 7860)")
    parser.add_argument("--open", action="store_true",
                        help="공유 모드에서도 로컬 브라우저를 자동으로 연다")
    args = parser.parse_args()

    print(f"[정보] 실행환경 device={DEVICE} / YOLO={YOLO_DEVICE} / 검출={LAYOUT_AVAILABLE} / OCR={OCR_AVAILABLE}", flush=True)
    if args.share:
        print("[배포] 임시 공개 링크(share)를 생성합니다. ⚠️ 사내 매뉴얼은 올리지 마세요(외부 공개됨).", flush=True)
    elif args.host == "0.0.0.0":
        print(f"[배포] 사내망 공유 모드. 같은 네트워크에서 http://<내PC IP>:{args.port} 로 접속하세요.", flush=True)
    else:
        print(f"[준비] 웹 서버를 시작합니다. 브라우저가 자동으로 열립니다 (http://127.0.0.1:{args.port})", flush=True)
    open_browser = True if not args.share else args.open  # 공유 모드는 기본 미오픈, --open 시 로컬 오픈
    build_ui().launch(server_name=args.host, server_port=args.port,
                      share=args.share, inbrowser=open_browser)
