"""Per-op layer descriptions generated from REAL ONNX operator metadata.

Every fact in a description comes from the operator itself: its `op_type`,
its real attributes (kernel_shape, epsilon, hidden_size, transB, …) and the
real tensor shapes measured by the analyzer. The old synthetic-ID pattern
tables (t0_gate_f, DiT blocks, …) are gone — real exporter names like
`/features/0/conv/Conv` never matched them, so every model silently fell
through to misleading generic blurbs.

`describe_op` is a pure function over OperatorInfo-ish data. Unknown op_types
get an honest fallback that names the real op_type instead of pretending.
`describe_kind_fallback` covers nodes that have no ONNX op at all (synthetic
input/output nodes and weights-only checkpoints, whose kinds are inferred
from real tensor shapes).

Shape of every result (matches the LayerNode.description contract):
    {"title": str, "summary": {"en": str, "ko": str}, "formula": str | None}
"""

from __future__ import annotations

# ── tiny helpers over real op data ──────────────────────────────────────────


def _dims(v: object) -> str | None:
    """[3, 3] → "3×3" — only when every entry is a positive int."""
    if isinstance(v, (list, tuple)) and v and all(isinstance(d, int) and d > 0 for d in v):
        return "×".join(str(d) for d in v)
    return None


def _channels(shape: list[int] | None) -> int | None:
    """Channel dim for NCHW-ish tensors (rank ≥ 3), feature dim otherwise."""
    if not shape:
        return None
    return shape[1] if len(shape) >= 3 else shape[-1]


def _features(shape: list[int] | None) -> int | None:
    return shape[-1] if shape else None


def _io(cin: int | None, cout: int | None, *, unit_en: str, unit_ko: str) -> tuple[str, str]:
    """("64→128 channels", "64→128 채널") or empty strings when unknown."""
    if cin and cout:
        return f", {cin}→{cout} {unit_en}", f" ({cin}→{cout} {unit_ko})"
    if cout:
        return f", {cout} {unit_en}", f" ({cout} {unit_ko})"
    return "", ""


def _desc(title: str, en: str, ko: str, formula: str | None = None) -> dict:
    return {"title": title, "summary": {"en": en, "ko": ko}, "formula": formula}


# ── the generator ────────────────────────────────────────────────────────────


def describe_op(  # noqa: C901, PLR0911, PLR0912, PLR0915 — one flat op_type dispatch
    op_type: str,
    *,
    attributes: dict | None = None,
    input_shape: list[int] | None = None,
    output_shape: list[int] | None = None,
    params: int = 0,
    is_attention: bool = False,
) -> dict:
    """Describe one real ONNX operator from its own metadata (pure function)."""
    a = attributes or {}
    cin, cout = _channels(input_shape), _channels(output_shape)
    fin, fout = _features(input_shape), _features(output_shape)

    # ---- dense compute -----------------------------------------------------
    if op_type in ("Conv", "ConvInteger"):
        k = _dims(a.get("kernel_shape"))
        s = _dims(a.get("strides"))
        g = a.get("group")
        depthwise = isinstance(g, int) and g > 1 and g == cin
        title = f"Conv {k}" if k else "Convolution"
        if depthwise:
            title = f"Depthwise Conv {k}" if k else "Depthwise Conv"
        kt_en = f"{k} kernel" if k else "learned kernel"
        kt_ko = f"{k} 커널" if k else "학습된 커널"
        st_en = f", stride {s}" if s and s != "1×1" and s != "1" else ""
        grp_en = " Each input channel is filtered independently (depthwise)." if depthwise else (
            f" Grouped into {g} channel groups." if isinstance(g, int) and g > 1 else "")
        io_en, io_ko = _io(cin, cout, unit_en="channels", unit_ko="채널")
        grp_ko = " 채널별로 독립 필터링하는 depthwise 구조." if depthwise else (
            f" {g}개 채널 그룹으로 나눠 연산." if isinstance(g, int) and g > 1 else "")
        st_ko = f" (stride {s})" if st_en else ""
        return _desc(
            title,
            f"Slides a shared {kt_en} over the input to extract local features{st_en}{io_en}.{grp_en}",
            f"공유 {kt_ko}을 입력 위로 슬라이드하며 local feature를 추출{io_ko}{st_ko}.{grp_ko}",
            r"Y = W \ast X + b",
        )
    if op_type == "ConvTranspose":
        k = _dims(a.get("kernel_shape"))
        io_en, io_ko = _io(cin, cout, unit_en="channels", unit_ko="채널")
        return _desc(
            f"Transposed Conv {k}" if k else "Transposed Conv",
            f"Learned upsampling: the transpose of a convolution increases spatial resolution{io_en}.",
            f"학습된 업샘플링: convolution의 전치 연산으로 spatial 해상도를 키움{io_ko}.",
        )
    if op_type in ("MatMul", "MatMulInteger"):
        if is_attention:
            return _desc(
                "Attention MatMul",
                "Matrix multiply inside a detected attention block. Computes the "
                "QKᵀ score map or applies the softmaxed weights to V.",
                "감지된 attention 블록 내부의 행렬곱, QKᵀ score 계산 또는 softmax된 "
                "weight를 V에 적용하는 단계.",
                r"\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V",
            )
        io_en, io_ko = _io(fin, fout, unit_en="features", unit_ko="features")
        return _desc(
            "MatMul",
            f"Dense matrix multiplication{io_en}: a linear projection without bias.",
            f"행렬곱 연산{io_ko}: bias 없는 linear projection.",
            r"Y = X W",
        )
    if op_type == "Gemm":
        trans_b = bool(a.get("transB"))
        io_en, io_ko = _io(fin, fout, unit_en="features", unit_ko="features")
        return _desc(
            "Gemm (Fully connected)",
            f"Fully-connected linear layer{io_en} with {params:,} real parameters"
            f"{' (weight stored transposed, transB=1)' if trans_b else ''}.",
            f"Fully-connected linear 레이어{io_ko}, 실제 파라미터 {params:,}개"
            f"{' (weight를 전치 저장, transB=1)' if trans_b else ''}.",
            r"Y = X W^\top + b" if trans_b else r"Y = X W + b",
        )

    # ---- activations ---------------------------------------------------------
    if op_type == "Relu":
        return _desc(
            "ReLU",
            "Rectified linear activation. Clips negatives to 0, keeping the network nonlinear at zero cost.",
            "음수를 0으로 잘라내는 비선형 활성 함수. 연산 비용이 거의 없음.",
            r"f(x) = \max(0,\, x)",
        )
    if op_type == "LeakyRelu":
        alpha = a.get("alpha", 0.01)
        return _desc(
            "LeakyReLU",
            f"ReLU variant that keeps a small slope α={alpha:g} for negative inputs instead of zeroing them.",
            f"음수 입력을 0으로 만들지 않고 기울기 α={alpha:g}로 약하게 통과시키는 ReLU 변형.",
            r"f(x) = \max(\alpha x,\, x)",
        )
    if op_type == "Sigmoid":
        return _desc(
            "Sigmoid",
            "Squashes each value into (0, 1); used for gates and binary probabilities.",
            "각 값을 (0, 1) 구간으로 압축. 게이트나 이진 확률에 사용.",
            r"\sigma(x) = \frac{1}{1 + e^{-x}}",
        )
    if op_type == "Tanh":
        return _desc(
            "Tanh",
            "Squashes each value into (−1, 1); zero-centered, common in recurrent cells.",
            "각 값을 (−1, 1) 구간으로 압축. 0 중심이라 recurrent 셀에서 흔히 사용.",
            r"f(x) = \tanh(x)",
        )
    if op_type == "Gelu":
        return _desc(
            "GELU",
            "Gaussian-error linear unit: the smooth activation used by transformer FFNs.",
            "Gaussian 오차 함수 기반의 부드러운 활성, transformer FFN의 표준 활성 함수.",
            r"f(x) = x \cdot \Phi(x)",
        )
    if op_type == "Elu":
        alpha = a.get("alpha", 1.0)
        return _desc(
            "ELU",
            f"Exponential linear unit (α={alpha:g}): smooth negative saturation instead of a hard zero.",
            f"지수 선형 유닛 (α={alpha:g}). 음수 영역을 0으로 자르지 않고 부드럽게 포화시킴.",
        )
    if op_type == "Selu":
        return _desc(
            "SELU",
            "Self-normalizing ELU with fixed scale/alpha that keeps activations near zero mean and unit variance.",
            "고정 scale·alpha를 쓰는 self-normalizing ELU로, 활성값을 평균 0, 분산 1 근처로 유지.",
        )
    if op_type == "Clip":
        return _desc(
            "Clip",
            "Clamps values into a [min, max] range. Exporters emit this for ReLU6-style activations.",
            "값을 [min, max] 구간으로 자르는 연산. ReLU6 계열 활성이 이 형태로 export됨.",
        )
    if op_type in ("Softmax", "LogSoftmax"):
        axis = a.get("axis")
        ax_en = f" along axis {axis}" if isinstance(axis, int) else ""
        ax_ko = f" (axis {axis})" if isinstance(axis, int) else ""
        log = op_type == "LogSoftmax"
        return _desc(
            op_type,
            f"Normalizes scores into a probability distribution{ax_en}"
            f"{' and takes the log (numerically stable for NLL loss)' if log else ''}.",
            f"점수를 합이 1인 확률 분포로 정규화{ax_ko}"
            f"{'한 뒤 log를 취함 (NLL loss에 수치적으로 안정적)' if log else ''}.",
            r"\mathrm{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}",
        )

    # ---- normalization -------------------------------------------------------
    if op_type in ("BatchNormalization", "InstanceNormalization", "LayerNormalization",
                   "GroupNormalization"):
        eps = a.get("epsilon")
        eps_en = f" (ε={eps:g})" if isinstance(eps, float) else ""
        scope = {
            "BatchNormalization": (
                "Batch normalization", "per-channel statistics learned over training batches",
                "학습 배치에서 추정한 채널별 통계"),
            "InstanceNormalization": (
                "Instance normalization", "statistics computed per sample and channel",
                "샘플·채널 단위로 계산한 통계"),
            "LayerNormalization": (
                "Layer normalization", "statistics computed across the feature dimension of each token",
                "각 토큰의 feature 차원 전체에서 계산한 통계"),
            "GroupNormalization": (
                "Group normalization", "statistics computed over channel groups",
                "채널 그룹 단위로 계산한 통계"),
        }[op_type]
        ch = f" over {cin} channels" if cin else ""
        ch_ko = f"{cin}개 채널의 " if cin else ""
        return _desc(
            scope[0],
            f"Normalizes activations{ch} using {scope[1]}{eps_en}, then applies a learned scale γ and shift β.",
            f"{scope[2]}{eps_en}로 {ch_ko}활성값을 정규화한 뒤 학습된 γ(scale)·β(shift)를 적용.",
            r"y = \gamma\,\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta",
        )

    # ---- pooling --------------------------------------------------------------
    if op_type in ("MaxPool", "AveragePool"):
        k = _dims(a.get("kernel_shape"))
        s = _dims(a.get("strides"))
        mode_en = "the strongest activation" if op_type == "MaxPool" else "the average"
        mode_ko = "최댓값" if op_type == "MaxPool" else "평균값"
        kt_en = f"{k} window" if k else "window"
        kt_ko = f"{k} 윈도우" if k else "윈도우"
        st_en = f", stride {s}" if s else ""
        st_ko = f" (stride {s})" if s else ""
        return _desc(
            f"{op_type} {k}" if k else op_type,
            f"Downsamples spatially by keeping {mode_en} in each {kt_en}{st_en}.",
            f"각 {kt_ko}에서 {mode_ko}을 취해 spatial 해상도를 줄임{st_ko}.",
        )
    if op_type == "GlobalAveragePool":
        return _desc(
            "Global average pool",
            f"Averages each channel over all spatial positions into a single value"
            f"{f' ({cin} channels → vector)' if cin else ''}: the classifier-head input.",
            f"각 채널의 모든 spatial 위치를 평균해 값 하나로 압축"
            f"{f' ({cin}개 채널 → 벡터)' if cin else ''}. 분류 head의 입력.",
            r"y_c = \frac{1}{HW} \sum_{i,j} x_{c,ij}",
        )
    if op_type == "GlobalMaxPool":
        return _desc(
            "Global max pool",
            "Takes the maximum of each channel over all spatial positions.",
            "각 채널의 모든 spatial 위치에서 최댓값 하나를 취함.",
        )

    # ---- embedding / elementwise ----------------------------------------------
    if op_type in ("Gather", "GatherElements"):
        emb = f" ({fout}-dim vectors)" if fout else ""
        emb_ko = f" ({fout}차원 벡터)" if fout else ""
        return _desc(
            "Gather (lookup)",
            f"Indexes rows out of a tensor. As the first op on integer ids this is the embedding lookup{emb}.",
            f"텐서에서 인덱스로 행을 꺼내는 연산. 정수 id 입력의 첫 op이면 embedding lookup{emb_ko}에 해당.",
        )
    if op_type == "Add":
        return _desc(
            "Add",
            "Element-wise addition. In exported graphs this is typically a residual (skip) join or a bias add.",
            "원소별 덧셈. export된 그래프에서는 보통 residual(skip) 합류 지점이거나 bias 덧셈.",
            r"y = x_1 + x_2",
        )
    if op_type == "Mul":
        return _desc(
            "Mul",
            "Element-wise multiplication. Used for gating, attention scaling, or normalization scale.",
            "원소별 곱셈. 게이팅, attention 스케일링, 정규화 scale 등에 사용.",
            r"y = x_1 \odot x_2",
        )

    # ---- data movement ----------------------------------------------------------
    if op_type in ("Concat", "Reshape", "Transpose", "Flatten", "Slice", "Split",
                   "Pad", "Squeeze", "Unsqueeze"):
        detail_en = {
            "Concat": f"Joins tensors along axis {a.get('axis')}" if isinstance(a.get("axis"), int)
                      else "Joins tensors along an axis",
            "Reshape": "Reinterprets the tensor with a new shape",
            "Transpose": f"Permutes dimensions (perm={a.get('perm')})" if a.get("perm")
                         else "Permutes tensor dimensions",
            "Flatten": "Collapses dimensions into a 2-D matrix for the dense head",
            "Slice": "Extracts a sub-range of the tensor",
            "Split": "Splits the tensor into parts along an axis",
            "Pad": "Pads the tensor borders",
            "Squeeze": "Removes size-1 dimensions",
            "Unsqueeze": "Inserts size-1 dimensions",
        }[op_type]
        detail_ko = {
            "Concat": f"axis {a.get('axis')} 방향으로 텐서들을 이어붙임" if isinstance(a.get("axis"), int)
                      else "한 축을 따라 텐서들을 이어붙임",
            "Reshape": "데이터 이동 없이 텐서를 새 shape로 재해석",
            "Transpose": f"차원 순서를 재배열 (perm={a.get('perm')})" if a.get("perm")
                         else "텐서의 차원 순서를 재배열",
            "Flatten": "dense head 입력을 위해 차원을 2-D 행렬로 펼침",
            "Slice": "텐서의 일부 구간을 잘라냄",
            "Split": "한 축을 따라 텐서를 여러 조각으로 분할",
            "Pad": "텐서 가장자리를 패딩",
            "Squeeze": "크기 1인 차원을 제거",
            "Unsqueeze": "크기 1인 차원을 삽입",
        }[op_type]
        return _desc(
            op_type,
            f"{detail_en}: pure data movement, no learned weights or math.",
            f"{detail_ko}: 학습 가중치나 연산이 없는 순수 데이터 이동.",
        )

    # ---- recurrent -----------------------------------------------------------------
    if op_type in ("LSTM", "GRU", "RNN"):
        hidden = a.get("hidden_size")
        direction = a.get("direction")
        h_en = f" with hidden size {hidden}" if isinstance(hidden, int) and hidden > 0 else ""
        h_ko = f" (hidden {hidden})" if isinstance(hidden, int) and hidden > 0 else ""
        bi = " Bidirectional." if direction == "bidirectional" else ""
        bi_ko = " 양방향." if direction == "bidirectional" else ""
        gates = {"LSTM": "input/forget/cell/output gates",
                 "GRU": "update/reset gates", "RNN": "a single tanh/ReLU cell"}[op_type]
        gates_ko = {"LSTM": "input·forget·cell·output 게이트",
                    "GRU": "update·reset 게이트", "RNN": "단일 tanh/ReLU 셀"}[op_type]
        return _desc(
            f"{op_type} cell",
            f"Fused recurrent layer{h_en}. Runs the full sequence through {gates} in one op.{bi}",
            f"퓨즈드 recurrent 레이어{h_ko}. 전체 시퀀스를 {gates_ko}로 한 op 안에서 처리.{bi_ko}",
            r"h_t = o_t \odot \tanh(c_t)" if op_type == "LSTM" else None,
        )
    if op_type in ("Loop", "Scan"):
        return _desc(
            f"{op_type} (traced recurrence)",
            f"Generic ONNX {op_type} carrying a recurrent subgraph: tf2onnx emits this instead of a "
            "fused LSTM/GRU op; the recurrent weights live on this node.",
            f"recurrent 서브그래프를 담은 범용 ONNX {op_type}. tf2onnx가 퓨즈드 LSTM/GRU 대신 "
            "이 형태로 export하며, recurrent 가중치가 이 노드에 실려 있음.",
        )

    # ---- resize / upsample -------------------------------------------------------
    if op_type in ("Resize", "Upsample"):
        mode = a.get("mode")
        m_en = f" ({mode} interpolation)" if isinstance(mode, str) else ""
        return _desc(
            op_type,
            f"Resamples the spatial resolution{m_en}. No learned weights.",
            f"spatial 해상도를 리샘플링{m_en}. 학습 가중치 없음.",
        )

    # ---- classical ML (ai.onnx.ml) ----------------------------------------------
    if op_type in ("TreeEnsembleClassifier", "TreeEnsembleRegressor", "TreeEnsemble"):
        kind_en = "class scores" if "Classifier" in op_type else "regression values"
        kind_ko = "클래스 점수" if "Classifier" in op_type else "회귀 값"
        return _desc(
            "Tree ensemble",
            f"Gradient-boosted decision tree ensemble evaluated natively by ONNX Runtime that "
            f"sums every tree's leaf output into {kind_en}.",
            f"ONNX Runtime이 네이티브로 평가하는 gradient-boosted 결정 트리 앙상블로, "
            f"모든 트리의 리프 출력을 합산해 {kind_ko}를 산출.",
        )
    if op_type in ("LinearClassifier", "LinearRegressor"):
        return _desc(
            "Linear model",
            "A single learned linear transform over the input features.",
            "입력 feature에 대한 단일 학습 linear 변환.",
            r"y = X W + b",
        )
    if op_type in ("SVMClassifier", "SVMRegressor"):
        kernel = a.get("kernel_type")
        k_en = f" ({kernel} kernel)" if isinstance(kernel, str) else ""
        return _desc(
            "SVM",
            f"Support vector machine{k_en}: decision from kernel distances to the stored support vectors.",
            f"서포트 벡터 머신{k_en}: 저장된 support vector와의 커널 거리로 판별.",
        )

    # ---- quantization / casting ---------------------------------------------------
    if op_type == "DequantizeLinear":
        return _desc(
            "DequantizeLinear",
            "Converts quantized integers back to float using the stored scale/zero-point; "
            "this op exists because the served artifact is quantized.",
            "저장된 scale·zero-point로 양자화된 정수를 float로 복원. 서빙 아티팩트가 "
            "양자화되어 있어 존재하는 연산.",
            r"y = (q - z) \cdot s",
        )
    if op_type == "QuantizeLinear":
        return _desc(
            "QuantizeLinear",
            "Quantizes float values to integers with a scale/zero-point.",
            "scale·zero-point로 float 값을 정수로 양자화.",
            r"q = \mathrm{round}(x / s) + z",
        )
    if op_type == "Cast":
        return _desc(
            "Cast",
            "Converts the tensor's data type; no values are learned or transformed beyond dtype.",
            "텐서의 데이터 타입만 변환. dtype 외에는 값을 바꾸지 않음.",
        )
    if op_type in ("ReduceMean", "ReduceSum", "ReduceMax"):
        what = {"ReduceMean": ("mean", "평균"), "ReduceSum": ("sum", "합"),
                "ReduceMax": ("max", "최댓값")}[op_type]
        return _desc(
            op_type,
            f"Reduces the tensor by taking the {what[0]} along the given axes.",
            f"지정한 축을 따라 {what[1]}을 취해 텐서를 축소.",
        )

    # ---- honest fallback — names the real op, invents nothing ---------------------
    return _desc(
        op_type,
        f"ONNX `{op_type}` operator. No curated explanation yet. The shapes, parameter "
        "count and FLOPs shown are still measured from the real graph.",
        f"ONNX `{op_type}` 연산자입니다. 아직 준비된 설명이 없습니다. 함께 표시되는 shape·파라미터·"
        "FLOPs 값은 실제 그래프에서 측정된 값입니다.",
    )


# ── fallback for nodes with no ONNX op (input/output, weights-only layers) ──

_KIND_FALLBACK: dict[str, tuple[str, str, str]] = {
    # kind: (title, en, ko) — kinds for weights-only layers are inferred from
    # REAL tensor shapes (4-D weight → conv, running stats → bn, …).
    "conv": ("Convolution",
             "Convolution layer, identified from its real 4-D weight tensor.",
             "합성곱 레이어. 실제 4차원 weight 텐서 shape로 식별됨."),
    "bn": ("Batch normalization",
           "Normalization layer, identified from its real running-mean/variance tensors.",
           "정규화 레이어. 실제 running mean/variance 텐서로 식별됨."),
    "norm": ("Normalization",
             "Normalization layer, identified from its real 1-D scale/shift tensors.",
             "정규화 레이어. 실제 1차원 scale/shift 텐서로 식별됨."),
    "dense": ("Dense (Linear)",
              "Fully-connected layer, identified from its real 2-D weight matrix.",
              "Fully-connected 레이어. 실제 2차원 weight 행렬로 식별됨."),
    "embed": ("Embedding",
              "Embedding table, identified from its real 2-D lookup weight.",
              "임베딩 테이블. 실제 2차원 lookup weight로 식별됨."),
    "lstm": ("Recurrent cell",
             "Recurrent layer, identified from its real input/hidden weight tensors.",
             "Recurrent 레이어. 실제 input/hidden weight 텐서로 식별됨."),
    "attn": ("Attention",
             "Attention block, identified from its layer name.",
             "Attention 블록. 레이어 이름으로 식별됨."),
    "relu": ("Activation", "Nonlinear activation layer.", "비선형 활성 레이어."),
    "pool": ("Pooling", "Spatial downsampling layer.", "Spatial 다운샘플링 레이어."),
    "softmax": ("Softmax", "Probability normalization layer.", "확률 정규화 레이어."),
    "upsample": ("Upsample", "Spatial upsampling layer.", "Spatial 업샘플링 레이어."),
    "ffn": ("Feed-forward", "Feed-forward block.", "Feed-forward 블록."),
}


def describe_kind_fallback(kind: str, name: str, model_type: str) -> dict:
    """Honest fallback for nodes without per-op ONNX metadata."""
    if kind == "input":
        return _desc("Input",
                     f"Input tensor of this {model_type} model.",
                     f"이 {model_type} 모델의 입력 텐서.")
    if kind == "output":
        return _desc("Output",
                     f"Final output of this {model_type} model.",
                     f"이 {model_type} 모델의 최종 출력.")
    found = _KIND_FALLBACK.get(kind)
    if found:
        return _desc(found[0], found[1], found[2])
    return _desc(name,
                 f"A layer of this {model_type} model.",
                 f"이 {model_type} 모델의 레이어.")
