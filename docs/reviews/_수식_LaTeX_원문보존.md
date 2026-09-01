# 실증보고서 개정본 — 수식 LaTeX 원문 보존

생성 2026-09-01 · 대상 `AI실증보고서_다재약물_DDI_위험예측_개정.docx`

개정본에서 Word 네이티브 수식(OMML)으로 대체된 모든 수식의 LaTeX 원문 기록이다. 변환 후 문서에는 LaTeX 소스가 남지 않으므로, 재편집·재조판·검증 시 이 파일을 정본으로 삼는다.

- 독립 수식 **36개** (`m:oMathPara`)
- 인라인 수식 — 본문 **75개** · 표 셀 **45개**
- 문서 전체 `m:oMath` **156개** (독립 36 + 인라인 120)
- 원문 = 변환 직전 문서 텍스트, 복원 = 실제 변환 입력

---

## 1. 파손·정정 대조 (원문 ≠ 변환 입력)

변환 전에 수식을 재작성한 항목이다. 대부분 docx 변환 과정에서 `$$` 구분자와 첨자가 뒤엉켜 파손된 것이며, 일부는 집합 중괄호 미이스케이프·설정값 평문화다.

### P94 — $$ 구분자 뒤엉킴  (본문)

원문
```latex
$$\hat{y} = \arg\max_{c \in {0..85}} \; p(c \mid \mathbf{s}{1}, \mathbf{s}$$}), \qquad \mathbf{s}_i \in [0,1]^{1705
```
복원 / 대체
```latex
$$\hat{y} = \arg\max_{c \in \{0,\dots,85\}} \; p(c \mid \mathbf{s}_1, \mathbf{s}_2), \qquad \mathbf{s}_i \in [0,1]^{1705}$$
```

### P122 — $$ 구분자 뒤엉킴  (본문)

원문
```latex
\mathrm{head}^{(h)} = \mathrm{softmax}!\left(\frac{Q^{(h)} {K^{(h)}}^{\top}}{\sqrt{d_h}}\right) V^{(h)}, \qquad \mathbf{a} = \big[\mathrm{head}^{(1)}\Vert\cdots\Vert\mathrm{head}^{(8)}\big] W_O
```
복원 / 대체
```latex
\mathrm{head}^{(h)} = \mathrm{softmax}\!\left(\frac{Q^{(h)} {K^{(h)}}^{\top}}{\sqrt{d_h}}\right) V^{(h)}, \qquad \mathbf{a} = \big[\mathrm{head}^{(1)}\Vert\cdots\Vert\mathrm{head}^{(8)}\big] W_O
```

### P205 — $$ 구분자 뒤엉킴  (본문)

원문
```latex
$$w_{+} = \frac{|{y=0}|{\text{train}}}{\max(|{y=1}| \approx 272$$}},\,1)
```
복원 / 대체
```latex
$$w_{+} = \frac{|\{y=0\}|_{\text{train}}}{\max\!\big(|\{y=1\}|_{\text{train}},\,1\big)} \approx 272$$
```

### P207 — $$ 구분자 뒤엉킴  (본문)

원문
```latex
$$w_i = \underbrace{\frac{N}{K \cdot n_{c(i)}}}{\text{balanced (sklearn)}} \times \underbrace{\rho$$}}_{\text{cost_ratio (optional)}
```
복원 / 대체
```latex
$$w_i = \underbrace{\frac{N}{K \cdot n_{c(i)}}}_{\text{balanced (sklearn)}} \times \underbrace{\rho_{c(i)}}_{\text{cost ratio (optional)}}$$
```

### P277 — $$ 구분자 뒤엉킴  (본문)

원문
```latex
$$\hat{p}c = \frac{\exp(z_c/T)}{\sum_k \exp(z_k/T)}, \qquad T^{*} = \arg\min_T \mathrm{NLL}(T)$$}
```
복원 / 대체
```latex
$$\hat{p}_c = \frac{\exp(z_c/T)}{\sum_k \exp(z_k/T)}, \qquad T^{*} = \arg\min_T \mathrm{NLL}(T)$$
```

### P281 — $$ 구분자 뒤엉킴  (본문)

원문
```latex
$$\mathrm{Brier} = \frac{1}{N}\sum_{i}(\hat{p}i - y_i)^2 \;=\; \underbrace{\text{Reliability}}} - \underbrace{\text{Resolution}{\uparrow} + \underbrace{\text{Uncertainty}}$$}
```
복원 / 대체
```latex
$$\mathrm{Brier} = \frac{1}{N}\sum_{i}(\hat{p}_i - y_i)^2 \;=\; \underbrace{\text{Reliability}}_{\downarrow} - \underbrace{\text{Resolution}}_{\uparrow} + \underbrace{\text{Uncertainty}}_{\text{const}}$$
```

### P300 — $$ 구분자 뒤엉킴  (본문)

원문
```latex
$$\mathbf{e}j = \mathbf{E}}}[d_j] + \mathbf{E{\text{inst}}[\iota_j] + \mathrm{PE}_\delta \delta_j$$}}(t_j) + \mathbf{W
```
복원 / 대체
```latex
$$\mathbf{e}_j = \mathbf{E}_{\text{drug}}[d_j] + \mathbf{E}_{\text{inst}}[\iota_j] + \mathrm{PE}_{\text{time}}(t_j) + \mathbf{W}_{\delta}\,\delta_j$$
```

### P302 — $$ 구분자 뒤엉킴  (본문)

원문
```latex
\mathrm{PE}_{\text{time}}(t)[i] = \begin{cases} \omega_0 t + \varphi_0 & i=0 \ \sin(\omega_i t + \varphi_i) & 1 \le i < d \end{cases}
```
복원 / 대체
```latex
\mathrm{PE}_{\text{time}}(t)[i] = \begin{cases} \omega_0 t + \varphi_0 & i=0 \\ \sin(\omega_i t + \varphi_i) & 1 \le i < d \end{cases}
```

### P646 — $$ 구분자 뒤엉킴  (본문)

원문
```latex
$$h_v^{(l+1)} = \sigma\Bigg(W_0^{(l)}h_v^{(l)} + \sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}r(v)} \frac{1}{c\Bigg)$$}} W_r^{(l)} h_u^{(l)
```
복원 / 대체
```latex
$$h_v^{(l+1)} = \sigma\Bigg(W_0^{(l)}h_v^{(l)} + \sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_r(v)} \frac{1}{c_{v,r}} W_r^{(l)} h_u^{(l)}\Bigg)$$
```

### P656 — $$ 구분자 뒤엉킴  (본문)

원문
```latex
$$\mathbf{z}(x) = \Big[\underbrace{\hat{p}{\text{xgb}}, \hat{p}}}, \hat{p{\text{rf}}, \hat{p}}}{\text{tabular}} \;\Big\Vert\; \underbrace{\hat{p}}}, \hat{p{\text{seq}}}\Big]$$}} \;\Big\Vert\; \underbrace{\mathbb{1}[\text{RED_CONTRA}], \mathbb{1}[\text{DDI_MAJOR}], \mathbb{1}[\text{SEV_*}], \mathrm{ord}(L_{\text{rule}})}_{\text{룰 신호}
```
복원 / 대체
```latex
$$\mathbf{z}(x) = \Big[\underbrace{\hat{p}_{\text{xgb}}, \hat{p}_{\text{lgbm}}, \hat{p}_{\text{rf}}}_{\text{tabular}} \;\Big\Vert\; \underbrace{\hat{p}_{\text{gat}}, \hat{p}_{\text{seq}}}_{\text{표현학습}} \;\Big\Vert\; \underbrace{\mathbb{1}[\text{RED CONTRA}],\, \mathbb{1}[\text{DDI MAJOR}],\, \mathbb{1}[\text{SEV}],\, \mathrm{ord}(L_{\text{rule}})}_{\text{룰 신호}}\Big]$$
```

### P657 — $$ 구분자 뒤엉킴  (본문)

원문
```latex
$$\hat{p}{\text{final}} = g(x))$$}(\mathbf{z
```
복원 / 대체
```latex
$$\hat{p}_{\text{final}} = g_{\theta}\big(\mathbf{z}(x)\big)$$
```

### P149 — 인라인 수식 분절  (본문)

원문
```latex
논증 1 — 특징공간의 폐쇄성(closedness). 임베딩 테이블 방식(drug_id → embedding)은 정의역이 학습 어휘 $\mathcal{V}{\text{train}} 에 국한되어, d \notin \mathcal{V}$ 이면 표현이 }정의되지 않는다(OOV). 반면 SSP의 정의역은 "유효한 SMILES를 갖는 모든 분자"이며, 좌표계는 고정된 기준집합 \mathcal{R} 이다. 신약 $d^{} 가 등장해도 \mathrm{SSP}(d^{}) 는 \mathcal{R} 에 대한 M$ 회 Tanimoto 계산만으로 즉시 산출되며, 학습 시점과 동일한 좌표계의 벡터를 얻는다. 즉 SSP는 학습 데이터에 의존하는 파라메트릭 표현이 아니라 데이터 독립적 사상(deterministic map) 이다.
```
복원 / 대체
```latex
논증 1 — 특징공간의 폐쇄성(closedness). 임베딩 테이블 방식(drug_id → embedding)은 정의역이 학습 어휘  에 국한되어,  이면 표현이 정의되지 않는다(OOV). 반면 SSP의 정의역은 "유효한 SMILES를 갖는 모든 분자"이며, 좌표계는 고정된 기준집합  이다. 신약  가 등장해도  는  에 대한  회 Tanimoto 계산만으로 즉시 산출되며, 학습 시점과 동일한 좌표계의 벡터를 얻는다. 즉 SSP는 학습 데이터에 의존하는 파라메트릭 표현이 아니라 데이터 독립적 사상(deterministic map) 이다.
```

### P303 — 인라인 수식 분절  (본문)

원문
```latex
Causal masking: \mathrm{Attn} = \mathrm{softmax}!\big(\frac{QK^\top}{\sqrt{d_k}} + M\big)V, M_{jk} = -\infty if t_k > t_j. 미래 정보 누수를 구조적으로 차단한다 — 이는 sparse_linear 모델 카드가 지적한 same-window 라벨 문제(모델이 시계열 예측 우위를 보이지 못한 원인)의 직접 대응이다.
```
복원 / 대체
```latex
Causal masking: ,  if . 미래 정보 누수를 구조적으로 차단한다 — 이는 sparse_linear 모델 카드가 지적한 same-window 라벨 문제(모델이 시계열 예측 우위를 보이지 못한 원인)의 직접 대응이다.
```

### P633 — 인라인 수식 분절  (본문)

원문
```latex
보정셋 $\mathcal{D}{\text{cal}} (n개, 학습셋과 교환가능·독립)에서 비적합도 점수 계산. - 이진(Stage 1): s_i = 1 - \hat{p}(y_i \mid x_i)$ - 다중분류(Stage 2 / 86-class): APS(Adaptive Prediction Sets) — 확률 내림차순 누적합 $s_i = \sumc \ge \hat{p}_c$}} \hat{p
```
복원 / 대체
```latex
보정셋  (개, 학습셋과 교환가능·독립)에서 비적합도 점수 계산. — 이진(Stage 1):   — 다중분류(Stage 2 / 86-class): APS(Adaptive Prediction Sets), 확률 내림차순 누적합
```

### P150 — 파손 복원  (본문)

원문
```latex
$\max_j T(d^{}, r_j)$
support 내부에 착지*한다
```
복원 / 대체
```latex
\max_j T(d^{*}, r_j)
support 내부에 착지한다
```

### P210 — 설정값 평문화  (본문)

원문
```latex
\rho = {Normal:1.0, Green:1.5, Yellow:0.7\cdot\frac{c_{FN}}{c_{FP}}, Red:\frac{c_{FN}}{c_{FP}}}
c_{FP}{=}1.0, c_{FN}{=}5.0
```
복원 / 대체
```latex
ρ = {Normal: 1.0, Green: 1.5, Yellow: 0.7 × (c_FN / c_FP), Red: c_FN / c_FP}
c_FP = 1.0, c_FN = 5.0
```

### P139 — 집합 중괄호 미이스케이프  (본문)

원문
```latex
\phi(d) \in {0,1}^{2048}, \qquad \phi(d)_j = 1 \iff \text{반경 2 이내 부분구조 해시가 } j \text{에 매핑}
```
복원 / 대체
```latex
\phi(d) \in \{0,1\}^{2048}, …(중괄호 이스케이프)
```

### P214 — 집합 중괄호 미이스케이프  (본문)

원문
```latex
\tau_{\text{review}} = \max{\,t \in \mathcal{T} \;:\; R(t) \ge 0.98\,}
```
복원 / 대체
```latex
\tau_{\text{review}} = \max\{\,t \in \mathcal{T} \;:\; R(t) \ge 0.98\,\}
```

### P649 — 파손 복원  (본문)

원문
```latex
\alpha^{r}{vu} = \frac{\exp\big(\mathrm{LeakyReLU}(\mathbf{a}_r^{\top}[W_r h_v \Vert W_r h_u])\big)}{\sumr(v)}\exp\big(\mathrm{LeakyReLU}(\mathbf{a}_r^{\top}[W_r h_v \Vert W_r h_k])\big)}, \qquad h_v = \sum}\beta_r \sum_{u\in\mathcal{Nr(v)}\alpha^{r}W_r h_u
```
복원 / 대체
```latex
\alpha^{r}_{vu} = \frac{\exp\big(\mathrm{LeakyReLU}(\mathbf{a}_r^{\top}[W_r h_v \Vert W_r h_u])\big){\sum_{k \in \mathcal{N}_r(v)}\exp\big(\mathrm{LeakyReLU}(\mathbf{a}_r^{\top}[W_r h_v \Vert W_r h_k])\big)}, \qquad h_v = \sum_{r \in \mathcal{R}}\beta_r \sum_{u \in \mathcal{N}_r(v)}\alpha^{r}_{vu} W_r h_u
```

### T38 r2c1 — 파손 복원  (표)

원문
```latex
$\mathcal{L} = \mathcal{L}{\text{CE}} + \lambda!!\sum\big)^2$ (미분가능 surrogate)}!\big(R_a - R_{a'
```
복원 / 대체
```latex
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda\sum_{a,a'}\big(R_a - R_{a'}\big)^2 (미분가능 surrogate)
```

### T27 r18c4 — 설정값 평문화  (표)

원문
```latex
w={3A4:3.0,\,2D6:2.0,\,2C9:2.0,\,2C19:1.5,\,1A2:1.0}
```
복원 / 대체
```latex
w = {3A4: 3.0, 2D6: 2.0, 2C9: 2.0, 2C19: 1.5, 1A2: 1.0}
```

### T40 r1c1 — 간격 매크로 정리  (표)

원문
```latex
p{=}3.9\times10^{-35}
```
복원 / 대체
```latex
p = 3.9\times10^{-35}
```

---

## 2. 독립 수식 (문단 전체 → `m:oMathPara`)

### 2.1 1차 복원분 11개

**P94**
```latex
$$\hat{y} = \arg\max_{c \in \{0,\dots,85\}} \; p(c \mid \mathbf{s}_1, \mathbf{s}_2), \qquad \mathbf{s}_i \in [0,1]^{1705}$$
```
**P122**
```latex
\mathrm{head}^{(h)} = \mathrm{softmax}\!\left(\frac{Q^{(h)} {K^{(h)}}^{\top}}{\sqrt{d_h}}\right) V^{(h)}, \qquad \mathbf{a} = \big[\mathrm{head}^{(1)}\Vert\cdots\Vert\mathrm{head}^{(8)}\big] W_O
```
**P205**
```latex
$$w_{+} = \frac{|\{y=0\}|_{\text{train}}}{\max\!\big(|\{y=1\}|_{\text{train}},\,1\big)} \approx 272$$
```
**P207**
```latex
$$w_i = \underbrace{\frac{N}{K \cdot n_{c(i)}}}_{\text{balanced (sklearn)}} \times \underbrace{\rho_{c(i)}}_{\text{cost ratio (optional)}}$$
```
**P277**
```latex
$$\hat{p}_c = \frac{\exp(z_c/T)}{\sum_k \exp(z_k/T)}, \qquad T^{*} = \arg\min_T \mathrm{NLL}(T)$$
```
**P281**
```latex
$$\mathrm{Brier} = \frac{1}{N}\sum_{i}(\hat{p}_i - y_i)^2 \;=\; \underbrace{\text{Reliability}}_{\downarrow} - \underbrace{\text{Resolution}}_{\uparrow} + \underbrace{\text{Uncertainty}}_{\text{const}}$$
```
**P300**
```latex
$$\mathbf{e}_j = \mathbf{E}_{\text{drug}}[d_j] + \mathbf{E}_{\text{inst}}[\iota_j] + \mathrm{PE}_{\text{time}}(t_j) + \mathbf{W}_{\delta}\,\delta_j$$
```
**P302**
```latex
\mathrm{PE}_{\text{time}}(t)[i] = \begin{cases} \omega_0 t + \varphi_0 & i=0 \\ \sin(\omega_i t + \varphi_i) & 1 \le i < d \end{cases}
```
**P646**
```latex
$$h_v^{(l+1)} = \sigma\Bigg(W_0^{(l)}h_v^{(l)} + \sum_{r \in \mathcal{R}} \sum_{u \in \mathcal{N}_r(v)} \frac{1}{c_{v,r}} W_r^{(l)} h_u^{(l)}\Bigg)$$
```
**P656**
```latex
$$\mathbf{z}(x) = \Big[\underbrace{\hat{p}_{\text{xgb}}, \hat{p}_{\text{lgbm}}, \hat{p}_{\text{rf}}}_{\text{tabular}} \;\Big\Vert\; \underbrace{\hat{p}_{\text{gat}}, \hat{p}_{\text{seq}}}_{\text{표현학습}} \;\Big\Vert\; \underbrace{\mathbb{1}[\text{RED CONTRA}],\, \mathbb{1}[\text{DDI MAJOR}],\, \mathbb{1}[\text{SEV}],\, \mathrm{ord}(L_{\text{rule}})}_{\text{룰 신호}}\Big]$$
```
**P657**
```latex
$$\hat{p}_{\text{final}} = g_{\theta}\big(\mathbf{z}(x)\big)$$
```

### 2.2 2차 변환분 20개

**P98**
```latex
\mathbf{x} = [\mathbf{s}_1 \Vert \mathbf{s}_2] \in \mathbb{R}^{3410}
```
**P100**
```latex
\mathbf{h}^{(l)} = \mathrm{Dropout}_{0.1}\Big(\mathrm{ReLU}\big(\mathrm{BN}(W^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})\big)\Big), \quad \mathbf{h}^{(0)}=\mathbf{x}
```
**P101**
```latex
\mathrm{BN}(z) = \gamma \odot \frac{z-\mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}}+\epsilon}} + \beta
```
**P107**
```latex
\mathcal{F}(\mathbf{h}) = \mathrm{BN}_2\Big(W_2 \cdot \mathrm{Dropout}\big(\mathrm{ReLU}(\mathrm{BN}_1(W_1\mathbf{h}+\mathbf{b}_1))\big) + \mathbf{b}_2\Big)
```
**P108**
```latex
\mathbf{h}' = \mathrm{ReLU}\big(\mathbf{h} + \mathcal{F}(\mathbf{h})\big)
```
**P110**
```latex
\mathbf{h}^{(0)} = \mathrm{ReLU}(W_{\text{proj}}\mathbf{x} + \mathbf{b}_{\text{proj}}), \qquad \mathbf{h}^{(l)} = \mathrm{ReLU}\big(\mathbf{h}^{(l-1)} + \mathcal{F}_l(\mathbf{h}^{(l-1)})\big)
```
**P112**
```latex
\frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}} = \mathrm{diag}\big(\mathbb{1}[\mathbf{h}^{(l-1)}+\mathcal{F}_l > 0]\big)\Big(I + \frac{\partial \mathcal{F}_l}{\partial \mathbf{h}^{(l-1)}}\Big)
```
**P114**
```latex
\frac{\partial L}{\partial \mathbf{h}^{(0)}} \approx \frac{\partial L}{\partial \mathbf{h}^{(L)}}\prod_{l=1}^{L}\Big(I + \frac{\partial \mathcal{F}_l}{\partial \mathbf{h}^{(l-1)}}\Big)
```
**P119**
```latex
\mathbf{u}_1 = W_p\mathbf{s}_1 + \mathbf{b}_p, \qquad \mathbf{u}_2 = W_p\mathbf{s}_2 + \mathbf{b}_p, \qquad \mathbf{u}_i \in \mathbb{R}^{2048}
```
**P124**
```latex
\mathbf{h} = \mathrm{Dropout}\big(\mathrm{ReLU}(W_f[\mathbf{u}_1 \Vert \mathbf{a}] + \mathbf{b}_f)\big), \qquad \mathbf{z} = W_c\mathbf{h}+\mathbf{b}_c
```
**P139**
```latex
\phi(d) \in {0,1}^{2048}, \qquad \phi(d)_j = 1 \iff \text{반경 2 이내 부분구조 해시가 } j \text{에 매핑}
```
**P141**
```latex
\mathrm{SSP}(d) = \big[T(d,r_1),\; T(d,r_2),\; \dots,\; T(d,r_M)\big] \in [0,1]^{M}
```
**P213**
```latex
\tau_{\text{red}} = \arg\max_{\;t \in \mathcal{T},\; R(t) \ge 0.90} \; P(t)
```
**P214**
```latex
\tau_{\text{review}} = \max{\,t \in \mathcal{T} \;:\; R(t) \ge 0.98\,}
```
**P224**
```latex
\tau_{\text{red}} - \tau_{\text{review}} = 1.0 \times 10^{-6}
```
**P273**
```latex
\hat{p}_{\text{cal}} = \sigma(a \cdot z + b), \qquad (a,b) = \arg\min \; -\sum_i \big[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\big]
```
**P280**
```latex
\mathrm{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{N}\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|, \qquad \mathrm{MCE} = \max_{m}\big|\mathrm{acc}(B_m)-\mathrm{conf}(B_m)\big|
```
**P294**
```latex
\max_{\pi} \; \mathbb{E}[U] \quad \text{s.t.} \quad \mathbb{E}\big[\mathbb{1}[a=\text{약사 전화}]\big] \cdot N_{\text{daily}} \le B
```
**P312**
```latex
\Delta_{\text{EOpp}} = \max_{a, a' \in \mathcal{A}} \big| \mathbb{P}(\hat{Y}{=}1 \mid Y{=}1, A{=}a) - \mathbb{P}(\hat{Y}{=}1 \mid Y{=}1, A{=}a') \big|
```
**P649**
```latex
\alpha^{r}{vu} = \frac{\exp\big(\mathrm{LeakyReLU}(\mathbf{a}_r^{\top}[W_r h_v \Vert W_r h_u])\big)}{\sumr(v)}\exp\big(\mathrm{LeakyReLU}(\mathbf{a}_r^{\top}[W_r h_v \Vert W_r h_k])\big)}, \qquad h_v = \sum}\beta_r \sum_{u\in\mathcal{Nr(v)}\alpha^{r}W_r h_u
```

### 2.3 3차 — 초기 분류 누락분 5개

역슬래시로 시작하지 않아(`Q^{(h)} = …` 등) 최초 분류에서 빠졌던 독립 수식이다.

**P121**
```latex
Q^{(h)} = \mathbf{u}_1 W_Q^{(h)},\quad K^{(h)} = \mathbf{u}_2 W_K^{(h)},\quad V^{(h)} = \mathbf{u}_2 W_V^{(h)}
```
**P142**
```latex
T(d, r) = \frac{|\phi(d) \cap \phi(r)|}{|\phi(d) \cup \phi(r)|} = \frac{\phi(d)^{\top}\phi(r)}{|\phi(d)|_1 + |\phi(r)|_1 - \phi(d)^{\top}\phi(r)}
```
**P162**
```latex
L_{\text{final}} = \max\big(L_{\text{rule}},\; L_{\text{ML}},\; L_{\text{backstop}}\big), \qquad \mathrm{ord}(\text{Normal}{=}0 < \text{Green}{=}1 < \text{Yellow}{=}2 < \text{Red}{=}3)
```
**P275**
```latex
g^{*} = \arg\min_{g \in \mathcal{G}_{\uparrow}} \sum_i \big(y_i - g(p_i)\big)^2
```
**P289**
```latex
a^{*}(x) = \arg\max_{a \in \mathcal{A}} \; \sum_{y} \hat{p}_{\text{cal}}(y \mid x)\, U(a, y)
```

---

## 3. 본문 인라인 수식

문단 원문 전체와 수식으로 변환한 구간을 함께 남긴다.

### 3.1 1차 복원분 3개

**P149** (수식 7개) — 복원 후 전문
```latex
논증 1 — 특징공간의 폐쇄성(closedness). 임베딩 테이블 방식(drug_id → embedding)은 정의역이 학습 어휘  에 국한되어,  이면 표현이 정의되지 않는다(OOV). 반면 SSP의 정의역은 "유효한 SMILES를 갖는 모든 분자"이며, 좌표계는 고정된 기준집합  이다. 신약  가 등장해도  는  에 대한  회 Tanimoto 계산만으로 즉시 산출되며, 학습 시점과 동일한 좌표계의 벡터를 얻는다. 즉 SSP는 학습 데이터에 의존하는 파라메트릭 표현이 아니라 데이터 독립적 사상(deterministic map) 이다.
```
**P303** (수식 3개) — 복원 후 전문
```latex
Causal masking: ,  if . 미래 정보 누수를 구조적으로 차단한다 — 이는 sparse_linear 모델 카드가 지적한 same-window 라벨 문제(모델이 시계열 예측 우위를 보이지 못한 원인)의 직접 대응이다.
```
**P633** (수식 4개) — 복원 후 전문
```latex
보정셋  (개, 학습셋과 교환가능·독립)에서 비적합도 점수 계산. — 이진(Stage 1):   — 다중분류(Stage 2 / 86-class): APS(Adaptive Prediction Sets), 확률 내림차순 누적합
```

### 3.2 자동 판정 21개 · 수동 지정 15개

**P93** [수동] 수식 2개
```latex
약물 쌍 (d_1, d_2) 에 대해 상호작용 유형 y \in {0,\dots,85} 를 예측하는 다중 클래스 단일 라벨 분류로 정식화했다. 86개 클래스는 domain_team/labeling/ddi_taxonomy.py 에 DDIType(class_id, short_name, template, mechanism, severity, direction) 구조로 정의되어 있으며, 각 클래스는 기전(mechanism) 과 심각도(severity ∈ {mild, moderate, severe, contraindicated}) 메타데이터를 동반한다. 즉 모델의 86-class 출력은 단순 라벨이 아니라 자연어 설명문(template) + 임상 심각도로 자동 번역 가능한 구조다.
```
변환 구간: `(d_1, d_2)` · `y \in \{0,\dots,85\}`

**P102** [자동] 수식 2개
```latex
출력은 \mathbf{z} = W^{\text{cls}}\mathbf{h}^{(3)} + \mathbf{b}^{\text{cls}} \in \mathbb{R}^{86}, 확률은 p_c = \exp(z_c)/\sum_k \exp(z_k).
```
**P115** [수동] 수식 2개
```latex
가 되어, |\partial\mathcal{F}/\partial\mathbf{h}| \ll 1 인 초기 학습 구간에서도 곱이 0 으로 지수 감쇠하지 않는다. 평문 MLP에서는 동일 위치의 야코비안이 \prod W^{(l)} 형태라 스펙트럼 반경이 1 미만이면 지수적으로 소멸한다. 이것이 실측에서 ResDeepDDI가 best_epoch 22 로 baseline보다 빠르게 수렴하고 Macro-F1이 +0.0805 개선된 메커니즘적 이유다.
```
변환 구간: `|\partial\mathcal{F}/\partial\mathbf{h}| \ll 1` · `\prod W^{(l)}`

**P123** [수동] 수식 2개
```latex
본 구현에서 각 약물은 시퀀스 길이 1의 토큰(unsqueeze(1) → shape (B,1,H))으로 취급되므로 attention 가중치 행렬은 1\times1 이고 softmax 결과는 항등적으로 1이 된다. 따라서 실질적으로 \mathbf{a} \approx \mathbf{u}_2 W_V W_O 로, 학습 가능한 값 변환(value projection) 으로 기능한다. 이후 원본과 결합해 FFN에 통과시킨다.
```
변환 구간: `1\times1` · `\mathbf{a} \approx \mathbf{u}_2 W_V W_O`

**P132** [수동] 수식 1개
```latex
최종 선택 = ResDeepDDI. 근거는 세 가지다. ① 4개 지표 전부에서 우위이며 특히 AUPR +0.0492, Macro-F1 +0.0805 로 희귀 클래스 성능 개선이 큼(불균형 데이터에서 Accuracy보다 신뢰할 지표). ② McNemar 검정 b=708, c=312, p=3.90\times10^{-35} 로 baseline 대비 개선이 통계적으로 유의(validation_summary.md). ③ Bootstrap 95% CI(n=1000)에서 Accuracy [0.9316, 0.9386]로 구간이 baseline을 포함하지 않음. 반면 클래스 가중을 적용한 ResDeepDDI (Weighted) 는 AUPR은 0.94로 유지되나 Accuracy·Macro-F1이 모두 하락해 채택하지 않았다.
```
변환 구간: `p=3.90\times10^{-35}`

**P140** [수동] 수식 2개
```latex
기준 약물 집합 \mathcal{R} = {r_1,\dots,r_M} (M=1{,}705, DeepDDI 논문 관례에 따라 전체 약물 집합) 에 대해, SSP는 Tanimoto(Jaccard) 유사도 벡터로 정의된다.
```
변환 구간: `\mathcal{R} = \{r_1,\dots,r_M\}` · `M = 1{,}705`

**P146** [수동] 수식 2개
```latex
이진 벡터에서 |\phi_a \cap \phi_b| = \phi_a^\top\phi_b 이고 |\phi_a \cup \phi_b| = |\phi_a| + |\phi_b| - |\phi_a\cap\phi_b| 라는 포함배제 항등식을 사용해, 단일 GEMM으로 1,705×1,705 유사도 행렬을 계산한다. U > 0 가드는 빈 fingerprint(파싱 실패 분자)로 인한 0-division을 차단한다.
```
변환 구간: `|\phi_a \cap \phi_b| = \phi_a^\top\phi_b` · `|\phi_a \cup \phi_b| = |\phi_a| + |\phi_b| - |\phi_a\cap\phi_b|`

**P150** [수동] 수식 3개
```latex
논증 2 — 분포 이동의 부재. 모델이 학습한 함수는 f: [0,1]^{3410} \to \Delta^{85} 이다. 미등장 약물의 SSP 벡터도 기존 약물과 동일한 [0,1]^{1705} 하이퍼큐브 안에 놓인다. 신약이 기존 약물과 완전히 이질적 골격이라면 SSP는 0에 가까운 희소 벡터가 되어 분포 밖(out-of-support)이 되겠지만, 실제 승인 약물은 리드 최적화 과정상 기존 약물과 골격을 공유하는 경우가 지배적이므로 $\max_j T(d^{}, r_j)$ 가 충분히 커서 학습 분포의 support 내부에 착지*한다.
```
변환 구간: `f: [0,1]^{3410} \to \Delta^{85}` · `[0,1]^{1705}` · `\max_j T(d^{*}, r_j)`

**P191** [자동] 수식 1개
```latex
Stage 1 (이진): y_1 = \mathbb{1}[\texttt{risk_level} = \text{Red}]. 정확히 ddi_contraindicated ≥ 1 에 대응.
```
**P198** [수동] 수식 1개
```latex
② global↔local 재매핑: XGBoost는 y \in [0, \text{num_class}) 연속 정수를 요구하므로, 학습셋에 없는 클래스가 있으면 classes_present 로 압축 매핑하고 그 배열을 번들에 저장한다.
```
변환 구간: `y \in [0, \text{num\_class})`

**P215** [자동] 수식 3개
```latex
여기서 P(t), R(t) 는 sklearn.precision_recall_curve 의 Precision·Recall이다. \tau_{\text{red}} 는 "Red 재현율 90% 하한을 지키면서 정밀도 최대", \tau_{\text{review}} 는 "재현율 98%를 보장하는 가장 엄격한(높은) 임계값"으로, 설계상 \tau_{\text{review}} < \tau_{\text{red}} 가 성립해야 한다.
```
**P220** [자동] 수식 2개
```latex
더미는 predict_proba(X)[:,1] ≡ 0.0 이며 임계값을 \tau_{\text{red}}{=}1.0, \tau_{\text{review}}{=}0.5 로 강제한다(불변식 \tau_{\text{review}}<\tau_{\text{red}} 유지, 모든 샘플이 Stage 2로 분기, red_suspect=False). 이 상태는 stage_meta.stage1_trained=False 로 투명하게 기록되며, 배포 시 Red 미탐지 한계를 UI·메타에서 경고한다. 운영 번들은 stage1_trained=true, stage1_red_count=1660 으로 정상 학습되었다.
```
**P227** [자동] 수식 1개
```latex
진단 1 — 임계값 붕괴(threshold collapse)의 원인. Stage 1의 정답 라벨 y_1 = \mathbb{1}[\texttt{ddi_contraindicated} \ge 1] 은 입력 피처 4번(ddi_contraindicated)의 결정론적 함수다. 즉 라벨이 피처 안에 그대로 들어 있다(구조적 라벨 누수, rulefeat leakage). XGBoost는 이 규칙을 거의 완벽히 학습하고, 그 결과 예측 확률 분포가 0 근처와 1 근처로 양극화된다. PR 곡선상에서 Recall 0.90 지점과 0.98 지점이 사실상 같은 임계값에 대응하므로 두 τ가 동일해지고, 코드가 1e-6 간격을 강제한다. 이는 모델 결함이 아니라 라벨-피처 관계에서 필연적으로 유도되는 현상이다(작업일지 2026-06-07: "Red 가 룰파생 결정적 라벨이라 ML 이 룰 학습→점수 양극화→threshold collapse(정상)").
```
**P228** [자동] 수식 2개
```latex
진단 2 — 운영상 귀결. review band 폭이 10^{-6} 이므로 \tau_{\text{review}} \le p_{\text{red}} < \tau_{\text{red}} 구간에 들어오는 환자는 사실상 0명이다. 즉 점수 기반 red_suspect 검수 큐가 무력화(inert) 되어 있다. 계층 구조의 안전장치 하나가 설계 의도대로 작동하지 않는다.
```
**P229** [자동] 수식 1개
```latex
진단 3 — 과확신(overconfidence)의 실체. p_{\text{red}} \approx 0.99997 는 캘리브레이션된 확률이 아니다. 모델은 "이 환자가 Red일 확률 99.997%"라고 말하고 있으나, 이는 규칙을 재현한 결과일 뿐 불확실성 정보를 담고 있지 않다. Brier score·reliability diagram 등 보정도 지표는 현행 파이프라인에서 compute_stage1_metrics() 가 계산은 하지만(pr_auc, roc_auc, brier), 보정 자체는 수행하지 않는다. 초기모델 쪽에서도 동일 현상이 관찰되어 ECE=0.0406에 "P50 신뢰도 0.9999, 중간 구간(0.80~0.93) 과신 gap +21%p"로 기록되어 있다.
```
**P238** [자동] 수식 1개
```latex
τ 분산 노출: _tau_variance() 가 폴드별 \tau_{\text{red}}/\tau_{\text{review}} 의 mean·std·min·max를 반환. 변동이 크면 임계값 비안정 신호.
```
**P249** [자동] 수식 3개
```latex
연구 트랙 GAT (scripts/train/gat_model.py, gat_trainer.py): 2-layer GATConv(feature_dim=3 → hidden 64×heads 4 → out 32) + pair scorer \mathrm{MLP}([h_a \Vert h_b \Vert |h_a - h_b| \Vert h_a \odot h_b]). 노드 피처는 [log1p(degree), log1p(ddi_count), log1p(freq)], 엣지는 동일 환자·동일 처방일 공동처방 조합에 \log(1+\text{count}) 가중. 부정쌍 샘플링 NEG_POS_RATIO:1, 쌍 분할 train 80% / gat_val 10% / calibration 10%, BCELoss + Adam, gat_val AUC 기준 early stopping(patience 20). GraphBuilder.build() 는 train split 데이터만 허용하며 val/test 유입 시 RuntimeError 로 누출을 구조적으로 차단한다. EnsembleTrainer3Way 는 XGB+LGBM+GAT 소프트보팅 가중치를 Recall ≥ 0.90 제약 하 AUC 최대화(SLSQP) 로 최적화하고, 미지 약물 포함 시 w_{\text{gat}}=0 후 재정규화한다. 주목: GATTrainer.calibrate() 는 Platt scaling(LogisticRegression on raw scores)을 이미 구현하고 있어, 2.8절 ① 제안의 선례가 사내에 존재한다.
```
**P264** [자동] 수식 1개
```latex
confidence 는 이진에서는 예측 클래스의 확률, 다중분류에서는 \max_c p_c 로 산출한다. 이 사례집은 DOCX 보고서에 익명번호와 안전 피처 요약만으로 실린다. 2026-06-25 작업에서 모델 비교 섹션(5-5)이 저장 이력만 보던 결함을 수정해, 현재 세션의 ML·DL 학습 결과가 ML / DL / Hierarchical 구분 컬럼과 함께 Accuracy·F1·AUC 비교표·그래프로 통합되도록 개선했다.
```
**P271** [자동] 수식 2개
```latex
해결 대상: \tau_{\text{red}}=0.99997, review band 폭 10^{-6} 의 과확신(2.5.6). 초기모델 측 "중간 신뢰도 구간 과신 gap +21%p".
```
**P282** [수동] 수식 1개
```latex
Brier 분해를 함께 보고해 보정 개선이 판별력 희생 없이 이뤄졌는지 확인한다. Reliability diagram은 M=10 또는 M=15 등폭 구간으로 (\mathrm{conf}, \mathrm{acc}) 를 대각선 대비 시각화한다.
```
변환 구간: `(\mathrm{conf}, \mathrm{acc})`

**P288** [수동] 수식 2개
```latex
베이즈 최적 결정: 효용행렬 U(a, y) (행동 a \in \mathcal{A}, 실제 상태 y)에 대해
```
변환 구간: `U(a, y)` · `a \in \mathcal{A}`

**P290** [자동] 수식 1개
```latex
\hat{p}_{\text{cal}} 은 ①의 보정 확률이어야 한다. 미보정 확률로 기댓값을 계산하면 결정이론의 전제가 무너진다 — ①이 ③의 선결조건인 이유다.
```
**P292** [수동] 수식 2개
```latex
이진 특수해로는 \tau^{*} = \dfrac{U(\text{no-act},\,\text{neg}) - U(\text{act},\,\text{neg})}{\big[U(\text{act},\text{pos})-U(\text{no-act},\text{pos})\big] + \big[U(\text{no-act},\text{neg})-U(\text{act},\text{neg})\big]} 가 되며, c_{FN}/c_{FP} 비율이 커질수록 임계값이 낮아지는 관계가 명시적으로 도출된다.
```
변환 구간: `\tau^{*} = \dfrac{U(\text{no-act},\,\text{neg}) - U(\text{act},\,\text{neg})}{\big[U(\text{act},\text{pos})-U(\text{no-act},\text{pos})\big] + \big[U(\text{no-act},\text{neg})-U(\text{act},\text{neg})\big]}` · `c_{FN}/c_{FP}`

**P295** [자동] 수식 2개
```latex
이는 Net Benefit / Decision Curve Analysis 로 시각화한다: \mathrm{NB}(\tau) = \frac{TP(\tau)}{N} - \frac{FP(\tau)}{N}\cdot\frac{\tau}{1-\tau} 를 threshold probability 범위에서 그려, 어느 임계 구간에서 모델이 "전부 개입"·"전부 미개입" 전략을 능가하는지 제시한다. 기존 _optimize_threshold() (0.10~0.90 격자 탐색, \text{cost} = c_{FP}\cdot FP + c_{FN}\cdot FN)를 다중행동·효용행렬로 일반화하는 형태가 된다.
```
**P299** [수동] 수식 3개
```latex
입력 표현: 환자 p 의 처방 이벤트 시퀀스 \mathcal{S}_p = \big[(d_1,t_1,\iota_1,\delta_1), \dots, (d_L,t_L,\iota_L,\delta_L)\big] (d=약물, t=처방일, \iota=기관, \delta=투여일수).
```
변환 구간: `\mathcal{S}_p = \big[(d_1,t_1,\iota_1,\delta_1), \dots, (d_L,t_L,\iota_L,\delta_L)\big]` · `\iota` · `\delta`

**P316** [자동] 수식 2개
```latex
리포팅: hierarchical_metrics 에 subgroup 섹션 추가 — 집단별 Recall·Precision·PR-AUC·Calibration-in-the-large(①과 결합: 집단별 ECE), 그리고 \Delta_{\text{EOpp}}. 수용 기준(안): \Delta_{\text{EOpp}} \le 0.05, 어느 집단도 Recall이 전체 대비 0.05 이상 낮지 않을 것. 위반 시 배포 게이트 차단.
```
**P347** [자동] 수식 2개
```latex
확률 보정 부재가 가장 시급하다. \tau_{\text{red}}{=}0.99997, band 폭 10^{-6} 은 review 큐를 무력화했고, 현행 시스템은 이를 결정적 백스톱으로 우회하고 있을 뿐 해결하지 않았다. ②③⑥⑧ 다수 제안이 ①을 선결조건으로 갖는다.
```
**P634** [자동] 수식 1개
```latex
분위수 \hat{q} = \mathrm{Quantile}\big({s_i}; \tfrac{\lceil (n+1)(1-\alpha)\rceil}{n}\big)
```
**P635** [수동] 수식 1개
```latex
신규 x 의 예측집합 C(x) = {c : s(x,c) \le \hat{q}}
```
변환 구간: `C(x) = \{c : s(x,c) \le \hat{q}\}`

**P636** [자동] 수식 1개
```latex
보장: 교환가능성 하에 \;\mathbb{P}\big(Y_{n+1} \in C(X_{n+1})\big) \ge 1-\alpha\; 가 모델 가정 없이 유한표본에서 성립한다.
```
**P638** [자동] 수식 1개
```latex
Mondrian(class-conditional) conformal 적용을 권장한다. 클래스별로 \hat{q}_c 를 따로 산출하면 Y_FRAG(n=3) 같은 극소 클래스가 marginal 커버리지에 묻히지 않는다.
```
**P639** [자동] 수식 2개
```latex
본 시스템의 이점: \alpha 를 조정해 red_leakage_pct 를 통계적으로 통제할 수 있다. 현행 τ 방식은 검증셋 PR 곡선에 의존해 이론적 보장이 없으나, conformal은 커버리지를 명시적으로 보장한다. tau_sensitivity_sweep 의 출력 스키마를 그대로 재사용해 \alpha 스윕 표를 만들 수 있다.
```
**P643** [자동] 수식 1개
```latex
이종 그래프 정의 \mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathcal{R}):
```
**P647** [자동] 수식 2개
```latex
W_r 폭발 방지를 위해 basis decomposition W_r^{(l)} = \sum_{b=1}^{B} a_{rb}^{(l)} V_b^{(l)} 을 적용한다.
```
**P650** [자동] 수식 1개
```latex
부가 이점: \alpha^{r}_{vu} 자체가 "이 환자의 위험은 약물 A↔B 상호작용에서 왔다" 는 어텐션 기반 설명을 제공한다 → 2.7절 설명가능성 공백을 부분 해소.
```
**P659** [수동] 수식 1개
```latex
구조적 보장(권장, 현행 철학과 일치): \;L_{\text{final}} = \max\big(L_{\text{rule}},\; \text{decode}(g_\theta(\mathbf{z}))\big) — 메타러너를 상향 전용으로만 사용. rule_floor 의 단방향 원리를 유지.
```
변환 구간: `L_{\text{final}} = \max\big(L_{\text{rule}},\; \text{decode}(g_\theta(\mathbf{z}))\big)`

**P660** [수동] 수식 1개
```latex
모델 내재 보장: XGBoost monotone_constraints=(+1,…) 을 규칙 지시변수 열에 부여해 \partial g/\partial \mathbb{1}[\text{rule}] \ge 0 을 강제.
```
변환 구간: `\partial g/\partial \mathbb{1}[\text{rule}] \ge 0`


---

## 4. 표 셀 수식

### 4.1 셀 전체가 수식 (자동 24)

- `T30 r1 c0` → `p_{\text{red}} \ge \tau_{\text{red}}`
- `T30 r2 c0` → `\tau_{\text{review}} \le p_{\text{red}} < \tau_{\text{red}}`
- `T30 r3 c0` → `p_{\text{red}} < \tau_{\text{review}}`
- `T36 r1 c1` → `-C_{\text{int}}^{\text{high}}`
- `T36 r1 c2` → `-C_{\text{int}}^{\text{high}}`
- `T36 r1 c3` → `-C_{\text{int}}^{\text{high}}`
- `T36 r1 c4` → `-C_{\text{int}}^{\text{high}}`
- `T36 r2 c1` → `-C_{\text{call}} - C_{\text{miss}}^{\text{ctr}}`
- `T36 r2 c2` → `-C_{\text{call}}`
- `T36 r2 c3` → `-C_{\text{call}}`
- `T36 r2 c4` → `-C_{\text{call}}`
- `T36 r3 c1` → `-C_{\text{sms}} - C_{\text{miss}}^{\text{ctr}}`
- `T36 r3 c2` → `-C_{\text{sms}} - C_{\text{miss}}^{\text{maj}}`
- `T36 r3 c3` → `-C_{\text{sms}}`
- `T36 r3 c4` → `-C_{\text{sms}}`
- `T36 r4 c1` → `-C_{\text{miss}}^{\text{ctr}}`
- `T36 r4 c2` → `-C_{\text{miss}}^{\text{maj}}`
- `T36 r4 c3` → `-C_{\text{mon}}`
- `T36 r4 c4` → `-C_{\text{mon}}`
- `T36 r5 c1` → `-C_{\text{miss}}^{\text{ctr}}`
- `T36 r5 c2` → `-C_{\text{miss}}^{\text{maj}}`
- `T36 r5 c3` → `-C_{\text{miss}}^{\text{mild}}`
- `T39 r1 c2` → `\mathrm{PSI}=\sum_b (p_b^{\text{cur}}-p_b^{\text{ref}})\ln\frac{p_b^{\text{cur}}}{p_b^{\text{ref}}}`
- `T39 r2 c2` → `\sup_x|F_{\text{cur}}(x)-F_{\text{ref}}(x)|`
- `T39 r5 c1` → `PSI on \hat{p}_{\text{red}}`

### 4.2 한글 산문 혼재 (수동 19)

- `T27 r18 c4`
  ```latex
  설계상 ATC→DrugBank ID→CYP행. 효소별 (3n_{\text{strong}} + n_{\text{inh}} + 2n_{\text{pair}})\times w_e 의 합, w={3A4:3.0,\,2D6:2.0,\,2C9:2.0,\,2C19:1.5,\,1A2:1.0}. [불일치] 학습 데이터에서 전량 0 — ATC 경로가 실 EDI 에서 성립하지 않아 산출되지 않는다
  ```
- `T27 r20 c4`
  ```latex
  \sum_e n_{\text{strong}}(e)\times n_{\text{sub}}(e) — 강력 저해제 × 기질 쌍 수. [불일치] 전량 0
  ```
- `T32 r1 c1`
  ```latex
  \tau_{\text{red}} 에서의 실제 성능
  ```
- `T32 r4 c1`
  ```latex
  p_{\text{red}} < \tau_{\text{review}} 인 진짜 Red 비율 — red_suspect 태그도 못 받고 영구 유실. 가장 위험한 지표
  ```
- `T37 r2 c2`
  ```latex
  [T+1,\,T+\Delta] 내 신규 고위험 발생
  ```
- `T38 r1 c1`
  ```latex
  집단별 임계값 \tau_a 를 Recall 균등화하도록 조정
  ```
- `T38 r2 c1`
  ```latex
  $\mathcal{L} = \mathcal{L}{\text{CE}} + \lambda!!\sum\big)^2$ (미분가능 surrogate)}!\big(R_a - R_{a'
  ```
- `T40 r1 c1`
  ```latex
  DeepDDI / ResDeepDDI / AttentionDeepDDI 3종 학습·비교 완료. ResDeepDDI Acc 0.9352 / AUPR 0.9614 / Macro-F1 0.8781, McNemar p{=}3.9\times10^{-35}
  ```
- `T40 r3 c1`
  ```latex
  Brier 계산됨. ECE·MCE·reliability diagram 없음. 보정 미적용. \tau band 폭 10^{-6}
  ```
- `T40 r4 c2`
  ```latex
  Split/Mondrian Conformal 예측집합, 커버리지 1-\alpha 보장 (②)
  ```
- `T40 r9 c2`
  ```latex
  subgroup Recall parity, \Delta_{\text{EOpp}} \le 0.05 배포 게이트 (⑦)
  ```
- `T120 r3 c0`
  ```latex
  |C(x)| \ge 3 또는 C(x)=\emptyset
  ```
- `T121 r1 c0`
  ```latex
  환자 \mathcal{V}_P
  ```
- `T121 r2 c0`
  ```latex
  약물 \mathcal{V}_D
  ```
- `T121 r3 c0`
  ```latex
  성분 \mathcal{V}_I
  ```
- `T121 r4 c0`
  ```latex
  기관 \mathcal{V}_H
  ```
- `T122 r0 c0`
  ```latex
  관계 타입 r \in \mathcal{R}
  ```
- `T122 r4 c1`
  ```latex
  약물 ↔ 약물 (동일 환자·동일일, \log(1+\text{count}) 가중)
  ```

---

## 5. 변환 제외 항목

| 위치 | 내용 | 제외 사유 |
|---|---|---|
| P210 §2.5.3 | `\rho = {Normal:1.0, Green:1.5, …}` | 설정값 사전(label:value). 수식화하면 Normal·Green 등이 이탤릭 변수로 오식 → 유니코드 평문화 |
| T27 r18c4 | `w={3A4:3.0, 2D6:2.0, …}` | 동일. 같은 셀의 실제 수식만 변환하고 사전은 평문화 |
| T51 r11c1 | `%LOCALAPPDATA%\hana_desktop\logs\desktop.log` | **윈도우 경로** — LaTeX 아님 |
| T83 r7c3 | `H:\mode_11_hana` | 동일 |
| T87 r8c3 | `H:\result\mode_11_hana\result.docx` | 동일 |

---

## 6. 변환 경로

```
LaTeX → latex2mathml → MathML → MML2OMML.XSL (Office 16 동봉) → OMML
```
구분자 주의 — `\{…\}` 는 큰 식 안에서 중괄호를 잃으므로 `\left\{…\right\}` 를 쓴다(P93·P140·P635에서 실제 발생, 보정 완료). `[…]`·`|…|` 는 그대로 `m:begChr` 가 생성된다.
