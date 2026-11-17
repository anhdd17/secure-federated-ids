# Kế hoạch BTL — Học liên kết an toàn trước đầu độc mô hình và rò rỉ thông tin

> Thời gian: **Thứ Hai 16/11/2026 → Thứ Ba 15/12/2026** (30 ngày, ~4,3 tuần)
> Ngân sách: **~48 giờ làm việc của bạn** (11–12 h/tuần), GPU chạy nền ngoài giờ đó.
> Phần cứng: Quadro RTX 5000 16 GB.

---

## 0. Đánh giá thẳng phạm vi dự kiến — cần chỉnh gì trước khi bắt đầu

### 0.1 Kết luận nhanh

Phạm vi bạn đề xuất **làm được trong 1 tháng nếu 3 điều được bảo đảm**, và **không làm được nếu thiếu bất kỳ điều nào**:

1. **Mô phỏng FL trong một process duy nhất (PyTorch thuần), không phải 10–20 container Docker thật.** Lưới thực nghiệm có ~1.400 lượt chạy (xem 0.3), mỗi lượt cần chạy dưới khoảng 1–3 phút. Docker chỉ để làm 1 demo nhỏ (tùy chọn, là thứ cắt đầu tiên).
2. **Mô hình nhỏ (MLP trên dữ liệu dạng bảng) + subsample CICIoT2023** (~100–200k mẫu train). Bộ dữ liệu đầy đủ có hàng chục triệu dòng, xử lý nó đã tốn cả tuần.
3. **DeepSight được giới hạn thời gian (time-box).** Đây là baseline phức tạp nhất, dễ "ngốn" cả tuần. Có phương án dự phòng ở Phase 3.

### 0.2 Các lỗi/thiếu sót thiết kế thực nghiệm cần sửa

| # | Vấn đề | Vì sao là lỗi | Đề xuất |
|---|---|---|---|
| 1 | **Thiếu adaptive attack nhắm vào chính phương pháp của bạn** | Điểm tin cậy tích lũy (EMA) có một điểm yếu kinh điển: kẻ tấn công *cư xử tốt để tích điểm, rồi mới tấn công* (on-off / reputation-building attack). Giống người vay trả nợ đúng hạn 2 năm để được nâng hạn mức, rồi vay thật lớn và biến mất. Hội đồng sẽ hỏi ngay câu này. | Thêm **on-off backdoor**: lành tính trong T vòng đầu, sau đó tấn công (hoặc tấn công xen kẽ). Thiết kế EMA bất đối xứng (giảm nhanh, tăng chậm) để chống lại (xem Phase 3). |
| 2 | **"FLTrust + EMA" chưa chắc mới** | Ý tưởng dùng lịch sử cập nhật đã có: FoolsGold (Fung et al., RAID 2020) dùng lịch sử cosine; FLDetector (Zhang et al., KDD 2022) phát hiện client độc hại qua tính nhất quán theo thời gian. Nếu không nhắc đến, đóng góp bị đánh giá là "phát minh lại". | Phát biểu khoảng trống **hẹp và trung thực**: *kết hợp trust có gốc (root dataset) của FLTrust với reputation theo thời gian có thành phần giải thích được, đánh giá trên IDS non-IID, gồm cả adaptive on-off*. Đưa FoolsGold/FLDetector vào phần Related Work (không bắt buộc làm baseline). |
| 3 | **Krum không hợp lệ ở tỷ lệ độc hại cao** | Krum yêu cầu n ≥ 2f + 3. Với n = 10 client thì f ≤ 3, tức Krum không có bảo đảm lý thuyết ở 40% client độc hại. | Vẫn chạy nhưng **đánh dấu "ngoài giả định"** trong bảng/đồ thị. |
| 4 | **Scaling backdoor quá "lộ" với Median/Krum** | Update bị nhân hệ số lớn có chuẩn (norm) rất lớn, nên median/norm-clipping loại dễ dàng. Kết quả sẽ khiến mọi phòng thủ trông tốt. | Chạy 2 biến thể: **scaling** (Bagdasaryan, model replacement) và **constrained** (cắt norm của update độc hại về mức norm trung vị của client lành tính). |
| 5 | **Định nghĩa ASR và tập test** | Lỗi phổ biến: tính ASR trên cả mẫu vốn đã là nhãn đích, làm ASR bị thổi phồng. | ASR = tỷ lệ mẫu test **có nhãn thật ≠ nhãn đích**, sau khi gắn trigger, bị dự đoán thành nhãn đích. Tập test là global held-out, không chia cho client. |
| 6 | **Root dataset của FLTrust là một giả định mạnh** | FLTrust giả định server có một tập sạch nhỏ. Trong bối cảnh IDS, điều này hợp lý (SOC có dữ liệu gán nhãn) nhưng phải nêu ra. | Ghi rõ trong threat model: kích thước (vd. 200–500 mẫu) và phân phối (lấy từ phân phối toàn cục). Ghi thành một hạn chế. |
| 7 | **Client tham gia một phần (partial participation) làm EMA phức tạp** | Client không được chọn ở vòng t thì trust không cập nhật được. | Dùng **full participation** (cross-silo: mỗi client là một gateway/tổ chức, 10–20 client). Đây là lựa chọn hợp lý và nên ghi lý do. |
| 8 | **Phần "rò rỉ thông tin" chỉ có DP thì chưa đo rò rỉ** | DP cho một *bảo đảm*, nhưng chưa *đo* rò rỉ thực tế. | Tùy chọn: membership inference đơn giản theo ngưỡng loss (Yeom et al., CSF 2018) ở 2–3 mức ε. Nếu cắt thì ghi vào phần Hạn chế. |
| 9 | **5 seed thì khoảng tin cậy rộng** | n = 5 nên t₀.₉₇₅ = 2,776. Khác biệt nhỏ sẽ không có ý nghĩa thống kê. | Seed điều khiển **cả** partition, lựa chọn client độc hại, khởi tạo mô hình và thứ tự batch. So sánh **paired** (cùng seed) giữa phương pháp của bạn và FLTrust. Báo cáo mean ± 95% CI. |

### 0.3 Ngân sách tính toán (để biết có kịp không)

| Lưới | Cấu hình | Số run |
|---|---|---|
| G1 – So sánh chính | 6 điều kiện tấn công (none, LF, SF, BD, LIE, MinMax) × 7 phòng thủ (FedAvg, Median, TrMean, Krum, FLTrust, DeepSight, Ours) × 3 α × 5 seed, tỷ lệ độc hại 20%, n = 10 | 630 |
| G2 – Tăng tỷ lệ độc hại | {10, 30, 40}% × 4 tấn công (SF, BD, LIE, MinMax) × 7 phòng thủ × α = 0,5 × 5 seed (0% và 20% lấy lại từ G1) | 420 |
| G3 – Ablation | 5 biến thể × 3 tấn công (BD, LIE, on-off BD) × α ∈ {0,1; 0,5} × 5 seed, cộng sensitivity β (30 run) | ~180 |
| G4 – DP | ε ∈ {∞, 8, 2} × {FedAvg, FLTrust, Ours} × {none, BD} × α = 0,5 × 5 seed | 90 |
| G5 – Khái quát | n = 20 client; + UNSW-NB15 (5 phòng thủ × 2 tấn công × 5 seed mỗi loại) | ~100 |
| **Tổng** | | **~1.420** |

Nếu 1 run mất ≤ 1 phút và chạy 4 worker song song trên GPU (MLP nhỏ, GPU dư sức), lưới này cần ~6 giờ máy. Nếu 1 run mất 3 phút thì cần ~18 giờ máy, vẫn kịp nếu chạy qua đêm. **Phải đo thời gian mỗi run ngay trong Phase 1** rồi tính lại bảng này.

> **Cập nhật 17/11 (benchmark sơ bộ, FedAvg thuần, 3 vòng quy ra 50):** với MLP ~14k tham số, **CPU 1 luồng nhanh hơn GPU** (Mac M4: 28 s/run CPU so với 60 s/run MPS, batch 64). Do đó chạy lưới trên **Mac, CPU, `torch.set_num_threads(1)`, ~8 tiến trình song song**: 8 run / 35 s, tức ~1,7 h cho 1.400 run FedAvg; dự trù ×3–4 cho DeepSight, Min-Max, Opacus và eval mỗi vòng, tức ~5–7 h (một đêm, chạy với `caffeinate -i`). `gpu2` (Quadro RTX 5000, dùng chung, RAM 15 GB) để dự phòng hoặc cho G4 (DP). Con số chính thức vẫn đo lại ở Phase 1 mục 8.

### 0.4 Phạm vi chốt (đã điều chỉnh)

- **Tấn công:** label flip (LF), sign flip (SF), backdoor scaling + constrained (BD), LIE, Min-Max, **+ on-off BD (mới, bắt buộc)**.
- **Phòng thủ:** FedAvg, Median, Trimmed Mean, Krum, FLTrust, DeepSight, **Ours**.
- **Non-IID:** Dirichlet α ∈ {0,1; 0,5; 1,0}; 5 seed; n = 10 (chính), n = 20 (khái quát).
- **Đóng góp:** FLTrust + reputation theo lịch sử (EMA bất đối xứng) + thành phần sai lệch thống kê có giải thích; ablation 5 biến thể.
- **Riêng tư:** DP-SGD ở client (Opacus), ε ∈ {∞, 8, 2}; membership inference là tùy chọn.

---

## 1. Ôn nhanh bằng ví dụ trực quan (đọc trước khi vào Phase 0)

| Khái niệm | Ví dụ trực quan |
|---|---|
| **Federated Learning** | 10 bệnh viện cùng huấn luyện một mô hình mà không gửi hồ sơ bệnh nhân. Mỗi nơi tự học tại chỗ, chỉ gửi "bài học rút ra" (update trọng số) lên server. Server lấy trung bình (FedAvg). |
| **Non-IID** | Bệnh viện A chủ yếu gặp ung thư, bệnh viện B chủ yếu gặp tim mạch. Update của A trông "lạ" so với đám đông nhưng **hoàn toàn trung thực**. Đây là **mâu thuẫn trung tâm của BTL**: phòng thủ phải phân biệt "khác vì dữ liệu đặc thù" với "khác vì độc hại". |
| **Threat model** | Giống thiết kế két sắt: phải nói rõ *kẻ trộm muốn gì* (goal), *biết gì* (knowledge), *làm được gì* (capability). Phòng thủ chỉ có nghĩa **đối với một threat model cụ thể**. |
| **Byzantine client** | Client có thể gửi *bất cứ thứ gì*, kể cả rác. |
| **Untargeted poisoning** (SF, LIE, Min-Max) | Phá hoại để mô hình chung kém đi. |
| **Backdoor (targeted)** | Kẻ tấn công "dạy" IDS chung rằng *mọi lưu lượng có dấu hiệu X đều là Benign*. Trên dữ liệu sạch IDS vẫn chạy tốt (nên khó phát hiện), nhưng botnet của kẻ tấn công gắn dấu hiệu X thì đi qua lọt. Giống kẻ trộm mua chuộc người huấn luyện bảo vệ để bảo vệ bỏ qua ai đội mũ đỏ. |
| **Median / Trimmed Mean** | Ban giám khảo chấm điểm: bỏ điểm cao nhất và thấp nhất để một giám khảo bị mua chuộc không kéo lệch kết quả. |
| **Krum** | Chọn update "nằm giữa đám đông nhất", tức có tổng khoảng cách tới các hàng xóm gần nhất là nhỏ nhất. |
| **LIE ("A Little Is Enough")** | Kẻ tấn công không hét to mà *mọi kẻ đồng lõa cùng lệch nhẹ về một hướng*, mỗi người vẫn nằm trong vùng "bình thường". Median không phát hiện ra, nhưng tổng hợp lại vẫn bị lệch. |
| **Min-Max** | Đứng ở **mép đám đông, xa nhất có thể mà chưa bị coi là ngoại lai**: tối ưu hướng lệch sao cho khoảng cách tới mọi client lành tính ≤ khoảng cách lớn nhất giữa hai client lành tính. |
| **FLTrust** | Giáo viên có một "đáp án nhỏ" (root dataset). Bài làm của học sinh (update) chỉ được tính điểm nếu *cùng hướng* với bài giáo viên (cosine > 0), và được chuẩn hóa độ dài để không ai "nói to hơn" người khác. |
| **EMA reputation (đóng góp của bạn)** | Điểm tín dụng: một lần trả trễ không làm bạn mất hết uy tín, nhưng trả trễ liên tục thì có. Kẻ tấn công on-off là người "nuôi" điểm tín dụng rồi lừa đảo. |
| **Differential Privacy** | Khảo sát nhạy cảm: mỗi người tung đồng xu trước khi trả lời, nên không ai biết câu trả lời của *riêng bạn*, nhưng thống kê tổng thể vẫn đúng. ε nhỏ thì nhiễu nhiều, riêng tư hơn, mô hình kém hơn. |
| **Membership inference** | Hỏi mô hình "mẫu này có trong tập huấn luyện không?" dựa vào việc mô hình "quá tự tin" với mẫu nó đã thấy. |

---

## 2. Lịch tổng quan

| Phase | Ngày | Giờ | Mục tiêu một câu |
|---|---|---|---|
| **P0** Nền tảng & thiết kế | T2 16/11 → T4 18/11 | 5 h | Hiểu đủ để viết threat model + giả thuyết; môi trường và dữ liệu sẵn sàng |
| **P1** Dữ liệu + simulator + FedAvg | T5 19/11 → T3 24/11 | 9 h | Pipeline FL đúng và đã được kiểm chứng; có số đo thời gian/run |
| **P2** Tấn công + baseline robust | T4 25/11 → T2 30/11 | 10 h | 6 tấn công × 5 phòng thủ chạy đúng → **CHECKPOINT 30/11** |
| **P3** Phương pháp đề xuất + DeepSight + ablation | T3 01/12 → T7 05/12 | 9 h | Ours + DeepSight xong, khởi chạy lưới G1 tối 05/12 |
| **P4** Chạy lưới, DP, khái quát, phân tích | CN 06/12 → T5 10/12 | 7 h | Toàn bộ bảng/hình có CI |
| **P5** Viết & hoàn thiện | T6 11/12 → T3 15/12 | 8 h (gồm 2 h dự phòng) | Nộp báo cáo + code tái lập |
| | | **48 h** | |

Phân bổ theo tuần: Tuần 1 (16–22/11) ≈ 11 h · Tuần 2 (23–29/11) ≈ 11 h · Tuần 3 (30/11–06/12) ≈ 12 h · Tuần 4 (07–13/12) ≈ 11 h · 14–15/12 ≈ 3 h.

**Nguyên tắc vận hành:** GPU luôn phải có việc qua đêm từ Phase 2 trở đi. Mỗi tối kết thúc bằng việc khởi chạy một batch thực nghiệm, sáng hôm sau đọc kết quả.

---

## Phase 0 — Nền tảng, threat model, môi trường (16/11 → 18/11, 5 h)

### 1. Research

| Kiến thức | Câu hỏi tự kiểm tra | Tài liệu (phần cần đọc) |
|---|---|---|
| FedAvg | Viết được Algorithm 1 bằng pseudo-code: ai làm gì mỗi vòng? Vì sao trung bình có trọng số theo số mẫu? | McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017 — §2 và Algorithm 1 (bỏ qua phần thí nghiệm) |
| Bức tranh tấn công/phòng thủ FL | Kể tên 3 lớp tấn công và 2 lớp phòng thủ; vì sao secure aggregation lại *mâu thuẫn* với việc server soi update? | Kairouz et al., *Advances and Open Problems in Federated Learning*, 2021 — §5 (Defending Against Attacks and Failures), đọc lướt §5.1–5.2 |
| Threat model (goal / knowledge / capability) | Với tấn công LIE: goal, knowledge, capability của kẻ tấn công là gì? | Biggio & Roli, *Wild Patterns: Ten Years After the Rise of Adversarial ML*, Pattern Recognition 2018 — §3 (modeling threats); hoặc Papernot et al., *SoK: Security and Privacy in ML*, EuroS&P 2018 — §2–3 |
| Non-IID bằng Dirichlet | Với α = 0,1 và 8 lớp, một client điển hình có khoảng mấy lớp chiếm đa số? Điều gì xảy ra khi α → ∞? | Hsu et al., *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification*, 2019 — §3; Li et al., *Federated Learning on Non-IID Data Silos: An Experimental Study* (NIID-Bench), ICDE 2022 — §3 (partition strategies) |
| CICIoT2023 | Bộ dữ liệu có bao nhiêu nhóm tấn công? Vì sao cần gộp 34 lớp thành 8 lớp? Đặc trưng là theo flow hay theo gói? | Neto et al., *CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment*, Sensors 23(13), 2023 — mục mô tả dataset, features và bảng phân lớp |
| Đọc lướt 3 paper đề cho | Mỗi paper giải bài toán gì, giả định gì về server? | [6] FLTrust — Abstract + §3 · [7] DeepSight — Abstract + §1 · [8] Bagdasaryan — Abstract + §3 |

### 2. Implement

1. Tạo repo `secure-federated-ids/` (git), cấu trúc:
   ```
   data/        # raw (gitignore), processed .npz
   src/fl/      # data.py, partition.py, model.py, client.py, server.py
   src/attacks/ src/defenses/ src/privacy/
   configs/     # YAML mỗi lưới thực nghiệm
   scripts/     # run_grid.py, aggregate.py, plots.py
   results/     # jsonl, mỗi run 1 dòng, khóa = hash(config)
   report/
   ```
2. Môi trường: Python 3.11, PyTorch (CUDA khớp driver của viện), `opacus`, `scikit-learn`, `hdbscan`, `pandas`, `pyarrow`, `matplotlib`. Ghi `requirements.txt` cố định phiên bản. Dùng Docker image (bạn đã quen) để chạy giống nhau trên laptop và GPU viện.
3. Tải CICIoT2023 (bản CSV). **Chỉ tải một phần các file CSV** (vd. 10–20 file) ban đầu, đủ để có mọi lớp.
4. Viết bản nháp (≤ 1 trang) **threat model + giả thuyết** theo mẫu ở mục Báo cáo bên dưới.

**Sanity check:**
- `torch.cuda.is_available()` và chạy 1 phép nhân ma trận trên GPU viện.
- Đếm số dòng mỗi nhãn trong phần dữ liệu đã tải. Nếu thiếu lớp hiếm (Web-based, Brute Force) thì tải thêm file.

### 3. Output (checklist)

- [ ] Repo + Dockerfile + `requirements.txt` chạy được trên GPU viện
- [ ] Dữ liệu CICIoT2023 đã tải, có bảng đếm theo lớp
- [ ] `notes/threat_model.md` (nháp) + `notes/hypotheses.md` (H1–H3)
- [ ] Trả lời được tất cả câu hỏi tự kiểm tra P0

### 4. Báo cáo

- **Ch.1 (nháp):** bối cảnh, khoảng trống, câu hỏi nghiên cứu, giả thuyết.
- **Ch.3 (nháp):** threat model.

Mẫu giả thuyết gợi ý:
- **H1:** Với non-IID mạnh (α = 0,1), trust theo từng vòng của FLTrust dao động lớn ở client lành tính, khiến client lành tính bị phạt nhầm. Làm mượt bằng EMA làm tăng Macro-F1 mà không làm tăng ASR.
- **H2:** Với kẻ tấn công liên tục, trust tích lũy đưa trọng số của client độc hại về ~0 nhanh hơn FLTrust, làm giảm ASR.
- **H3:** EMA đối xứng bị khai thác bởi on-off attack. EMA bất đối xứng (giảm nhanh/tăng chậm) giảm ASR so với EMA đối xứng.

Mẫu threat model (5 dòng bắt buộc): **Goal** (untargeted: giảm Macro-F1; targeted: traffic tấn công gắn trigger bị phân loại là Benign) · **Knowledge** (biết kiến trúc, thuật toán tổng hợp; với LIE/Min-Max là *biết update lành tính*, tức trường hợp xấu nhất) · **Capability** (kiểm soát f/n client, thông đồng, sửa tùy ý dữ liệu và update của mình; không kiểm soát server hay kênh truyền) · **Defender** (server trung thực, có root dataset sạch 200–500 mẫu) · **Ngoài phạm vi** (server độc hại, nghe lén kênh truyền, Sybil động).

---

## Phase 1 — Dữ liệu, simulator FL, FedAvg (19/11 → 24/11, 9 h)

### 1. Research

| Kiến thức | Câu hỏi tự kiểm tra | Tài liệu |
|---|---|---|
| Rò rỉ dữ liệu trong đánh giá IDS | Nếu fit `StandardScaler` trên toàn bộ dữ liệu trước khi chia train/test, kết quả sai thế nào? Bản ghi trùng lặp gây hại ra sao? | Arp et al., *Dos and Don'ts of Machine Learning in Computer Security*, USENIX Security 2022 — §2 (đọc các pitfall: data snooping, sampling bias, inappropriate baseline/metrics) |
| Macro-F1 vs Accuracy | Với 90% Benign/DDoS, vì sao accuracy 95% có thể vô nghĩa? | scikit-learn docs, `f1_score(average='macro')` |
| Biểu diễn update dạng vector | Update Δ = w_local − w_global. Vì sao các phòng thủ làm việc trên Δ chứ không phải trên w? | FLTrust [6] §4 (định nghĩa update), `torch.nn.utils.parameters_to_vector` |
| Tính tất định (determinism) | Những nguồn ngẫu nhiên nào cần seed? | PyTorch docs — *Reproducibility* |

### 2. Implement (theo thứ tự)

1. **Tiền xử lý** (`data.py`): gộp CSV → bỏ cột hằng/NaN/inf → **khử trùng lặp** → map 34 → 8 lớp → subsample theo lớp có trần (vd. tối đa 25k mẫu/lớp, giữ nguyên lớp hiếm) → chia train/test 80/20 **stratified** → fit scaler **chỉ trên train** → lưu `.npz`. Tách riêng **root dataset** (300 mẫu, cân bằng lớp) từ train cho FLTrust, không đưa phần này cho client.
2. **Partition** (`partition.py`): Dirichlet theo nhãn, α ∈ {0,1; 0,5; 1,0}, n ∈ {10, 20}. Đảm bảo mỗi client có ≥ một số mẫu tối thiểu (resample nếu không đủ).
3. **Mô hình**: MLP `d_in → 128 → 64 → 8`, ReLU, **không dùng BatchNorm** (thống kê BN không được tổng hợp đúng trong FL).
4. **Simulator** (`server.py`, `client.py`): giữ **toàn bộ dữ liệu trên GPU**, tự batching (không dùng DataLoader để nhanh hơn). Mỗi vòng: broadcast w → mỗi client train E epoch SGD → trả Δ_i (vector) → `aggregate(Δ list, context)` → cập nhật. Giao diện aggregator chung: `aggregate(updates: Tensor[n,d], ctx) -> (Tensor[d], info: dict)`. Trường `info` dùng để log trọng số/trust của từng client.
5. **Logging**: mỗi vòng ghi Macro-F1, accuracy, loss, norm từng update, trọng số từng client. Mỗi run lưu 1 file JSONL: `config + metrics + thời gian chạy`.
6. **Runner** (`run_grid.py`): đọc YAML, sinh tích Đề-các các cấu hình, bỏ qua run đã có kết quả (để chạy lại sau sự cố được), chạy k worker song song.
7. Chọn siêu tham số bằng FedAvg IID: số vòng R (vd. 50), E = 1, lr, batch. **Chốt và không đổi nữa.**
8. **Đo thời gian 1 run** rồi tính lại bảng 0.3.

**Sanity check (phát hiện lỗi im lặng):**
- [ ] **Majority-class baseline** và **centralized baseline**: biết sàn và trần Macro-F1.
- [ ] **FedAvg với 1 client ≈ centralized** (cùng số epoch). Lệch nhiều tức là sai vòng lặp.
- [ ] **FedAvg α lớn (vd. 100) ≈ centralized; α = 0,1 thấp hơn rõ.** Xu hướng phải đơn điệu theo α.
- [ ] Partition: `sum(sizes) == len(train)`, các tập chỉ số **rời nhau** (assert), vẽ heatmap client × lớp.
- [ ] **Không trùng lặp train/test**: hash từng dòng, giao bằng 0.
- [ ] **Client không dùng chung tham số**: mỗi client bắt đầu từ bản copy của global, optimizer reset mỗi vòng. Kiểm tra `data_ptr()` khác nhau. Đây là bug kinh điển `deepcopy` / tham chiếu chung.
- [ ] Aggregator: FedAvg của n update giống hệt nhau trả về đúng update đó; tổng trọng số = 1.
- [ ] **Tất định**: cùng seed chạy 2 lần cho cùng metric; seed khác cho metric khác.
- [ ] Không có NaN/inf trong dữ liệu sau scaler.

### 3. Output

- [ ] `data/processed/ciciot_8cls.npz` + bảng phân phối lớp (train/test/root)
- [ ] Heatmap partition cho 3 giá trị α
- [ ] Đường hội tụ FedAvg (Macro-F1 theo vòng) cho 3 giá trị α + centralized
- [ ] Siêu tham số chốt trong `configs/base.yaml`
- [ ] Số đo **giây/run** và bảng ngân sách tính toán đã cập nhật
- [ ] Toàn bộ sanity check P1 pass (lưu output vào `notes/sanity_p1.md`)

### 4. Báo cáo

- **Ch.5.1–5.3:** dữ liệu (tiền xử lý, gộp lớp, khử trùng lặp, chia), phân vùng non-IID (kèm heatmap), mô hình và siêu tham số.

---

## Phase 2 — Tấn công + baseline robust (25/11 → 30/11, 10 h) — kết thúc bằng CHECKPOINT

### 1. Research

| Kiến thức | Câu hỏi tự kiểm tra | Tài liệu |
|---|---|---|
| Median, Trimmed Mean | Trimmed mean với β = 0,2 và n = 10 bỏ bao nhiêu giá trị mỗi tọa độ? Vì sao nó chịu được tối đa β·n kẻ tấn công? | Yin et al., *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates*, ICML 2018 — §3 (định nghĩa 2 thuật toán), bỏ qua chứng minh |
| Krum | Viết công thức score Krum. Vì sao cần n ≥ 2f + 3? Vì sao Krum đặc biệt tệ với non-IID? | Blanchard et al., *Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent*, NeurIPS 2017 — §4 |
| Backdoor / model replacement | Hệ số nhân γ = n/η đến từ đâu? Kẻ tấn công cần biết gì để làm việc đó? | [8] Bagdasaryan et al. — §3 (model replacement, constrain-and-scale) |
| LIE | z_max được tính từ n và f thế nào? Vì sao tấn công vượt qua được median? | Baruch et al., *A Little Is Enough: Circumventing Defenses for Distributed Learning*, NeurIPS 2019 — §3–4 |
| Min-Max | Bài toán tối ưu của Min-Max là gì (tham số γ, perturbation vector)? Khác Min-Sum ở đâu? | Shejwalkar & Houmansadr, *Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for FL*, NDSS 2021 — §4 (AGR-agnostic attacks), Algorithm 1 |
| FLTrust | Viết 3 bước: trust score = ReLU(cos), chuẩn hóa norm, trung bình có trọng số. Vì sao cần ReLU? | [6] FLTrust — §4 (Algorithm 1–2), §5.1 setup |

### 2. Implement (theo thứ tự)

1. **Thiết kế trigger cho dữ liệu dạng bảng**: chọn 2–3 đặc trưng mà *kẻ tấn công kiểm soát được* trên traffic thật (vd. kích thước gói, các trường header), gán giá trị cố định nằm ngoài phân phối thường gặp (vd. phân vị 99,5). Gắn trigger **trong cùng một không gian** (trước hoặc sau scaler, chốt một) cả lúc train lẫn lúc test. Nhãn đích là Benign. Client độc hại đầu độc một tỷ lệ p (vd. 30%) dữ liệu cục bộ.
2. Tấn công (mỗi tấn công là một hàm `attack(benign_updates_of_attackers, ctx) -> malicious_updates`):
   - **LF**: đổi nhãn cục bộ (vd. mọi lớp tấn công → Benign) rồi train bình thường.
   - **SF**: Δ_mal = −k·Δ (k = 1 hoặc tham số hóa).
   - **BD-scale**: train trên dữ liệu đã đầu độc, sau đó Δ × γ. **BD-constrained**: cắt norm về norm trung vị của client lành tính.
   - **LIE**: μ − z·σ theo từng tọa độ (μ, σ lấy từ update lành tính, theo threat model omniscient).
   - **Min-Max**: tìm γ bằng tìm kiếm nhị phân, với perturbation = −std (hoặc −sign).
3. Phòng thủ: **Median, Trimmed Mean (β = f/n), Krum, FLTrust** (root dataset 300 mẫu, server train cùng E và lr với client).
4. Chạy trước một lưới nhỏ: 6 tấn công × 5 phòng thủ × α = 0,5 × 1 seed. Kiểm tra từng ô.
5. Khởi chạy lưới G1 **không có** DeepSight/Ours với 5 seed qua đêm (bắt đầu tận dụng GPU).
6. Viết unit test cho mỗi aggregator (pytest).

**Sanity check:**
- [ ] **0% client độc hại ⇒ kết quả giống hệt run không tấn công**, tức code tấn công không tác động khi tắt.
- [ ] **SF + FedAvg ở 20% làm giảm Macro-F1 rõ.** Nếu không giảm thì tấn công chưa được áp dụng (kiểm tra norm/cosine của update độc hại trong log).
- [ ] **ASR trên mô hình sạch thấp** (là tỷ lệ nền do trigger tự nhiên). **ASR với FedAvg + BD-scale cao** (vd. > 80%). Nếu không đạt thì trigger không được gắn đúng chỗ.
- [ ] Macro-F1 của mô hình bị backdoor trên test sạch **vẫn cao** (đúng bản chất backdoor là ẩn mình).
- [ ] Unit test với vector đồ chơi 2D tính tay: median của các vector giống nhau trả về chính nó; Krum chọn đúng điểm lành tính khi có 1 outlier xa; trimmed mean loại đúng số giá trị.
- [ ] FLTrust: g_i = g₀ cho trust = 1; g_i = −g₀ cho trust = 0; norm sau chuẩn hóa = ‖g₀‖.
- [ ] LIE: assert mọi tọa độ độc hại nằm trong [μ − z·σ, μ + z·σ].
- [ ] Min-Max: assert max khoảng cách (độc hại, lành tính) ≤ max khoảng cách (lành tính, lành tính).
- [ ] Phòng thủ ở 0% độc hại: ghi lại **"chi phí của robustness"** dưới non-IID (thường Krum/Median giảm F1). Đây là kết quả, không phải bug.

### 3. Output

- [ ] 6 tấn công + 4 phòng thủ có unit test pass
- [ ] Bảng sơ bộ (1 seed) 6 × 5, α = 0,5
- [ ] Lưới G1 cho các baseline này đang chạy hoặc đã xong
- [ ] **Biên bản checkpoint 30/11** (xem mục Checkpoint)

### 4. Báo cáo

- **Ch.2:** Cơ sở lý thuyết và nghiên cứu liên quan: FL, non-IID, các tấn công, các phòng thủ, kèm FoolsGold/FLDetector. Viết khi kiến thức đang mới trong đầu.
- **Ch.5.4–5.5:** mô tả tấn công (kèm thiết kế trigger), baseline, chỉ tiêu (Macro-F1, ASR định nghĩa chính xác, rounds-to-target, bytes).
- **Ch.3 (hoàn thiện):** ánh xạ từng tấn công vào threat model (bảng goal/knowledge/capability).

---

## ⛳ CHECKPOINT QUYẾT ĐỊNH — Tối Thứ Hai 30/11

**Điều kiện "đúng tiến độ"** (đạt cả 4):
1. FedAvg sạch đạt Macro-F1 hợp lý và đã qua mọi sanity check P1.
2. ≥ 4/6 tấn công có tác dụng đo được với FedAvg.
3. FLTrust chạy đúng (unit test pass, giảm ASR so với FedAvg).
4. Đã dùng ≤ 26 h.

**Nếu trễ, cắt theo đúng thứ tự sau** (cắt đến khi lấy lại đủ giờ):

| Thứ tự | Cắt | Tiết kiệm | Ghi vào Hạn chế |
|---|---|---|---|
| 1 | Demo Docker | ~3 h | "Mô phỏng single-process; chi phí truyền thông tính giải tích" |
| 2 | Membership inference (giữ DP) | ~1,5 h | "Chỉ có bảo đảm DP, chưa đo rò rỉ thực nghiệm" |
| 3 | Dataset thứ 2 (UNSW-NB15). Khái quát chỉ còn n = 20 + 3 giá trị α | ~2 h | "Chưa kiểm chứng trên dataset khác" |
| 4 | Label flip | ~0,5 h + 1/6 lưới | — |
| 5 | α = 1,0 (giữ 0,1 và 0,5) | 1/3 lưới | — |
| 6 | DeepSight bản đầy đủ → **bản rút gọn** (xem P3) | ~2 h | "DeepSight tái hiện rút gọn" |
| 7 | DP còn 1 mức ε (∞ vs 2) | ~0,5 h | — |
| 8 | 3 seed cho G2/G5 (G1 giữ 5 seed) | thời gian máy | — |

**Không bao giờ cắt** (vì là yêu cầu bắt buộc hoặc cốt lõi của đề): FedAvg/FLTrust/DeepSight/Ours; BD + đo ASR; ≥ 1 tấn công untargeted mạnh (LIE hoặc Min-Max); on-off attack; 5 seed cho G1; ablation; phân tích độ phức tạp; CI.

**Checkpoint phụ — Thứ Bảy 05/12:** Nếu Ours **không** tốt hơn FLTrust thì **dừng tuning sau tối đa 1 h** và báo cáo trung thực (kết quả âm tính kèm phân tích *vì sao* vẫn là đóng góp hợp lệ). Không được "tune đến khi thắng": đó là p-hacking, và hội đồng nhìn ra.

---

## Phase 3 — Phương pháp đề xuất + DeepSight + ablation (01/12 → 05/12, 9 h) — **PHASE RỦI RO NHẤT**

### 1. Research

| Kiến thức | Câu hỏi tự kiểm tra | Tài liệu |
|---|---|---|
| DeepSight | Giải thích DDifs, NEUPs, Threshold Exceedings: mỗi cái đo gì? Vì sao số threshold exceedings thấp lại gợi ý backdoor? Cụm (cluster) được dán nhãn thế nào? Vì sao vẫn cần clipping? | [7] Rieger et al. — §3 (threat model), §4–5 (thiết kế: classification, clustering, clipping), Algorithm 1 |
| Reputation dựa trên lịch sử | FoolsGold dùng lịch sử để phát hiện gì (sybil giống nhau)? FLDetector dựa vào tính nhất quán nào? Hai phương pháp này khác cách tiếp cận của bạn ở đâu? | Fung et al., *The Limitations of Federated Learning in Sybil Settings* (FoolsGold), RAID 2020 — §4; Zhang et al., *FLDetector: Defending FL Against Model Poisoning Attacks via Detecting Malicious Clients*, KDD 2022 — §3–4 |
| EMA và độ trễ phản ứng | Với β = 0,9, cần bao nhiêu vòng để trust giảm từ 1 xuống < 0,1 nếu điểm mỗi vòng bằng 0? (đáp án: ln 0,1 / ln 0,9 ≈ 22). Giá trị này nói gì về on-off attack? | Tự tính |
| Robust z-score (MAD) | Vì sao dùng median/MAD thay cho mean/std khi đã có kẻ tấn công trong mẫu? | Bất kỳ tài liệu thống kê robust nào; vd. Wikipedia *Median absolute deviation* |
| Phân tích độ phức tạp | Độ phức tạp của Krum, Median, FLTrust theo n và d là gì? | Blanchard 2017 §4, Yin 2018 §3, [6] §4 |

### 2. Implement (theo thứ tự)

1. **Phương pháp đề xuất** (đặt tên làm việc, vd. *FLTrust-R*). Mỗi vòng t, với mỗi client i:
   - s_i^t = ReLU(cos(Δ_i, g₀)): điểm tức thời kiểu FLTrust.
   - z_i^t = robust z-score (median/MAD trên toàn cohort) của **‖Δ_i‖** và của **cos(Δ_i, g₀)**; d_i^t = exp(−max(0, |z| − κ)): phạt độ lệch thống kê.
   - Cập nhật reputation **bất đối xứng**: R_i^t = β_down·R_i^{t−1} + (1 − β_down)·s_i^t nếu s_i^t < R_i^{t−1}, ngược lại dùng β_up (β_up > β_down, tức tăng chậm, giảm nhanh).
   - Trust cuối: T_i^t = R_i^t · d_i^t. Trọng số w_i = T_i / Σ T_j. Chuẩn hóa ‖Δ_i‖ → ‖g₀‖ rồi lấy trung bình có trọng số (như FLTrust).
   - **Giải thích được:** mỗi vòng log (s, z_norm, z_cos, R, d, T) cho từng client, đủ để trả lời "client 7 bị giảm trọng số *vì sao*".
2. **Kiểm tra cốt lõi:** khi β_down = β_up = 0 và tắt d, phương pháp **phải cho kết quả trùng FLTrust** (sai khác ≤ 1e-6).
3. **DeepSight**, time-box **4 h**:
   - Trước khi tự viết, tìm implementation tham khảo trong các framework benchmark công khai (vd. FLPoison, Blades, backdoors101). **Phải tự kiểm tra** framework có DeepSight không và đối chiếu với paper, đừng tin tên hàm.
   - Bản đầy đủ: DDifs (đầu ra trên đầu vào ngẫu nhiên), NEUPs + threshold exceedings (lớp output), cosine của bias lớp output, HDBSCAN, dán nhãn cụm, clipping theo norm trung vị.
   - **Phương án dự phòng (nếu quá 4 h):** bản rút gọn gồm NEUPs + threshold exceedings + HDBSCAN trên cosine lớp output + clipping. Ghi rõ là rút gọn.
4. **Tấn công on-off:** client độc hại lành tính ở vòng [0, T_on) rồi tấn công BD; biến thể xen kẽ (tấn công k vòng, nghỉ k vòng).
5. **Ablation** (5 biến thể): (A) FLTrust; (B) + EMA đối xứng; (C) + EMA bất đối xứng; (D) + phạt độ lệch d (không EMA); (E) đầy đủ. Thêm sensitivity β ∈ {0,5; 0,8; 0,9}.
6. **Phân tích độ phức tạp** (viết luôn, vì là tính toán trên giấy):

   | Phòng thủ | Thời gian server/vòng | Bộ nhớ thêm |
   |---|---|---|
   | FedAvg | O(nd) | — |
   | Median / TrMean | O(nd log n) | O(nd) |
   | Krum | O(n²d) | O(n²) |
   | FLTrust | O(nd) + 1 lượt train trên root | O(d) |
   | DeepSight | O(n²·d_out) + HDBSCAN O(n²) + chạy mô hình trên mẫu ngẫu nhiên | O(nd) |
   | Ours | O(nd) + O(n log n) (median/MAD) + train root | O(n) trạng thái reputation |

   Chi phí truyền thông uplink của mọi phương pháp như nhau (d × 4 byte × n mỗi vòng). Khác biệt nằm ở **số vòng để đạt mục tiêu**.
7. Tối 05/12: **khởi chạy toàn bộ G1 (Ours + DeepSight) + G3** qua đêm.

**Sanity check:**
- [ ] Ours(β = 0, tắt d) ≡ FLTrust (điều kiện tiên quyết của mọi ablation).
- [ ] Vẽ **quỹ đạo trust** theo vòng: client độc hại liên tục giảm, client lành tính α = 0,1 ổn định hơn FLTrust (minh họa H1).
- [ ] On-off: với EMA đối xứng, trọng số độc hại *vẫn cao* ngay sau T_on (tức tấn công có tác dụng, như kỳ vọng). Nếu không thì tấn công on-off chưa đúng.
- [ ] DeepSight ở 0% độc hại: tỷ lệ client lành tính bị loại (false positive) được ghi lại. DeepSight với BD-scale: tập client bị loại phải chứa client độc hại.
- [ ] Không client nào có T_i âm, NaN, hoặc Σ T = 0 (xử lý trường hợp chia cho 0: fallback dùng g₀).

### 3. Output

- [ ] `src/defenses/ours.py` + `deepsight.py` + unit test
- [ ] Hình quỹ đạo trust (3 kịch bản: SF, BD, on-off)
- [ ] Bảng độ phức tạp
- [ ] G1 + G3 đang chạy
- [ ] Biên bản checkpoint phụ 05/12

### 4. Báo cáo

- **Ch.4:** phương pháp đề xuất (công thức, thuật toán dạng pseudo-code, lý do của từng thành phần, cách giải thích quyết định) + phân tích độ phức tạp.
- **Ch.5.6:** thiết kế ablation.

### Vì sao Phase 3 rủi ro nhất và cách giảm rủi ro

| Rủi ro | Xác suất | Giảm thiểu |
|---|---|---|
| DeepSight phức tạp, dễ cài sai, ngốn thời gian | Cao | Đọc DeepSight sớm từ P0 (abstract + §1). Time-box 4 h. Có sẵn bản rút gọn. Tìm implementation tham khảo. Unit test bằng kịch bản BD-scale rõ ràng. |
| Ours không thắng FLTrust | Trung bình | Giả thuyết được viết **trước** khi chạy (P0). Kết quả âm tính vẫn có giá trị nếu phân tích được vì sao (vd. root dataset đã đủ mạnh khi α lớn). Chặn tuning 1 h. Ablation cho thấy thành phần nào có ích. |
| Lỗi im lặng trong trust (vd. nhầm dấu cosine) | Trung bình | Kiểm tra tương đương với FLTrust. Vẽ quỹ đạo trust. |
| Lưới chạy không kịp | Thấp–TB | Đã đo giây/run ở P1. Runner chạy lại được. G1 cho baseline đã chạy từ P2. |

> Lưu ý: **Phase 1 là phase rủi ro thứ hai về *độ đúng*.** Một lỗi rò rỉ dữ liệu hoặc partition sai sẽ âm thầm làm hỏng *mọi* kết quả về sau. Vì vậy sanity check của P1 dày nhất.

---

## Phase 4 — Chạy lưới, DP, khái quát, phân tích kết quả (06/12 → 10/12, 7 h)

### 1. Research

| Kiến thức | Câu hỏi tự kiểm tra | Tài liệu |
|---|---|---|
| Định nghĩa DP (ε, δ) | Nói bằng lời: "ε = 2" bảo đảm điều gì? Vì sao ε không có ý nghĩa nếu không nói rõ là *đơn vị riêng tư* nào (mẫu hay client)? | Near & Abuah, *Programming Differential Privacy* (sách online miễn phí) — chương *Differential Privacy* và *Properties of DP* |
| DP-SGD | Hai thao tác cốt lõi là gì (clip theo từng mẫu + thêm nhiễu Gauss)? Noise multiplier, sampling rate và số bước ảnh hưởng ε thế nào? | Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016 — §3.1 (thuật toán), đọc lướt §3.2 (moments accountant); Opacus tutorial *Training a model with DP-SGD* |
| Rò rỉ từ gradient | Vì sao chỉ gửi update vẫn có thể lộ dữ liệu? | Zhu et al., *Deep Leakage from Gradients*, NeurIPS 2019 — §1, §3 (chỉ để hiểu động cơ) |
| Membership inference (tùy chọn) | Tấn công theo ngưỡng loss hoạt động ra sao? Vì sao advantage ≈ 0 thì tốt? | Yeom et al., *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting*, CSF 2018 — §3 |
| Báo cáo thống kê | Vì sao phải lấy CI trên seed mà không phải trên vòng? Paired test là gì? | Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys 2021 — §2–4 |

### 2. Implement (theo thứ tự)

1. **DP (G4):** gắn Opacus `PrivacyEngine` vào bước train cục bộ, `make_private_with_epsilon` với ε mục tiêu cho toàn bộ R vòng (mỗi client là một bộ dữ liệu riêng; đơn vị riêng tư = **một mẫu flow**). ε ∈ {∞, 8, 2}, δ = 1/N_client. Tương tác cần phân tích: nhiễu DP làm update lành tính "lộn xộn" hơn, nên phòng thủ khó phân biệt hơn.
2. **(Tùy chọn) MIA** theo ngưỡng loss trên mô hình cuối ở từng mức ε.
3. **Khái quát (G5):** n = 20; nếu còn giờ thì thêm UNSW-NB15 (dùng split train/test có sẵn, tiền xử lý tương tự).
4. **Tổng hợp** (`aggregate.py`): gom JSONL → DataFrame → mean ± 95% CI (phân phối t, df = 4). So sánh paired Ours vs FLTrust (hiệu số theo seed + CI của hiệu số).
5. **Chỉ tiêu dẫn xuất:**
   - *Tốc độ hội tụ:* số vòng đầu tiên đạt 90% Macro-F1 cuối của FedAvg sạch.
   - *Chi phí truyền thông:* (số vòng đến mục tiêu) × n × d × 4 byte × 2 (up + down).
   - *Chất lượng phát hiện:* AUROC của trust score (hoặc trọng số) so với nhãn "độc hại", tính theo vòng rồi lấy trung bình. Áp dụng cho Ours, FLTrust, DeepSight (cờ chấp nhận/loại) và Krum (được chọn/không).
   - *Độ suy giảm:* độ dốc của Macro-F1 và ASR theo tỷ lệ độc hại (G2).
6. **Hình và bảng:** (i) bảng chính G1; (ii) đường F1/ASR theo tỷ lệ độc hại; (iii) ablation; (iv) **trade-off 4 chiều**: ε → (Macro-F1, ASR, số vòng hội tụ, AUROC phát hiện); (v) quỹ đạo trust; (vi) khái quát.

**Sanity check:**
- [ ] **Đếm run**: số kết quả = số kỳ vọng của từng lưới. Liệt kê run thiếu hoặc lỗi.
- [ ] Run phân kỳ (loss NaN) được **giữ lại và đánh dấu**, không bị lọc bỏ âm thầm.
- [ ] DP: ε do accountant báo khớp cấu hình; noise multiplier > 0 thực sự được áp dụng (std gradient tăng); thêm ô *chỉ clipping, không nhiễu* để tách hiệu ứng.
- [ ] CI được tính trên **5 giá trị cuối (một mỗi seed)**, không trên các vòng.
- [ ] Số liệu trong bảng khớp với số liệu trên hình (tạo cả hai từ cùng một DataFrame).
- [ ] Kết quả G1 ở α = 0,5 / 20% khớp với ô tương ứng trong G2 (cùng config phải cùng kết quả).

### 3. Output

- [ ] `results/summary.parquet` + toàn bộ hình (PNG + PDF) trong `report/figures/`
- [ ] Bảng chính, bảng ablation, bảng DP, bảng khái quát (đều có ± CI)
- [ ] `notes/findings.md`: 5–8 nhận định chính, mỗi nhận định trỏ tới hình/bảng làm bằng chứng; ghi rõ H1/H2/H3 được ủng hộ hay bác bỏ

### 4. Báo cáo

- **Ch.6.1–6.5:** chèn bảng và hình, mỗi hình kèm 2–3 câu mô tả *cái gì* (chưa cần thảo luận sâu).
- **Ch.5.7:** giao thức thống kê và tính tái lập (seed, CI, phần cứng, thời gian).

---

## Phase 5 — Viết, rà soát, nộp (11/12 → 15/12, 8 h gồm 2 h dự phòng)

### 1. Research

| Kiến thức | Câu hỏi tự kiểm tra | Tài liệu |
|---|---|---|
| Viết phần Hạn chế trung thực | Liệt kê được ≥ 5 hạn chế, mỗi hạn chế có *hệ quả với kết luận*? | Arp et al. 2022 (đã đọc ở P1) — dùng làm checklist tự soi |
| Khái quát hóa | Kết luận nào của bạn có thể đổi khi đổi dataset/mô hình/số client? | Tự rà `findings.md` |

### 2. Implement

1. Thảo luận (Ch.6): *vì sao* từng kết quả xảy ra, liên hệ với giả thuyết, chỉ ra trường hợp phương pháp **thua**.
2. Ch.7 Hạn chế: root dataset; mô phỏng thay vì hệ thống thật; threat model omniscient; trigger có thực hiện được trên traffic thật không; MLP nhỏ; một dataset (nếu đã cắt); full participation; DP ở mức mẫu, không phải mức client; mọi mục đã cắt ở checkpoint.
3. Tóm tắt, kết luận, phụ lục (siêu tham số đầy đủ, lệnh tái lập 1 dòng).
4. **Kiểm tra tái lập:** clone repo sạch → chạy 1 cấu hình → ra đúng số trong bảng.
5. (Nếu chưa bị cắt) demo Docker Compose: 1 server + 5 client container chạy 5 vòng, chỉ để minh họa kiến trúc.
6. Rà soát cuối theo bảng ánh xạ bên dưới: mỗi yêu cầu bắt buộc đều có chỗ trong báo cáo.

**Sanity check:**
- [ ] Mọi con số trong văn bản trùng với bảng (grep thủ công 10 số ngẫu nhiên).
- [ ] Không có câu "phương pháp tốt hơn" nếu CI của hiệu số paired chứa 0 (theo đúng yêu cầu của đề: "chỉ được kết luận sau thử nghiệm").
- [ ] Mỗi tài liệu trích dẫn đều đã thực sự đọc phần được trích.

### 3. Output

- [ ] Báo cáo hoàn chỉnh (PDF)
- [ ] Repo code có README tái lập + `results/summary.parquet`
- [ ] (Nếu có) slide bảo vệ 10–12 trang

### 4. Báo cáo

- **Ch.6 (thảo luận), Ch.7 (hạn chế + hướng phát triển), Kết luận, Tóm tắt, Phụ lục.** Hoàn thiện Ch.1 theo kết quả thật.

---

## 3. Cấu trúc báo cáo và bảng ánh xạ

| Chương báo cáo | Phase viết | Yêu cầu bắt buộc của đề được đáp ứng |
|---|---|---|
| **Ch.1** Giới thiệu: bối cảnh, khoảng trống, câu hỏi nghiên cứu, giả thuyết H1–H3, đóng góp | P0 (nháp), P5 (chốt) | **Phát biểu khoảng trống và giả thuyết** |
| **Ch.2** Cơ sở lý thuyết và nghiên cứu liên quan (FL, non-IID, tấn công, phòng thủ, reputation, DP) | P2 | Nền cho **so sánh baseline**; định vị đóng góp so với FoolsGold/FLDetector |
| **Ch.3** Threat model và phát biểu bài toán | P0 (nháp), P2 (chốt) | **Threat model / giả định** |
| **Ch.4** Phương pháp đề xuất + **phân tích độ phức tạp** | P3 | Nội dung (1) *mô hình hóa tin cậy từ lịch sử + sai lệch thống kê có giải thích*; (2) *tổng hợp Byzantine-robust non-IID*; **Phân tích độ phức tạp** |
| **Ch.5** Thiết lập thực nghiệm: dữ liệu, partition, mô hình, tấn công, baseline, chỉ tiêu, giao thức thống kê | P1, P2, P3, P4 | **So sánh baseline** (FedAvg, FLTrust, DeepSight + Median/TrMean/Krum); **nhiều seed + CI**; dữ liệu IDS công khai; 5–20 client |
| **Ch.6.1** Kết quả chính (G1) | P4 | **So sánh baseline**; chỉ tiêu F1/Macro-F1, ASR |
| **Ch.6.2** Suy giảm theo tỷ lệ độc hại (G2) | P4 | Chỉ tiêu *độ suy giảm khi tỷ lệ client độc hại tăng* |
| **Ch.6.3** Ablation (G3) + adaptive on-off | P3–P4 | **Ablation** |
| **Ch.6.4** Đánh đổi riêng tư – hội tụ – truyền thông – phát hiện (G4) | P4 | Nội dung (3); chỉ tiêu *tốc độ hội tụ, chi phí truyền thông* |
| **Ch.6.5** Khả năng khái quát (n = 20, α, dataset thứ 2) | P4 | **Khả năng khái quát** |
| **Ch.7** Hạn chế, hướng phát triển, kết luận | P5 | **Nêu hạn chế** |
| **Phụ lục** Siêu tham số, tái lập | P1, P5 | Tính tái lập (gián tiếp cho **nhiều seed + CI**) |

---

## 4. Theo dõi tiến độ hằng tuần (tự điền)

| Tuần | Giờ kế hoạch | Giờ thực tế | Output đạt | Ghi chú/cắt |
|---|---|---|---|---|
| 16–22/11 | 11 | | | |
| 23–29/11 | 11 | | | |
| 30/11–06/12 | 12 | | | Checkpoint 30/11, 05/12 |
| 07–13/12 | 11 | | | |
| 14–15/12 | 3 | | | Nộp |
