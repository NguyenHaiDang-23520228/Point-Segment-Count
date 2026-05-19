# Cursor Prompts — PseCo Zero-Shot Counting
> Paste từng prompt vào Cursor theo thứ tự. Mỗi prompt là một task độc lập.

---

## PROMPT 1 — Phase 5: Hoàn thiện Grad-CAM & Concept Drift Visualization

```
Tôi đang làm đề tài PseCo Zero-Shot Object Counting trên notebook Google Colab.
File notebook là PseCo_ZeroShot_Counting.ipynb.

CONTEXT:
- Pipeline đã chạy xong: Point → Segment (SAM1) → Count (CLIP + ROIHead)
- Baseline đã có: MAE=18.11, RMSE=130.55 trên FSC-147 Test
- Đã có các biến: `sam`, `point_decoder`, `cls_head`, `clip_model`, `clip_preprocess`
- Đã có hàm: `pseco_count(image_path, text_prompt)` trả về dict với keys: count, boxes, scores, points, image, heatmap, all_points
- Đã có hàm: `run_gradcam(image_pil, box, text_prompt, target_layers)` trả về (grayscale_cam, crop)
- Đã có class: `ClipSimilarityWrapper(nn.Module)` với forward tính cosine sim
- target_layers đã định nghĩa: `[clip_model.visual.transformer.resblocks[-1].ln_1]`

NHIỆM VỤ: Viết các cell notebook hoàn chỉnh cho Phase 5.

YÊU CẦU CỤ THỂ:

**Cell 1 — Chuẩn bị ảnh test cho Concept Drift:**
- Tìm trong annotations FSC-147 một ảnh có class_name chứa 2 đối tượng có thể nhầm (ví dụ: ảnh có nhiều loại object nhỏ, hoặc dùng 1 ảnh từ tập test bất kỳ)
- Nếu không tìm được ảnh phù hợp, tải 1 ảnh từ internet (dùng requests + PIL) có 2 loại đối tượng rõ ràng (ví dụ: apples and oranges, hoặc coins and buttons)
- Lưu vào biến `test_image_path` và in ra class_name + số object

**Cell 2 — Chạy pseco_count với 2 prompt khác nhau:**
- Chạy `result_prompt1 = pseco_count(test_image_path, PROMPT_1)` với PROMPT_1 là đúng class
- Chạy `result_prompt2 = pseco_count(test_image_path, PROMPT_2)` với PROMPT_2 là sai class (ví dụ class khác trong cùng ảnh hoặc một object không liên quan)
- In ra: count của cả 2, chênh lệch bao nhiêu

**Cell 3 — Chạy Grad-CAM và visualize:**
- Lấy box có score cao nhất từ result_prompt1['boxes'][0]
- Chạy run_gradcam cho cả 2 prompt trên cùng box đó
- Dùng matplotlib tạo figure 1x4 gồm:
  * Cột 1: Ảnh gốc với box highlight
  * Cột 2: Crop của box đó
  * Cột 3: Grad-CAM overlay với PROMPT_1 (đúng) — dùng cv2.applyColorMap + cv2.addWeighted
  * Cột 4: Grad-CAM overlay với PROMPT_2 (sai)
- Title mỗi subplot phải ghi rõ prompt và similarity score
- Lưu figure ra file: `gradcam_concept_drift.png`

**Cell 4 — Phân tích và kết luận (markdown cell):**
Viết markdown cell giải thích:
- Tại sao Grad-CAM ở PROMPT_2 activate sai vùng → đây là Concept Drift
- Liên kết với lý do SAM1 cắt mask bị dính → CLIP bị nhiễu feature
- 2-3 câu kết luận ngắn gọn dùng cho báo cáo

CONSTRAINTS:
- Import thêm: `import cv2` nếu cần, hoặc dùng `matplotlib.cm` nếu không có cv2
- torch.set_grad_enabled(True) phải được gọi trước khi chạy Grad-CAM
- Wrap tất cả inference trong `with torch.no_grad()` ngoại trừ phần Grad-CAM
- Không thay đổi các hàm/class đã có, chỉ viết cell mới gọi chúng
```

---

## PROMPT 2 — Phase 6: Integrate SAM2 vào pseco_count()

```
Tôi đang làm đề tài PseCo Zero-Shot Object Counting trên Google Colab.
File notebook là PseCo_ZeroShot_Counting.ipynb.

CONTEXT:
- Đã có hàm pseco_count(image_path, text_prompt, score_threshold=0.1, nms_threshold=0.5, point_threshold=0.05, max_points=1000) — dùng SAM1
- Đã có hàm segment_with_sam2(predictor, image_np, points_xy, anchor_size=8) — trả về (pred_boxes, pred_ious) cùng shape như SAM1
- Đã có: `sam`, `point_decoder`, `cls_head`, `clip_model`, `sam2_predictor`
- Đã có hàm: `read_image(path)` → PIL Image 1024x1024
- Import sẵn: torch, torchvision.ops as vision_ops, numpy as np, PIL, transforms

NHIỆM VỤ: Refactor pseco_count() và thêm point caching.

YÊU CẦU CỤ THỂ:

**Cell 1 — Refactor pseco_count() thêm tham số segmenter:**
Viết lại hàm pseco_count() với chữ ký:
```python
def pseco_count(image_path, text_prompt, score_threshold=0.1, nms_threshold=0.5,
                point_threshold=0.05, max_points=1000, segmenter='sam1',
                cached_points=None):
```
Logic thay đổi:
- Phase 1 (Point Decoder): Nếu cached_points không None thì dùng luôn, không chạy lại
- Phase 2 (Segment): 
  * Nếu segmenter='sam1': dùng code SAM1 cũ (forward_sam_with_embeddings)
  * Nếu segmenter='sam2': gọi segment_with_sam2(sam2_predictor, np.array(img), pts)
- Phase 3 (CLIP + ROIHead): giữ nguyên, không đổi
- Return dict thêm key 'segmenter' và 'pred_points' (để cache)

**Cell 2 — Hàm cache và load point prompts:**
```python
import os

def save_point_cache(result, cache_dir, fname_key):
    """Lưu point prompts ra disk để dùng lại cho SAM2."""
    ...

def load_point_cache(cache_dir, fname_key):
    """Load cached points. Return None nếu chưa có."""
    ...
```
- cache_dir mặc định: f'{PROJECT_ROOT}/cache/point_prompts/'
- fname_key: tên file ảnh (dùng làm key)
- Lưu dưới dạng .pt với dict: {'points': tensor, 'score': tensor}

**Cell 3 — Test nhanh trên 1 ảnh:**
- Chạy pseco_count với segmenter='sam1', lưu cache
- Chạy lại với segmenter='sam2', load cache (cached_points=...)
- Assert rằng pred_points giống nhau (ablation control)
- In side-by-side: count SAM1 vs count SAM2 trên cùng ảnh
- Visualize 2 kết quả trên 1 figure: ảnh trái SAM1, ảnh phải SAM2, ghi count lên title

CONSTRAINTS:
- pseco_count() mới phải backward compatible: gọi không có segmenter vẫn chạy như cũ
- Khi segmenter='sam2', convert image PIL sang numpy RGB uint8 trước khi truyền vào segment_with_sam2
- Wrap SAM2 inference trong try/except, nếu lỗi thì fallback về SAM1 và log warning
- Không import thêm thư viện mới ngoài os
```

---

## PROMPT 3 — Phase 7: Ablation Study hoàn chỉnh

```
Tôi đang làm đề tài PseCo Zero-Shot Object Counting trên Google Colab.
File notebook là PseCo_ZeroShot_Counting.ipynb.

CONTEXT:
- Đã có pseco_count(image_path, text_prompt, segmenter='sam1'/'sam2', cached_points=None)
- Đã có evaluate_split(split_name, eval_cls_head, use_zeroshot=True) — chỉ chạy SAM1
- Đã có benchmark_image(image_path, text_prompt, segmenter=...) — đo FPS + VRAM
- Đã có mask_adhesion_rate(boxes, iou_thresh=0.5) — tỷ lệ mask bị dính
- Đã có: all_data, all_image_list, clip_text_prompts, cls_head_zs
- Baseline đã có: SAM1 → MAE=18.11, RMSE=130.55

NHIỆM VỤ: Viết toàn bộ ablation experiment và bảng kết quả.

YÊU CẦU CỤ THỂ:

**Cell 1 — Refactor evaluate_split() thêm segmenter và cache:**
Viết lại evaluate_split():
```python
def evaluate_split(split_name, eval_cls_head, use_zeroshot=True,
                   segmenter='sam1', cache_dir=None, max_images=None):
```
- Thêm tham số segmenter truyền xuống pseco_count()
- Thêm cache_dir: nếu không None thì load/save point prompts
- Thêm max_images: nếu không None thì chỉ chạy N ảnh đầu (dùng để test nhanh)
- Trong vòng lặp, đo thêm: thời gian inference mỗi ảnh, VRAM peak, mask_adhesion_rate
- Return: mae, rmse, preds, gts, avg_fps, avg_vram_gb, avg_mask_adhesion

**Cell 2 — Chạy ablation trên test set:**
```python
# Bước 1: Tạo cache point prompts (chạy SAM1 trước)
print("=== Bước 1: Tạo point cache (SAM1) ===")
mae1, rmse1, _, _, fps1, vram1, adh1 = evaluate_split(
    'test', cls_head_zs, use_zeroshot=True,
    segmenter='sam1', cache_dir=f'{PROJECT_ROOT}/cache/point_prompts/'
)

# Bước 2: Chạy SAM2 dùng cùng cache
print("=== Bước 2: Chạy SAM2 (dùng cùng point cache) ===")
mae2, rmse2, _, _, fps2, vram2, adh2 = evaluate_split(
    'test', cls_head_zs, use_zeroshot=True,
    segmenter='sam2', cache_dir=f'{PROJECT_ROOT}/cache/point_prompts/'
)
```
- Có progress bar tqdm với prefix [SAM1] / [SAM2]
- In kết quả sau mỗi bước

**Cell 3 — Bảng kết quả:**
Dùng pandas tạo bảng kết quả và display:
```
| Method           | MAE ↓  | RMSE ↓   | FPS ↓  | VRAM (GB) ↑ | Mask Adhesion ↓ |
|------------------|--------|----------|--------|-------------|-----------------|
| PseCo (SAM1)     | 18.11  | 130.55   | x.xx   | x.xx        | x.xx%           |
| PseCo (SAM2-Ours)| xx.xx  | xxx.xx   | x.xx   | x.xx        | x.xx%           |
```
- Highlight cell MAE/RMSE tốt hơn bằng màu xanh, tệ hơn bằng màu đỏ (dùng df.style)
- Thêm hàng "Δ (SAM2 vs SAM1)" ghi chênh lệch với dấu +/-
- Lưu bảng ra file: `ablation_results.csv`

**Cell 4 — Phân tích kết quả (markdown):**
Viết markdown cell template phân tích:
- Mask Adhesion cải thiện X% → SAM2 cắt rời rạc hơn
- Đánh đổi: FPS giảm X%, VRAM tăng X GB
- Kết luận trung thực: mask tốt hơn (điều kiện cần) nhưng chưa đủ nếu CLIP vẫn nhầm
- Đây là template, giữ placeholder [X] để điền số thực sau khi chạy

CONSTRAINTS:
- Nếu SAM2 chưa cài được trong môi trường, wrap evaluate_split SAM2 trong try/except và in hướng dẫn cài: `!pip install git+https://github.com/facebookresearch/sam2.git`
- Dùng torch.cuda.reset_peak_memory_stats() và torch.cuda.max_memory_allocated() để đo VRAM
- Dùng time.perf_counter() để đo FPS (1/elapsed)
- Không chạy cả 2 segmenter trong cùng 1 vòng lặp (chạy riêng từng bước để tránh OOM)
```

---

## PROMPT 4 — Phase 8: Curated Demo & compare_sam1_sam2()

```
Tôi đang làm đề tài PseCo Zero-Shot Object Counting trên Google Colab.
File notebook là PseCo_ZeroShot_Counting.ipynb.

CONTEXT:
- Đã có pseco_count(image_path, text_prompt, segmenter='sam1'/'sam2', cached_points=None)
- Đã có hàm compare_sam1_sam2() còn rỗng — chỉ có 2 dòng gọi pseco_count
- Đã có kết quả ablation: mae1, rmse1, mae2, rmse2, fps1, fps2, vram1, vram2, adh1, adh2
- annotations, all_image_list, PROJECT_ROOT đã có sẵn

NHIỆM VỤ: Implement compare_sam1_sam2() và chọn ảnh demo có chủ đích.

YÊU CẦU CỤ THỂ:

**Cell 1 — Implement compare_sam1_sam2() đầy đủ:**
```python
def compare_sam1_sam2(image_path, text_prompt, cached_points=None,
                      figsize=(20, 10), save_path=None):
```
Figure layout 2x3:
- Hàng 1: [Ảnh gốc] [SAM1 result] [SAM2 result]  
- Hàng 2: [Heatmap điểm] [SAM1 mask adhesion highlight] [SAM2 mask adhesion highlight]

Trong SAM1/SAM2 result subplot:
- Vẽ tất cả bounding boxes
- Màu box: xanh lá nếu score > 0.5, vàng nếu 0.2-0.5, đỏ nếu < 0.2
- Title ghi: "SAM1: count=X, adhesion=Y%" hoặc "SAM2: count=X, adhesion=Y%"

Trong mask adhesion highlight subplot:
- Tìm các cặp box có IoU > 0.5 (mask bị dính)
- Vẽ box bình thường màu xanh, box bị dính màu đỏ đậm + fill đỏ nhạt (alpha=0.3)
- Thêm chú thích số cặp bị dính

Nếu save_path không None: lưu figure ra file đó
Return: dict với count_sam1, count_sam2, adhesion_sam1, adhesion_sam2

**Cell 2 — Tìm ảnh demo Case 1 & 2 (Mask Adhesion Fix):**
Viết hàm tìm ảnh phù hợp:
```python
def find_demo_images(annotations, image_list, criteria='crowded', n=3):
    """
    criteria='crowded': ảnh có nhiều object nhỏ sát nhau (count > 50, avg box size < 50px)
    criteria='failure': ảnh mà SAM1 đếm sai nhiều nhất (cần preds và gts từ evaluate_split)
    """
```
- Với criteria='crowded': tìm theo thống kê từ annotations
- Chạy hàm này, lấy 3 ảnh crowded nhất
- Chạy compare_sam1_sam2() trên mỗi ảnh, lưu ra file: `demo_case1_crowded_1.png`, `demo_case1_crowded_2.png`, `demo_case1_crowded_3.png`

**Cell 3 — Tìm ảnh demo Case 4 (Failure Case):**
```python
# Dùng preds và gts từ evaluate_split đã chạy ở Phase 7
errors_sam2 = [(abs(p-g), fname) for (p, g, fname) in zip(preds2, gts2, all_image_list['test'])]
errors_sam2.sort(reverse=True)

# Lấy top 3 ảnh SAM2 đếm sai nhiều nhất
failure_cases = errors_sam2[:3]
for err, fname in failure_cases:
    img_path = f'{PROJECT_ROOT}/data/fsc147/images_384_VarV2/{fname}'
    class_name = annotations[fname]['class_name']
    gt = annotations[fname]['annotations']
    result = compare_sam1_sam2(img_path, class_name,
                                save_path=f'demo_case4_failure_{fname}.png')
    print(f'{fname}: GT={len(gt)}, SAM1={result["count_sam1"]}, SAM2={result["count_sam2"]}, Error={err}')
```

**Cell 4 — Summary figure cho báo cáo:**
Tạo 1 figure tổng hợp dạng infographic:
- Góc trên trái: bảng số liệu (MAE, RMSE, FPS, VRAM) dạng text table trên nền trắng
- Góc trên phải: bar chart so sánh MAE và RMSE (SAM1 vs SAM2), dùng màu xanh/cam
- Dưới: 1 ví dụ case crowded side-by-side (SAM1 | SAM2)
- Title lớn ở trên: "PseCo: SAM1 vs SAM2 — FSC-147 Test Set"
- Lưu ra: `summary_report_figure.png` với dpi=200

CONSTRAINTS:
- Tất cả matplotlib figure phải có plt.tight_layout() và bbox_inches='tight' khi save
- Dùng matplotlib.patches.Rectangle cho bounding boxes, không dùng cv2.rectangle
- Nếu preds2/gts2 chưa có (SAM2 chưa chạy xong), thay bằng placeholder random để test layout
- So sánh Grad-CAM (từ Phase 5) vào Case 3 nếu đã có file gradcam_concept_drift.png: load và nhúng vào summary figure bằng plt.imread
```

---

## Thứ tự chạy đề xuất

1. **Prompt 1** (Phase 5 — Grad-CAM) → Chạy độc lập, không phụ thuộc SAM2
2. **Prompt 2** (Phase 6 — Integrate SAM2) → Cần cài SAM2 trước: `!pip install git+https://github.com/facebookresearch/sam2.git`
3. **Prompt 3** (Phase 7 — Ablation) → Chạy sau khi Prompt 2 xong, tốn nhiều thời gian nhất
4. **Prompt 4** (Phase 8 — Demo) → Chạy cuối cùng, cần kết quả từ Prompt 3

## Lưu ý khi dùng Cursor

- Mở file `PseCo_ZeroShot_Counting.ipynb` trong Cursor trước
- Paste prompt vào chat Cursor, kèm theo "@PseCo_ZeroShot_Counting.ipynb" để Cursor đọc context
- Sau mỗi prompt, yêu cầu Cursor: *"Chỉ thêm cell mới, không xóa hoặc sửa cell cũ"*
- Nếu Cursor hỏi về biến chưa có (vd: `preds2`), trả lời: *"Đó là kết quả từ Phase 7, dùng placeholder list rỗng để test layout trước"*
