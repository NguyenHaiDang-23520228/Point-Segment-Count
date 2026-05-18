---
name: PseCo Zero-Shot Research & Improvement Plan
overview: "Phase 1: Reproduce PseCo Zero-Shot Baseline (SAM1). Phase 2: Explainable AI analysis via Grad-CAM to identify Concept Drift. Phase 3: Integrate SAM2 to fix mask adhesion, followed by strict Ablation study on FSC-147."
todos:
  # Phần Baseline (Đã hoàn thành)
  - id: baseline-setup
    content: "Phase 0: Setup env, download FSC-147, SAM1, CLIP weights."
    status: completed
  - id: baseline-point
    content: "Phase 1: Run Point Decoder for class-agnostic localization."
    status: completed
  - id: baseline-segment
    content: "Phase 2: Run SAM1 Mask Decoder to generate hierarchical masks."
    status: completed
  - id: baseline-count
    content: "Phase 3: Run CLIP Text-Image matching (Zero-Shot only) & HKD filtering."
    status: completed
  - id: baseline-eval
    content: "Phase 4: Evaluate Baseline (SAM1) -> MAE=18.11, RMSE=130.55 on Test set."
    status: completed
  # Phần Nghiên cứu & Cải tiến (Sẽ làm)
  - id: gradcam-analysis
    content: "Phase 5: Apply Grad-CAM on CLIP similarity logit to visualize 'Concept Drift' when shifting Text Prompts."
    status: pending
  - id: sam2-integration
    content: "Phase 6: Replace SAM1 with SAM2. Ensure exact same Point Prompts (same seed) are fed into SAM2."
    status: pending
  - id: ablation-study
    content: "Phase 7: Run strict Ablation experiment (SAM1 vs SAM2). Measure MAE, RMSE, FPS, and VRAM usage."
    status: pending
  - id: curated-demo
    content: "Phase 8: Select intentional demo images (Mask adhesion, Prompt shift, Failure case) for final report."
    status: pending
isProject: false
---
Nghiên cứu và Cải tiến Đếm Đối tượng Zero-Shot PseCo
Goal
Không chỉ tái hiện mô hình PseCo (Zero-Shot), mà phân tích điểm yếu của nó khi đối mặt với sự trôi khái niệm (Concept Drift) khi đổi Text Prompt, và đề xuất cải tiến bằng cách nâng cấp linh kiện phân đoạn từ SAM 1 lên SAM 2 để giảm lỗi dính mask (Mask Adhesion Error).

PART I: BASELINE REPRODUCTION (Hoàn thành)
(Phần này giữ nguyên logic gốc của tác giả, chỉ tập trung vào Text Prompt)

Phase 0-4: Đã chạy xong
Đã chạy thành công luồng: Image Encoder -> Point Decoder -> SAM1 Mask Decoder -> CLIP Text Embeddings -> Cosine Similarity -> Count.
Kết quả Baseline trên FSC-147 (Test): MAE = 18.11, RMSE = 130.55.
Phát hiện thực tế: Đổi Text Prompt (VD: từ "apple" sang "leaf") khiến mô hình đếm sai, Bounding Box/Mask không chuẩn xác do SAM1 bị dính các vật thể nhỏ/sát nhau.
PART II: EXPLAINABLE AI & ANALYSIS (Phase 5)
Mục tiêu: Chứng minh lý thuyết "Tại sao đổi prompt lại sai" không phải do code lỗi, mà do đặc thù của CLIP + SAM1.

Phase 5: Grad-CAM Targeted Analysis
Thư viện sử dụng: pytorch-gradcam
Target Layer & Model: Phải gắn Grad-CAM vào CLIP Image Encoder, cụ thể là trích xuất Gradient của Cosine Similarity Score giữa Image Feature (của mask đang xét) và Text Feature (của từ khóa nhập vào).
Thực nghiệm:
Chạy 1 bức ảnh có 2 đối tượng (VD: Quả táo và Cái lá).
Nhập Prompt 1: "A photo of an apple". Chụp Grad-CAM (Vùng đỏ phải nằm ở quả táo).
Nhập Prompt 2: "A photo of a leaf". Chụp Grad-CAM.
Phân tích kỳ vọng: Nếu vùng đỏ nhảy lung tung hoặc bao phủ cả cái lá khi prompt là "apple", chứng tỏ SAM1 cắt mask bị dính lá → CLIP bị nhiễu đặc trưng (Concept Drift).
PART III: IMPROVEMENT & ABLATION (Phase 6 & 7)
Mục tiêu: Vá lỗi "Mask dính" bằng SAM2, nhưng phải chứng minh bằng Ablation nghiêm ngặt.

Phase 6: Tích hợp SAM 2
Nguyên tắc cốt lõi (Ablation Control): TUYỆT ĐỐI KHÔNG đổi bất kỳ thông số nào ở Phase 1 (Point Decoder). Phải lưu lại exact tọa độ các điểm (Point Prompts) mà Baseline (SAM1) đã sinh ra, và ném đúng các điểm đó vào SAM2.
Thay đổi duy nhất: Thay thư viện segment_anything bằng sam2. Dùng SAM2 Predictor để nhận Point Prompts và sinh ra Mask Proposals mới.
Giữ nguyên bộ lọc CLIP và HKD ở Phase 3.
Phase 7: Thí nghiệm Ablation Khắt khe
Chạy lại toàn bộ FSC-147 Test set với cấu hình SAM2.Bảng báo cáo phải có đủ các cột sau:

Method: PseCo (SAM1) vs. PseCo (SAM2 - Ours)
MAE ↓ & RMSE ↓: So sánh chỉ số đếm. (Lưu ý: Không hứa hẹn sẽ giảm mạnh, chỉ báo cáo số liệu thực tế).
Inference Speed (FPS) ↓: Đo thời gian chạy mỗi ảnh. (SAM2 sẽ chậm hơn, phải trung thực báo cáo đây là đánh đổi).
VRAM Usage (GB) ↑: Đo bộ nhớ tải (SAM2 nặng hơn).
Mask Adhesion Rate (Tùy chọn): Đếm số lượng mask bị dính (IoU với nhau > 0.5) ở SAM1 so với SAM2 để chứng minh mask thực sự đẹp hơn.
PART IV: REPORTING & DEMO (Phase 8)
Mục tiêu: Trình bày kết quả có chủ đích, không "cherry-pick" 100% case tốt.

Phase 8: Curated Demo Images
Chọn ra khoảng 5-6 bức ảnh đưa vào Báo cáo/Slide thuyết trình:

Case 1 & 2 (Mask Adhesion Fix): Ảnh đông đúc. Trình bày Side-by-side: SAM1 cắt dính bét → SAM2 cắt rời rạc rõ ràng → Đếm đúng.
Case 3 (Concept Drift Visualization): Ảnh mà SAM1 cắt dính. Trình bày Side-by-side: Ảnh gốc → Ảnh có Grad-CAM overlay (chỉ ra vùng CLIP đang nhìn sai do mask dính).
Case 4 (Failure Case - RẤT QUAN TRỌNG): 1 bức ảnh mà SAM2 cắt đẹp hơn SAM1, nhưng VẪN ĐẾM SAI.
Ghi chú trong báo cáo: "Điều này chứng tỏ việc cải thiện chất lượng Mask (điều kiện cần) chưa đủ để giải quyết triệt để bài toán Zero-shot nếu Classifier (CLIP) vẫn bị nhầm lẫn giữa foreground và background. Đây là hướng mở cho nghiên cứu tương lai."
Key References for Improvement
SAM 2: Meta AI Segment Anything 2 Repository.
Grad-CAM: pytorch-gradcam library (Tài liệu hướng dẫn targetting custom similarity scores).
Lưu ý cực kỳ quan trọng:
Phần nằm giữa 2 dấu --- ở trên cùng gọi là YAML Frontmatter. Bạn TUYỆT ĐỐI KHÔNG được thêm bớt khoảng trắng (space) ở đầu các dòng trong phần này (như name:, todos:, - id:). Nếu bạn lỡ thụt lề (indent) sai, phần mềm đọc file plan của bạn sẽ báo lỗi. Cứ copy y nguyên từ chữ --- đầu tiên đến chữ --- thứ hai là chuẩn!