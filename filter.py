import os
import cv2
import torch
import numpy as np
import argparse
import pandas as pd
import shutil
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ---------------- #
MODEL_NAME = "facebook/dino-vits16"
FRAME_SIZE = (224, 224)
TRANSITION_THRESHOLD = 0.75  # Giảm xuống để ít strict hơn trong việc phát hiện chuyển cảnh
MIN_SCENE_LENGTH = 2         # Giảm xuống để chấp nhận cảnh ngắn hơn

# Quality filtering parameters (điều chỉnh RẤT mềm)
BLUR_PERCENTILE = 10.0       # Chỉ loại bỏ 10% frame mờ nhất 
EDGE_PERCENTILE = 10.0       # Chỉ loại bỏ 10% frame có edge thấp nhất
ENABLE_ADAPTIVE_FILTERING = True  # Sử dụng percentile thay vì ngưỡng cố định

# Fixed thresholds (rất thấp để giữ nhiều frames)
BLUR_THRESHOLD = 10.0        # Giảm xuống rất thấp
EDGE_THRESHOLD = 5.0         # Giảm xuống rất thấp
ENABLE_BLUR_DETECTION = True
ENABLE_EDGE_DETECTION = True

# NEW: Similarity deduplication parameters
ENABLE_SIMILARITY_FILTERING = True  # Bật tính năng lọc frame tương tự
SIMILARITY_THRESHOLD = 0.95         # Ngưỡng similarity để coi là "gần giống nhau"
MIN_FRAME_DISTANCE = 1              # Khoảng cách tối thiểu giữa các frame được giữ lại
SIMILARITY_WINDOW_SIZE = 5          # Kích thước window để so sánh similarity

# Default folders
DEFAULT_INPUT_KEYFRAMES_DIR = "keyframes"
DEFAULT_INPUT_MAP_DIR = "map"
DEFAULT_OUTPUT_KEYFRAMES_DIR = "keyframesv2.0"
DEFAULT_OUTPUT_MAP_DIR = "mapv2.0"
# ---------------------------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DINO model
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def extract_embedding(image_path):
    """Trích xuất embedding từ file ảnh"""
    try:
        img = Image.open(image_path).convert('RGB').resize(FRAME_SIZE)
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return embedding
    except Exception as e:
        print(f"❌ Lỗi khi trích xuất embedding từ {image_path}: {e}")
        return None

def calculate_blur_score(image_path):
    """Tính toán độ mờ của ảnh sử dụng Laplacian variance"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        return laplacian_var
    except Exception as e:
        return 0.0

def calculate_edge_density(image_path):
    """Tính toán mật độ edge để phát hiện frame mờ"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sử dụng threshold RẤT thấp cho Canny để detect nhiều edge hơn
        edges = cv2.Canny(gray, 20, 80)  # Giảm từ 30,100 xuống 20,80
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]) * 100
        
        return edge_density
    except Exception as e:
        return 0.0

def calculate_frame_quality_scores(image_path):
    """Tính toán các chỉ số chất lượng frame"""
    blur_score = calculate_blur_score(image_path)
    edge_density = calculate_edge_density(image_path)
    
    return {
        'blur_score': blur_score,
        'edge_density': edge_density
    }

def determine_adaptive_thresholds(all_quality_scores, config):
    """Tự động xác định ngưỡng dựa trên phân phối dữ liệu"""
    if not all_quality_scores:
        return None, None
    
    blur_scores = [q['blur_score'] for q in all_quality_scores]
    edge_scores = [q['edge_density'] for q in all_quality_scores]
    
    # Tính percentile thresholds
    blur_threshold = np.percentile(blur_scores, config['blur_percentile'])
    edge_threshold = np.percentile(edge_scores, config['edge_percentile'])
    
    return blur_threshold, edge_threshold

def is_frame_acceptable_adaptive(quality_scores, blur_threshold, edge_threshold, config):
    """Kiểm tra frame có đạt chất lượng không (với adaptive thresholds)"""
    if config['enable_blur_detection'] and blur_threshold is not None:
        if quality_scores['blur_score'] < blur_threshold:
            return False, "blur"
    
    if config['enable_edge_detection'] and edge_threshold is not None:
        if quality_scores['edge_density'] < edge_threshold:
            return False, "low_edge"
    
    return True, "acceptable"

def is_frame_acceptable_fixed(quality_scores, config):
    """Kiểm tra frame có đạt chất lượng không (với fixed thresholds)"""
    if config['enable_blur_detection']:
        if quality_scores['blur_score'] < config['blur_threshold']:
            return False, "blur"
    
    if config['enable_edge_detection']:
        if quality_scores['edge_density'] < config['edge_threshold']:
            return False, "low_edge"
    
    return True, "acceptable"

def calculate_similarities(embeddings):
    """Tính toán cosine similarity giữa các frame liên tiếp"""
    similarities = []
    for i in range(1, len(embeddings)):
        if embeddings[i-1] is not None and embeddings[i] is not None:
            sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
            similarities.append(sim)
        else:
            similarities.append(1.0)
    return similarities

def detect_scene_transitions(similarities, threshold):
    """Phát hiện các điểm chuyển cảnh dựa trên similarity thấp"""
    transition_points = []
    for i, sim in enumerate(similarities):
        if sim < threshold:
            transition_points.append(i + 1)
    return transition_points

def group_into_scenes(transition_points, total_frames, min_length):
    """Nhóm các frame thành các cảnh và loại bỏ cảnh quá ngắn"""
    scenes = []
    start_idx = 0
    
    for transition_idx in transition_points:
        scene_length = transition_idx - start_idx
        if scene_length >= min_length:
            scenes.append((start_idx, transition_idx - 1))
        start_idx = transition_idx
    
    final_scene_length = total_frames - start_idx
    if final_scene_length >= min_length:
        scenes.append((start_idx, total_frames - 1))
    
    return scenes

def filter_similar_frames_in_scene(scene_embeddings, scene_indices, config):
    """
    Lọc bỏ các frame tương tự nhau trong một scene
    
    Args:
        scene_embeddings: List embeddings của các frame trong scene
        scene_indices: List các index tương ứng với embeddings
        config: Dictionary chứa các tham số cấu hình
    
    Returns:
        List các indices được giữ lại sau khi lọc
    """
    if not config['enable_similarity_filtering'] or len(scene_embeddings) <= 1:
        return scene_indices
    
    similarity_threshold = config['similarity_threshold']
    min_distance = config['min_frame_distance']
    
    kept_indices = [0]  # Luôn giữ frame đầu tiên của scene
    last_kept_idx = 0
    
    for i in range(1, len(scene_embeddings)):
        current_embedding = scene_embeddings[i]
        
        # Kiểm tra khoảng cách tối thiểu
        if i - last_kept_idx < min_distance:
            continue
            
        # So sánh với frame được giữ gần nhất
        last_kept_embedding = scene_embeddings[last_kept_idx]
        similarity = cosine_similarity([current_embedding], [last_kept_embedding])[0][0]
        
        # Nếu similarity thấp hơn threshold, giữ frame này
        if similarity < similarity_threshold:
            kept_indices.append(i)
            last_kept_idx = i
    
    # Luôn giữ frame cuối cùng của scene nếu chưa được giữ
    if kept_indices[-1] != len(scene_embeddings) - 1:
        kept_indices.append(len(scene_embeddings) - 1)
    
    # Chuyển đổi local indices thành global indices
    global_kept_indices = [scene_indices[i] for i in kept_indices]
    
    return global_kept_indices

def filter_similar_frames_advanced(scene_embeddings, scene_indices, config):
    """
    Lọc frame tương tự với thuật toán sliding window advanced
    """
    if not config['enable_similarity_filtering'] or len(scene_embeddings) <= 1:
        return scene_indices
    
    similarity_threshold = config['similarity_threshold']
    window_size = min(config['similarity_window_size'], len(scene_embeddings))
    
    kept_indices = [0]  # Luôn giữ frame đầu tiên
    
    for i in range(1, len(scene_embeddings)):
        current_embedding = scene_embeddings[i]
        should_keep = True
        
        # So sánh với các frame trong window
        window_start = max(0, i - window_size)
        for j in range(window_start, i):
            if j in kept_indices:  # Chỉ so sánh với frames đã được giữ lại
                local_j = kept_indices.index(j) if j in kept_indices else -1
                if local_j >= 0:
                    past_embedding = scene_embeddings[j]
                    similarity = cosine_similarity([current_embedding], [past_embedding])[0][0]
                    
                    if similarity >= similarity_threshold:
                        should_keep = False
                        break
        
        if should_keep:
            kept_indices.append(i)
    
    # Chuyển đổi local indices thành global indices
    global_kept_indices = [scene_indices[i] for i in kept_indices]
    
    return global_kept_indices

def apply_similarity_filtering_to_scenes(embeddings, valid_rows, scenes, config):
    """
    Áp dụng lọc similarity cho tất cả các scenes
    
    Returns:
        Tuple (filtered_embeddings, filtered_rows, similarity_stats)
    """
    if not config['enable_similarity_filtering']:
        # Tạo indices cho tất cả frames trong scenes
        all_scene_indices = []
        for scene_start, scene_end in scenes:
            for i in range(scene_start, scene_end + 1):
                all_scene_indices.append(i)
        
        filtered_embeddings = [embeddings[i] for i in all_scene_indices]
        filtered_rows = [valid_rows[i] for i in all_scene_indices]
        return filtered_embeddings, filtered_rows, {'original': len(all_scene_indices), 'filtered': len(all_scene_indices), 'removed': 0}
    
    print("🔄 Đang lọc các frame tương tự nhau trong từng scene...")
    
    all_kept_indices = []
    similarity_stats = {'original': 0, 'filtered': 0, 'removed': 0}
    
    for scene_idx, (scene_start, scene_end) in enumerate(scenes):
        scene_length = scene_end - scene_start + 1
        scene_embeddings = embeddings[scene_start:scene_end + 1]
        scene_indices = list(range(scene_start, scene_end + 1))
        
        similarity_stats['original'] += scene_length
        
        # Áp dụng lọc similarity cho scene này
        if config.get('use_advanced_similarity_filtering', False):
            kept_indices = filter_similar_frames_advanced(scene_embeddings, scene_indices, config)
        else:
            kept_indices = filter_similar_frames_in_scene(scene_embeddings, scene_indices, config)
        
        all_kept_indices.extend(kept_indices)
        similarity_stats['filtered'] += len(kept_indices)
        
        print(f"   🎬 Scene {scene_idx + 1}: {scene_length} → {len(kept_indices)} frames "
              f"(loại bỏ {scene_length - len(kept_indices)})")
    
    similarity_stats['removed'] = similarity_stats['original'] - similarity_stats['filtered']
    
    # Tạo danh sách kết quả
    filtered_embeddings = [embeddings[i] for i in all_kept_indices]
    filtered_rows = [valid_rows[i] for i in all_kept_indices]
    
    print(f"📊 Kết quả lọc similarity:")
    print(f"   📥 Frames đầu vào: {similarity_stats['original']}")
    print(f"   📤 Frames giữ lại: {similarity_stats['filtered']}")
    print(f"   🗑️ Frames loại bỏ: {similarity_stats['removed']}")
    print(f"   📊 Tỷ lệ loại bỏ: {(similarity_stats['removed'] / similarity_stats['original'] * 100):.1f}%")
    
    return filtered_embeddings, filtered_rows, similarity_stats

def filter_transition_frames_for_video(video_name, config):
    """Lọc các frame chuyển cảnh và frame mờ cho một video cụ thể"""
    print(f"\n🎬 Đang xử lý video: {video_name}")
    
    # Đường dẫn các file và folder
    csv_path = os.path.join(config['input_map_dir'], f"{video_name}.csv")
    input_frames_dir = os.path.join(config['input_keyframes_dir'], video_name)
    output_frames_dir = os.path.join(config['output_keyframes_dir'], video_name)
    output_csv_path = os.path.join(config['output_map_dir'], f"{video_name}.csv")
    
    # Kiểm tra file CSV tồn tại
    if not os.path.exists(csv_path):
        print(f"⚠️ Không tìm thấy file CSV: {csv_path}")
        return None
    
    if not os.path.exists(input_frames_dir):
        print(f"⚠️ Không tìm thấy folder frames: {input_frames_dir}")
        return None
    
    # Đọc CSV mapping
    try:
        df = pd.read_csv(csv_path)
        print(f"📄 Đọc được {len(df)} frames từ CSV")
    except Exception as e:
        print(f"❌ Lỗi khi đọc CSV {csv_path}: {e}")
        return None
    
    os.makedirs(output_frames_dir, exist_ok=True)
    
    # Phase 1: Tính toán chất lượng cho TẤT CẢ frames trước
    print("🔍 Đang phân tích chất lượng tất cả frames...")
    all_quality_scores = []
    all_valid_paths = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing all frames"):
        frame_filename = f"{row['frame_idx']}.jpg"
        frame_path = os.path.join(input_frames_dir, frame_filename)
        
        if os.path.exists(frame_path):
            quality = calculate_frame_quality_scores(frame_path)
            all_quality_scores.append(quality)
            all_valid_paths.append((frame_path, row))
        else:
            print(f"⚠️ Không tìm thấy frame: {frame_path}")
    
    if not all_quality_scores:
        print("❌ Không có frame nào để phân tích")
        return None
    
    # Hiển thị thống kê chất lượng
    blur_scores = [q['blur_score'] for q in all_quality_scores]
    edge_scores = [q['edge_density'] for q in all_quality_scores]
    
    print(f"📊 Thống kê chất lượng:")
    print(f"   🌫️ Blur score: min={min(blur_scores):.1f}, max={max(blur_scores):.1f}, "
          f"median={np.median(blur_scores):.1f}")
    print(f"   📐 Edge density: min={min(edge_scores):.1f}%, max={max(edge_scores):.1f}%, "
          f"median={np.median(edge_scores):.1f}%")
    
    # Phase 2: Xác định ngưỡng (adaptive hoặc fixed)
    if config['enable_adaptive_filtering']:
        blur_threshold, edge_threshold = determine_adaptive_thresholds(all_quality_scores, config)
        print(f"📏 Ngưỡng adaptive:")
        if blur_threshold is not None:
            print(f"   🌫️ Blur threshold (percentile {config['blur_percentile']}): {blur_threshold:.1f}")
        if edge_threshold is not None:
            print(f"   📐 Edge threshold (percentile {config['edge_percentile']}): {edge_threshold:.1f}%")
    else:
        blur_threshold = config['blur_threshold']
        edge_threshold = config['edge_threshold']
        print(f"📏 Ngưỡng cố định:")
        print(f"   🌫️ Blur threshold: {blur_threshold}")
        print(f"   📐 Edge threshold: {edge_threshold}%")
    
    # Phase 3: Lọc theo chất lượng
    print("🔍 Đang lọc theo chất lượng...")
    embeddings = []
    valid_rows = []
    quality_filter_stats = {'blur': 0, 'low_edge': 0, 'acceptable': 0, 'embedding_error': 0}
    
    for i, (frame_path, row) in enumerate(tqdm(all_valid_paths, desc="Quality filtering")):
        quality = all_quality_scores[i]
        
        if config['enable_adaptive_filtering']:
            is_acceptable, reason = is_frame_acceptable_adaptive(quality, blur_threshold, edge_threshold, config)
        else:
            is_acceptable, reason = is_frame_acceptable_fixed(quality, config)
        
        if is_acceptable:
            emb = extract_embedding(frame_path)
            if emb is not None:
                embeddings.append(emb)
                valid_rows.append(row)
                quality_filter_stats['acceptable'] += 1
            else:
                quality_filter_stats['embedding_error'] += 1
        else:
            quality_filter_stats[reason] += 1
    
    print(f"📊 Kết quả lọc chất lượng:")
    print(f"   ✅ Chấp nhận được: {quality_filter_stats['acceptable']}")
    print(f"   🌫️ Bị mờ: {quality_filter_stats['blur']}")
    print(f"   📉 Edge thấp: {quality_filter_stats['low_edge']}")
    print(f"   ❌ Lỗi embedding: {quality_filter_stats['embedding_error']}")
    
    # Cảnh báo nếu loại bỏ quá nhiều
    total_processed = len(all_valid_paths)
    kept_percentage = (quality_filter_stats['acceptable'] / total_processed) * 100 if total_processed > 0 else 0
    if kept_percentage < 30:
        print(f"⚠️ CẢNH BÁO: Chỉ giữ lại {kept_percentage:.1f}% frames! Hãy thử:")
        print(f"   🌸 --ultra_gentle (chỉ lọc chuyển cảnh)")
        print(f"   🌿 --gentle (chỉ loại bỏ 5% frame tệ nhất)")
        print(f"   📐 --blur_percentile 5.0 --edge_percentile 5.0")
    
    if len(embeddings) < config['min_scene_length']:
        print(f"⚠️ Không đủ frames chất lượng tốt để xử lý (cần ít nhất {config['min_scene_length']} frames)")
        return None
    
    # Phase 4: Phát hiện chuyển cảnh
    print("📊 Đang tính toán similarities...")
    similarities = calculate_similarities(embeddings)
    avg_similarity = np.mean(similarities)
    print(f"📈 Similarity trung bình: {avg_similarity:.3f}")
    
    transition_points = detect_scene_transitions(similarities, config['transition_threshold'])
    print(f"🎭 Phát hiện {len(transition_points)} điểm chuyển cảnh")
    
    scenes = group_into_scenes(transition_points, len(embeddings), config['min_scene_length'])
    print(f"🎞️ Tìm thấy {len(scenes)} cảnh hợp lệ")
    
    if not scenes:
        print("⚠️ Không tìm thấy cảnh nào hợp lệ")
        return None
    
    # Phase 5: NEW - Lọc các frame tương tự nhau trong từng scene
    filtered_embeddings, filtered_rows, similarity_stats = apply_similarity_filtering_to_scenes(
        embeddings, valid_rows, scenes, config
    )
    
    # Phase 6: Tạo DataFrame cuối cùng
    final_df = pd.DataFrame(filtered_rows)
    final_df = final_df.reset_index(drop=True)
    final_df['n'] = range(len(final_df))
    
    # Phase 7: Copy frames
    print(f"💾 Đang copy {len(final_df)} frames chất lượng cao...")
    copied_count = 0
    copy_errors = 0
    
    for _, row in tqdm(final_df.iterrows(), total=len(final_df), desc="Copying frames"):
        source_frame = f"{row['frame_idx']}.jpg"
        source_path = os.path.join(input_frames_dir, source_frame)
        dest_path = os.path.join(output_frames_dir, source_frame)
        
        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                copied_count += 1
            else:
                copy_errors += 1
        except Exception as e:
            copy_errors += 1
    
    # Phase 8: Lưu CSV
    try:
        final_df.to_csv(output_csv_path, index=False)
        print(f"📄 Đã lưu CSV mới: {output_csv_path}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu CSV: {e}")
        return None
    
    # Thống kê kết quả
    original_count = len(df)
    quality_passed = quality_filter_stats['acceptable']
    scene_filtered = similarity_stats['filtered']
    final_count = len(final_df)
    
    quality_removed = original_count - quality_passed
    similarity_removed = similarity_stats['removed']
    total_removed = original_count - final_count
    removal_percentage = (total_removed / original_count) * 100 if original_count > 0 else 0
    
    print(f"✅ Hoàn thành video {video_name}!")
    print(f"📊 Frames gốc: {original_count}")
    print(f"📊 Loại bỏ do chất lượng kém: {quality_removed}")
    print(f"📊 Loại bỏ do similarity cao: {similarity_removed}")
    print(f"📊 Frames cuối cùng: {final_count}")
    print(f"📊 Tổng loại bỏ: {total_removed} ({removal_percentage:.1f}%)")
    print(f"📊 Frames đã copy: {copied_count}")
    if copy_errors > 0:
        print(f"⚠️ Lỗi copy: {copy_errors}")
    
    return {
        'video_name': video_name,
        'original_frames': original_count,
        'quality_passed': quality_passed,
        'scene_filtered': scene_filtered,
        'filtered_frames': final_count,
        'quality_removed': quality_removed,
        'similarity_removed': similarity_removed,
        'total_removed': total_removed,
        'removal_percentage': removal_percentage,
        'scenes_count': len(scenes),
        'transitions_count': len(transition_points),
        'avg_similarity': avg_similarity,
        'copied_frames': copied_count,
        'copy_errors': copy_errors,
        'quality_stats': quality_filter_stats,
        'similarity_stats': similarity_stats
    }

def process_all_videos(config):
    """Xử lý tất cả các video trong folder keyframes và map"""
    print("🚀 BẮT ĐẦU XỬ LÝ ADAPTIVE FRAME FILTERING")
    print("=" * 70)
    
    if not os.path.exists(config['input_keyframes_dir']):
        print(f"❌ Không tìm thấy folder keyframes: {config['input_keyframes_dir']}")
        return
    
    if not os.path.exists(config['input_map_dir']):
        print(f"❌ Không tìm thấy folder map: {config['input_map_dir']}")
        return
    
    os.makedirs(config['output_keyframes_dir'], exist_ok=True)
    os.makedirs(config['output_map_dir'], exist_ok=True)
    
    csv_files = [f for f in os.listdir(config['input_map_dir']) if f.endswith('.csv')]
    video_names = [os.path.splitext(f)[0] for f in csv_files]
    
    if not video_names:
        print(f"❌ Không tìm thấy file CSV nào trong folder: {config['input_map_dir']}")
        return
    
    print(f"🎯 Tìm thấy {len(video_names)} video để xử lý:")
    for i, name in enumerate(video_names, 1):
        print(f"   {i}. {name}")
    
    print(f"\n⚙️ Cấu hình:")
    print(f"   📂 Input keyframes: {config['input_keyframes_dir']}")
    print(f"   📂 Output keyframes: {config['output_keyframes_dir']}")
    print(f"   🎭 Transition threshold: {config['transition_threshold']}")
    if config['enable_adaptive_filtering']:
        print(f"   🤖 Adaptive filtering: Bật")
        if config['enable_blur_detection']:
            print(f"      🌫️ Blur percentile: {config['blur_percentile']}%")
        if config['enable_edge_detection']:
            print(f"      📐 Edge percentile: {config['edge_percentile']}%")
    else:
        print(f"   🎯 Fixed thresholds:")
        if config['enable_blur_detection']:
            print(f"      🌫️ Blur threshold: {config['blur_threshold']}")
        if config['enable_edge_detection']:
            print(f"      📐 Edge threshold: {config['edge_threshold']}%")
    
    # NEW: Hiển thị cấu hình similarity filtering
    if config['enable_similarity_filtering']:
        print(f"   🔄 Similarity filtering: Bật")
        print(f"      🎯 Similarity threshold: {config['similarity_threshold']}")
        print(f"      📏 Min frame distance: {config['min_frame_distance']}")
        print(f"      🪟 Window size: {config['similarity_window_size']}")
    else:
        print(f"   🔄 Similarity filtering: Tắt")
    
    print(f"   🖥️ Device: {device}")
    print("=" * 70)
    
    results = []
    successful_videos = 0
    
    for i, video_name in enumerate(video_names, 1):
        print(f"\n[{i}/{len(video_names)}] 🎬 Đang xử lý: {video_name}")
        print("-" * 60)
        
        try:
            result = filter_transition_frames_for_video(video_name, config)
            if result:
                results.append(result)
                successful_videos += 1
            else:
                print(f"❌ Không thể xử lý video: {video_name}")
        except Exception as e:
            print(f"❌ Lỗi khi xử lý video {video_name}: {str(e)}")
    
    # Tổng kết
    if results:
        print("\n" + "=" * 70)
        print("📊 TỔNG KẾT KẾT QUẢ")
        print("=" * 70)
        
        total_original = sum(r['original_frames'] for r in results)
        total_final = sum(r['filtered_frames'] for r in results)
        total_removed = sum(r['total_removed'] for r in results)
        total_similarity_removed = sum(r['similarity_removed'] for r in results)
        avg_removal_rate = np.mean([r['removal_percentage'] for r in results])
        
        print(f"✅ Đã xử lý thành công: {successful_videos}/{len(video_names)} videos")
        print(f"🖼️ Tổng frames gốc: {total_original:,}")
        print(f"✅ Tổng frames cuối cùng: {total_final:,}")
        print(f"🗑️ Tổng frames đã loại bỏ: {total_removed:,}")
        print(f"   🔄 Loại bỏ do similarity: {total_similarity_removed:,}")
        print(f"📊 Tỷ lệ loại bỏ trung bình: {avg_removal_rate:.1f}%")
        
        print(f"\n🎉 HOÀN THÀNH! Kết quả đã được lưu tại:")
        print(f"   📂 {config['output_keyframes_dir']}")
        print(f"   📂 {config['output_map_dir']}")

def create_config(args):
    """Tạo dictionary config từ arguments"""
    return {
        'input_keyframes_dir': args.input_keyframes,
        'input_map_dir': args.input_map,
        'output_keyframes_dir': args.output_keyframes,
        'output_map_dir': args.output_map,
        'transition_threshold': args.threshold,
        'min_scene_length': args.min_scene_length,
        'blur_threshold': args.blur_threshold,
        'enable_blur_detection': args.enable_blur_detection,
        'edge_threshold': args.edge_threshold,
        'enable_edge_detection': args.enable_edge_detection,
        'enable_adaptive_filtering': args.enable_adaptive_filtering,
        'blur_percentile': args.blur_percentile,
        'edge_percentile': args.edge_percentile,
        # NEW: Similarity filtering parameters
        'enable_similarity_filtering': args.enable_similarity_filtering,
        'similarity_threshold': args.similarity_threshold,
        'min_frame_distance': args.min_frame_distance,
        'similarity_window_size': args.similarity_window_size,
        'use_advanced_similarity_filtering': args.use_advanced_similarity_filtering
    }

def main():
    parser = argparse.ArgumentParser(
        description="Lọc bỏ frame chuyển cảnh, frame mờ và frame tương tự với adaptive thresholds"
    )
    parser.add_argument("--input_keyframes", type=str, default=DEFAULT_INPUT_KEYFRAMES_DIR)
    parser.add_argument("--input_map", type=str, default=DEFAULT_INPUT_MAP_DIR)
    parser.add_argument("--output_keyframes", type=str, default=DEFAULT_OUTPUT_KEYFRAMES_DIR)
    parser.add_argument("--output_map", type=str, default=DEFAULT_OUTPUT_MAP_DIR)
    parser.add_argument("--threshold", type=float, default=TRANSITION_THRESHOLD)
    parser.add_argument("--min_scene_length", type=int, default=MIN_SCENE_LENGTH)
    
    # Adaptive filtering
    parser.add_argument("--enable_adaptive_filtering", action="store_true", default=ENABLE_ADAPTIVE_FILTERING)
    parser.add_argument("--disable_adaptive_filtering", action="store_true", default=False)
    parser.add_argument("--blur_percentile", type=float, default=BLUR_PERCENTILE,
                        help="Loại bỏ N% frame mờ nhất (mặc định: 10% - rất mềm)")
    parser.add_argument("--edge_percentile", type=float, default=EDGE_PERCENTILE,
                        help="Loại bỏ N% frame có edge thấp nhất (mặc định: 10% - rất mềm)")
    
    # Fixed thresholds
    parser.add_argument("--blur_threshold", type=float, default=BLUR_THRESHOLD)
    parser.add_argument("--edge_threshold", type=float, default=EDGE_THRESHOLD)
    parser.add_argument("--enable_blur_detection", action="store_true", default=ENABLE_BLUR_DETECTION)
    parser.add_argument("--disable_blur_detection", action="store_true", default=False)
    parser.add_argument("--enable_edge_detection", action="store_true", default=ENABLE_EDGE_DETECTION)
    parser.add_argument("--disable_edge_detection", action="store_true", default=False)
    
    # NEW: Similarity filtering parameters
    parser.add_argument("--enable_similarity_filtering", action="store_true", default=ENABLE_SIMILARITY_FILTERING,
                        help="Bật tính năng lọc frame tương tự nhau")
    parser.add_argument("--disable_similarity_filtering", action="store_true", default=False,
                        help="Tắt tính năng lọc frame tương tự nhau")
    parser.add_argument("--similarity_threshold", type=float, default=SIMILARITY_THRESHOLD,
                        help="Ngưỡng similarity để coi là gần giống nhau (0.0-1.0, mặc định: 0.95)")
    parser.add_argument("--min_frame_distance", type=int, default=MIN_FRAME_DISTANCE,
                        help="Khoảng cách tối thiểu giữa các frame được giữ lại")
    parser.add_argument("--similarity_window_size", type=int, default=SIMILARITY_WINDOW_SIZE,
                        help="Kích thước window để so sánh similarity")
    parser.add_argument("--use_advanced_similarity_filtering", action="store_true", default=False,
                        help="Sử dụng thuật toán similarity filtering nâng cao")
    
    # Gentle filtering options
    parser.add_argument("--gentle", action="store_true", default=False,
                        help="Chế độ lọc nhẹ: chỉ loại bỏ 5% frame tệ nhất")
    parser.add_argument("--ultra_gentle", action="store_true", default=False,
                        help="Chế độ lọc cực nhẹ: chỉ lọc chuyển cảnh, không lọc chất lượng")
    parser.add_argument("--similarity_only", action="store_true", default=False,
                        help="Chỉ lọc frame tương tự, không lọc chất lượng và chuyển cảnh")
    
    parser.add_argument("--video", type=str, default=None)
    
    args = parser.parse_args()
    
    # Xử lý gentle modes
    if args.ultra_gentle:
        args.enable_blur_detection = False
        args.enable_edge_detection = False
        print("🌸 Chế độ Ultra Gentle: Chỉ lọc chuyển cảnh, giữ nguyên tất cả frames chất lượng")
    elif args.gentle:
        args.blur_percentile = 5.0
        args.edge_percentile = 5.0
        print("🌿 Chế độ Gentle: Chỉ loại bỏ 5% frame tệ nhất")
    elif args.similarity_only:
        args.enable_blur_detection = False
        args.enable_edge_detection = False
        args.threshold = 0.0  # Không phát hiện chuyển cảnh
        print("🔄 Chế độ Similarity Only: Chỉ lọc frame tương tự")
    
    # Xử lý flags
    if args.disable_adaptive_filtering:
        args.enable_adaptive_filtering = False
    if args.disable_blur_detection:
        args.enable_blur_detection = False
    if args.disable_edge_detection:
        args.enable_edge_detection = False
    if args.disable_similarity_filtering:
        args.enable_similarity_filtering = False
    
    config = create_config(args)
    
    print("🚀 ADAPTIVE TRANSITION & BLUR FRAME FILTER WITH SIMILARITY DEDUPLICATION")
    print("=" * 80)
    
    if args.video:
        os.makedirs(config['output_keyframes_dir'], exist_ok=True)
        os.makedirs(config['output_map_dir'], exist_ok=True)
        result = filter_transition_frames_for_video(args.video, config)
        if result:
            print(f"\n✅ Hoàn thành video '{args.video}'!")
    else:
        process_all_videos(config)

if __name__ == "__main__":
    main()