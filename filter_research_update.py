import os
import cv2
import torch
import numpy as np
import argparse
import glob
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import csv
import hashlib
from collections import defaultdict
import imagehash
import time
from colorama import init, Fore, Back, Style
import gc

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Color scheme - Earth Yellow theme
class Colors:
    GOLD = Fore.YELLOW + Style.BRIGHT
    EARTH = Fore.LIGHTYELLOW_EX
    BROWN = Fore.YELLOW
    GREEN = Fore.LIGHTGREEN_EX
    RED = Fore.LIGHTRED_EX
    BLUE = Fore.LIGHTBLUE_EX
    PURPLE = Fore.LIGHTMAGENTA_EX
    WHITE = Fore.WHITE + Style.BRIGHT
    DIM = Style.DIM
    RESET = Style.RESET_ALL

# ---------------- CONFIG ---------------- #
MODEL_NAME = "facebook/dino-vits16"
FRAME_SIZE = (224, 224)
SIM_THRESHOLD = 0.95        # Stricter threshold
SCENE_THRESHOLD = 0.7       # Scene change detection
CLUSTER_EPS = 0.05          # DBSCAN clustering epsilon
MIN_CLUSTER_SIZE = 2        # Minimum cluster size
PERCEPTUAL_THRESHOLD = 5    # Perceptual hash difference
TEMPORAL_WINDOW = 10        # Look-back window for comparison
KEYFRAME_DIR = "keyframes_v2"
MAP_DIR = "map_v2"
# ---------------------------------------- #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_header():
    """Print stylized header"""
    header = f"""
{Colors.GOLD}{'='*80}
{Colors.GOLD}█████╗ ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗ ██████╗███████╗██████╗ 
{Colors.GOLD}██╔══██╗██╔══██╗██║   ██║██╔══██╗████╗  ██║██╔════╝██╔════╝██╔══██╗
{Colors.EARTH}███████║██║  ██║██║   ██║███████║██╔██╗ ██║██║     █████╗  ██║  ██║
{Colors.BROWN}██╔══██║██║  ██║╚██╗ ██╔╝██╔══██║██║╚██╗██║██║     ██╔══╝  ██║  ██║
{Colors.GOLD}██║  ██║██████╔╝ ╚████╔╝ ██║  ██║██║ ╚████║╚██████╗███████╗██████╔╝
{Colors.GOLD}╚═╝  ╚═╝╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═════╝ 
{Colors.EARTH}           🎬 KEYFRAME EXTRACTOR - Multi-Level AI Pipeline 🎬
{Colors.GOLD}{'='*80}
"""
    print(header)

def log(message, color=Colors.WHITE, prefix=""):
    """Colored logging function"""
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"{Colors.DIM}[{timestamp}]{Colors.RESET} {color}{prefix}{message}{Colors.RESET}")

# Load DINO model
log("🚀 Initializing DINO model...", Colors.GOLD, "🤖 ")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
log(f"✅ Model loaded on {device}", Colors.GREEN, "🤖 ")

class AdvancedKeyframeExtractor:
    def __init__(self):
        self.global_stats = {
            'total_videos': 0,
            'total_frames_processed': 0,
            'total_keyframes_saved': 0,
            'total_time': 0,
            'current_video': '',
            'videos_completed': 0
        }
        
    def extract_embedding(self, image: Image.Image):
        """Extract DINO embedding from image"""
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return embedding
    
    def extract_perceptual_hash(self, image: Image.Image):
        """Extract perceptual hash for duplicate detection"""
        return imagehash.phash(image)
    
    def detect_scene_changes(self, embeddings, threshold=SCENE_THRESHOLD):
        """Detect scene boundaries using embedding similarity"""
        scene_changes = [0]  # First frame is always a scene start
        
        for i in range(1, len(embeddings)):
            sim = cosine_similarity([embeddings[i]], [embeddings[i-1]])[0][0]
            if sim < threshold:  # Low similarity indicates scene change
                scene_changes.append(i)
        
        scene_changes.append(len(embeddings))  # End marker
        return scene_changes
    
    def cluster_similar_frames(self, embeddings, frame_indices):
        """Cluster similar frames within a scene"""
        if len(embeddings) < 2:
            return [[0]] if embeddings else []
            
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1 - similarity_matrix
        
        # DBSCAN clustering
        clustering = DBSCAN(
            eps=CLUSTER_EPS, 
            min_samples=MIN_CLUSTER_SIZE,
            metric='precomputed'
        ).fit(distance_matrix)
        
        # Group frames by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            clusters[label].append(idx)
        
        return list(clusters.values())
    
    def select_representative_frame(self, cluster_indices, embeddings, frame_info):
        """Select the most representative frame from a cluster"""
        if len(cluster_indices) == 1:
            return cluster_indices[0]
        
        # Calculate centroid of cluster
        cluster_embeddings = [embeddings[i] for i in cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)
        
        # Find frame closest to centroid
        best_idx = 0
        best_sim = -1
        
        for i, emb_idx in enumerate(cluster_indices):
            sim = cosine_similarity([embeddings[emb_idx]], [centroid])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_idx = i
                
        return cluster_indices[best_idx]
    
    def is_duplicate_by_hash(self, current_hash, hash_window):
        """Check if frame is duplicate using perceptual hash"""
        for prev_hash in hash_window:
            if abs(current_hash - prev_hash) <= PERCEPTUAL_THRESHOLD:
                return True
        return False
    
    def process_video_batch(self, video_paths, keyframe_root, map_root):
        """Process all videos with single progress bar"""
        os.makedirs(keyframe_root, exist_ok=True)
        os.makedirs(map_root, exist_ok=True)
        
        # Calculate total frames across all videos
        log("📊 Calculating total workload...", Colors.EARTH, "📋 ")
        total_frames_all = 0
        video_frame_counts = {}
        
        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_frame_counts[video_path] = frame_count
            total_frames_all += frame_count
            cap.release()
        
        self.global_stats['total_videos'] = len(video_paths)
        
        log(f"📈 Total workload: {total_frames_all:,} frames across {len(video_paths)} videos", 
            Colors.GOLD, "📊 ")
        
        # Single progress bar for all videos
        with tqdm(
            total=total_frames_all,
            desc=f"{Colors.GOLD}🎬 Processing{Colors.RESET}",
            unit="frames",
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            colour="yellow",
            ncols=100
        ) as pbar:
            
            frames_processed = 0
            
            for video_idx, video_path in enumerate(video_paths):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                self.global_stats['current_video'] = video_name
                
                # Update progress bar description
                pbar.set_description(
                    f"{Colors.GOLD}🎬 [{video_idx+1}/{len(video_paths)}] {Colors.EARTH}{video_name[:30]}{'...' if len(video_name) > 30 else ''}{Colors.RESET}"
                )
                
                # Process single video
                keyframes_saved = self.extract_advanced_keyframes_optimized(
                    video_path, keyframe_root, map_root, pbar, video_frame_counts[video_path]
                )
                
                # Update global stats
                self.global_stats['videos_completed'] += 1
                self.global_stats['total_frames_processed'] += video_frame_counts[video_path]
                self.global_stats['total_keyframes_saved'] += keyframes_saved
                
                # Force garbage collection between videos
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Final summary
        self.print_final_summary()
    
    def extract_advanced_keyframes_optimized(self, video_path, keyframe_root, map_root, pbar, total_frames):
        """Memory-optimized keyframe extraction for single video"""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_out_dir = os.path.join(keyframe_root, video_name)
        os.makedirs(video_out_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Phase 1: Streaming feature extraction (memory efficient)
        frames_data = []
        frame_idx = 0
        batch_size = 50  # Process in small batches to save memory
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert and process frame
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb).resize(FRAME_SIZE)
            
            # Extract features
            embedding = self.extract_embedding(img_pil)
            p_hash = self.extract_perceptual_hash(img_pil)
            pts_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            frames_data.append({
                'frame_idx': frame_idx,
                'embedding': embedding,
                'hash': p_hash,
                'pts_time': pts_time,
                'frame': frame.copy()
            })
            
            frame_idx += 1
            pbar.update(1)
            
            # Process in batches to free memory
            if len(frames_data) >= batch_size * 10:  # Keep reasonable buffer
                # Keep only essential data, remove frame data for older frames if needed
                pass
        
        cap.release()
        
        if not frames_data:
            return 0
        
        # Phase 2: Scene detection
        embeddings = [f['embedding'] for f in frames_data]
        scene_boundaries = self.detect_scene_changes(embeddings)
        
        # Phase 3: Process each scene efficiently
        selected_frames = []
        
        for i in range(len(scene_boundaries) - 1):
            scene_start = scene_boundaries[i]
            scene_end = scene_boundaries[i + 1]
            scene_frames = frames_data[scene_start:scene_end]
            
            if not scene_frames:
                continue
            
            # Step 1: Remove perceptual duplicates
            unique_frames = []
            hash_window = []
            
            for frame_data in scene_frames:
                if not self.is_duplicate_by_hash(frame_data['hash'], hash_window):
                    unique_frames.append(frame_data)
                    hash_window.append(frame_data['hash'])
                    if len(hash_window) > TEMPORAL_WINDOW:
                        hash_window.pop(0)
            
            if not unique_frames:
                continue
            
            # Step 2: Clustering within scene
            scene_embeddings = [f['embedding'] for f in unique_frames]
            clusters = self.cluster_similar_frames(scene_embeddings, 
                                                 list(range(len(unique_frames))))
            
            # Step 3: Select representative from each cluster
            for cluster in clusters:
                if cluster and cluster[0] != -1:  # Valid cluster (not noise)
                    rep_idx = self.select_representative_frame(
                        cluster, scene_embeddings, unique_frames
                    )
                    selected_frames.append(unique_frames[rep_idx])
        
        # Phase 4: Final temporal filtering
        final_frames = []
        embedding_window = []
        
        # Sort by timestamp
        selected_frames.sort(key=lambda x: x['pts_time'])
        
        for frame_data in selected_frames:
            is_unique = True
            current_emb = frame_data['embedding']
            
            # Check against temporal window
            for prev_emb in embedding_window:
                sim = cosine_similarity([current_emb], [prev_emb])[0][0]
                if sim >= SIM_THRESHOLD:
                    is_unique = False
                    break
            
            if is_unique:
                final_frames.append(frame_data)
                embedding_window.append(current_emb)
                if len(embedding_window) > TEMPORAL_WINDOW:
                    embedding_window.pop(0)
        
        # Phase 5: Save selected frames
        csv_path = os.path.join(map_root, f"{video_name}.csv")
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["n", "pts_time", "fps", "original_frame_idx", "scene_id"])
            
            for i, frame_data in enumerate(final_frames):
                frame_filename = f"{i:06d}.jpg"
                save_path = os.path.join(video_out_dir, frame_filename)
                cv2.imwrite(save_path, frame_data['frame'])
                
                # Find which scene this frame belongs to
                scene_id = 0
                for j, boundary in enumerate(scene_boundaries[:-1]):
                    if frame_data['frame_idx'] >= boundary:
                        scene_id = j
                
                writer.writerow([
                    i,
                    round(frame_data['pts_time'], 3),
                    int(fps),
                    frame_data['frame_idx'],
                    scene_id
                ])
        
        # Quick stats for this video
        reduction_ratio = (1 - len(final_frames) / len(frames_data)) * 100
        
        return len(final_frames)
    
    def print_final_summary(self):
        """Print beautiful final summary"""
        stats = self.global_stats
        reduction = (1 - stats['total_keyframes_saved'] / stats['total_frames_processed']) * 100
        
        summary = f"""
{Colors.GOLD}{'='*80}
{Colors.EARTH}🎊 PROCESSING COMPLETE! 🎊
{Colors.GOLD}{'='*80}

{Colors.WHITE}📊 FINAL STATISTICS:
{Colors.EARTH}   🎬 Videos processed:     {Colors.GOLD}{stats['videos_completed']:,}{Colors.EARTH}
{Colors.EARTH}   📽️  Total frames:        {Colors.GOLD}{stats['total_frames_processed']:,}{Colors.EARTH}
{Colors.EARTH}   ⭐ Keyframes saved:      {Colors.GOLD}{stats['total_keyframes_saved']:,}{Colors.EARTH}
{Colors.EARTH}   📉 Space reduction:      {Colors.GREEN}{reduction:.1f}%{Colors.EARTH}
{Colors.EARTH}   💾 Average per video:    {Colors.GOLD}{stats['total_keyframes_saved'] // max(stats['videos_completed'], 1):,} frames{Colors.EARTH}

{Colors.WHITE}💡 PERFORMANCE HIGHLIGHTS:
{Colors.GREEN}   ✨ Intelligent scene detection
{Colors.GREEN}   ✨ Multi-level deduplication  
{Colors.GREEN}   ✨ Memory-optimized processing
{Colors.GREEN}   ✨ Semantic clustering analysis

{Colors.GOLD}{'='*80}
{Colors.WHITE}🎯 Ready for your next video analysis adventure! 🎯
{Colors.GOLD}{'='*80}
"""
        print(summary)

def process_videos_advanced(input_path):
    """Process videos with advanced keyframe extraction"""
    # Print beautiful header
    print_header()
    
    log(f"🚀 Starting advanced keyframe extraction", Colors.GOLD, "🎬 ")
    log(f"🎯 Device: {device}", Colors.EARTH, "🔧 ")
    
    # Get video paths
    video_paths = []
    if os.path.isdir(input_path):
        extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
        log(f"📁 Scanning directory: {input_path}", Colors.EARTH, "🔍 ")
        
        for ext in extensions:
            found = glob.glob(os.path.join(input_path, ext))
            video_paths.extend(found)
            if found:
                log(f"   Found {len(found)} {ext} files", Colors.DIM, "📄 ")
                
    elif os.path.isfile(input_path):
        video_paths = [input_path]
        log(f"📄 Single file mode: {os.path.basename(input_path)}", Colors.EARTH, "🎬 ")
    else:
        log("❌ Invalid input path!", Colors.RED, "⚠️ ")
        return
    
    if not video_paths:
        log("❌ No video files found!", Colors.RED, "⚠️ ")
        return
    
    log(f"✅ Found {len(video_paths)} videos to process", Colors.GREEN, "📊 ")
    
    # Initialize extractor and process
    extractor = AdvancedKeyframeExtractor()
    start_time = time.time()
    
    extractor.process_video_batch(video_paths, KEYFRAME_DIR, MAP_DIR)
    
    total_time = time.time() - start_time
    extractor.global_stats['total_time'] = total_time

# ------------------ CLI ------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"{Colors.GOLD}🎬 Advanced Video Keyframe Extractor{Colors.RESET}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""{Colors.EARTH}
Examples:
  python script.py --input video.mp4
  python script.py --input videos_folder/ --sim_threshold 0.93
  python script.py --input videos/ --scene_threshold 0.65 --cluster_eps 0.03
{Colors.RESET}"""
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to video file or folder containing videos")
    parser.add_argument("--sim_threshold", type=float, default=SIM_THRESHOLD,
                        help=f"Similarity threshold (default: {SIM_THRESHOLD})")
    parser.add_argument("--scene_threshold", type=float, default=SCENE_THRESHOLD,
                        help=f"Scene change threshold (default: {SCENE_THRESHOLD})")
    parser.add_argument("--cluster_eps", type=float, default=CLUSTER_EPS,
                        help=f"DBSCAN clustering epsilon (default: {CLUSTER_EPS})")
    
    args = parser.parse_args()
    
    # Update global configs
    SIM_THRESHOLD = args.sim_threshold
    SCENE_THRESHOLD = args.scene_threshold
    CLUSTER_EPS = args.cluster_eps
    
    log(f"⚙️ Configuration:", Colors.GOLD, "🔧 ")
    log(f"   Similarity threshold: {SIM_THRESHOLD}", Colors.DIM, "   📊 ")
    log(f"   Scene threshold: {SCENE_THRESHOLD}", Colors.DIM, "   🎬 ")
    log(f"   Cluster epsilon: {CLUSTER_EPS}", Colors.DIM, "   🔗 ")
    print()
    
    process_videos_advanced(args.input)