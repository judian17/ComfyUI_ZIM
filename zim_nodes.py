import torch
import numpy as np
import json
import os
import time # 导入 time 模块
import hashlib # 导入 hashlib 用于计算哈希
from PIL import Image

# 假设 zim_anything 库已经安装并且可以导入
# 如果 zim_anything 不在标准 Python 路径中，可能需要调整 sys.path
try:
    from zim_anything import zim_model_registry, ZimPredictor
    from zim_anything.utils import ResizeLongestSide # 导入必要的工具
except ImportError:
    print("Warning: zim_anything library not found. Please ensure it is installed.")
    # 提供一个假的类结构，以避免在库缺失时完全崩溃
    class ZimPredictor:
        def __init__(self, model): pass
        def set_image(self, image): pass
        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
            print("Error: ZimPredictor not available.")
            # 返回一个符合预期的空/默认形状的 numpy 数组
            # 这里假设原始图像大小为 512x512，需要根据实际情况调整或从输入获取
            dummy_mask = np.zeros((1, 512, 512), dtype=np.uint8)
            dummy_iou = np.zeros((1,), dtype=np.float32)
            dummy_low_res = np.zeros((1, 256, 256), dtype=np.uint8) # ZIM 低分辨率通常是 256x256
            return dummy_mask, dummy_iou, dummy_low_res

    class ModelRegistryMock:
        def __getitem__(self, key):
            print(f"Warning: Model registry '{key}' accessed, but zim_anything not found.")
            # 返回一个假的 model 对象
            return lambda checkpoint: None

    zim_model_registry = ModelRegistryMock()


# ComfyUI 张量到 PIL/NumPy 图像的转换函数
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2numpy(image):
    return np.array(image).astype(np.uint8)

def numpy2pil(img_np):
    return Image.fromarray(img_np)

# 全局变量来缓存模型和预测器，避免每次都重新加载
loaded_models = {}
loaded_predictors = {}

# --- 新增：全局变量来缓存图像特征 ---
# key: (model_key, image_hash)
# value: (features, interm_feats, original_size, input_size)
cached_image_features = {}

# --- 新增：计算 Tensor 哈希的函数 ---
def calculate_tensor_hash(tensor):
    """计算 PyTorch Tensor 的 SHA256 哈希值"""
    if tensor is None:
        return None
    # 使用第一个 batch item 计算哈希，假设 batch 内图像相同或只关心第一个
    tensor_item = tensor[0].cpu().numpy() 
    return hashlib.sha256(tensor_item.tobytes()).hexdigest()

class ZimSegment:
    """
    使用 ZIM 模型根据点或 BBOX 提示分割图像。
    """
    @classmethod
    def INPUT_TYPES(cls):
        # 获取 ComfyUI 根目录下的 models/zim 路径
        try:
            base_path = os.path.dirname(__file__)
            comfyui_root = os.path.abspath(os.path.join(base_path, "..", ".."))
            zim_models_dir = os.path.join(comfyui_root, "models", "zim")
            
            if os.path.isdir(zim_models_dir):
                available_models = [d for d in os.listdir(zim_models_dir) if os.path.isdir(os.path.join(zim_models_dir, d))]
                # 优先使用常见的模型名称，如果找不到则使用扫描到的目录
                default_models = ["zim_vit_l_2092", "zim_vit_b_2043"]
                model_list = sorted(list(set(default_models + available_models)))
                if not model_list: # 如果扫描后列表仍为空
                     model_list = default_models # 使用默认列表作为后备
            else:
                print(f"[ZIM WARNING] Directory not found: {zim_models_dir}. Using default model list.")
                model_list = ["zim_vit_l_2092", "zim_vit_b_2043"]
        except Exception as e:
            print(f"[ZIM WARNING] Error scanning for models: {e}. Using default model list.")
            model_list = ["zim_vit_l_2092", "zim_vit_b_2043"]


        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (model_list, ), # 使用扫描或默认的模型列表
                "backbone": (["vit_l", "vit_b", "default"], {"default": "vit_l"}), # 保留 backbone 选择，可能与 model_name 关联
            },
            "optional": {
                "positive_points": ("STRING", {"multiline": False, "default": "[]"}),
                "negative_points": ("STRING", {"multiline": False, "default": "[]"}), # JSON 格式: [[x1, y1], [x2, y2]] 或 [{"x":x1, "y":y1}, ...]
                "bbox": ("BBOX",), # 输入格式 (x, y, width, height)
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "segment"
    CATEGORY = "ComfyUI_ZIM" # Changed category

    def segment(self, image, model_name, backbone, positive_points="[]", negative_points="[]", bbox=None):
        global loaded_models, loaded_predictors, cached_image_features # 确保全局变量被引用
        
        t_start = time.time()
        print(f"\n[ZIM DEBUG] Node execution started.") # 添加换行符以便区分日志块

        # --- 输入验证 ---
        t0 = time.time()
        try:
            pos_pts = json.loads(positive_points)
        except json.JSONDecodeError:
            raise ValueError("Positive points input is not valid JSON.")
        try:
            neg_pts = json.loads(negative_points)
        except json.JSONDecodeError:
            raise ValueError("Negative points input is not valid JSON.")

        has_points = (isinstance(pos_pts, list) and len(pos_pts) > 0) or \
                     (isinstance(neg_pts, list) and len(neg_pts) > 0)
        # 移除初步的、过于严格的 has_bbox 检查
        # has_bbox = bbox is not None and len(bbox) == 4
        # if not has_points and not has_bbox:
        #     raise ValueError("At least one positive point, negative point, or a bbox must be provided.")
        # 验证将在 BBOX 解析后进行
        print(f"[ZIM DEBUG] Input validation (points only) took {time.time() - t0:.4f} seconds.")

        # --- 模型路径构建与加载缓存 ---
        t0 = time.time()
        
        # 构建模型绝对路径
        try:
            base_path = os.path.dirname(__file__)
            comfyui_root = os.path.abspath(os.path.join(base_path, "..", ".."))
            model_abs_path = os.path.join(comfyui_root, "models", "zim", model_name)
            print(f"[ZIM DEBUG] Constructed model path: {model_abs_path}")
        except Exception as e:
            raise RuntimeError(f"Error constructing model path: {e}")

        # 使用绝对路径和 backbone 生成缓存 key
        model_key = f"{model_abs_path}_{backbone}" 
        predictor = None # 初始化 predictor

        if model_key not in loaded_predictors:
            print(f"[ZIM DEBUG] Predictor cache miss for {model_key}. Loading model...")
            # 检查模型路径是否存在
            if not os.path.isdir(model_abs_path):
                 raise FileNotFoundError(f"ZIM model directory not found at the standard location: '{model_abs_path}'. Please ensure the model '{model_name}' exists in '{os.path.join(comfyui_root, 'models', 'zim')}'")
            
            # 加载模型
            t_load_start = time.time()
            try:
                # 使用绝对路径加载
                model = zim_model_registry[backbone](checkpoint=model_abs_path) 
                print(f"[ZIM DEBUG] Model loaded from registry using path '{model_abs_path}' in {time.time() - t_load_start:.4f} seconds.")
                
                # 检查模型初始设备 (如果 zim_anything 内部设置了)
                model_device = "N/A"
                try:
                     # 假设 model 对象本身或其子模块有 device 属性
                     if hasattr(model, 'device'):
                         model_device = model.device
                     elif hasattr(model, 'encoder') and hasattr(model.encoder, 'device'): # 尝试访问内部 encoder
                         model_device = model.encoder.device
                     elif hasattr(model, 'decoder') and hasattr(model.decoder, 'device'): # 尝试访问内部 decoder
                         model_device = model.decoder.device
                     print(f"[ZIM DEBUG] Initial model device (best guess): {model_device}")
                except AttributeError:
                     print("[ZIM DEBUG] Could not determine initial model device attribute.")

                # 移动到 GPU (如果可用)
                t_move_start = time.time()
                if torch.cuda.is_available():
                    print("[ZIM DEBUG] CUDA available. Moving model to CUDA...")
                    model.cuda() # 调用 zim_anything 库的 cuda 方法
                    print(f"[ZIM DEBUG] model.cuda() call took {time.time() - t_move_start:.4f} seconds.")
                    # 再次检查设备
                    try:
                         if hasattr(model, 'device'):
                             model_device = model.device
                         elif hasattr(model, 'encoder') and hasattr(model.encoder, 'device'):
                             model_device = model.encoder.device
                         elif hasattr(model, 'decoder') and hasattr(model.decoder, 'device'):
                             model_device = model.decoder.device
                         print(f"[ZIM DEBUG] Model device after .cuda() (best guess): {model_device}")
                    except AttributeError:
                         print("[ZIM DEBUG] Could not determine model device attribute after .cuda().")
                else:
                    print("[ZIM DEBUG] CUDA not available, using CPU for ZIM model.")
                
                # 创建 Predictor
                t_pred_init_start = time.time()
                predictor = ZimPredictor(model)
                loaded_predictors[model_key] = predictor
                print(f"[ZIM DEBUG] ZimPredictor initialized in {time.time() - t_pred_init_start:.4f} seconds.")

            except Exception as e:
                 raise RuntimeError(f"Failed to load ZIM model from {model_abs_path}: {e}")
            print(f"[ZIM DEBUG] Total model loading and setup took {time.time() - t0:.4f} seconds.")
        else:
            print(f"[ZIM DEBUG] Predictor cache hit for {model_key}. Using cached predictor.")
            predictor = loaded_predictors[model_key]
            # 检查缓存模型的设备
            model_device = "N/A"
            try:
                 if hasattr(predictor.model, 'device'):
                     model_device = predictor.model.device
                 elif hasattr(predictor.model, 'encoder') and hasattr(predictor.model.encoder, 'device'):
                     model_device = predictor.model.encoder.device
                 elif hasattr(predictor.model, 'decoder') and hasattr(predictor.model.decoder, 'device'):
                     model_device = predictor.model.decoder.device
                 print(f"[ZIM DEBUG] Cached predictor's model device (best guess): {model_device}")
            except AttributeError:
                 print("[ZIM DEBUG] Could not determine cached predictor's model device attribute.")


        # --- 图像特征缓存与处理 ---
        t0 = time.time()
        print(f"[ZIM DEBUG] Input image tensor shape: {image.shape}, dtype: {image.dtype}, device: {image.device}")
        
        # --- 图像特征缓存与处理 ---
        # (这部分逻辑不变，但 feature_cache_key 现在基于绝对路径了)
        t0 = time.time()
        print(f"[ZIM DEBUG] Input image tensor shape: {image.shape}, dtype: {image.dtype}, device: {image.device}")
        
        # 计算图像哈希值
        image_hash = calculate_tensor_hash(image)
        # 使用新的 model_key (基于绝对路径)
        feature_cache_key = (model_key, image_hash) 
        print(f"[ZIM DEBUG] Calculated image hash: {image_hash}")
        print(f"[ZIM DEBUG] Using feature cache key: {feature_cache_key}")

        # 检查特征缓存
        if feature_cache_key in cached_image_features:
            print(f"[ZIM DEBUG] Image feature cache hit for key: {feature_cache_key}")
            # 从缓存加载特征和尺寸信息
            features, interm_feats, original_size, input_size = cached_image_features[feature_cache_key]
            
            # 手动设置 predictor 状态，跳过 set_image
            predictor.features = features
            predictor.interm_feats = interm_feats
            predictor.original_size = original_size
            predictor.input_size = input_size
            predictor.is_image_set = True # 标记图像已设置
            print("[ZIM DEBUG] Predictor state set from cached features.")
            
            # 确保特征在正确的设备上 (如果模型在 GPU 上)
            # predictor.predict 会处理输入点/框的设备转换，但特征需要匹配模型设备
            model_device = "N/A"
            try:
                 if hasattr(predictor.model, 'device'): model_device = predictor.model.device
                 elif hasattr(predictor.model, 'encoder') and hasattr(predictor.model.encoder, 'device'): model_device = predictor.model.encoder.device
                 elif hasattr(predictor.model, 'decoder') and hasattr(predictor.model.decoder, 'device'): model_device = predictor.model.decoder.device
            except AttributeError: pass

            if model_device != "N/A" and model_device != features.device:
                 print(f"[ZIM DEBUG] Moving cached features from {features.device} to model device {model_device}")
                 predictor.features = features.to(model_device)
                 predictor.interm_feats = [feat.to(model_device) for feat in interm_feats]

        else:
            print(f"[ZIM DEBUG] Image feature cache miss for key: {feature_cache_key}. Proceeding with set_image.")
            # --- 正常图像处理流程 ---
            img_tensor = image[0] # 取 batch 中的第一张图
            t_conv_start = time.time()
            img_np_uint8 = np.clip(255. * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            print(f"[ZIM DEBUG] Image tensor to NumPy conversion took {time.time() - t_conv_start:.4f} seconds. Shape: {img_np_uint8.shape}")
            
            print("[ZIM DEBUG] Calling predictor.set_image...")
            t_setimg_start = time.time()
            model_device_before_set = "N/A"
            try:
                 if hasattr(predictor.model, 'device'): model_device_before_set = predictor.model.device
                 elif hasattr(predictor.model, 'encoder') and hasattr(predictor.model.encoder, 'device'): model_device_before_set = predictor.model.encoder.device
                 elif hasattr(predictor.model, 'decoder') and hasattr(predictor.model.decoder, 'device'): model_device_before_set = predictor.model.decoder.device
                 print(f"[ZIM DEBUG] Device before set_image (best guess): {model_device_before_set}")
            except AttributeError: print("[ZIM DEBUG] Could not determine device before set_image.")

            predictor.set_image(img_np_uint8, image_format="RGB") # ComfyUI 通常是 RGB
            print(f"[ZIM DEBUG] predictor.set_image took {time.time() - t_setimg_start:.4f} seconds.")

            # --- 缓存新计算的特征 ---
            if predictor.is_image_set:
                print(f"[ZIM DEBUG] Caching new image features for key: {feature_cache_key}")
                # 注意：确保特征张量是可缓存的（例如，不在计算图中）
                # predictor.features 和 interm_feats 应该是 no_grad 上下文中计算的，通常没问题
                cached_features_to_store = (
                    predictor.features.clone().detach(), # 克隆并分离计算图
                    [feat.clone().detach() for feat in predictor.interm_feats], # 对列表中的每个张量执行相同操作
                    predictor.original_size,
                    predictor.input_size
                )
                cached_image_features[feature_cache_key] = cached_features_to_store
            else:
                print("[ZIM DEBUG] Warning: predictor.is_image_set is False after set_image call. Features not cached.")

        print(f"[ZIM DEBUG] Total image feature handling took {time.time() - t0:.4f} seconds.")

        # --- 提示处理 ---
        t0 = time.time()
        input_points = []
        input_labels = []

        # 处理点
        def parse_points(points_data):
            parsed = []
            if isinstance(points_data, list):
                for p in points_data:
                    if isinstance(p, list) and len(p) == 2:
                        parsed.append([int(p[0]), int(p[1])])
                    elif isinstance(p, dict) and 'x' in p and 'y' in p:
                        parsed.append([int(p['x']), int(p['y'])])
            return parsed

        pos_coords = parse_points(pos_pts)
        neg_coords = parse_points(neg_pts)

        if pos_coords:
            input_points.extend(pos_coords)
            input_labels.extend([1] * len(pos_coords)) # 1 表示前景点
        if neg_coords:
            input_points.extend(neg_coords)
            input_labels.extend([0] * len(neg_coords)) # 0 表示背景点

        input_points_np = np.array(input_points) if input_points else None
        input_labels_np = np.array(input_labels) if input_labels else None

        # 处理 BBOX (兼容多种格式)
        input_box_np = None
        has_bbox_parsed = False 
        if bbox is not None:
            print(f"[ZIM DEBUG] Received bbox input: {bbox}, type: {type(bbox)}")
            
            # 格式 1: 直接元组 (x, y, width, height)
            if isinstance(bbox, tuple) and len(bbox) == 4 and all(isinstance(n, (int, float)) for n in bbox):
                x, y, w, h = map(int, bbox) # Ensure integers
                input_box_np = np.array([x, y, x + w, y + h]) # Convert xywh to xyxy
                has_bbox_parsed = True
                print(f"[ZIM DEBUG] Interpreted bbox as direct (x,y,w,h) tuple: {bbox} -> xyxy: {input_box_np}")

            # 格式 2: MTB_Bbox 输出格式: tuple containing one tuple ((x, y, width, height),)
            elif isinstance(bbox, tuple) and len(bbox) == 1 and isinstance(bbox[0], tuple) and len(bbox[0]) == 4:
                 inner_bbox = bbox[0]
                 x, y, w, h = map(int, inner_bbox) # Ensure integers
                 input_box_np = np.array([x, y, x + w, y + h]) # Convert xywh to xyxy
                 has_bbox_parsed = True
                 print(f"[ZIM DEBUG] Interpreted bbox as tuple containing (x,y,w,h): {bbox} -> xyxy: {input_box_np}")

            # 格式 3: PointsEditor 输出格式: list containing one tuple [(xmin, ymin, xmax, ymax)]
            elif isinstance(bbox, list) and len(bbox) == 1 and isinstance(bbox[0], tuple) and len(bbox[0]) == 4:
                # 假设列表内的元组是 XYXY 格式
                inner_bbox = bbox[0]
                x_min, y_min, x_max, y_max = map(int, inner_bbox) # Ensure integers
                input_box_np = np.array([x_min, y_min, x_max, y_max]) # Already xyxy
                has_bbox_parsed = True
                print(f"[ZIM DEBUG] Interpreted bbox as list containing xyxy: {bbox} -> xyxy: {input_box_np}")
                
            else:
                print(f"[ZIM WARNING] Received bbox in an unexpected format: {bbox}. Ignoring bbox input.")
        
        print(f"[ZIM DEBUG] Prompt processing (including bbox parsing) took {time.time() - t0:.4f} seconds.")

        # --- 重新进行输入验证 (基于解析后的结果) ---
        if not has_points and not has_bbox_parsed:
             raise ValueError("At least one positive point, negative point, or a *valid* bbox must be provided.")

        # --- 预测 ---
        t0 = time.time()
        print(f"[ZIM DEBUG] Calling predictor.predict with points: {input_points_np is not None}, bbox: {input_box_np is not None}")
        # 检查 predictor 内部模型的设备
        model_device_before_predict = "N/A"
        try:
             if hasattr(predictor.model, 'device'): model_device_before_predict = predictor.model.device
             elif hasattr(predictor.model, 'encoder') and hasattr(predictor.model.encoder, 'device'): model_device_before_predict = predictor.model.encoder.device
             elif hasattr(predictor.model, 'decoder') and hasattr(predictor.model.decoder, 'device'): model_device_before_predict = predictor.model.decoder.device
             print(f"[ZIM DEBUG] Device before prediction (best guess): {model_device_before_predict}")
        except AttributeError: print("[ZIM DEBUG] Could not determine device before prediction.")
        
        # 确保只在解析出有效 bbox 时才传递 box 参数
        masks_np, _, _ = predictor.predict(
            point_coords=input_points_np,
            point_labels=input_labels_np,
            box=input_box_np if has_bbox_parsed else None, # 使用解析后的 bbox
            multimask_output=False, 
        )
        print(f"[ZIM DEBUG] predictor.predict took {time.time() - t0:.4f} seconds.")
        print(f"[ZIM DEBUG] Output mask shape from predict: {masks_np.shape}, dtype: {masks_np.dtype}")

        # --- 输出处理 ---
        t0 = time.time()
        # masks_np 是 (C, H, W) NumPy 数组 (0-1 float)
        # ComfyUI MASK 需要 (batch, H, W) Torch Tensor (0-1 float)
        
        # predictor.predict 返回的是 (C, H, W) numpy 数组，C=1 因为 multimask_output=False
        # 我们需要 (H, W)
        if masks_np.ndim == 3 and masks_np.shape[0] == 1:
             mask_result_np = masks_np.squeeze(0) # 降维 (1, H, W) -> (H, W)
        elif masks_np.ndim == 2: # 如果已经是 (H, W)
             mask_result_np = masks_np
        else:
             raise ValueError(f"Unexpected mask shape from predictor: {masks_np.shape}")

        # 转换为 Torch Tensor 并添加 batch 维度
        t_conv_start = time.time()
        mask_torch = torch.from_numpy(mask_result_np).unsqueeze(0).float()
        print(f"[ZIM DEBUG] NumPy mask to Torch tensor conversion took {time.time() - t_conv_start:.4f} seconds. Shape: {mask_torch.shape}, dtype: {mask_torch.dtype}, device: {mask_torch.device}")
        
        # 最终输出的 Tensor 默认在 CPU 上，ComfyUI 会处理后续移动
        print(f"[ZIM DEBUG] Output processing took {time.time() - t0:.4f} seconds.")
        
        print(f"[ZIM DEBUG] Node execution finished. Total time: {time.time() - t_start:.4f} seconds.\n") # 添加换行符
        return (mask_torch,)


# --- 节点映射 ---
NODE_CLASS_MAPPINGS = {
    "ZimSegment": ZimSegment
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZimSegment": "ZIM Segmenter"
}

print("ZIM Custom Nodes Loaded")
