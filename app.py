import gradio as gr
import torch
from PIL import Image
import numpy as np
from distillanydepth.modeling.archs.dam.dam import DepthAnything
from distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
from distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose
import cv2
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# 画像処理関数（カラーマップを引数で受け取る）
def process_image(image, model, device, cmap):
    if model is None:
        return None

    # 画像の前処理
    image = image.convert("RGB")
    image_np = np.array(image)[..., ::-1] / 255

    # Dynamically resize based on the image's short side
    h, w = image_np.shape[:2]
    short_side = min(h, w)
    # Ensure the target size is a multiple of 14 and at least 14
    target_size = max(1, short_side // 14) * 14

    resize_transform = Resize(
        width=target_size,
        height=target_size,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC
    )
    
    normalize_transform = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    prepare_for_net_transform = PrepareForNet()

    # Apply transformations sequentially
    sample = {'image': image_np}
    sample = resize_transform(sample)
    sample = normalize_transform(sample)
    sample = prepare_for_net_transform(sample)
    
    image_tensor = torch.from_numpy(sample['image']).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_disp, _ = model(image_tensor)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 深度マップの整形と正規化
    pred_disp_np = pred_disp.cpu().detach().numpy()[0, 0, :, :]
    pred_disp = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())

    # カラーマップで色付け
    depth_colored = colorize_depth_maps(pred_disp[None, ..., None], 0, 1, cmap=cmap).squeeze()

    # 0-255 の uint8 に変換し、チャネル順を変換
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)

    # 元画像と同じサイズにリサイズ
    h, w = image_np.shape[:2]
    depth_colored_hwc = cv2.resize(depth_colored_hwc, (w, h), cv2.INTER_LINEAR)

    # PIL 画像に変換して返す
    depth_image = Image.fromarray(depth_colored_hwc)
    return depth_image

# モデルのロード処理（シングルトンにすることで複数関数から使い回し可能）
def load_depth_model(device):
    model_kwargs = dict(
        vitb=dict(
            encoder='vitb',
            features=128,
            out_channels=[96, 192, 384, 768],
        ),
        vitl=dict(
            encoder="vitl", 
            features=256, 
            out_channels=[256, 512, 1024, 1024], 
            use_bn=False, 
            use_clstoken=False, 
            max_depth=150.0, 
            mode='disparity',
            pretrain_type='dinov2',
            del_mask_token=False
        )
    )
    model = DepthAnything(**model_kwargs['vitl']).to(device)
    # hf_hub_download を使う場合の例（コメントアウト）
    # checkpoint_path = hf_hub_download(repo_id=f"xingyang1/Distill-Any-Depth", filename=f"large/model.safetensors", repo_type="model")
    
    # ローカルパスの場合（環境に合わせて変更）
    checkpoint_path = "D:\\AI\\Distill-Any-Depth\\model.safetensors"
    model_weights = load_file(checkpoint_path)
    model.load_state_dict(model_weights)
    model = model.to(device)
    return model

# 単一画像処理用のGradioインターフェイス関数
def gradio_interface_single(image, cmap):
    device = get_device()
    model = load_depth_model(device)
    
    # 深度画像の生成
    depth_image = process_image(image, model, device, cmap)
    
    # 入力画像と出力画像の保存処理（ファイル名に現在時刻を利用）
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.join(output_dir, f"{timestamp}.png")
    output_filename = os.path.join(output_dir, f"{timestamp}_depth.png")
    image.save(input_filename)
    depth_image.save(output_filename)
    
    return depth_image

# フォルダ内の画像を一括処理する関数
def gradio_interface_batch(folder_path, cmap):
    device = get_device()
    model = load_depth_model(device)
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    # 対象となる画像拡張子
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if not files:
        return "指定されたフォルダに画像が見つかりませんでした。"
    
    # 各画像を処理
    for file in files:
        try:
            image = Image.open(file)
        except Exception as e:
            print(f"Failed to open {file}: {e}")
            continue
        depth_image = process_image(image, model, device, cmap)
        # 元ファイル名に _depth を付加して保存
        base = os.path.basename(file)
        name, ext = os.path.splitext(base)
        output_filename = os.path.join(output_dir, f"{name}_depth{ext}")
        depth_image.save(output_filename)
    return f"バッチ処理が完了しました。出力は {output_dir} フォルダに保存されました。"

# Gradio インターフェイスのタブ化
with gr.Blocks() as demo:
    gr.Markdown("# Depth Estimation Demo")
    
    with gr.Tab("単一画像処理"):
        gr.Markdown("入力画像とカラーマップを選択して深度推定を行います。")
        with gr.Row():
            input_image = gr.Image(type="pil", label="入力画像")
            cmap_single = gr.Dropdown(choices=["Spectral_r", "gray", "viridis", "plasma", "inferno", "magma"],
                                      value="Spectral_r", label="カラーマップ")
        output_image = gr.Image(type="pil", label="深度推定結果")
        single_btn = gr.Button("処理開始")
        single_btn.click(gradio_interface_single, inputs=[input_image, cmap_single], outputs=output_image)
    
    with gr.Tab("フォルダ内一括処理"):
        gr.Markdown("フォルダパスを入力して、そのフォルダ内のすべての画像を一括で処理します。\n出力画像は各ファイル名に _depth を付加して保存されます。")
        folder_path = gr.Textbox(label="画像フォルダのパス", placeholder="例: D:\\Images")
        cmap_batch = gr.Dropdown(choices=["Spectral_r", "gray", "viridis", "plasma", "inferno", "magma"],
                                 value="Spectral_r", label="カラーマップ")
        batch_btn = gr.Button("バッチ処理開始")
        batch_output = gr.Textbox(label="処理結果メッセージ")
        batch_btn.click(gradio_interface_batch, inputs=[folder_path, cmap_batch], outputs=batch_output)

# Gradio インターフェイスの起動
demo.launch()
