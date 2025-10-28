# sherpa_asr.py
from pathlib import Path
import sherpa_onnx
import soundfile as sf
from io import BytesIO
from py.get_setting import DEFAULT_ASR_DIR
async def sherpa_recognize(audio_bytes: bytes):
    """
    使用Sherpa ASR引擎进行语音识别
    
    参数:
        audio_bytes: 音频字节数据
        settings: 包含模型路径等配置的字典
        
    返回:
        识别结果文本
    """
    # 从设置中获取模型路径，如果没有则使用默认路径
    model_path = DEFAULT_ASR_DIR
    
    model = f"{model_path}/model.int8.onnx"
    tokens = f"{model_path}/tokens.txt"
    
    # 验证模型文件是否存在
    if not Path(model).is_file() or not Path(tokens).is_file():
        raise ValueError(
            "Sherpa模型文件未找到，请确保配置了正确的sherpa_model_path"
        )
    
    # 创建识别器
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model,
        tokens=tokens,
        use_itn=True,
        debug=False,
    )
    
    # 将字节数据转换为音频数组
    try:
        # 使用BytesIO将字节数据转换为文件对象
        with BytesIO(audio_bytes) as audio_file:
            audio, sample_rate = sf.read(audio_file, dtype="float32", always_2d=True)
            audio = audio[:, 0]  # 只使用第一个声道
            
            # 创建识别流
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio)
            recognizer.decode_stream(stream)
            
            return stream.result.text
    except Exception as e:
        raise RuntimeError(f"Sherpa ASR处理失败: {str(e)}")