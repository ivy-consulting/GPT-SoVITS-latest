import os
import sys
import traceback
from typing import Generator
import random
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
import wave
import subprocess
import uvicorn
import signal
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config, NO_PROMPT_ERROR
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
from process_ckpt import get_sovits_version_from_path_fast
from config import pretrained_sovits_name, name2gpt_path, name2sovits_path
import torch
import argparse
import re


now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

i18n = I18nAuto()
cut_method_names = get_cut_method_names()

# Language dictionaries from GUI
dict_language_v1 = {
    i18n("中文"): "all_zh",
    i18n("英文"): "en",
    i18n("日文"): "all_ja",
    i18n("中英混合"): "zh",
    i18n("日英混合"): "ja",
    i18n("多语种混合"): "auto",
}
dict_language_v2 = {
    i18n("中文"): "all_zh",
    i18n("英文"): "en",
    i18n("日文"): "all_ja",
    i18n("粤语"): "all_yue",
    i18n("韩文"): "all_ko",
    i18n("中英混合"): "zh",
    i18n("日英混合"): "ja",
    i18n("粤英混合"): "yue",
    i18n("韩英混合"): "ko",
    i18n("多语种混合"): "auto",
    i18n("多语种混合(粤语)"): "auto_yue",
}

# Extended language dictionary from api.py
dict_language = {
    "中文": "all_zh",
    "粤语": "all_yue",
    "英文": "en",
    "日文": "all_ja",
    "韩文": "all_ko",
    "中英混合": "zh",
    "粤英混合": "yue",
    "日英混合": "ja",
    "韩英混合": "ko",
    "多语种混合": "auto",
    "多语种混合(粤语)": "auto_yue",
    "all_zh": "all_zh",
    "all_yue": "all_yue",
    "en": "en",
    "all_ja": "all_ja",
    "all_ko": "all_ko",
    "zh": "zh",
    "yue": "yue",
    "ja": "ja",
    "ko": "ko",
    "auto": "auto",
    "auto_yue": "auto_yue",
}

cut_method = {
    i18n("不切"): "cut0",
    i18n("凑四句一切"): "cut1",
    i18n("凑50字一切"): "cut2",
    i18n("按中文句号。切"): "cut3",
    i18n("按英文句号.切"): "cut4",
    i18n("按标点符号切"): "cut5",
}

# Pretrained model paths
path_sovits_v3 = pretrained_sovits_name["v3"]
path_sovits_v4 = pretrained_sovits_name["v4"]
is_exist_s2gv3 = os.path.exists(path_sovits_v3)
is_exist_s2gv4 = os.path.exists(path_sovits_v4)


# Argument parsing
parser = argparse.ArgumentParser(description="GPT-SoVITS API")
parser.add_argument("-dr", "--default_refer_path", type=str, default="", help="Default reference audio path")
parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="Default reference audio text")
parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="Default reference audio language")
parser.add_argument("-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda / cpu")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=9880, help="default: 9880")
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="Use full precision")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="Use half precision")
parser.add_argument("-sm", "--stream_mode", type=str, default="close", help="Stream mode: close / normal")
parser.add_argument("-mt", "--media_type", type=str, default="wav", help="Audio format: wav / ogg / aac")
parser.add_argument("-st", "--sub_type", type=str, default="int16", help="Audio data type: int16 / int32")
parser.add_argument("-cp", "--cut_punc", type=str, default=",.;?!、，。？！；：…", help="Text segmentation punctuation")
args = parser.parse_args()

# Apply argument configurations
device = args.device
port = args.port
host = args.bind_addr
is_half = not args.full_precision if args.full_precision or args.half_precision else torch.cuda.is_available()
stream_mode = "normal" if args.stream_mode.lower() in ["normal", "n"] else "close"
media_type = args.media_type.lower() if args.media_type.lower() in ["aac", "ogg"] else ("wav" if stream_mode == "close" else "ogg")
is_int32 = args.sub_type.lower() == "int32"
default_cut_punc = args.cut_punc 
# Default reference audio
class DefaultRefer:
    def __init__(self, path, text, language):
        self.path = path
        self.text = text
        self.language = language

    def is_ready(self) -> bool:
        return all(item and item != "" for item in [self.path, self.text, self.language])

default_refer = DefaultRefer(args.default_refer_path, args.default_refer_text, args.default_refer_language)


# Character configurations
character_configs = {
    "kurari": {
        "v1": {
            "gpt_path": "/workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/kurari-e40.ckpt",
            "sovits_path": "/workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/kurari_e20_s1800_l32.pth",
            "ref_audio": "idols/kurari/kurari.wav",
            "ref_text": "おはよう〜。今日はどんな1日過ごすー？くらりはね〜いつでもあなたの味方だよ",
            "ref_language": "日文"
        },
        "v2": {
            "gpt_path": "/workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/kurari-hql-e40.ckpt",
            "sovits_path": "/workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/kurari-hql_e20_s1240.pth",
            "ref_audio": "/workspace/GPT-SoVITS/idols/kurari/kurari.wav",
            "ref_text": "おはよう〜。今日はどんな1日過ごすー？くらりはね〜いつでもあなたの味方だよ",
            "ref_language": "日文"
        },
        "v3": {
            "gpt_path": "/workspace/GPT-SoVITS/GPT_SoVITS/pretrained_modelskurari-high-e45.ckpt",
            "sovits_path": "GPT_SoVITS/pretrained_models/kurari-high_e25_s325.pth",
            "ref_audio": "idols/kurari/kurari.wav",
            "ref_text": "おはよう〜。今日はどんな1日過ごすー？くらりはね〜いつでもあなたの味方だよ",
            "ref_language": "日文"
        }
    },
    "saotome": {
        "v1": {
            "gpt_path": "GPT_SoVITS/pretrained_models/saotome-e30.ckpt",
            "sovits_path": "GPT_SoVITS/pretrained_models/saotome_e9_s522_l32.pth",
            "ref_audio": "idols/saotome/saotome.wav",
            "ref_text": "朝ごはんにはトーストと卵、そしてコーヒーを飲みました。簡単だけど、朝の時間が少し幸せに感じられる瞬間でした。",
            "ref_language": "日文"
        }
    },
    "baacharu": {
        "v1": {
            "gpt_path": "/workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/baacharu-e40.ckpt",
            "sovits_path": "/workspace/GPT-SoVITS/GPT_SoVITS/pretrained_models/baacharu_e15_s1320_l32.pth",
            "ref_audio": "/workspace/GPT-SoVITS/idols/baacharu/baacharu.wav",
            "ref_text": "どーもー、世界初男性バーチャルユーチューバーのばあちゃるです",
            "ref_language": "日文"
        }
    },
    "ikka": {
        "v1": {
            "gpt_path": "GPT_SoVITS/pretrained_models/ikko-san-e45.ckpt",
            "sovits_path": "GPT_SoVITS/pretrained_models/s2Gv2ProPlus.pth",
            "ref_audio": "idols/ikka/ikko.wav",
            "ref_text": "せおいなげ、まじばな、らぶらぶ、あげあげ、まぼろし",
            "ref_language": "日文",
            "ref_audio_boost": "idols/ikka/ikko_boost.wav"
        }
    }
}

APP = FastAPI()

class TTS_Request(BaseModel):
    text: str = None
    text_language: str = None
    refer_wav_path: str = None
    prompt_text: str = None
    prompt_language: str = None
    cut_punc: str = None
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    speed: float = 1.0
    sample_steps: int = 32
    if_sr: bool = False
    character: str = "kurari"
    version: str = "v1"
    loudness_boost: bool = False
    gain: float = 0.0
    normalize: bool = False
    energy_scale: float = 1.0
    volume_scale: float = 1.0
    strain_effect: float = 0.0

def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer

def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer

def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    subtype = 'PCM_32' if is_int32 else 'PCM_16'
    sf.write(io_buffer, data, rate, format='wav', subtype=subtype)
    return io_buffer

def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    pcm = 's32le' if is_int32 else 's16le'
    bit_rate = '256k' if is_int32 else '128k'
    process = subprocess.Popen([
        'ffmpeg',
        '-f', pcm,
        '-ar', str(rate),
        '-ac', '1',
        '-i', 'pipe:0',
        '-c:a', 'aac',
        '-b:a', bit_rate,
        '-vn',
        '-f', 'adts',
        'pipe:1'
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer

def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()

def read_clean_buffer(audio_bytes):
    audio_chunk = audio_bytes.getvalue()
    audio_bytes.truncate(0)
    audio_bytes.seek(0)
    return audio_bytes, audio_chunk

def cut_text(text, punc):
    punc_list = [p for p in punc if p in {",", ".", ";", "?", "!", "、", "，", "。", "？", "！", "；", "：", "…"}]
    if len(punc_list) > 0:
        punds = r"[" + "".join(punc_list) + r"]"
        text = text.strip("\n")
        items = re.split(f"({punds})", text)
        mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
        if len(items) % 2 == 1:
            mergeitems.append(items[-1])
        text = "\n".join(mergeitems)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    return text

def only_punc(text):
    return not any(t.isalnum() or t.isalpha() for t in text)

def get_tts_wav(
    text,
    text_lang,
    ref_audio_path,
    aux_ref_audio_paths,
    prompt_text,
    prompt_lang,
    top_k,
    top_p,
    temperature,
    text_split_method,
    batch_size,
    speed_factor,
    ref_text_free,
    split_bucket,
    fragment_interval,
    seed,
    keep_random,
    parallel_infer,
    repetition_penalty,
    sample_steps,
    super_sampling,
    gpt_weights,
    sovits_weights,
    loudness_boost=False,
    gain=0.0,
    normalize=False,
    energy_scale=1.0,
    volume_scale=1.0,
    strain_effect=0.0
):
    seed = -1 if keep_random else seed
    actual_seed = seed if seed not in [-1, "", None] else random.randint(0, 2**32 - 1)
    
    # Initialize TTS configuration
    tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
    tts_config.device = device
    tts_config.is_half = is_half
    
    # Determine model version from SoVITS weights
    version, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_weights)
    tts_config.version = version
    
    # Validate LoRA weights
    is_exist = is_exist_s2gv3 if model_version == "v3" else is_exist_s2gv4
    path_sovits = path_sovits_v3 if model_version == "v3" else path_sovits_v4
    if if_lora_v3 and not is_exist:
        raise FileExistsError(path_sovits + "SoVITS %s" % model_version + i18n("底模缺失，无法加载相应 LoRA 权重"))
    
    # Map weights with special characters
    if "！" in gpt_weights or "!" in gpt_weights:
        gpt_weights = name2gpt_path.get(gpt_weights, gpt_weights)
    if "！" in sovits_weights or "!" in sovits_weights:
        sovits_weights = name2sovits_path.get(sovits_weights, sovits_weights)
    
    # Set weights paths
    tts_config.t2s_weights_path = gpt_weights
    tts_config.vits_weights_path = sovits_weights
    
    # Initialize TTS pipeline
    tts_pipeline = TTS(tts_config)
    
    # Prepare input dictionary
    dict_language_map = dict_language_v1 if version == "v1" else dict_language_v2
    inputs = {
        "text": text,
        "text_lang": dict_language_map.get(text_lang, text_lang),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths if aux_ref_audio_paths is not None else [],
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language_map.get(prompt_lang, prompt_lang),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method.get(text_split_method, text_split_method),
        "batch_size": int(batch_size),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "return_fragment": stream_mode == "normal",
        "fragment_interval": fragment_interval,
        "seed": actual_seed,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
        "loudness_boost": loudness_boost,
        "gain": gain,
        "normalize": normalize,
        "energy_scale": energy_scale,
        "volume_scale": volume_scale,
        "strain_effect": strain_effect
    }
    
    # Apply audio effects if specified
    if loudness_boost or gain != 0 or normalize or energy_scale != 1.0 or volume_scale != 1.0 or strain_effect != 0.0:
        inputs["postprocess"] = True
    
    try:
        for item in tts_pipeline.run(inputs):
            yield item, actual_seed
    except NO_PROMPT_ERROR:
        raise ValueError(i18n("V3不支持无参考文本模式，请填写参考文本！"))

# def check_params(req: dict):
#     text: str = req.get("text") or "default text"  # Fallback if None or empty
#     text_language: str = req.get("text_language") or "auto"
#     refer_wav_path: str = req.get("refer_wav_path") or default_refer.path
#     prompt_text: str = req.get("prompt_text") or default_refer.text
#     prompt_language: str = req.get("prompt_language") or default_refer.language
#     cut_punc: str = req.get("cut_punc") or default_cut_punc
#     sample_steps: int = req.get("sample_steps", 32)
#     super_sampling: bool = req.get("super_sampling", False)
#     character: str = req.get("character", "kurari").lower()
#     version: str = req.get("version", "v1")

#     if not text:
#         return JSONResponse(status_code=400, content={"message": "text cannot be empty"})
#     if not text_language:
#         return JSONResponse(status_code=400, content={"message": "text_language cannot be empty"})
#     if text_language not in dict_language:
#         return JSONResponse(status_code=400, content={"message": f"text_language: {text_language} is not supported"})
#     if not prompt_text and not default_refer.is_ready():
#         return JSONResponse(status_code=400, content={"message": "prompt_text is required when default reference is not set"})
#     if not prompt_language and not default_refer.is_ready():
#         return JSONResponse(status_code=400, content={"message": "prompt_language is required when default reference is not set"})
#     if prompt_language not in dict_language:
#         return JSONResponse(status_code=400, content={"message": f"prompt_language: {prompt_language} is not supported"})
#     if sample_steps not in [4, 8, 16, 32, 64, 128]:
#         return JSONResponse(status_code=400, content={"message": f"sample_steps: {sample_steps} is not supported"})
#     if character not in character_configs:
#         return JSONResponse(status_code=400, content={"message": f"character: {character} is not supported"})
#     if version not in character_configs[character]:
#         return JSONResponse(status_code=400, content={"message": f"version: {version} is not supported for character {character}"})

#     # Validate super_sampling for v3
#     gpt_weights = character_configs[character][version]["gpt_path"]
#     sovits_weights = character_configs[character][version]["sovits_path"]
#     _, model_version, if_lora_v3 = get_sovits_version_from_path_fast(sovits_weights)
#     if super_sampling and model_version != "v3":
#         return JSONResponse(status_code=400, content={"message": "super_sampling is only supported in v3"})

#     return None

async def tts_handle(req: dict):
    # check_res = check_params(req)
    # if check_res is not None:
    #     return check_res

    character = req.get("character", "kurari").lower()
    version = req.get("version", "v1")
    text = req.get("text")
    cut_punc = req.get("cut_punc")
    if cut_punc is not None:
        text = cut_text(text, cut_punc)
    elif default_cut_punc:
        text = cut_text(text, default_cut_punc)

    # Use default reference if not provided
    refer_wav_path = req.get("refer_wav_path", default_refer.path)
    prompt_text = req.get("prompt_text", default_refer.text)
    prompt_language = req.get("prompt_language", default_refer.language)

    # Character-specific configuration
    config = character_configs[character][version]
    gpt_weights = config["gpt_path"]
    sovits_weights = config["sovits_path"]
    ref_audio_path = config["ref_audio"]
    if character == "ikka" and req.get("loudness_boost", False):
        ref_audio_path = config.get("ref_audio_boost", ref_audio_path)
    prompt_text = req.get("prompt_text", config["ref_text"])
    prompt_language = req.get("prompt_language", config["ref_language"])

    # Map language codes
    text_language = dict_language.get(req.get("text_language"), req.get("text_language"))
    prompt_language = dict_language.get(prompt_language, prompt_language)

    try:
        tts_generator = get_tts_wav(
            text=text,
            text_lang=text_language,
            ref_audio_path=refer_wav_path or ref_audio_path,
            aux_ref_audio_paths=req.get("inp_refs", []),
            prompt_text=prompt_text,
            prompt_lang=prompt_language,
            top_k=req.get("top_k", 15),
            top_p=req.get("top_p", 1.0),
            temperature=req.get("temperature", 1.0),
            text_split_method=req.get("text_split_method", "cut5"),
            batch_size=req.get("batch_size", 1),
            speed_factor=req.get("speed", 1.0),
            ref_text_free=req.get("ref_text_free", False),
            split_bucket=req.get("split_bucket", True),
            fragment_interval=req.get("fragment_interval", 0.3),
            seed=req.get("seed", -1),
            keep_random=req.get("keep_random", True),
            parallel_infer=req.get("parallel_infer", True),
            repetition_penalty=req.get("repetition_penalty", 1.35),
            sample_steps=req.get("sample_steps", 32),
            super_sampling=req.get("if_sr", False),
            gpt_weights=gpt_weights,
            sovits_weights=sovits_weights,
            loudness_boost=req.get("loudness_boost", False),
            gain=req.get("gain", 0.0),
            normalize=req.get("normalize", False),
            energy_scale=req.get("energy_scale", 1.0),
            volume_scale=req.get("volume_scale", 1.0),
            strain_effect=req.get("strain_effect", 0.0)
        )

        if stream_mode == "normal":
            def streaming_generator(tts_generator: Generator):
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type_local = "raw"
                else:
                    media_type_local = media_type
                for (sr, chunk), _ in tts_generator:
                    if is_int32:
                        chunk = (chunk * 2147483647).astype(np.int32)
                    else:
                        chunk = (chunk * 32768).astype(np.int16)
                    audio_bytes = pack_audio(BytesIO(), chunk, sr, media_type_local)
                    audio_bytes, audio_chunk = read_clean_buffer(audio_bytes)
                    yield audio_chunk
            return StreamingResponse(streaming_generator(tts_generator), media_type=f"audio/{media_type}")
        else:
            (sr, audio_data), actual_seed = next(tts_generator)
            if is_int32:
                audio_data = (audio_data * 2147483647).astype(np.int32)
            else:
                audio_data = (audio_data * 32768).astype(np.int16)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"tts failed", "Exception": str(e)})

@APP.post("/change_refer")
async def change_refer(request: Request):
    json_post_raw = await request.json()
    return handle_change(
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("prompt_language")
    )

@APP.get("/change_refer")
async def change_refer(
    refer_wav_path: str = None,
    prompt_text: str = None,
    prompt_language: str = None
):
    return handle_change(refer_wav_path, prompt_text, prompt_language)

def handle_change(path, text, language):
    if not any([path, text, language]):
        return JSONResponse(status_code=400, content={"message": "At least one of refer_wav_path, prompt_text, or prompt_language is required"})
    
    if path:
        default_refer.path = path
    if text:
        default_refer.text = text
    if language:
        default_refer.language = language


    return JSONResponse(status_code=200, content={"code": 0, "message": "Success"})

@APP.post("/control")
async def control(request: Request):
    json_post_raw = await request.json()
    return handle_control(json_post_raw.get("command"))

@APP.get("/control")
async def control(command: str = None):
    return handle_control(command)

def handle_control(command):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    if command == "restart":
        os.execl(sys.executable, sys.executable, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
    return JSONResponse(status_code=400, content={"message": f"Unknown command: {command}"})

@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None,
    text_language: str = None,
    refer_wav_path: str = None,
    prompt_text: str = None,
    prompt_language: str = None,
    cut_punc: str = None,
    top_k: int = 15,
    top_p: float = 1.0,
    temperature: float = 1.0,
    speed: float = 1.0,
    sample_steps: int = 32,
    if_sr: bool = False,
    character: str = "kurari",
    version: str = "v1",
    loudness_boost: bool = False,
    gain: float = 0.0,
    normalize: bool = False,
    energy_scale: float = 1.0,
    volume_scale: float = 1.0,
    strain_effect: float = 0.0,
    text_split_method: str = "cut5",
    batch_size: int = 1,
    fragment_interval: float = 0.3,
    seed: int = -1,
    keep_random: bool = True,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35,
    ref_text_free: bool = False
):
    req = {
        "text": text,
        "text_language": text_language,
        "refer_wav_path": refer_wav_path,
        "prompt_text": prompt_text,
        "prompt_language": prompt_language,
        "cut_punc": cut_punc,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "speed": speed,
        "sample_steps": sample_steps,
        "if_sr": if_sr,
        "character": character,
        "version": version,
        "loudness_boost": loudness_boost,
        "gain": gain,
        "normalize": normalize,
        "energy_scale": energy_scale,
        "volume_scale": volume_scale,
        "strain_effect": strain_effect,
        "text_split_method": text_split_method,
        "batch_size": batch_size,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "keep_random": keep_random,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
        "ref_text_free": ref_text_free,
        "inp_refs": [f"idols/{character}/refs/{file}" for file in os.listdir(f"idols/{character}/refs") if file.endswith('.wav')] if os.path.exists(f"idols/{character}/refs") else []
    }
    return await tts_handle(req)

@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)

if __name__ == "__main__":
    try:
        if host == 'None':
            host = None
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)