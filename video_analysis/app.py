import os
import json
import tempfile
import warnings
import os
import re
import subprocess
import shutil
from flask import Flask, render_template, request, jsonify, send_file, url_for, redirect, session
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip, concatenate_videoclips
import sys
sys.path.append("../")
import numpy as np
from PIL import Image
import uuid

# 尝试导入模型相关库，如果失败则使用模拟模式
try:
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from omni_utils import process_mm_info  # Assuming this is a custom utility
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Model libraries not available. Running in simulation mode.")

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['CONVERSATIONS_FOLDER'] = 'conversations'
app.config['HIGHLIGHTS_FOLDER'] = 'static/highlights'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 限制上传大小为500MB
app.config['SECRET_KEY'] = 'super_secret_key'  # 用于session
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CONVERSATIONS_FOLDER'], exist_ok=True)
os.makedirs(app.config['HIGHLIGHTS_FOLDER'], exist_ok=True)

# 设置静态文件缓存过期时间为0，避免缓存问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# 支持的视频格式
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# 集锦生成配置
SEGMENT_DURATION = 10  # 每个切片的时长（秒）
TOP_N_SEGMENTS = 3     # 选择评分最高的前N个切片
USE_AUDIO_IN_VIDEO = True  # 是否在视频分析中使用音频

def check_ffmpeg_available():
    """检查ffmpeg是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False

def check_moviepy_dependencies():
    """检查MoviePy的依赖是否完整"""
    try:
        from moviepy.editor import VideoFileClip
        # 尝试创建一个简单的测试
        return True
    except ImportError as e:
        print(f"MoviePy导入失败: {e}")
        return False
    except Exception as e:
        print(f"MoviePy依赖检查失败: {e}")
        return False

# --- Model Initialization ---
# Load Qwen2.5-Omni model and processor
if MODEL_AVAILABLE:
    try:
        print("Loading Qwen2.5-Omni model and processor...")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "/data2/dyl/LLMCheckpoint/Qwen2.5-Omni-3B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_safetensors=True,
            attn_implementation="flash_attention_2",
        )
        model.disable_talker()  # As per your original code

        processor = Qwen2_5OmniProcessor.from_pretrained("/data2/dyl/LLMCheckpoint/Qwen2.5-Omni-3B")
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        processor = None
else:
    model = None
    processor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_duration(video_path):
    """获取视频时长"""
    try:
        with VideoFileClip(video_path) as clip:
            return clip.duration
    except Exception as e:
        print(f"获取视频时长失败: {e}")
        return 0

def split_video_into_segments(video_path, segment_duration=SEGMENT_DURATION):
    """将视频分割成指定时长的片段"""
    try:
        duration = get_video_duration(video_path)
        if duration <= 0:
            return []
        
        segments = []
        current_time = 0
        
        while current_time < duration:
            end_time = min(current_time + segment_duration, duration)
            segments.append((current_time, end_time))
            current_time = end_time
        
        return segments
    except Exception as e:
        print(f"分割视频失败: {e}")
        return []

def score_video_segment(video_path, start_time, end_time):
    """使用Qwen模型为视频片段打分"""
    if not MODEL_AVAILABLE or model is None or processor is None:
        # 模拟模式：返回随机分数
        import random
        return random.randint(1, 10)
    
    try:
        # 创建临时视频片段
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            with VideoFileClip(video_path) as clip:
                segment_clip = clip.subclip(start_time, end_time)
                segment_clip.write_videofile(temp_path, verbose=False, logger=None)
                segment_clip.close()
        except Exception as e:
            print(f"创建临时片段失败: {e}")
            return 0
        
        # 使用模型分析片段
        segment_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": temp_path},
                    {"type": "text", "text": "请为这个视频片段的精彩程度打分，分数范围1-10分，10分最精彩。只需要回复：评分：X分"}
                ],
            },
        ]
        
        from qwen_omni_utils import process_mm_info
        text = processor.apply_chat_template(segment_conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(segment_conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=USE_AUDIO_IN_VIDEO
        )
        
        inputs = inputs.to(model.device).to(model.dtype)
        
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=USE_AUDIO_IN_VIDEO,
            return_audio=False,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1
        )
        
        prompt_token_len = inputs['input_ids'].shape[-1]
        new_tokens = text_ids[0, prompt_token_len:]
        response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        print(f"🤖 Qwen 对片段 ({start_time:.1f}s-{end_time:.1f}s) 的回复: {response}")
        
        # 清理临时文件
        try:
            os.unlink(temp_path)
        except:
            pass
        
        # 提取分数
        match = re.search(r'[评分P]分[:：]\s*(\d+)', response)
        if match:
            try:
                score = int(match.group(1))
                return score
            except ValueError:
                return 0
        return 0
        
    except Exception as e:
        print(f"对片段 ({start_time:.1f}s-{end_time:.1f}s) 打分时发生错误: {e}")
        return 0

def create_highlight_video(original_video_path, scored_segments, output_filename_base="highlight"):
    """根据评分创建集锦视频"""
    # 检查依赖
    if not check_ffmpeg_available():
        print("❌ 错误：未找到ffmpeg，无法进行视频编码")
        print("💡 解决方案：")
        print("   1. 下载并安装ffmpeg: https://ffmpeg.org/download.html")
        print("   2. 确保ffmpeg已添加到系统PATH环境变量中")
        print("   3. 重启应用程序")
        return None
    
    if not check_moviepy_dependencies():
        print("❌ 错误：MoviePy依赖不完整")
        print("💡 解决方案：pip install moviepy")
        return None
    
    if not scored_segments:
        print("没有可评分的切片，无法创建集锦。")
        return None
    
    # 按照分数降序排列，并取前N个
    scored_segments.sort(key=lambda x: x[1], reverse=True)
    selected_segments_info = [s[0] for s in scored_segments[:TOP_N_SEGMENTS]]
    
    # 按照原始时间顺序重新排序，以保证集锦的逻辑性
    selected_segments_info.sort(key=lambda x: x[0])  # x[0] 是开始时间
    
    if not selected_segments_info:
        print("没有足够的切片来创建集锦。")
        return None
    
    try:
        full_video_clip = VideoFileClip(original_video_path)
    except Exception as e:
        print(f"加载原始视频 {original_video_path} 失败: {e}")
        return None
    
    adjusted_clips = []
    last_segment_end_time = -1
    
    for start_time, end_time in selected_segments_info:
        current_segment_start_time = max(start_time, last_segment_end_time)
        
        if current_segment_start_time >= end_time:
            continue
        
        try:
            clip_to_add = full_video_clip.subclip(current_segment_start_time, end_time)
            adjusted_clips.append(clip_to_add)
            last_segment_end_time = end_time
        except Exception as e:
            print(f"截取视频片段 ({current_segment_start_time:.2f}s - {end_time:.2f}s) 失败: {e}")
            continue
    
    if not adjusted_clips:
        print("没有可用于合并的调整后切片。")
        full_video_clip.close()
        return None
    
    output_filename = os.path.join(app.config['HIGHLIGHTS_FOLDER'], f"{output_filename_base}_{uuid.uuid4().hex}.mp4")
    
    try:
        print(f"📹 开始合并 {len(adjusted_clips)} 个视频片段...")
        final_clip = concatenate_videoclips(adjusted_clips)
        
        print(f"💾 开始写入集锦视频到: {output_filename}")
        print("⚠️ 注意：视频编码可能需要较长时间，请耐心等待...")
        
        # 添加更详细的参数和错误处理
        final_clip.write_videofile(
            output_filename,
            codec="libx264",
            audio_codec="aac",
            verbose=True,  # 改为True以显示进度
            logger='bar',  # 显示进度条
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            fps=24,  # 设置固定帧率
            preset='medium'  # 编码预设
        )
        
        print(f"✅ 集锦视频写入完成: {output_filename}")
        
        final_clip.close()
        full_video_clip.close()
        
        # 清理临时片段
        for clip in adjusted_clips:
            clip.close()
        
        print(f"🎉 集锦视频创建成功，文件大小: {os.path.getsize(output_filename) / (1024*1024):.2f} MB")
        return output_filename
        
    except Exception as e:
        print(f"❌ 合并视频失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        
        # 清理资源
        try:
            full_video_clip.close()
        except:
            pass
        
        for clip in adjusted_clips:
            try:
                clip.close()
            except:
                pass
        
        # 如果输出文件已部分创建，尝试删除
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
                print(f"🗑️ 已清理部分创建的文件: {output_filename}")
            except:
                pass
        
        return None

def generate_response(conversation_history, use_audio_in_video=True):
    """Generates a text response from the Qwen-Omni model or simulation."""
    if not MODEL_AVAILABLE or model is None or processor is None:
        # 模拟模式：返回基于关键词的简单回复
        last_message = ""
        for msg in reversed(conversation_history):
            if msg.get('role') == 'user':
                for content in msg.get('content', []):
                    if content.get('type') == 'text':
                        last_message = content.get('text', '')
                        break
                break
        
        # 简单的关键词匹配回复
        if '摘要' in last_message or '总结' in last_message:
            return "这是一个关于AI技术的视频，主要介绍了机器学习、深度学习等概念，以及它们在图像识别和自然语言处理中的应用。"
        elif '思维导图' in last_message or 'mindmap' in last_message.lower():
            return "# 视频内容思维导图\n\n## AI技术介绍\n- 机器学习基础\n- 深度学习概念\n\n## 应用场景\n- 图像识别\n- 自然语言处理"
        else:
            return f"基于您的问题：{last_message}，这个视频主要讨论了AI相关技术。如需更详细的分析，请确保模型库已正确安装。"

    try:
        text = processor.apply_chat_template(conversation_history, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation_history, use_audio_in_video=use_audio_in_video)
        
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video
        )
        inputs = inputs.to(model.device).to(model.dtype)

        text_ids = model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            return_audio=False,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1
        )
        prompt_token_len = inputs['input_ids'].shape[-1]

        new_tokens = text_ids[0, prompt_token_len:]  # 截取新生成的部分
        response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response
    except Exception as e:
        print(f"Error during model generation: {e}")
        return "An error occurred during response generation."

def get_conversation_file_path(session_id):
    return os.path.join(app.config['CONVERSATIONS_FOLDER'], f"{session_id}.json")

def load_conversation(session_id):
    file_path = get_conversation_file_path(session_id)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_conversation(session_id, conversation):
    file_path = get_conversation_file_path(session_id)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=4)

def process_video_with_qwen(video_path):
    """Process video using Qwen2.5-Omni model"""
    if not MODEL_AVAILABLE or model is None or processor is None:
        # 模拟模式：返回示例数据
        demo_summary = """# 📹 视频内容摘要

## 🎯 概述
这是一个关于**AI技术**的教育视频。视频内容深入介绍了机器学习和深度学习的核心概念，涵盖了从基础理论到实际应用的完整知识体系。

## 📊 视频统计
- **时长**: 35分钟
- **主要概念**: 6个
- **应用案例**: 3个
- **技术深度**: 中级

## 🔑 核心要点

### 1. 机器学习基础
- **监督学习**: 使用标记数据进行训练的学习方法
- **无监督学习**: 从未标记数据中发现模式的技术
- **强化学习**: 通过与环境交互学习最优策略

### 2. 深度学习技术
- **神经网络**: 模拟人脑神经元连接的计算模型
- **卷积神经网络**: 专门用于图像处理的深度学习架构
- **循环神经网络**: 处理序列数据的神经网络结构

### 3. 实际应用领域
- **图像识别**: 计算机视觉中的核心应用
- **自然语言处理**: 理解和生成人类语言的技术
- **语音识别**: 将语音信号转换为文本的技术

## 💡 重点提示
> 🚀 **关键洞察**: 视频强调了理论与实践相结合的重要性，展示了AI技术在各个领域的广泛应用前景。

## 🏷️ 内容标签
`🤖 机器学习` `🧠 深度学习` `🔬 技术教育` `💼 实际应用`"""
        
        return {
            "summary": demo_summary,
            "mindmap": {
                "root": {
                    "text": "AI技术视频内容",
                    "children": [
                        {
                            "text": "机器学习基础",
                            "children": [
                                {"text": "监督学习"},
                                {"text": "无监督学习"},
                                {"text": "强化学习"}
                            ]
                        },
                        {
                            "text": "深度学习技术",
                            "children": [
                                {"text": "神经网络"},
                                {"text": "卷积神经网络"},
                                {"text": "循环神经网络"}
                            ]
                        },
                        {
                            "text": "应用领域",
                            "children": [
                                {"text": "图像识别"},
                                {"text": "自然语言处理"},
                                {"text": "语音识别"}
                            ]
                        }
                    ]
                }
            }
        }
    
    try:
        # Create conversation for video analysis
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "请详细分析这个视频的内容，包括主要主题、关键信息和重要观点。"}
                ],
            },
        ]
        
        summary = generate_response(conversation, use_audio_in_video=True)
        
        # Generate mindmap based on summary
        mindmap_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"基于以下视频分析内容，生成一个结构化的思维导图JSON格式：\n{summary}\n\n请严格按照以下格式返回：{{\"root\": {{\"text\": \"中心主题\", \"children\": [{{\"text\": \"分支1\", \"children\": [{{\"text\": \"子分支1\"}}, {{\"text\": \"子分支2\"}}]}}, {{\"text\": \"分支2\", \"children\": [{{\"text\": \"子分支3\"}}, {{\"text\": \"子分支4\"}}]}}]}}}}"}
                ],
            },
        ]
        
        mindmap_response = generate_response(mindmap_conversation, use_audio_in_video=True)
        
        # Try to parse mindmap JSON
        try:
            # Clean the response to extract JSON
            mindmap_response = mindmap_response.strip()
            if mindmap_response.startswith('```json'):
                mindmap_response = mindmap_response[7:]
            if mindmap_response.endswith('```'):
                mindmap_response = mindmap_response[:-3]
            mindmap_response = mindmap_response.strip()
            
            mindmap_data = json.loads(mindmap_response)
        except json.JSONDecodeError:
            # Fallback mindmap if JSON parsing fails
            mindmap_data = {
                "root": {
                    "text": "视频内容分析",
                    "children": [
                        {
                            "text": "主要内容",
                            "children": [
                                {"text": "核心主题"},
                                {"text": "关键信息"}
                            ]
                        },
                        {
                            "text": "重要观点",
                            "children": [
                                {"text": "观点1"},
                                {"text": "观点2"}
                            ]
                        }
                    ]
                }
            }
        
        return {
            "summary": summary,
            "mindmap": mindmap_data
        }
        
    except Exception as e:
        print(f"Error processing video: {e}")
        return {
            "summary": f"视频处理出错: {str(e)}",
            "mindmap": {"root": {"text": "错误", "children": []}}
        }



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    from flask import session
    
    # 检查session中是否有分析结果
    if 'analysis_result' not in session:
        return redirect(url_for('index'))
    
    result_data = session['analysis_result']
    
    # 构建视频URL
    video_url = url_for('serve_video', filename=result_data['video_filename'])
    
    return render_template('result.html',
                         summary=result_data['summary'],
                         mindmap_data=result_data['mindmap'],
                         video_url=video_url)

@app.route('/highlights')
def highlights():
    from flask import session
    
    # 检查session中是否有分析结果
    if 'analysis_result' not in session:
        return redirect(url_for('index'))
    
    result_data = session['analysis_result']
    
    # 构建视频URL
    video_url = url_for('serve_video', filename=result_data['video_filename'])
    
    return render_template('highlights.html', video_url=video_url)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    if file and allowed_file(file.filename):
        # 保存上传的文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 初始化session
        if 'session_id' not in session:
            session['session_id'] = os.urandom(16).hex()
        
        session_id = session['session_id']
        
        # 使用真实的模型处理视频
        try:
            result = process_video_with_qwen(filepath)
            
            # 初始化对话历史
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": filepath},
                    ],
                },
            ]
            save_conversation(session_id, conversation)
            
            session['analysis_result'] = {
                'summary': result['summary'],
                'mindmap': result['mindmap'],
                'video_filename': filename
            }
            
            return jsonify({
                "status": "success",
                "redirect_url": "/result"
            })
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return jsonify({"error": f"视频处理失败: {str(e)}"}), 500
    
    return jsonify({"error": "不支持的文件格式"}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')
    use_audio = data.get('use_audio', True)
    
    session_id = session.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session not initialized. Please upload a video first.'}), 400

    if not question:
        return jsonify({'error': 'No message provided'}), 400

    try:
        conversation = load_conversation(session_id)
        
        # Add user message to conversation history
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": question}],
        })

        # Generate response using the model
        response_text = generate_response(conversation, use_audio_in_video=use_audio)

        # Add assistant response to conversation history
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}],
        })

        save_conversation(session_id, conversation)

        return jsonify({
            "answer": response_text
        })
        
    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({"error": f"对话处理失败: {str(e)}"}), 500

@app.route('/summarize_video', methods=['POST'])
def summarize_video():
    """生成视频摘要"""
    session_id = session.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session not initialized. Please upload a video first.'}), 400
    
    try:
        conversation = load_conversation(session_id)
        
        # 创建临时对话用于生成摘要
        temp_conversation = conversation[:1]  # 只包含视频上下文
        temp_conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": "请为这个视频生成一个详细的Markdown格式摘要，要求：\n1. 使用标题和子标题组织内容结构\n2. 用列表展示关键点\n3. 突出重要信息\n4. 包含视频的主要内容、核心概念和实践要点\n5. 使用适当的Markdown语法（如**粗体**、*斜体*、- 列表等）"}],
        })
        
        # 生成摘要
        summary = generate_response(temp_conversation)
        
        # 可选：将摘要请求和响应保存到主对话历史
        save_to_main = request.json.get('save_to_main', False) if request.json else False
        if save_to_main:
            conversation.extend(temp_conversation[1:])  # 添加用户请求
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": summary}],
            })
            save_conversation(session_id, conversation)
        
        return jsonify({'summary': summary})
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return jsonify({'error': f'摘要生成失败: {str(e)}'}), 500

@app.route('/generate_mindmap', methods=['POST'])
def generate_mindmap():
    """生成思维导图"""
    session_id = session.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session not initialized. Please upload a video first.'}), 400
    
    try:
        conversation = load_conversation(session_id)
        
        # 检查是否有视频在当前对话中
        if not conversation or not any(msg.get('role') == 'user' and 
                                     any(content.get('type') == 'video' for content in msg.get('content', [])) 
                                     for msg in conversation):
            return jsonify({'error': 'No video found in current conversation'}), 400
        
        # 创建临时对话用于生成思维导图
        temp_conversation = conversation[:1]  # 只包含视频上下文
        temp_conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": "请为这个视频生成一个Markdown格式的思维导图，包含主要主题和子主题的层次结构。"}],
        })
        
        # 生成思维导图
        mindmap_text = generate_response(temp_conversation)
        
        # 可选：将思维导图请求和响应保存到主对话历史
        save_to_main = request.json.get('save_to_main', False) if request.json else False
        if save_to_main:
            conversation.extend(temp_conversation[1:])  # 添加用户请求
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": mindmap_text}],
            })
            save_conversation(session_id, conversation)
        
        return jsonify({'mindmap': mindmap_text})
        
    except Exception as e:
        print(f"Error generating mindmap: {e}")
        return jsonify({'error': f'思维导图生成失败: {str(e)}'}), 500

@app.route('/template-data', methods=['GET'])
def template_data():
    """提供模板数据，无需上传视频"""
    # 创建示例思维导图数据
    mindmap_data = {
        "root": {
            "text": "示例思维导图",
            "children": [
                {
                    "text": "主题1",
                    "children": [
                        {"text": "子主题1.1"},
                        {"text": "子主题1.2"}
                    ]
                },
                {
                    "text": "主题2",
                    "children": [
                        {"text": "子主题2.1"},
                        {"text": "子主题2.2"}
                    ]
                }
            ]
        }
    }
    
    # 初始化session
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    
    # 创建示例Markdown格式摘要
    demo_summary = """# 📹 视频内容摘要

## 🎯 概述
这个视频深入介绍了**ChatGPT Prompt工程**的核心概念和实践技巧，主要由*OpenAI的Isa Fulford*和*DeepLearning.AI的Andrew Ng*联合讲解。课程内容涵盖了从基础概念到高级应用的完整知识体系。

## 📊 视频统计
- **时长**: 45分钟
- **核心概念**: 8个
- **实践技巧**: 15个
- **内容覆盖**: 95%

## 🔑 核心要点

### 1. LLM API应用
- 如何利用大语言模型API构建智能软件应用
- 接口调用和参数优化技巧
- 实际项目中的应用场景

### 2. Prompt设计原则
- 掌握提示词工程的重要性
- 学习编写**高效提示词**的技巧
- 避免常见的设计误区

### 3. 模型特性理解
- 深入了解ChatGPT模型的特性
- 不同场景下的适用性分析
- 模型局限性和使用注意事项

### 4. 优化策略
- 如何优化提示词以获得更好的结果
- 提升模型输出质量的方法
- 迭代改进的最佳实践

### 5. 安全考量
- AI安全的重要性
- 如何防范潜在风险和误用
- 负责任的AI使用原则

## 💡 重点提示
> 💫 **关键洞察**: 讲者强调良好的prompt设计可以显著提高LLM的效率和安全性，这是构建可靠AI应用的关键基础。

## 🏷️ 内容标签
`🤖 AI技术` `💡 Prompt工程` `🔧 实践应用` `📚 教程指南`"""
    
    # 将结果存储到session中
    session['analysis_result'] = {
        'summary': demo_summary,
        'mindmap': mindmap_data,
        'video_filename': 'template_video.mp4'  # 模拟视频文件名
    }
    
    # 返回简单的示例数据
    return jsonify({
        "status": "success",
        "summary": demo_summary,
        "mindmap": mindmap_data
    })



@app.route('/video/<filename>')
def serve_video(filename):
    """直接提供视频文件的访问"""
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "视频文件不存在"}), 404
    
    # 返回视频文件，使用合适的MIME类型
    if filename.lower().endswith('.mp4'):
        mimetype = 'video/mp4'
    elif filename.lower().endswith('.avi'):
        mimetype = 'video/x-msvideo'
    elif filename.lower().endswith('.mov'):
        mimetype = 'video/quicktime'
    else:
        mimetype = 'application/octet-stream'
    
    return send_file(video_path, mimetype=mimetype)

@app.route('/fixed-video')
def serve_fixed_video():
    """提供固定的视频文件"""
    fixed_video = "test2.mp4"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], fixed_video)
    
    if not os.path.exists(video_path):
        return jsonify({"error": "固定视频文件不存在"}), 404
    
    return send_file(video_path, mimetype='video/mp4')

@app.route('/generate_highlights', methods=['POST'])
def generate_highlights():
    """生成视频集锦"""
    session_id = session.get('session_id')
    
    if not session_id or 'analysis_result' not in session:
        return jsonify({'error': '请先上传视频进行分析'}), 400
    
    try:
        # 检查是否有模型可用
        if not MODEL_AVAILABLE or model is None or processor is None:
            print("🎭 模型不可用，使用示例集锦视频")
            
            # 使用原视频作为示例集锦（模拟集锦效果）
            result_data = session['analysis_result']
            video_filename = result_data['video_filename']
            original_video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            
            if not os.path.exists(original_video_path):
                return jsonify({'error': '原视频文件不存在'}), 404
            
            # 直接使用原视频作为集锦视频的URL
            highlight_url = url_for('serve_video', filename=video_filename)
            
            print(f"✅ 示例集锦视频准备完成: {video_filename}")
            
            return jsonify({
                'status': 'success',
                'highlight_url': highlight_url,
                'highlight_filename': f"highlight_{video_filename}",
                'message': '集锦视频生成成功！'
            })
        
        # 获取原视频文件路径
        result_data = session['analysis_result']
        video_filename = result_data['video_filename']
        original_video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        
        if not os.path.exists(original_video_path):
            return jsonify({'error': '原视频文件不存在'}), 404
        
        # 分割视频为片段
        print(f"🎬 开始分割视频: {original_video_path}")
        segments = split_video_into_segments(original_video_path)
        
        if not segments:
            return jsonify({'error': '视频分割失败'}), 500
        
        print(f"📊 视频已分割为 {len(segments)} 个片段")
        
        # 为每个片段打分
        scored_segments = []
        total_segments = len(segments)
        
        for i, (start_time, end_time) in enumerate(segments):
            print(f"🔍 正在分析片段 {i+1}/{total_segments}: {start_time:.1f}s - {end_time:.1f}s")
            score = score_video_segment(original_video_path, start_time, end_time)
            scored_segments.append(((start_time, end_time), score))
            print(f"⭐ 片段 {i+1} 得分: {score}")
        
        # 创建集锦视频
        print("🎥 开始创建集锦视频...")
        print("🔧 正在检查视频编码依赖...")
        
        highlight_video_path = create_highlight_video(original_video_path, scored_segments)
        
        if not highlight_video_path:
            error_msg = "集锦视频创建失败。可能的原因：\n"
            error_msg += "1. ffmpeg未安装或未添加到PATH环境变量\n"
            error_msg += "2. MoviePy依赖不完整\n"
            error_msg += "3. 视频文件格式不支持\n"
            error_msg += "4. 磁盘空间不足\n"
            error_msg += "请检查控制台输出获取详细错误信息。"
            print(f"❌ {error_msg}")
            return jsonify({'error': error_msg}), 500
        
        # 获取集锦视频的相对路径用于前端访问
        highlight_filename = os.path.basename(highlight_video_path)
        highlight_url = url_for('serve_highlight_video', filename=highlight_filename)
        
        print(f"✅ 集锦视频创建成功: {highlight_video_path}")
        
        return jsonify({
            'status': 'success',
            'highlight_url': highlight_url,
            'highlight_filename': highlight_filename,
            'message': '集锦视频生成成功！'
        })
        
    except Exception as e:
        print(f"❌ 生成集锦时发生错误: {e}")
        return jsonify({'error': f'集锦生成失败: {str(e)}'}), 500

@app.route('/highlights/<filename>')
def serve_highlight_video(filename):
    """提供集锦视频文件的访问"""
    video_path = os.path.join(app.config['HIGHLIGHTS_FOLDER'], filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "集锦视频文件不存在"}), 404
    
    return send_file(video_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False) # 将 use_reloader 设置为 False
