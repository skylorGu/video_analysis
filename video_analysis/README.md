# 智能视频分析平台

一个基于Web的智能视频内容解析工具，集成了先进的AI模型，能够自动分析上传的视频内容，并提供文字总结、思维导图生成、智能对话和视频集锦等多种功能。

## ✨ 主要功能

### 🎥 视频处理
- **多格式支持**：支持MP4、AVI、MOV、MKV格式视频上传
- **智能播放**：内置播放器，支持播放控制和速度调节
- **视频集锦**：自动生成视频精彩片段集锦
- **大文件支持**：最大支持500MB视频文件上传

### 🤖 AI智能分析
- **内容总结**：使用Qwen2.5-Omni模型自动生成视频内容的文字摘要
- **思维导图**：基于视频内容自动生成直观的思维导图可视化
- **智能对话**：根据视频内容提供问答交互功能
- **多模态理解**：结合视觉和音频信息进行综合分析

### 🎨 用户体验
- **现代化UI**：基于Bootstrap 5的响应式设计
- **实时反馈**：上传和处理进度实时显示
- **会话管理**：支持多轮对话历史记录
- **结果导出**：支持分析结果的保存和分享

## 🛠 技术栈

### 后端技术
- **框架**：Python Flask
- **AI模型**：Qwen2.5-Omni（3B参数的多模态大语言模型）
- **视频处理**：MoviePy, FFmpeg
- **图像处理**：PIL, NumPy
- **深度学习**：PyTorch, Transformers

### 前端技术
- **UI框架**：Bootstrap 5
- **样式**：CSS3, Bootstrap Icons
- **交互**：JavaScript (ES6+)
- **数据可视化**：D3.js
- **Markdown渲染**：Marked.js

### 开发工具
- **依赖管理**：pip, requirements.txt
- **版本控制**：Git
- **IDE支持**：PyCharm配置文件

## 📋 环境要求

### 系统要求
- **操作系统**：Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python版本**：3.8 或更高版本
- **内存**：建议8GB以上（AI模型运行需要）
- **存储空间**：至少10GB可用空间

### 依赖软件
- **FFmpeg**：用于视频处理（自动检测）
- **CUDA**：可选，用于GPU加速（推荐）

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd video_analysis

# 创建虚拟环境（推荐）
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装额外的AI模型依赖（如果需要）
pip install torch transformers flash-attn
```

### 3. 模型配置

```bash
# 下载Qwen2.5-Omni模型（可选，支持模拟模式）
# 模型路径需要在app.py中配置
```

### 4. 启动应用

```bash
# 进入项目目录
cd video_analysis

# 启动Flask应用
python app.py
```

### 5. 访问应用

在浏览器中打开：`http://localhost:5000`

## 📁 项目结构

```
video_analysis/
├── omni_utils/                # AI模型工具包
│   ├── __init__.py
│   ├── .gitignore
│   └── v2_5/                  # Qwen2.5相关工具
│       ├── __init__.py
│       ├── audio_process.py   # 音频处理模块
│       └── vision_process.py  # 视觉处理模块
└── video_analysis/            # 主项目目录
    ├── .gitignore
    ├── .idea/                 # PyCharm配置文件
    │   ├── .gitignore
    │   ├── inspectionProfiles/
    │   ├── misc.xml
    │   ├── modules.xml
    │   ├── vcs.xml
    │   └── video_analysis.iml
    ├── README.md              # 项目说明文档
    ├── app.py                 # Flask主应用文件
    ├── requirements.txt       # Python依赖列表
    ├── static/                # 静态资源目录
    │   ├── css/               # 样式文件
    │   ├── js/                # JavaScript文件
    │   ├── images/            # 图片资源
    │   ├── uploads/           # 上传文件存储
    │   └── highlights/        # 视频集锦存储
    └── templates/             # HTML模板
        ├── index.html         # 主页模板
        ├── result.html        # 结果页面模板
        └── highlights.html    # 集锦页面模板
```

## 📖 使用指南

### 基本使用流程

1. **上传视频**
   - 在主页点击"选择文件"按钮
   - 选择支持格式的视频文件（MP4、AVI、MOV、MKV）
   - 等待上传完成

2. **视频分析**
   - 上传完成后自动开始AI分析
   - 系统会提取视频关键帧进行内容理解
   - 分析过程可能需要几分钟，请耐心等待

3. **查看结果**
   - **内容摘要**：查看AI生成的视频内容总结
   - **思维导图**：浏览可视化的内容结构图
   - **智能问答**：与AI就视频内容进行对话
   - **视频集锦**：观看自动生成的精彩片段

### 高级功能

#### 智能对话
- 支持多轮对话，可以深入询问视频细节
- AI会基于视频内容回答问题
- 对话历史会自动保存

#### 视频集锦生成
- 自动识别视频中的精彩片段
- 可配置集锦时长和片段数量
- 支持导出集锦视频

## ⚙️ 配置说明

### 模型配置

在 `app.py` 中可以配置AI模型相关参数：

```python
# 模型路径配置
MODEL_PATH = "/data2/dyl/LLMCheckpoint/Qwen2.5-Omni-3B"

# 集锦生成配置
SEGMENT_DURATION = 10  # 每个切片的时长（秒）
TOP_N_SEGMENTS = 3     # 选择评分最高的前N个切片
USE_AUDIO_IN_VIDEO = True  # 是否在视频分析中使用音频
```

### 上传限制配置

```python
# 文件大小限制
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# 支持的文件格式
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
```

## 🔧 API 文档

### 主要端点

#### 1. 文件上传
```http
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: 视频文件
```

#### 2. 视频分析
```http
POST /analyze
Content-Type: application/json

Body:
{
    "filename": "video.mp4",
    "analysis_type": "summary|mindmap|chat"
}
```

#### 3. 智能对话
```http
POST /chat
Content-Type: application/json

Body:
{
    "message": "用户问题",
    "video_context": "视频上下文"
}
```

#### 4. 生成集锦
```http
POST /generate_highlights
Content-Type: application/json

Body:
{
    "filename": "video.mp4",
    "duration": 10,
    "segments": 3
}
```

## 🚨 常见问题

### Q: 支持哪些视频格式？
A: 目前支持 MP4、AVI、MOV、MKV 格式。推荐使用 MP4 格式以获得最佳兼容性。

### Q: 视频文件大小有限制吗？
A: 是的，目前限制为 500MB。如需处理更大文件，请联系管理员调整配置。

### Q: AI分析需要多长时间？
A: 分析时间取决于视频长度和复杂度，通常在 1-5 分钟之间。

### Q: 是否需要GPU？
A: GPU不是必需的，但强烈推荐使用GPU以获得更快的处理速度。

### Q: 模型文件在哪里下载？
A: Qwen2.5-Omni模型可以从Hugging Face或官方渠道下载。项目支持模拟模式，即使没有模型也可以运行。

### Q: 如何解决FFmpeg相关错误？
A: 请确保系统已安装FFmpeg，并且可以在命令行中访问。Windows用户可以从官网下载并添加到PATH。

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 开发规范

- 遵循 PEP 8 Python 代码规范
- 添加适当的注释和文档字符串
- 确保新功能有相应的测试
- 更新相关文档

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5) - 强大的多模态AI模型
- [Flask](https://flask.palletsprojects.com/) - 轻量级Web框架
- [Bootstrap](https://getbootstrap.com/) - 现代化UI组件库
- [D3.js](https://d3js.org/) - 数据可视化库
- [MoviePy](https://zulko.github.io/moviepy/) - 视频处理库

## 📞 联系我们

如有问题或建议，请通过以下方式联系：

- 提交 [Issue](../../issues)
- 发送邮件至：[your-email@example.com]
- 项目主页：[项目链接]

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！