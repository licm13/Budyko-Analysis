# src/utils/plotting_config.py
"""
统一的matplotlib中文字体配置模块

在所有需要matplotlib绘图的脚本开头导入此模块，
以确保中文字符正确显示。

Usage:
    from utils.plotting_config import setup_chinese_fonts
    setup_chinese_fonts()

或者直接导入即可自动配置：
    import utils.plotting_config
"""

import matplotlib.pyplot as plt
import matplotlib


def setup_chinese_fonts():
    """
    设置matplotlib支持中文字体
    
    优先级：
    1. SimHei（黑体）- Windows系统默认
    2. Microsoft YaHei（微软雅黑）- Windows系统常用
    3. PingFang SC（苹方）- macOS系统默认
    4. DejaVu Sans - 跨平台备选
    5. Arial Unicode MS - 跨平台备选
    """
    # 设置中文字体列表（按优先级排序）
    chinese_fonts = [
        'SimHei',           # 黑体 (Windows)
        'Microsoft YaHei',  # 微软雅黑 (Windows)
        'PingFang SC',      # 苹方 (macOS)
        'Hiragino Sans GB', # 冬青黑体 (macOS)
        'WenQuanYi Micro Hei',  # 文泉驿微米黑 (Linux)
        'DejaVu Sans',      # 跨平台
        'Arial Unicode MS', # 跨平台
        'sans-serif'        # 系统默认
    ]
    
    # 配置matplotlib
    plt.rcParams['font.sans-serif'] = chinese_fonts
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 可选：设置其他字体相关参数
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    
    print("✓ 中文字体配置已应用")


def get_available_chinese_fonts():
    """
    获取系统中可用的中文字体列表
    
    Returns
    -------
    list
        可用的中文字体名称列表
    """
    from matplotlib.font_manager import FontManager
    
    fm = FontManager()
    chinese_fonts = []
    
    chinese_font_keywords = [
        'SimHei', 'Microsoft YaHei', 'PingFang', 'Hiragino',
        'WenQuanYi', 'Source Han', 'Noto Sans CJK', 'Arial Unicode'
    ]
    
    for font in fm.ttflist:
        font_name = font.name
        if any(keyword in font_name for keyword in chinese_font_keywords):
            if font_name not in chinese_fonts:
                chinese_fonts.append(font_name)
    
    return sorted(chinese_fonts)


def test_chinese_display():
    """
    测试中文字符显示效果
    
    生成一个简单的测试图表来验证中文字体是否正确配置
    """
    import numpy as np
    
    # 创建测试数据
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, 'b-', linewidth=2, label='正弦曲线')
    
    # 添加中文标签和标题
    ax.set_xlabel('角度（弧度）')
    ax.set_ylabel('幅值')
    ax.set_title('中文字体测试图表')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加中文注释
    ax.annotate('最大值点', xy=(np.pi/2, 1), xytext=(2, 0.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    return fig


# 自动配置（导入模块时执行）
setup_chinese_fonts()


if __name__ == '__main__':
    # 测试脚本
    print("可用的中文字体:")
    fonts = get_available_chinese_fonts()
    for font in fonts:
        print(f"  - {font}")
    
    print("\n生成测试图表...")
    fig = test_chinese_display()
    plt.show()