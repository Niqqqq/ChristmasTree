import pygame  # type: ignore
import os
import sys
import math
import random
from PIL import Image, ImageDraw
# 预加载项目中 image 目录下的照片，用于照片粒子


def get_resource_path(relative_path):
    """获取资源文件的绝对路径，支持打包后的可执行文件"""
    try:
        # PyInstaller创建临时文件夹，将路径存储在_MEIPASS中
        base_path = sys._MEIPASS  # type: ignore
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def load_photo_images():
    photos = []
    # 支持打包后的路径
    image_dir = get_resource_path("image")
    if os.path.isdir(image_dir):
        for fname in os.listdir(image_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                path = os.path.join(image_dir, fname)
                try:
                    img = pygame.image.load(path).convert_alpha()
                    photos.append(img)
                except Exception:
                    continue
    return photos


PHOTO_IMAGES = []

# 初始化Pygame
pygame.init()  # type: ignore

# 屏幕设置（再增大20%）
WIDTH, HEIGHT = int(1200 * 1.2 * 1.2), int(800 * 1.2 * 1.2)  # 1728 x 1152
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D动态圣诞树 - 照片墙效果")

# 预加载照片资源（用于照片粒子）
PHOTO_IMAGES = load_photo_images()

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
DARK_GREEN = (0, 100, 0)
BRIGHT_GREEN = (0, 200, 0)
NEW_GREEN = (70, 133, 39)  # #468527 新颜色，占树的大部分
RED = (255, 50, 50)
BRIGHT_RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GOLD = (255, 215, 0)
ORANGE = (255, 165, 0)
LIGHT_YELLOW = (255, 255, 180)  # 淡黄色
PALE_YELLOW = (255, 250, 200)  # 更淡的黄色

# 圣诞色彩列表
CHRISTMAS_COLORS = [
    DARK_GREEN, BRIGHT_GREEN,  # 绿色
    RED, BRIGHT_RED,  # 红色
    YELLOW, GOLD, ORANGE  # 黄色/金色
]

# 3D相机设置


class Camera:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = -800  # 减小距离让树更大
        self.angle_x = 0
        self.angle_y = 0

    def project(self, x, y, z):
        """3D到2D投影"""
        # 旋转
        cos_x, sin_x = math.cos(self.angle_x), math.sin(self.angle_x)
        cos_y, sin_y = math.cos(self.angle_y), math.sin(self.angle_y)

        # Y轴旋转
        x, z = x * cos_y - z * sin_y, x * sin_y + z * cos_y

        # X轴旋转
        y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x

        # 透视投影
        # 相机在z=-800，物体在z=0附近
        depth = z - self.z  # z + 800

        if depth > 50:  # 只显示相机前方的物体
            # 使用透视投影公式 - 增大视野因子让物体更大
            fov = 1000  # 增大视野因子
            scale = fov / depth
            screen_x = x * scale + WIDTH // 2
            screen_y = y * scale + HEIGHT // 2
            return screen_x, screen_y, scale, depth
        return WIDTH // 2, HEIGHT // 2, 0, 0


# 粒子类型
PARTICLE_CUBE = 0
PARTICLE_SPHERE = 1
PARTICLE_PHOTO = 2

# 3D粒子类


# 雪花类
class Snowflake:
    def __init__(self):
        # 从屏幕上方随机位置开始
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(-100, -20)  # 从屏幕上方开始
        # 随机下落速度（加快速度）
        self.speed_y = random.uniform(2, 6)
        # 左右摆动幅度和速度（模拟雪花飘落）
        self.sway_amplitude = random.uniform(0.5, 2.0)
        self.sway_speed = random.uniform(0.02, 0.05)
        self.sway_offset = random.uniform(0, math.pi * 2)
        # 雪花颜色（白色和淡蓝色）
        colors = [
            (255, 255, 255),  # 纯白色
            (240, 248, 255),  # 淡蓝色
            (230, 240, 255),  # 更淡的蓝色
            (250, 250, 255),  # 略带蓝色的白
        ]
        self.color = random.choice(colors)
        # 大小（雪花有大有小，增大尺寸范围）
        self.size = random.randint(3, 12)
        # 旋转角度和速度
        self.rotation = random.uniform(0, math.pi * 2)
        self.rotation_speed = random.uniform(0.01, 0.03)

    def update(self, dt):
        """更新雪花位置"""
        # 垂直下落
        self.y += self.speed_y
        # 左右摆动（使用正弦波）
        self.sway_offset += self.sway_speed
        self.x += math.sin(self.sway_offset) * self.sway_amplitude
        # 旋转
        self.rotation += self.rotation_speed
        # 如果雪花移出屏幕，重置位置
        if self.y > HEIGHT + 50 or self.x < -50 or self.x > WIDTH + 50:
            self.x = random.randint(0, WIDTH)
            self.y = random.randint(-100, -20)
            self.speed_y = random.uniform(2, 6)
            self.sway_amplitude = random.uniform(0.5, 2.0)
            self.sway_speed = random.uniform(0.02, 0.05)
            self.sway_offset = random.uniform(0, math.pi * 2)
            colors = [
                (255, 255, 255), (240, 248, 255),
                (230, 240, 255), (250, 250, 255),
            ]
            self.color = random.choice(colors)
            self.size = random.randint(3, 12)
            self.rotation = random.uniform(0, math.pi * 2)
            self.rotation_speed = random.uniform(0.01, 0.03)

    def draw(self, surface):
        """绘制雪花"""
        # 创建半透明表面用于雪花
        snow_surf = pygame.Surface(
            (self.size * 4, self.size * 4), pygame.SRCALPHA)  # type: ignore

        # 绘制雪花形状（简单的星形）
        center_x, center_y = self.size * 2, self.size * 2
        points = []
        num_arms = 6  # 6个分支
        for i in range(num_arms * 2):
            angle = (i * math.pi / num_arms) + self.rotation
            if i % 2 == 0:
                radius = self.size
            else:
                radius = self.size * 0.5
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))

        # 绘制雪花主体
        if len(points) >= 3:
            pygame.draw.polygon(snow_surf, self.color, points)
        # 绘制中心点
        pygame.draw.circle(snow_surf, self.color,
                           (int(center_x), int(center_y)), max(1, self.size // 2))

        # 绘制到屏幕
        surface.blit(snow_surf, (int(self.x - self.size * 2),
                     int(self.y - self.size * 2)))
        bright_color = tuple(min(255, c + 100) for c in self.color)
        pygame.draw.circle(surface, bright_color, (int(
            self.x - 1), int(self.y - 1)), max(1, self.size // 2))


class Particle3D:
    def __init__(self, x, y, z, target_x, target_y, target_z, particle_type, size=15):
        self.x = x
        self.y = y
        self.z = z
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z
        self.start_x = x
        self.start_y = y
        self.start_z = z
        self.size = size
        self.particle_type = particle_type
        self.color = self.assign_color()
        self.photo_image = None
        self.is_large_photo = False  # 标记是否是大图（散开时显示更大）
        # 文字模式的目标位置（None表示正常散开）
        self.text_target_x = None
        self.text_target_y = None
        self.text_target_z = None
        if particle_type == PARTICLE_PHOTO:
            # 优先使用项目中的图片；没有则使用生成的占位图
            if PHOTO_IMAGES:
                self.photo_image = random.choice(PHOTO_IMAGES)
            else:
                self.photo_image = self.generate_photo_image()
            # 随机选择一些照片作为大图（约40%的照片是大图）
            if random.random() < 0.40:
                self.is_large_photo = True
        self.rotation_x = random.uniform(0, 360)
        self.rotation_y = random.uniform(0, 360)
        self.rotation_z = random.uniform(0, 360)
        self.rotation_speed_x = random.uniform(-3, 3)
        self.rotation_speed_y = random.uniform(-3, 3)
        self.rotation_speed_z = random.uniform(-3, 3)

        # 是否发光（散开时部分球体会发光）
        self.is_glowing = False
        if particle_type == PARTICLE_SPHERE and random.random() < 0.05:  # 5%的球体在散开时会发光（减少数量）
            self.is_glowing = True

        # 固定的散开参数（确保每次散开位置一致）
        # 使用球面坐标让散开形成球体分布
        # 生成均匀的球面分布角度
        self.scatter_theta = random.uniform(0, 2 * math.pi)  # 水平角度（0到2π）
        # 使用反余弦函数生成均匀的垂直角度分布
        self.scatter_phi = math.acos(
            2 * random.random() - 1)  # 垂直角度（0到π），球面均匀分布
        # 照片粒子距离略短一些，更容易经过屏幕中央
        if particle_type == PARTICLE_PHOTO:
            self.scatter_distance_factor = random.uniform(0.7, 0.95)
        else:
            self.scatter_distance_factor = random.uniform(0.9, 1.1)

    def generate_photo_image(self):
        """生成包含肖像的照片图片"""
        img_size = max(40, self.size * 2)
        img = Image.new('RGB', (img_size, int(img_size * 1.5)),
                        (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))
        draw = ImageDraw.Draw(img)

        # 绘制照片边框
        border_color = (random.randint(150, 200), random.randint(
            150, 200), random.randint(150, 200))
        draw.rectangle([0, 0, img_size-1, int(img_size * 1.5)-1],
                       outline=border_color, width=2)

        # 绘制简单的肖像（圆形脸 + 身体）
        center_x, center_y = img_size // 2, img_size // 2

        # 脸部（圆形）
        face_size = img_size // 3
        face_color = (random.randint(220, 255), random.randint(
            200, 240), random.randint(180, 220))
        draw.ellipse([center_x - face_size//2, center_y - face_size//2 - 10,
                     center_x + face_size//2, center_y + face_size//2 - 10],
                     fill=face_color)

        # 眼睛
        eye_size = face_size // 6
        draw.ellipse([center_x - face_size//3 - eye_size//2, center_y - face_size//3 - 10,
                     center_x - face_size//3 + eye_size//2, center_y - face_size//3 + eye_size - 10],
                     fill=(50, 50, 50))
        draw.ellipse([center_x + face_size//3 - eye_size//2, center_y - face_size//3 - 10,
                     center_x + face_size//3 + eye_size//2, center_y - face_size//3 + eye_size - 10],
                     fill=(50, 50, 50))

        # 嘴巴（微笑）
        mouth_y = center_y + face_size//4 - 10
        draw.arc([center_x - face_size//4, mouth_y - face_size//8,
                 center_x + face_size//4, mouth_y + face_size//8],
                 start=0, end=180, fill=(100, 50, 50), width=2)

        # 身体（矩形）
        body_color = (random.randint(100, 200), random.randint(
            100, 200), random.randint(100, 200))
        draw.rectangle([center_x - face_size//2, center_y + face_size//2 - 10,
                       center_x + face_size//2, int(img_size * 1.2) - 10],
                       fill=body_color)

        # 转换为pygame surface
        return pygame.image.fromstring(img.tobytes(), img.size, img.mode)

    def assign_color(self):
        """根据粒子类型和位置分配颜色"""
        if self.particle_type == PARTICLE_PHOTO:
            # 照片使用较浅的颜色
            return (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        elif self.particle_type == PARTICLE_SPHERE:
            # 球体：新颜色 #468527 占大部分（60%），其他颜色随机（绿色、黄色、红色等）
            rand_color = random.random()
            if rand_color < 0.60:  # 60% 新颜色
                return NEW_GREEN
            else:  # 40% 其他颜色，随机选择多种颜色
                # 随机选择：绿色、黄色、红色等
                color_choice = random.random()
                if color_choice < 0.3:  # 30% 其他绿色
                    return random.choice([DARK_GREEN, BRIGHT_GREEN])
                elif color_choice < 0.5:  # 20% 黄色
                    return random.choice([YELLOW, GOLD, ORANGE])
                elif color_choice < 0.7:  # 20% 红色
                    return random.choice([RED, BRIGHT_RED])
                else:  # 30% 随机绿色系（与基础颜色相似）
                    base_r, base_g, base_b = NEW_GREEN
                    r = max(0, min(255, base_r + random.randint(-40, 40)))
                    g = max(0, min(255, base_g + random.randint(-40, 40)))
                    b = max(0, min(255, base_b + random.randint(-40, 40)))
                    return (r, g, b)
        else:
            # 立方体：新颜色 #468527 占大部分（60%），其他颜色随机（绿色、黄色、红色等）
            rand_color = random.random()
            if rand_color < 0.60:  # 60% 新颜色
                return NEW_GREEN
            else:  # 40% 其他颜色，随机选择多种颜色
                # 随机选择：绿色、黄色、红色等
                color_choice = random.random()
                if color_choice < 0.3:  # 30% 其他绿色
                    return random.choice([DARK_GREEN, BRIGHT_GREEN])
                elif color_choice < 0.5:  # 20% 黄色
                    return random.choice([YELLOW, GOLD, ORANGE])
                elif color_choice < 0.7:  # 20% 红色
                    return random.choice([RED, BRIGHT_RED])
                else:  # 30% 随机绿色系（与基础颜色相似）
                    base_r, base_g, base_b = NEW_GREEN
                    r = max(0, min(255, base_r + random.randint(-40, 40)))
                    g = max(0, min(255, base_g + random.randint(-40, 40)))
                    b = max(0, min(255, base_b + random.randint(-40, 40)))
                    return (r, g, b)

    def update(self, t, scatter_mode=False):
        """更新粒子位置
        t: 0.0-1.0 动画进度
        scatter_mode: True=散开, False=聚合
        """
        # 根据屏幕大小和投影计算合适的散开范围
        # 屏幕宽度1200，高度800，相机距离800，视野1000
        # 要让粒子铺满整个可视区域，需要更大的散开范围
        # X方向需要约±600，Y方向需要约±400，Z方向需要更大的范围形成深度
        max_scatter_distance = 600  # 增大散开范围，铺满整个可视区域

        if scatter_mode:
            # 如果有文字目标位置，移动到文字位置（文字拼图模式）
            if self.text_target_x is not None:
                # 从原始目标位置平滑移动到文字位置
                ease_t = t * t * (3 - 2 * t)  # smoothstep缓动
                self.x = self.target_x + \
                    (self.text_target_x - self.target_x) * ease_t
                self.y = self.target_y + \
                    (self.text_target_y - self.target_y) * ease_t
                self.z = self.target_z + \
                    (self.text_target_z - self.target_z) * ease_t
                # 当t=1.0时，确保精确到达文字位置
                if t >= 1.0:
                    self.x = self.text_target_x
                    self.y = self.text_target_y
                    self.z = self.text_target_z
            else:
                # 正常散开模式：从目标位置向外扩散，铺满整个可视区域，形成由远到近的3D效果
                # 使用平滑的缓动函数让散开更丝滑
                # 使用 smoothstep 缓动函数，平衡性能和流畅度
                ease_t = t * t * (3 - 2 * t)  # smoothstep缓动
                scatter_distance = max_scatter_distance * ease_t * self.scatter_distance_factor

                # 使用球面坐标计算3D散开位置，形成球体分布
                # 球面坐标转笛卡尔坐标：
                # x = r * sin(phi) * cos(theta)
                # y = r * sin(phi) * sin(theta)
                # z = r * cos(phi)
                sin_phi = math.sin(self.scatter_phi)
                cos_phi = math.cos(self.scatter_phi)
                cos_theta = math.cos(self.scatter_theta)
                sin_theta = math.sin(self.scatter_theta)

                scatter_x_offset = scatter_distance * sin_phi * cos_theta
                scatter_y_offset = scatter_distance * sin_phi * sin_theta
                # Z轴使用更大的范围，形成由远到近的效果
                scatter_z_offset = scatter_distance * cos_phi * 1.5  # 增大Z轴范围

                # 照片粒子偏向中心，缩小散开半径，增加穿过中央的概率
                if self.particle_type == PARTICLE_PHOTO:
                    scatter_x_offset *= 0.7
                    scatter_y_offset *= 0.7
                    scatter_z_offset *= 0.9

                # 限制散开范围，铺满整个可视区域
                # X方向：±600（铺满屏幕宽度）
                # Y方向：±400（铺满屏幕高度）
                # Z方向：±500（形成深度感，由远到近）
                max_x_offset = 600
                max_y_offset = 400
                max_z_offset = 500

                scatter_x_offset = max(-max_x_offset,
                                       min(max_x_offset, scatter_x_offset))
                scatter_y_offset = max(-max_y_offset,
                                       min(max_y_offset, scatter_y_offset))
                scatter_z_offset = max(-max_z_offset,
                                       min(max_z_offset, scatter_z_offset))

                self.x = self.target_x + scatter_x_offset
                self.y = self.target_y + scatter_y_offset
                self.z = self.target_z + scatter_z_offset
        else:
            # 聚合模式：从散开位置回到目标位置
            if self.text_target_x is not None:
                # 从文字位置回到原始目标位置
                ease_t = t * t * (3 - 2 * t)  # smoothstep缓动
                self.x = self.text_target_x + \
                    (self.target_x - self.text_target_x) * ease_t
                self.y = self.text_target_y + \
                    (self.target_y - self.text_target_y) * ease_t
                self.z = self.text_target_z + \
                    (self.target_z - self.text_target_z) * ease_t
            else:
                # 正常聚合模式：从散开位置回到目标位置
                # 计算最大散开位置（使用球体分布，铺满整个可视区域）
                sin_phi = math.sin(self.scatter_phi)
                cos_phi = math.cos(self.scatter_phi)
                cos_theta = math.cos(self.scatter_theta)
                sin_theta = math.sin(self.scatter_theta)

                scatter_distance = max_scatter_distance * self.scatter_distance_factor
                scatter_x_offset = scatter_distance * sin_phi * cos_theta
                scatter_y_offset = scatter_distance * sin_phi * sin_theta
                # Z轴使用更大的范围
                scatter_z_offset = scatter_distance * cos_phi * 1.5

                if self.particle_type == PARTICLE_PHOTO:
                    scatter_x_offset *= 0.7
                    scatter_y_offset *= 0.7
                    scatter_z_offset *= 0.9

                max_x_offset = 600
                max_y_offset = 400
                max_z_offset = 500

                scatter_x_offset = max(-max_x_offset,
                                       min(max_x_offset, scatter_x_offset))
                scatter_y_offset = max(-max_y_offset,
                                       min(max_y_offset, scatter_y_offset))
                scatter_z_offset = max(-max_z_offset,
                                       min(max_z_offset, scatter_z_offset))

                scatter_x = self.target_x + scatter_x_offset
                scatter_y = self.target_y + scatter_y_offset
                scatter_z = self.target_z + scatter_z_offset

                # 从散开位置插值回目标位置（使用缓动函数让动画更平滑）
                ease_t = t * t * (3 - 2 * t)  # 平滑插值
                self.x = scatter_x + (self.target_x - scatter_x) * ease_t
                self.y = scatter_y + (self.target_y - scatter_y) * ease_t
                self.z = scatter_z + (self.target_z - scatter_z) * ease_t

        # 更新旋转
        self.rotation_x += self.rotation_speed_x
        self.rotation_y += self.rotation_speed_y
        self.rotation_z += self.rotation_speed_z


def draw_cube(surface, x, y, size, color, rotation=0):
    """绘制3D立方体（简化版）"""
    # 计算立方体的8个顶点（简化，只绘制可见面）
    half = size / 2
    points = [
        (-half, -half), (half, -half), (half, half), (-half, half)
    ]

    # 旋转点
    cos_r = math.cos(math.radians(rotation))
    sin_r = math.sin(math.radians(rotation))

    rotated_points = []
    for px, py in points:
        rx = px * cos_r - py * sin_r
        ry = px * sin_r + py * cos_r
        rotated_points.append((x + rx, y + ry))

    # 绘制立方体的前面（深色）和顶面（亮色）
    if len(rotated_points) >= 4:
        # 前面
        pygame.draw.polygon(surface, color, rotated_points)
        # 边框
        pygame.draw.polygon(surface, tuple(min(255, c + 30)
                            for c in color), rotated_points, 2)

        # 顶面（模拟3D效果）
        top_points = [(p[0], p[1] - size * 0.3) for p in rotated_points]
        lighter_color = tuple(min(255, c + 40) for c in color)
        pygame.draw.polygon(surface, lighter_color, top_points[:3])


def draw_sphere(surface, x, y, size, color):
    """绘制3D球体（用圆形和高光模拟）"""
    # 主体
    pygame.draw.circle(surface, color, (int(x), int(y)), size)
    # 高光
    highlight_color = tuple(min(255, c + 60) for c in color)
    pygame.draw.circle(surface, highlight_color, (int(
        x - size * 0.3), int(y - size * 0.3)), size // 3)
    # 边框
    pygame.draw.circle(surface, tuple(max(0, c - 30)
                       for c in color), (int(x), int(y)), size, 2)


PHOTO_CACHE = {}


def draw_photo(surface, x, y, size, photo_image, rotation=0):
    """绘制照片（使用预生成的图片），带缩放缓存减轻卡顿"""
    if photo_image is None:
        return

    try:
        scaled_size = max(10, min(int(size), 400))  # 减小图片最大尺寸到400
        cache_key = (id(photo_image), scaled_size)
        scaled_img = PHOTO_CACHE.get(cache_key)
        if scaled_img is None:
            scaled_img = pygame.transform.smoothscale(
                photo_image, (scaled_size, int(scaled_size * 1.2)))
            PHOTO_CACHE[cache_key] = scaled_img

        # 旋转比较耗时，散开时不要旋转，减少卡顿
        # 优化：只在旋转角度大于1度时才旋转，避免不必要的计算
        if abs(rotation) > 1:
            rotated_img = pygame.transform.rotate(scaled_img, rotation)
        else:
            rotated_img = scaled_img

        rect = rotated_img.get_rect(center=(int(x), int(y)))
        surface.blit(rotated_img, rect)
    except Exception:
        cos_r = math.cos(math.radians(rotation))
        sin_r = math.sin(math.radians(rotation))
        w, h = size * 1.2, size * 0.9
        points = [
            (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)
        ]
        rotated_points = []
        for px, py in points:
            rx = px * cos_r - py * sin_r
            ry = px * sin_r + py * cos_r
            rotated_points.append((x + rx, y + ry))
        pygame.draw.polygon(surface, (240, 240, 240), rotated_points)
        pygame.draw.polygon(surface, (180, 180, 180), rotated_points, 2)


def draw_star(surface, x, y, size, rotation=0, glow_intensity=1.0):
    """绘制发光的星星（可旋转）"""
    # 发光效果（多层光圈）
    for i in range(4, 0, -1):
        alpha = int(80 * glow_intensity / i)
        if alpha > 0:
            glow_surf = pygame.Surface(
                (size * i * 2, size * i * 2), pygame.SRCALPHA)  # type: ignore
            pygame.draw.circle(glow_surf, (*YELLOW, alpha),
                               (size * i, size * i), size * i)
            surface.blit(glow_surf, (x - size * i, y - size * i))

    # 绘制钻石形状的星星（一个角向上）
    base_points = [
        (0, -size),           # 上
        (size * 0.6, 0),      # 右
        (0, size),            # 下
        (-size * 0.6, 0)      # 左
    ]

    # 旋转点
    cos_r = math.cos(math.radians(rotation))
    sin_r = math.sin(math.radians(rotation))

    rotated_points = []
    for px, py in base_points:
        rx = px * cos_r - py * sin_r
        ry = px * sin_r + py * cos_r
        rotated_points.append((x + rx, y + ry))

    # 绘制主体
    pygame.draw.polygon(surface, YELLOW, rotated_points)
    # 绘制边框
    pygame.draw.polygon(surface, GOLD, rotated_points, 2)


def draw_glowing_sphere(surface, x, y, size, color, glow_intensity=1.0):
    """绘制闪闪发光的球体（优化性能：减少层数）"""
    # 优化：减少发光层数从5层到3层，提升性能
    for i in range(3, 0, -1):
        alpha = int(100 * glow_intensity / i)  # 增大发光强度
        if alpha > 0:
            glow_surf = pygame.Surface(
                (size * i * 2.5, size * i * 2.5), pygame.SRCALPHA)  # type: ignore
            # 使用更亮的发光颜色
            if color == RED or color == BRIGHT_RED:
                glow_color = (255, 100, 100)  # 红色球体用亮红色发光
            elif color == LIGHT_YELLOW or color == PALE_YELLOW:
                glow_color = YELLOW  # 淡黄色球体用亮黄色发光
            else:
                glow_color = tuple(min(255, c + 80) for c in color)  # 更亮的发光颜色
            pygame.draw.circle(glow_surf, (*glow_color, alpha),
                               (int(size * i * 1.25), int(size * i * 1.25)), int(size * i * 1.25))
            surface.blit(glow_surf, (x - size * i * 1.25, y - size * i * 1.25))

    # 主体（更亮）
    bright_color = tuple(min(255, c + 30) for c in color)  # 让主体更亮
    pygame.draw.circle(surface, bright_color, (int(x), int(y)), size)
    # 高光（更大更亮）
    highlight_color = tuple(min(255, c + 80) for c in color)
    pygame.draw.circle(surface, highlight_color, (int(
        x - size * 0.3), int(y - size * 0.3)), size // 2)
    # 边框（更亮）
    pygame.draw.circle(surface, tuple(min(255, c + 40)
                       for c in color), (int(x), int(y)), size, 2)

# 生成圣诞树形状的粒子位置


def generate_christmas_tree_particles():
    particles = []

    # 树的主体（圆锥形）- 优化形状让它更规整好看
    tree_height = 420  # 略高一点让比例更自然
    tree_width = 210   # 底部宽度
    tree_base_y = tree_height * 0.5  # 往下移，确保树完全在可视范围内

    # 从下到上生成层级 - 优化圆锥形状，让它更规整
    for layer in range(26):  # 更多层级让曲线更顺滑
        layer_y = tree_base_y - layer * (tree_height / 26)  # 反转，从下往上
        # 优化圆锥形状：更平滑的曲线并逐层稍微收紧
        layer_progress = layer / 26  # 0到1的进度
        width_factor = pow(1 - layer_progress, 1.2) * \
            (0.95 - 0.05 * layer_progress)
        layer_width = tree_width * width_factor

        # 每层的粒子数量（更密集，底部更多）- 优化分布，让形状更整齐
        # 底部层粒子更多，顶部层粒子更少，形成圆锥形状
        # 增加粒子数量（从140增加到180）
        base_particles = int(180 * (1 - layer_progress * 0.7))
        particles_per_layer = max(10, base_particles)  # 增加最小粒子数（从8增加到10）

        # 使用更整齐的分布方式
        for i in range(particles_per_layer):
            # 角度分布更均匀，减少随机性
            angle_step = 2 * math.pi / particles_per_layer
            angle = i * angle_step + random.uniform(-0.06, 0.06)  # 减小角度随机性

            # 使用分层半径分布，让形状更规整
            # 将半径分成几个区域，让粒子分布更均匀
            radius_zone = (i % 3) / 3.0  # 0, 0.33, 0.66三个区域
            radius_base = layer_width * (0.35 + radius_zone * 0.55)  # 35%到90%
            radius = radius_base + \
                random.uniform(-layer_width * 0.08, layer_width * 0.08)
            radius = max(0, min(radius, layer_width * 0.96))  # 限制在范围内

            x = math.cos(angle) * radius
            z = math.sin(angle) * radius
            y = layer_y + random.uniform(-3, 3)  # 更小的Y轴偏移让层次更整齐

            # 随机分配粒子类型 - 进一步减少照片比例，避免过多
            rand_val = random.random()
            if rand_val < 0.06:  # 6% 照片（再减少一点）
                ptype = PARTICLE_PHOTO
            elif rand_val < 0.84:  # 78% 球体
                ptype = PARTICLE_SPHERE
            else:  # 16% 立方体
                ptype = PARTICLE_CUBE

            particles.append((x, y, z, ptype))

    # 添加装饰球（红色和黄色球体）- 增加数量
    for _ in range(220):  # 增加装饰球数量（从180增加到220）
        layer = random.randint(2, 22)
        layer_progress = layer / 24
        layer_y = tree_base_y - layer * (tree_height / 24)  # 反转Y轴
        width_factor = pow(1 - layer_progress, 1.5)
        layer_width = tree_width * width_factor * 0.98
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(layer_width * 0.4, layer_width * 0.95)
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        y = layer_y + random.uniform(-8, 8)
        particles.append((x, y, z, PARTICLE_SPHERE))

    # 添加照片元素 - 少量
    for _ in range(100):  # 稍微增加照片数量（从80增加到100）
        layer = random.randint(1, 22)
        layer_progress = layer / 24
        layer_y = tree_base_y - layer * (tree_height / 24)  # 反转Y轴
        width_factor = pow(1 - layer_progress, 1.5)
        layer_width = tree_width * width_factor * 0.98
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(layer_width * 0.2, layer_width * 0.8)
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        y = layer_y + random.uniform(-8, 8)
        particles.append((x, y, z, PARTICLE_PHOTO))

    return particles


# 初始化
camera = Camera()
particle_positions = generate_christmas_tree_particles()
particles = []

# 预热照片缓存（预加载常用尺寸的照片，避免第一次散开时卡顿）


def prewarm_photo_cache():
    """预热照片缓存，预加载常用尺寸的照片（优化：异步预热，不阻塞启动）"""
    print("预热照片缓存...")
    if PHOTO_IMAGES:
        # 优化：只预热最常用的尺寸，减少预热时间
        # 散开时最常用的尺寸范围
        # 优化：只预热最关键的尺寸，减少预热时间
        common_sizes = [100, 150, 200, 250, 300, 350, 400]  # 减小预热尺寸范围
        # 只预热部分照片（前15张），避免启动太慢
        photos_to_warm = PHOTO_IMAGES[:15] if len(
            PHOTO_IMAGES) > 15 else PHOTO_IMAGES
        for photo_img in photos_to_warm:
            for size in common_sizes:
                cache_key = (id(photo_img), size)
                if cache_key not in PHOTO_CACHE:
                    try:
                        # 使用scale而不是smoothscale，更快但质量稍低（首次散开时再使用smoothscale）
                        scaled_img = pygame.transform.scale(
                            photo_img, (size, int(size * 1.2)))
                        PHOTO_CACHE[cache_key] = scaled_img
                    except Exception:
                        continue
    print(f"照片缓存预热完成，缓存了 {len(PHOTO_CACHE)} 张图片")


# 执行预热
prewarm_photo_cache()

# 创建粒子
star_particle = None
print(f"生成 {len(particle_positions)} 个粒子位置")
for i, pos in enumerate(particle_positions):
    # 初始位置就是目标位置（聚合状态）
    particle = Particle3D(
        pos[0], pos[1], pos[2],  # 初始位置 = 目标位置（聚合状态）
        pos[0], pos[1], pos[2],  # 目标位置
        pos[3],  # 粒子类型
        size=random.randint(3, 7)  # 进一步减小粒子尺寸，让颗粒更细
    )
    particles.append(particle)

# 添加顶部星星 - 调整位置匹配新的树位置，添加旋转
tree_height = 400
tree_base_y = tree_height * 0.5
star_y = tree_base_y - tree_height + 20  # 在树顶内部
star_particle = Particle3D(0, star_y, 0, 0, star_y,
                           0, PARTICLE_CUBE, size=15)  # 减小星星尺寸
star_rotation = 0  # 星星旋转角度
star_rotation_speed = 2  # 星星旋转速度
star_glow_intensity = 1.0  # 星星发光强度
star_glow_direction = 1  # 发光强度变化方向

# 动画状态
# 状态：0=聚合保持, 1=散开中, 2=散开保持, 3=聚合中
animation_state = 0  # 初始状态：聚合保持
animation_t = 0.0
animation_speed = 0.032  # 动画速度（加快散开速度，减少等待时间）
hold_time = 0.0  # 保持时间
hold_duration_aggregate = 5.0  # 聚合保持5秒后开始散开
hold_duration_scatter_first = 8.0  # 第一次散开（scatter_cycle=0）保持8秒
hold_duration_scatter_text = 3.0  # 文字模式（scatter_cycle=1或2）保持3秒
clock = pygame.time.Clock()

# 散开模式循环：0=正常散开, 1="LOVE", 2="happy"
scatter_cycle = 0

# 记录是否已经散开过（用于控制顶点显示）
has_scattered = False  # 一旦开始散开，就不再显示顶点

# 流星雨系统
snowflakes = []
snowflake_spawn_timer = 0.0
snowflake_spawn_interval = 0.03  # 每0.03秒生成一个雪花（增加生成频率）
max_snowflakes = 200  # 最大雪花数量（增加到200）

# 文字点阵定义（行排列）


def generate_text_positions(text_pattern, center_x=0, center_y=0, center_z=0, spacing=20):
    """根据文字点阵生成粒子位置列表（横排）
    text_pattern: 二维列表，每行是一个列表，1表示有粒子，0表示无粒子
    行从上到下，列从左到右（横排显示）
    """
    positions = []
    rows = len(text_pattern)
    cols = max(len(row) for row in text_pattern) if text_pattern else 0

    # 计算起始位置，使文字居中
    start_x = center_x - (cols * spacing) / 2
    start_y = center_y - (rows * spacing) / 2

    # 横排：row_idx是行（从上到下），col_idx是列（从左到右）
    for row_idx, row in enumerate(text_pattern):
        # 直接使用原始行数据，不反转（避免镜像）
        for col_idx, cell in enumerate(row):
            if cell == 1:
                x = start_x + col_idx * spacing  # 列从左到右
                y = start_y + row_idx * spacing  # 行从上到下
                z = center_z
                positions.append((x, y, z))
    return positions


# LOVE - 横排点阵（11行x38列，4个字母横排，每个字母9x9，字母间距1列）
text_christmas_simple = [
    # L O V E (修复V字母：V应该是两个斜线在底部相交，不是Y形状)
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
        1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
        1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
        1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
        1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
        1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],  # E中间横线
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
        0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],  # E底部横线，V底部相交
]

# 定义"HAPPY"的点阵（5个字母横排，11行x49列，每个字母9x9，字母间距1列）
# 重新设计清晰的HAPPY点阵，确保每个字母都清晰可辨
text_happy = [
    # H A P P Y - 清晰的字母形状
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],  # 行0: H顶部, A顶部, P顶部, P顶部, Y左边+右边（分叉顶部）
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0,
        0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],  # 行1: H左边+右边, A左边+右边, P左边+右边, P左边+右边, Y分叉（斜线）
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1,
        1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0],  # 行2: Y分叉（斜线继续）
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 行3: H中间横线, A中间横线, P顶部横线, P顶部横线, Y分叉点（相交）
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 行4: H左边+右边, A左边+右边, P中间横线, P中间横线, Y竖线
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 行5: H左边+右边, A左边+右边, P只有左边, P只有左边, Y竖线
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 行6: P只有左边
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 行7: P只有左边
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 行6: H左边+右边, A左边+右边, P只有左边, P只有左边, Y竖线
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 行7: P只有左边, Y竖线
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 行8: P只有左边, Y竖线
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 行9: P只有左边, Y竖线
    [1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 行10: P只有左边（底部没有横线）, Y竖线
]

# 主循环
running = True
auto_animate = True

while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # 手动切换状态
                if animation_state == 0 or animation_state == 2:
                    animation_state = (animation_state + 1) % 4
                animation_t = 0.0
                hold_time = 0.0
            elif event.key == pygame.K_a:
                auto_animate = not auto_animate
            elif event.key == pygame.K_LEFT:
                camera.angle_y -= 0.1
            elif event.key == pygame.K_RIGHT:
                camera.angle_y += 0.1
            elif event.key == pygame.K_UP:
                camera.angle_x -= 0.1
            elif event.key == pygame.K_DOWN:
                camera.angle_x += 0.1

    # 自动动画状态机
    if auto_animate:
        if animation_state == 0:  # 聚合保持状态
            hold_time += dt
            if hold_time >= hold_duration_aggregate:
                # 散开开始时，根据scatter_cycle分配文字位置
                print(f"散开开始，scatter_cycle={scatter_cycle}")  # 调试用
                if scatter_cycle == 1:  # "LOVE"
                    text_positions = generate_text_positions(
                        text_christmas_simple, 0, 0, 0, 25)  # 增大间距，让文字更舒适
                    print(
                        f"生成文字位置数量: {len(text_positions)}, 粒子总数: {len(particles)}")
                    # 使用所有粒子来拼成文字，多个粒子可以共享一个位置形成密集效果
                    available_particles = list(particles)
                    random.shuffle(available_particles)
                    # 分配所有粒子到文字位置（循环使用位置，形成密集文字）
                    assigned_count = 0
                    offset_range = 8  # 偏移范围，避免完全重叠
                    for i, p in enumerate(available_particles):
                        # 循环使用文字位置，让多个粒子共享位置形成密集效果
                        pos_idx = i % len(text_positions)
                        pos = text_positions[pos_idx]
                        # 添加小幅随机偏移，形成密集效果
                        p.text_target_x = pos[0] + \
                            random.uniform(-offset_range, offset_range)
                        p.text_target_y = pos[1] + \
                            random.uniform(-offset_range, offset_range)
                        p.text_target_z = pos[2] + \
                            random.uniform(-offset_range, offset_range)
                        assigned_count += 1
                    print(f"分配了 {assigned_count} 个粒子到文字位置")
                elif scatter_cycle == 2:  # "happy"
                    text_positions = generate_text_positions(
                        text_happy, 0, 0, 0, 25)  # 增大间距，让文字更舒适
                    print(
                        f"生成文字位置数量: {len(text_positions)}, 粒子总数: {len(particles)}")
                    # 使用所有粒子来拼成文字，多个粒子可以共享一个位置形成密集效果
                    available_particles = list(particles)
                    random.shuffle(available_particles)
                    # 分配所有粒子到文字位置（循环使用位置，形成密集文字）
                    assigned_count = 0
                    offset_range = 8  # 偏移范围，避免完全重叠
                    for i, p in enumerate(available_particles):
                        # 循环使用文字位置，让多个粒子共享位置形成密集效果
                        pos_idx = i % len(text_positions)
                        pos = text_positions[pos_idx]
                        # 添加小幅随机偏移，形成密集效果
                        p.text_target_x = pos[0] + \
                            random.uniform(-offset_range, offset_range)
                        p.text_target_y = pos[1] + \
                            random.uniform(-offset_range, offset_range)
                        p.text_target_z = pos[2] + \
                            random.uniform(-offset_range, offset_range)
                        assigned_count += 1
                    print(f"分配了 {assigned_count} 个粒子到文字位置")
                else:  # scatter_cycle == 0，正常散开
                    # 清除所有文字目标
                    for p in particles:
                        p.text_target_x = None
                        p.text_target_y = None
                        p.text_target_z = None

                animation_state = 1  # 切换到散开中
                animation_t = 0.0
                hold_time = 0.0
                # 标记已经散开过，之后不再显示顶点
                has_scattered = True
                # 第一次散开前，延迟排序到第一帧渲染时，避免阻塞状态切换
                if not hasattr(camera, '_sorted_particles_cache'):
                    camera._sort_frame_count = 0
        elif animation_state == 1:  # 散开中
            animation_t += animation_speed
            if animation_t >= 1.0:
                animation_t = 1.0
                animation_state = 2  # 切换到散开保持
                hold_time = 0.0
        elif animation_state == 2:  # 散开保持状态
            hold_time += dt
            # 根据scatter_cycle动态调整保持时间
            # scatter_cycle=0（第一次散开）保持8秒，其他（文字模式）保持3秒
            current_hold_duration = hold_duration_scatter_first if scatter_cycle == 0 else hold_duration_scatter_text
            if hold_time >= current_hold_duration:
                animation_state = 3  # 切换到聚合中
                animation_t = 0.0
                hold_time = 0.0
        elif animation_state == 3:  # 聚合中
            animation_t += animation_speed
            if animation_t >= 1.0:
                animation_t = 1.0
                # 循环散开模式：0->1->2->0（在聚合完成前更新，确保下次散开时使用新值）
                scatter_cycle = (scatter_cycle + 1) % 3
                print(f"聚合完成，scatter_cycle更新为={scatter_cycle}")  # 调试用
                animation_state = 0  # 切换到聚合保持
                hold_time = 0.0

    # 更新粒子
    scatter_mode = (animation_state == 1 or animation_state == 2)  # 散开中或散开保持

    # 如果是聚合保持状态，直接设置粒子到目标位置
    if animation_state == 0:
        for particle in particles:
            particle.x = particle.target_x
            particle.y = particle.target_y
            particle.z = particle.target_z
            # 文字模式下不旋转
            if scatter_cycle == 0:
                particle.rotation_x += particle.rotation_speed_x
                particle.rotation_y += particle.rotation_speed_y
                particle.rotation_z += particle.rotation_speed_z
    # 如果是散开保持状态，保持散开位置（包括文字位置）
    elif animation_state == 2:
        for particle in particles:
            # 如果有文字目标，保持在文字位置；否则保持在散开位置
            if particle.text_target_x is not None:
                # 保持在文字位置，文字模式下不旋转
                particle.x = particle.text_target_x
                particle.y = particle.text_target_y
                particle.z = particle.text_target_z
                # 文字模式下停止旋转
            else:
                # 保持在散开位置
                particle.update(1.0, True)  # 使用t=1.0保持最大散开状态
                # 正常散开模式下继续旋转
                if scatter_cycle == 0:
                    particle.rotation_x += particle.rotation_speed_x
                    particle.rotation_y += particle.rotation_speed_y
                    particle.rotation_z += particle.rotation_speed_z
    else:
        # 散开中或聚合中，使用动画插值
        # 优化：首次散开时，延迟更新粒子，让第一帧先渲染
        for particle in particles:
            if animation_state == 1 and scatter_cycle == 0 and not has_scattered and animation_t < 0.05:
                # 首次散开的前几帧，延迟更新，避免卡顿
                # 只快速更新位置，不更新旋转
                if particle.text_target_x is None:
                    # 快速更新位置（简化计算）
                    ease_t = animation_t * animation_t * (3 - 2 * animation_t)
                    max_scatter_distance = 600
                    scatter_distance = max_scatter_distance * \
                        ease_t * particle.scatter_distance_factor
                    sin_phi = math.sin(particle.scatter_phi)
                    cos_phi = math.cos(particle.scatter_phi)
                    cos_theta = math.cos(particle.scatter_theta)
                    sin_theta = math.sin(particle.scatter_theta)
                    scatter_x_offset = scatter_distance * sin_phi * cos_theta
                    scatter_y_offset = scatter_distance * sin_phi * sin_theta
                    scatter_z_offset = scatter_distance * cos_phi * 1.5
                    if particle.particle_type == PARTICLE_PHOTO:
                        scatter_x_offset *= 0.7
                        scatter_y_offset *= 0.7
                        scatter_z_offset *= 0.9
                    max_x_offset, max_y_offset, max_z_offset = 600, 400, 500
                    scatter_x_offset = max(-max_x_offset,
                                           min(max_x_offset, scatter_x_offset))
                    scatter_y_offset = max(-max_y_offset,
                                           min(max_y_offset, scatter_y_offset))
                    scatter_z_offset = max(-max_z_offset,
                                           min(max_z_offset, scatter_z_offset))
                    particle.x = particle.target_x + scatter_x_offset
                    particle.y = particle.target_y + scatter_y_offset
                    particle.z = particle.target_z + scatter_z_offset
            else:
                particle.update(animation_t, scatter_mode)
            # 文字模式下不旋转
            if scatter_cycle > 0 and particle.text_target_x is not None:
                # 文字模式下停止旋转，重置旋转角度
                particle.rotation_x = 0
                particle.rotation_y = 0
                particle.rotation_z = 0
            elif scatter_cycle == 0:
                # 正常模式下继续旋转
                particle.rotation_x += particle.rotation_speed_x
                particle.rotation_y += particle.rotation_speed_y
                particle.rotation_z += particle.rotation_speed_z

    # 更新星星旋转和发光效果
    star_rotation += star_rotation_speed
    if star_rotation >= 360:
        star_rotation -= 360

    # 更新星星发光强度（呼吸效果）
    star_glow_intensity += star_glow_direction * 0.02
    if star_glow_intensity >= 1.3:
        star_glow_direction = -1
    elif star_glow_intensity <= 0.7:
        star_glow_direction = 1

    # 更新流星雨（只在非第一次散开时显示）
    # 第一次散开：scatter_cycle == 0 且 animation_state == 1 或 2
    # 其他状态：显示流星雨
    show_snowflakes = not (scatter_cycle == 0 and (
        animation_state == 1 or animation_state == 2))

    if show_snowflakes:
        # 生成新雪花
        snowflake_spawn_timer += dt
        if snowflake_spawn_timer >= snowflake_spawn_interval and len(snowflakes) < max_snowflakes:
            snowflakes.append(Snowflake())
            snowflake_spawn_timer = 0.0

        # 更新所有雪花
        for snowflake in snowflakes:
            snowflake.update(dt)
    else:
        # 第一次散开时，清除所有雪花
        snowflakes.clear()
        snowflake_spawn_timer = 0.0

    # 旋转相机（自动）- 稍微向下看一点
    # 文字模式下不旋转相机，使文字平行屏幕
    if scatter_cycle > 0 and (animation_state == 1 or animation_state == 2):
        # 文字形成时，重置相机角度，使文字平行屏幕
        camera.angle_y = 0
        camera.angle_x = 0
    else:
        # 正常模式下继续旋转
        camera.angle_y += 0.003
        camera.angle_x = 0.1  # 调整角度，让树正立显示

    # 绘制
    screen.fill(BLACK)

    # 按Z轴深度排序（优化：使用更高效的排序，减少卡顿）
    # 只在散开模式下排序，聚合模式下可以跳过
    if scatter_mode and animation_state == 1:  # 只在散开中时排序
        # 优化：减少排序频率，每4帧排序一次
        if not hasattr(camera, '_sort_frame_count'):
            camera._sort_frame_count = 0
        # 第一次散开时，延迟排序，避免卡顿
        if not hasattr(camera, '_sorted_particles_cache'):
            # 优化：首次散开时不立即排序，延迟到第二帧
            # 第一帧使用简单的顺序渲染，不排序
            camera._sort_frame_count = 0
            camera._sorted_particles_cache = particles  # 暂时使用原始顺序
        elif camera._sort_frame_count == 0:
            # 第二帧开始排序（此时粒子已经更新过位置）
            # 只快速计算z值用于排序
            for p in particles:
                if p.text_target_x is None:  # 只处理正常散开的粒子
                    # 快速计算散开位置（只计算z值用于排序）
                    ease_t = animation_t * animation_t * (3 - 2 * animation_t)
                    max_scatter_distance = 600
                    scatter_distance = max_scatter_distance * ease_t * p.scatter_distance_factor
                    cos_phi = math.cos(p.scatter_phi)
                    scatter_z_offset = scatter_distance * cos_phi * 1.5
                    if p.particle_type == PARTICLE_PHOTO:
                        scatter_z_offset *= 0.9
                    p.z = p.target_z + scatter_z_offset
            camera._sorted_particles_cache = sorted(
                particles, key=lambda p: p.z, reverse=True)
        camera._sort_frame_count += 1
        if camera._sort_frame_count % 4 == 0:  # 每4帧排序一次（提高性能）
            sorted_particles = sorted(
                particles, key=lambda p: p.z, reverse=True)
            camera._sorted_particles_cache = sorted_particles
        else:
            sorted_particles = getattr(
                camera, '_sorted_particles_cache', particles)
    elif scatter_mode:
        # 散开保持状态，使用缓存的排序结果
        sorted_particles = getattr(
            camera, '_sorted_particles_cache', particles)
    else:
        sorted_particles = particles  # 聚合模式下不需要排序
        # 聚合时清除排序缓存，下次散开时重新初始化
        if hasattr(camera, '_sorted_particles_cache'):
            delattr(camera, '_sorted_particles_cache')
        if hasattr(camera, '_sort_frame_count'):
            delattr(camera, '_sort_frame_count')

    particle_count = 0
    visible_count = 0
    min_scale = 999
    max_scale = 0

    for particle in sorted_particles:
        screen_x, screen_y, scale, depth = camera.project(
            particle.x, particle.y, particle.z
        )
        # 优化：使用整数坐标，避免模糊
        screen_x = int(screen_x + 0.5)  # 快速四舍五入
        screen_y = int(screen_y + 0.5)

        particle_count += 1

        # 记录scale范围
        if scale > 0:
            min_scale = min(min_scale, scale)
            max_scale = max(max_scale, scale)

        # 放宽scale限制，允许更大的范围
        if scale > 0.2 and scale < 30:  # 放宽scale范围，让远近距离的粒子都能显示
            # 检查是否在屏幕范围内（大幅放宽边界以便看到散开的粒子）
            if -500 <= screen_x <= WIDTH + 500 and -500 <= screen_y <= HEIGHT + 500:
                # 计算缩放后的尺寸（进一步减小尺寸，让颗粒更细）
                # 优化：使用更精确的整数计算，避免模糊
                scaled_size = max(
                    2, min(int(particle.size * scale + 0.5), 25))  # 快速四舍五入

                if scaled_size > 0:
                    try:
                        visible_count += 1
                        effective_t = 1.0 if animation_state == 2 else animation_t
                        # 根据粒子类型绘制
                        if particle.particle_type == PARTICLE_CUBE:
                            # 文字模式下不旋转
                            rotation = 0 if (
                                scatter_cycle > 0 and particle.text_target_x is not None) else particle.rotation_z
                            draw_cube(screen, screen_x, screen_y, scaled_size,
                                      particle.color, rotation)
                        elif particle.particle_type == PARTICLE_SPHERE:
                            # 文字模式下不发光，正常模式下才发光
                            # 散开时，如果粒子会发光，使用发光球体
                            # 只有在散开进度达到一半（0.5）时才开始发光
                            # 在散开保持状态时，使用1.0作为animation_t
                            if scatter_mode and scatter_cycle == 0 and particle.is_glowing and effective_t >= 0.5:
                                # 计算发光强度（更明显的闪烁效果）
                                # 使用多个正弦波叠加，产生更复杂的闪烁效果
                                # 将effective_t从0.5-1.0映射到0-1.0用于闪烁计算
                                # 0.5->0, 1.0->1.0
                                normalized_t = (effective_t - 0.5) * 2.0
                                time_factor = normalized_t * 15  # 加快闪烁频率
                                glow_intensity = 0.7 + 0.3 * \
                                    (math.sin(time_factor) * 0.5 + 0.5)
                                # 添加快速闪烁效果
                                quick_flash = 0.2 * math.sin(time_factor * 3)
                                glow_intensity = min(
                                    1.5, max(0.5, glow_intensity + quick_flash))
                                draw_glowing_sphere(screen, screen_x, screen_y, scaled_size,
                                                    particle.color, glow_intensity)
                            else:
                                draw_sphere(screen, screen_x, screen_y, scaled_size,
                                            particle.color)
                        elif particle.particle_type == PARTICLE_PHOTO:
                            # 文字模式下：照片不放大，不旋转，保持正常大小
                            if scatter_cycle > 0 and particle.text_target_x is not None:
                                # 文字模式：照片保持正常大小，不旋转
                                photo_scaled_size = max(
                                    2, min(int(particle.size * scale), 25))
                                draw_photo(screen, screen_x, screen_y, photo_scaled_size,
                                           particle.photo_image, 0)  # 不旋转
                            else:
                                # 正常模式：照片随深度透视：远小近大，散开时再额外放大
                                base_factor = 1.0
                                if scatter_mode:
                                    # 使用平滑的缓动函数，让图片放大更丝滑
                                    eased_t = effective_t * effective_t * \
                                        (3 - 2 * effective_t)  # smoothstep缓动
                                    base_factor = 1.0 + 2.5 * eased_t  # 最大约3.5倍（从2.8倍增加）

                                # 如果是大图，只在散开时才放大3倍，聚合时保持正常大小
                                size_multiplier = 1.0
                                if particle.is_large_photo and scatter_mode:
                                    # 使用平滑的缓动函数，让大图放大更丝滑
                                    eased_t = effective_t * effective_t * \
                                        (3 - 2 * effective_t)  # smoothstep缓动
                                    size_multiplier = 1.0 + 2.0 * eased_t  # 从1倍平滑放大到3倍

                                photo_scaled_size = max(
                                    10, min(int(particle.size * scale * base_factor * size_multiplier), 400 * size_multiplier))  # 减小图片最大尺寸到400
                                draw_photo(screen, screen_x, screen_y, photo_scaled_size,
                                           particle.photo_image, particle.rotation_z)
                    except Exception as e:
                        # 打印错误以便调试
                        # print(f"绘制错误: {e}")
                        continue

    # 绘制顶部星星（只在未散开过且未形成文字时显示）
    # 显示条件：
    # 1. 从未散开过（has_scattered == False）
    # 2. animation_state == 0（聚合保持）或 animation_state == 3（聚合中）
    # 3. scatter_cycle == 0（正常模式，不是文字模式）
    if not has_scattered and (animation_state == 0 or animation_state == 3) and scatter_cycle == 0:
        # 绘制发光的星星
        star_screen_x, star_screen_y, star_scale, _ = camera.project(
            star_particle.target_x, star_particle.target_y, star_particle.target_z
        )
        if star_scale > 0.3:
            star_size = max(10, int(15 * star_scale))  # 减小星星显示尺寸
            if star_size > 0:
                draw_star(screen, star_screen_x, star_screen_y,
                          star_size, star_rotation, star_glow_intensity)

    # 绘制流星雨（只在非第一次散开时显示）
    if show_snowflakes:
        for snowflake in snowflakes:
            snowflake.draw(screen)

    pygame.display.flip()

pygame.quit()
