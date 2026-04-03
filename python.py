"""
Europa Lander Simulator
Точная физическая модель посадки на Европу (Юпитер IV)
Python 3.8+ + Pygame
"""

import pygame
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, List

# Инициализация Pygame
pygame.init()


# Константы физики Европы (научно достоверные)
class Physics:
    G_EUROPA = 1.314  # м/с² ускорение свободного падения
    R_EUROPA = 1.561e6  # м, радиус Европы (для поправки гравитации с высотой)
    MAX_THRUST = 12000  # Н, максимальная тяга двигателя (12 кН)
    ISP = 300  # с, удельный импульс (UDMH/NTO)
    G0 = 9.80665  # м/с², стандартная гравитация Земли

    # Массовые характеристики
    DRY_MASS = 500  # кг, сухая масса (конструкция + полезная нагрузка)
    MAX_FUEL = 1500  # кг, максимум топлива
    RTG_POWER = 200  # Вт, мощность РИТЭГа

    # Радиация (модель GIRE упрощенно)
    SURFACE_RADIATION = 0.00625  # рад/с (540 бэр/сутки)
    ORBIT_RADIATION = 0.1  # рад/с на высокой орбите
    SCALE_HEIGHT_RAD = 50000  # м, высотный масштаб радиации
    RAD_THRESHOLD = 1e6  # рад, отказ электроники
    RAD_SHIELD_EFFICIENCY = 0.5  # коэффициент ослабления защитой

    # Термодинамика
    TEMP_SURFACE = -160  # °C
    TEMP_MIN_OPERATING = -180  # °C
    TEMP_MAX_OPERATING = 50  # °C
    ENGINE_HEAT_COEF = 0.5  # °C на % тяги

    # Посадочные критерии
    MAX_V_VERTICAL = 3.0  # м/с
    MAX_V_HORIZONTAL = 2.0  # м/с
    MAX_SLOPE = 15.0  # градусов

    # Начальные условия
    START_ALTITUDE = 10000  # м
    START_VX = 30  # м/с
    START_VY = -50  # м/с (снижение)


@dataclass
class Vector2D:
    x: float
    y: float

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)

    def magnitude(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)


class Terrain:
    """Генерация рельефа Европы с трещинами и хребтами"""

    def __init__(self, width: float = 10000, resolution: int = 500):
        self.width = width
        self.resolution = resolution
        self.points = []
        self.features = []  # Трещины и хребты
        self.generate()

    def generate(self):
        """Процедурная генерация ледяной поверхности"""
        dx = self.width / self.resolution
        np.random.seed(42)  # Воспроизводимость

        # Базовый шум (Perlin-like)
        x_vals = np.linspace(0, self.width, self.resolution)
        heights = np.zeros(self.resolution)

        # Низкочастотные компоненты (холмы)
        heights += 100 * np.sin(x_vals * 0.001)
        heights += 50 * np.sin(x_vals * 0.003 + 1)

        # Высокочастотный шум (неровности)
        heights += 20 * np.sin(x_vals * 0.02)
        heights += np.random.normal(0, 5, self.resolution)

        # Добавляем трещины (cracks) - резкие провалы
        num_cracks = 5
        for _ in range(num_cracks):
            crack_x = np.random.uniform(1000, self.width - 1000)
            crack_width = np.random.uniform(50, 150)
            crack_depth = np.random.uniform(80, 200)
            self.features.append({'type': 'crack', 'x': crack_x,
                                  'width': crack_width, 'depth': crack_depth})

            # Применяем к высотам
            mask = np.abs(x_vals - crack_x) < crack_width / 2
            depths = crack_depth * np.exp(-((x_vals[mask] - crack_x) / (crack_width / 3)) ** 2)
            heights[mask] -= depths

        # Хребты (ridges) - повышения
        num_ridges = 8
        for _ in range(num_ridges):
            ridge_x = np.random.uniform(500, self.width - 500)
            ridge_width = np.random.uniform(200, 500)
            ridge_height = np.random.uniform(30, 80)
            mask = np.abs(x_vals - ridge_x) < ridge_width / 2
            heights[mask] += ridge_height * np.exp(-((x_vals[mask] - ridge_x) / (ridge_width / 3)) ** 2)

        self.points = [(x, y) for x, y in zip(x_vals, heights)]

    def get_height(self, x: float) -> float:
        """Получить высоту рельефа в точке x (линейная интерполяция)"""
        if x <= 0:
            return self.points[0][1]
        if x >= self.width:
            return self.points[-1][1]

        # Бинарный поиск или линейный для малых массивов
        for i in range(len(self.points) - 1):
            x1, y1 = self.points[i]
            x2, y2 = self.points[i + 1]
            if x1 <= x <= x2:
                t = (x - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)
        return self.points[-1][1]

    def get_slope(self, x: float) -> float:
        """Получить уклон в градусах"""
        dx = 10.0
        h1 = self.get_height(x - dx)
        h2 = self.get_height(x + dx)
        return math.degrees(math.atan2(h2 - h1, 2 * dx))


class Lander:
    """Физическая модель посадочного аппарата"""

    def __init__(self):
        self.pos = Vector2D(0, Physics.START_ALTITUDE)
        self.vel = Vector2D(Physics.START_VX, Physics.START_VY)
        self.mass = Physics.DRY_MASS + Physics.MAX_FUEL
        self.fuel = Physics.MAX_FUEL

        # Управление
        self.thrust_level = 0.0  # 0.0 - 1.0
        self.angle = 0.0  # градусы, 0 = вертикально вверх
        self.radiation_shield = False

        # Состояния
        self.radiation_dose = 0.0  # рад
        self.temperature = Physics.TEMP_SURFACE + 20  # начальная температура систем
        self.landed = False
        self.crashed = False
        self.landing_reason = ""

        # История для графиков
        self.trajectory = []

    def get_gravity(self, altitude: float) -> float:
        """Гравитация с учетом высоты (закон всемирного тяготения)"""
        r = Physics.R_EUROPA + altitude
        return Physics.G_EUROPA * ((Physics.R_EUROPA / r) ** 2)

    def get_radiation_rate(self, altitude: float) -> float:
        """Скорость накопления радиации в зависимости от высоты"""
        # Экспоненциальное убывание с высотой (упрощенная модель)
        rate = Physics.SURFACE_RADIATION + \
               (Physics.ORBIT_RADIATION - Physics.SURFACE_RADIATION) * \
               math.exp(-altitude / Physics.SCALE_HEIGHT_RAD)

        if self.radiation_shield:
            rate *= Physics.RAD_SHIELD_EFFICIENCY
        return rate

    def compute_derivatives(self, state: np.ndarray, terrain: Terrain) -> np.ndarray:
        """
        Вычисление производных для интегрирования Рунге-Кутты
        state = [x, y, vx, vy, mass]
        """
        x, y, vx, vy, mass = state

        # Высота над поверхностью
        terrain_h = terrain.get_height(x)
        altitude = max(0, y - terrain_h)

        # Гравитация
        g = self.get_gravity(altitude)

        # Сила тяги
        thrust_force = Physics.MAX_THRUST * self.thrust_level
        angle_rad = math.radians(self.angle)

        # Компоненты тяги (угол от вертикали)
        # angle > 0 -> тяга влево -> движение вправо
        thrust_x = -thrust_force * math.sin(angle_rad)
        thrust_y = thrust_force * math.cos(angle_rad)

        # Ускорения
        ax = thrust_x / mass if mass > 0 else 0
        ay = (thrust_y / mass - g) if mass > 0 else -g

        # Расход топлива: dm/dt = -F / (Isp * g0)
        if self.fuel > 0 and self.thrust_level > 0:
            dm = -thrust_force / (Physics.ISP * Physics.G0)
        else:
            dm = 0

        return np.array([vx, vy, ax, ay, dm])

    def update(self, dt: float, terrain: Terrain):
        """Обновление состояния методом Рунге-Кутты 4-го порядка"""
        if self.landed or self.crashed:
            return

        # Сохранение траектории для визуализации
        if len(self.trajectory) == 0 or \
                math.hypot(self.pos.x - self.trajectory[-1][0],
                           self.pos.y - self.trajectory[-1][1]) > 10:
            self.trajectory.append((self.pos.x, self.pos.y))
            if len(self.trajectory) > 500:
                self.trajectory.pop(0)

        # RK4 интегрирование
        state = np.array([self.pos.x, self.pos.y, self.vel.x, self.vel.y, self.mass])

        k1 = self.compute_derivatives(state, terrain)
        k2 = self.compute_derivatives(state + 0.5 * dt * k1, terrain)
        k3 = self.compute_derivatives(state + 0.5 * dt * k2, terrain)
        k4 = self.compute_derivatives(state + dt * k3, terrain)

        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self.pos.x, self.pos.y = new_state[0], new_state[1]
        self.vel.x, self.vel.y = new_state[2], new_state[3]

        # Обновление массы и топлива
        fuel_consumed = self.mass - new_state[4]
        if fuel_consumed > 0:
            self.fuel = max(0, self.fuel - fuel_consumed)
            self.mass = Physics.DRY_MASS + self.fuel
        else:
            self.mass = new_state[4]

        # Радиация
        terrain_h = terrain.get_height(self.pos.x)
        altitude = max(0, self.pos.y - terrain_h)
        rad_rate = self.get_radiation_rate(altitude)
        self.radiation_dose += rad_rate * dt

        # Термодинамика (упрощенная модель)
        # Тепловыделение двигателя + РИТЭГ - теплоотдача в космос
        engine_heat = self.thrust_level * Physics.ENGINE_HEAT_COEF * 100
        target_temp = Physics.TEMP_SURFACE + 20 + engine_heat
        # Тепловая инерция системы
        self.temperature += (target_temp - self.temperature) * 0.1 * dt

        # Проверка критических параметров
        if self.radiation_dose > Physics.RAD_THRESHOLD:
            self.crashed = True
            self.landing_reason = "Отказ электроники от радиации"

        if self.temperature < Physics.TEMP_MIN_OPERATING:
            self.crashed = True
            self.landing_reason = "Критическое охлаждение систем"

        if self.fuel <= 0 and altitude > 10:
            self.crashed = True
            self.landing_reason = "Исчерпание топлива"

        # Проверка посадки
        if altitude <= 0:
            self.pos.y = terrain_h
            self.check_landing(terrain)

    def check_landing(self, terrain: Terrain):
        """Проверка условий посадки"""
        v_vert = abs(self.vel.y)
        v_hor = abs(self.vel.x)
        slope = abs(terrain.get_slope(self.pos.x))

        if v_vert > Physics.MAX_V_VERTICAL:
            self.crashed = True
            self.landing_reason = f"Высокая вертикальная скорость: {v_vert:.1f} м/с (max {Physics.MAX_V_VERTICAL})"
        elif v_hor > Physics.MAX_V_HORIZONTAL:
            self.crashed = True
            self.landing_reason = f"Высокая горизонтальная скорость: {v_hor:.1f} м/с"
        elif slope > Physics.MAX_SLOPE:
            self.crashed = True
            self.landing_reason = f"Крутой уклон: {slope:.1f}°"
        else:
            self.landed = True
            self.landing_reason = "Успешная посадка"
            self.vel = Vector2D(0, 0)


class Visualizer:
    """Визуализация симуляции через Pygame"""

    def __init__(self, width: int = 1400, height: int = 900):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Europa Lander Simulator - Научная модель посадки")

        self.font = pygame.font.SysFont('consolas', 14)
        self.font_big = pygame.font.SysFont('consolas', 24, bold=True)

        # Цвета
        self.COLOR_ICE = (200, 230, 255)
        self.COLOR_CRACK = (100, 150, 180)
        self.COLOR_LANDER = (200, 200, 200)
        self.COLOR_THRUST = (255, 100, 50)
        self.COLOR_TEXT = (0, 255, 100)
        self.COLOR_WARNING = (255, 200, 0)
        self.COLOR_DANGER = (255, 50, 50)
        self.COLOR_TARGET = (50, 150, 255)

        # Масштаб и камера
        self.scale = 0.1  # пикселей на метр (начальный)
        self.offset_x = width // 2
        self.offset_y = height // 2
        self.target_x = 3000  # Целевая точка посадки

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Преобразование мировых координат в экранные"""
        screen_x = int(self.offset_x + (x - self.camera_x) * self.scale)
        screen_y = int(self.offset_y - (y - self.camera_y) * self.scale)
        return (screen_x, screen_y)

    def update_camera(self, lander: Lander):
        """Обновление положения камеры (следование за аппаратом)"""
        self.camera_x = lander.pos.x
        self.camera_y = lander.pos.y + 200  # Смотрим немного выше аппарата

        # Автомасштабирование при снижении
        altitude = lander.pos.y - terrain.get_height(lander.pos.x)
        if altitude < 1000:
            self.scale = 0.5  # Крупный план
        elif altitude < 5000:
            self.scale = 0.2
        else:
            self.scale = 0.1

    def draw_terrain(self, terrain: Terrain):
        """Отрисовка ледяной поверхности Европы"""
        points = []
        for x, y in terrain.points:
            sx, sy = self.world_to_screen(x, y)
            points.append((sx, sy))

        # Заливка поверхности
        if len(points) > 1:
            # Дно экрана
            _, bottom = self.world_to_screen(0, -1000)
            poly_points = points + [(points[-1][0], bottom), (points[0][0], bottom)]

            pygame.draw.polygon(self.screen, self.COLOR_ICE, poly_points)

            # Линия горизонта
            pygame.draw.lines(self.screen, (150, 200, 230), False, points, 2)

            # Трещины
            for feature in terrain.features:
                if feature['type'] == 'crack':
                    x = feature['x']
                    y = terrain.get_height(x) - feature['depth'] / 2
                    sx, sy = self.world_to_screen(x, y)
                    w = int(feature['width'] * self.scale)
                    h = int(feature['depth'] * self.scale)
                    pygame.draw.ellipse(self.screen, self.COLOR_CRACK,
                                        (sx - w // 2, sy - h // 2, w, h))

    def draw_lander(self, lander: Lander):
        """Отрисовка аппарата с визуализацией тяги"""
        x, y = lander.pos.x, lander.pos.y
        sx, sy = self.world_to_screen(x, y)

        # Поворот
        angle_rad = math.radians(lander.angle)

        # Точки корпуса (шестиугольник)
        size = 20
        points = []
        for i in range(6):
            ang = angle_rad + i * math.pi / 3
            px = sx + size * math.cos(ang)
            py = sy + size * math.sin(ang)
            points.append((px, py))

        pygame.draw.polygon(self.screen, self.COLOR_LANDER, points)
        pygame.draw.polygon(self.screen, (100, 100, 100), points, 2)

        # Опоры
        leg_length = 30
        left_leg = (sx - size * 0.7, sy + size * 0.5)
        right_leg = (sx + size * 0.7, sy + size * 0.5)
        left_foot = (sx - size * 0.9 - leg_length * math.sin(angle_rad),
                     sy + size * 0.5 + leg_length * math.cos(angle_rad))
        right_foot = (sx + size * 0.9 - leg_length * math.sin(angle_rad),
                      sy + size * 0.5 + leg_length * math.cos(angle_rad))

        pygame.draw.line(self.screen, (150, 150, 150), left_leg, left_foot, 3)
        pygame.draw.line(self.screen, (150, 150, 150), right_leg, right_foot, 3)

        # Струя двигателя
        if lander.thrust_level > 0 and lander.fuel > 0:
            flame_length = lander.thrust_level * 50 * (1 + 0.2 * np.random.randn())
            end_x = sx - flame_length * math.sin(angle_rad)
            end_y = sy + flame_length * math.cos(angle_rad)

            # Градиент пламени
            for i in range(5):
                t = i / 5.0
                color = (255, int(200 - t * 100), int(100 - t * 50))
                width = int((5 - i) * 2)
                mid_x = sx - (flame_length * t) * math.sin(angle_rad)
                mid_y = sy + (flame_length * t) * math.cos(angle_rad)
                pygame.draw.circle(self.screen, color, (int(mid_x), int(mid_y)), width)

        # Защитный экран (если включен)
        if lander.radiation_shield:
            pygame.draw.circle(self.screen, (100, 200, 255, 100), (sx, sy), 35, 2)
            pygame.draw.circle(self.screen, (100, 200, 255, 50), (sx, sy), 30)

    def draw_trajectory(self, lander: Lander):
        """Отрисовка пройденного пути"""
        if len(lander.trajectory) > 1:
            points = [self.world_to_screen(x, y) for x, y in lander.trajectory]
            pygame.draw.lines(self.screen, (255, 255, 0, 128), False, points, 2)

    def draw_target(self):
        """Отрисовка целевой точки посадки"""
        x, y = self.world_to_screen(self.target_x, terrain.get_height(self.target_x))

        # Пульсирующий круг
        pulse = 1 + 0.2 * math.sin(pygame.time.get_ticks() / 200)
        radius = int(20 * self.scale * pulse)

        pygame.draw.circle(self.screen, self.COLOR_TARGET, (x, y - 20), radius, 2)
        pygame.draw.line(self.screen, self.COLOR_TARGET, (x - 20, y - 20), (x + 20, y - 20), 2)
        pygame.draw.line(self.screen, self.COLOR_TARGET, (x, y - 40), (x, y), 2)

        # Текст
        text = self.font.render("ЦЕЛЬ", True, self.COLOR_TARGET)
        self.screen.blit(text, (x + 25, y - 30))

    def draw_ui(self, lander: Lander, terrain: Terrain):
        """Отрисовка телеметрии и панели управления"""
        # Фон панели
        panel_rect = pygame.Rect(10, 10, 350, 400)
        pygame.draw.rect(self.screen, (0, 0, 0, 200), panel_rect)
        pygame.draw.rect(self.screen, (50, 50, 50), panel_rect, 2)

        y_offset = 20
        line_height = 25

        def draw_line(label: str, value: str, color=None, warning=False, danger=False):
            nonlocal y_offset
            if color is None:
                if danger:
                    color = self.COLOR_DANGER
                elif warning:
                    color = self.COLOR_WARNING
                else:
                    color = self.COLOR_TEXT

            text = self.font.render(f"{label:<20} {value:>15}", True, color)
            self.screen.blit(text, (20, y_offset))
            y_offset += line_height

        # Телеметрия
        terrain_h = terrain.get_height(lander.pos.x)
        altitude = lander.pos.y - terrain_h

        draw_line("=== ТЕЛЕМЕТРИЯ ===", "")
        draw_line("Высота:", f"{altitude:7.1f} м")
        draw_line("Верт. скорость:", f"{lander.vel.y:7.1f} м/с",
                  warning=abs(lander.vel.y) > 5, danger=abs(lander.vel.y) > 10)
        draw_line("Гориз. скорость:", f"{lander.vel.x:7.1f} м/с",
                  warning=abs(lander.vel.x) > 1.5, danger=abs(lander.vel.x) > 3)

        # Перегрузка
        thrust_acc = (lander.thrust_level * Physics.MAX_THRUST) / lander.mass
        total_g = math.sqrt(thrust_acc ** 2 + Physics.G_EUROPA ** 2) / Physics.G0
        draw_line("Перегрузка:", f"{total_g:7.2f} g",
                  warning=total_g > 3, danger=total_g > 6)

        draw_line("Масса:", f"{lander.mass:7.1f} кг")
        draw_line("Топливо:", f"{lander.fuel:7.1f} кг ({100 * lander.fuel / Physics.MAX_FUEL:4.1f}%)",
                  warning=lander.fuel < 300, danger=lander.fuel < 100)

        # Радиация
        rad_percent = lander.radiation_dose / Physics.RAD_THRESHOLD * 100
        draw_line("Радиация:", f"{lander.radiation_dose / 1000:7.1f} кран ({rad_percent:4.2f}%)",
                  warning=rad_percent > 50, danger=rad_percent > 80)

        # Температура
        draw_line("Температура:", f"{lander.temperature:7.1f} °C",
                  warning=lander.temperature < -170 or lander.temperature > 30,
                  danger=lander.temperature < -175 or lander.temperature > 45)

        draw_line("Уклон:", f"{terrain.get_slope(lander.pos.x):7.1f}°")

        y_offset += 10
        draw_line("=== УПРАВЛЕНИЕ ===", "")
        draw_line("[↑/↓] Тяга:", f"{lander.thrust_level * 100:5.0f}%")
        draw_line("[←/→] Угол:", f"{lander.angle:5.1f}°")
        draw_line("[R] Защита:", "ВКЛ" if lander.radiation_shield else "ВЫКЛ")
        draw_line("[T] Цель:", f"{self.target_x:.0f} м")
        draw_line("[Пробел] Сброс", "")

        # Статус
        y_offset += 20
        if lander.landed:
            color = (0, 255, 0)
            status = f"УСПЕХ: {lander.landing_reason}"
        elif lander.crashed:
            color = (255, 0, 0)
            status = f"АВАРИЯ: {lander.landing_reason}"
        else:
            if altitude < 100:
                color = self.COLOR_WARNING
                status = "ВНИМАНИЕ: МАЛАЯ ВЫСОТА"
            else:
                color = self.COLOR_TEXT
                status = "ПОСАДКА В ПРОЦЕССЕ"

        text = self.font_big.render(status, True, color)
        self.screen.blit(text, (20, self.height - 50))

        # Указатели
        if not lander.landed and not lander.crashed:
            # Стрелка к цели
            dx = self.target_x - lander.pos.x
            if abs(dx) > 50:
                arrow_x = self.width // 2 + (100 if dx > 0 else -100)
                arrow_y = 100
                pygame.draw.polygon(self.screen, self.COLOR_TARGET, [
                    (arrow_x, arrow_y),
                    (arrow_x - 10, arrow_y - 10 if dx > 0 else arrow_y + 10),
                    (arrow_x - 10, arrow_y + 10 if dx > 0 else arrow_y - 10)
                ])


def main():
    """Основной цикл симуляции"""
    global terrain

    # Инициализация объектов
    terrain = Terrain(width=10000, resolution=1000)
    lander = Lander()
    visualizer = Visualizer()

    clock = pygame.time.Clock()
    running = True
    paused = False

    print("Europa Lander Simulator запущен")
    print("Управление:")
    print("  ВВЕРХ/ВНИЗ - регулировка тяги")
    print("  ВЛЕВО/ВПРАВО - угол наклона")
    print("  R - радиационная защита")
    print("  T - смена целевой точки (клик мышью)")
    print("  ПРОБЕЛ - перезапуск")
    print("  P - пауза")

    while running:
        keys = pygame.key.get_pressed()
        time_warp = 10 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1
        dt = (1 / 60.0) * time_warp # Фиксированный шаг для стабильности физики

        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Перезапуск
                    lander = Lander()
                    print("Симуляция перезапущена")
                elif event.key == pygame.K_r:
                    lander.radiation_shield = not lander.radiation_shield
                    print(f"Радиационная защита: {'ВКЛ' if lander.radiation_shield else 'ВЫКЛ'}")
                elif event.key == pygame.K_t:
                    # Режим выбора цели
                    print("Кликните на экран для выбора точки посадки...")
                elif event.key == pygame.K_p:
                    paused = not paused

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Левая кнопка
                    # Вычисление мировых координат
                    mx, my = event.pos
                    world_x = visualizer.camera_x + (mx - visualizer.offset_x) / visualizer.scale
                    if 0 <= world_x <= 10000:
                        visualizer.target_x = world_x
                        print(f"Новая цель посадки: {world_x:.0f} м")

        if not paused and not lander.landed and not lander.crashed:
            # Управление с клавиатуры
            keys = pygame.key.get_pressed()

            if keys[pygame.K_UP]:
                lander.thrust_level = min(1.0, lander.thrust_level + 0.01)
            if keys[pygame.K_DOWN]:
                lander.thrust_level = max(0.0, lander.thrust_level - 0.01)
            if keys[pygame.K_LEFT]:
                lander.angle = min(45, lander.angle + 0.5)
            if keys[pygame.K_RIGHT]:
                lander.angle = max(-45, lander.angle - 0.5)

            # Физический шаг (4 подшага для стабильности)
            sub_steps = 4
            for _ in range(sub_steps):
                lander.update(dt / sub_steps, terrain)

        # Визуализация
        visualizer.screen.fill((5, 5, 15))  # Космический фон
        visualizer.update_camera(lander)
        visualizer.draw_terrain(terrain)
        visualizer.draw_target()
        visualizer.draw_trajectory(lander)
        visualizer.draw_lander(lander)
        visualizer.draw_ui(lander, terrain)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()