# Настройка GPU и облако

> Обучать на CPU — нормально для учёбы. Обучать по-настоящему — нужен GPU.

**Тип:** Практика
**Языки:** Python
**Пресреквизиты:** Фаза 0, Урок 01
**Время:** ~45 минут

## Цели обучения

- Проверить наличие локального GPU с помощью `nvidia-smi` и CUDA API PyTorch
- Настроить Google Colab с GPU T4 для бесплатных облачных экспериментов
- Сравнить скорость матричного умножения на CPU и GPU и измерить ускорение
- Оценить максимальный размер модели, помещающейся в VRAM, используя правило fp16

## Проблема

Большинство уроков в фазах 1–3 нормально работают на CPU. Но когда вы начнёте обучать CNN, трансформеры или LLM (фазы 4+), вам понадобится ускорение GPU. Обучение, которое занимает 8 часов на CPU, занимает 10 минут на GPU.

У вас три варианта: локальный GPU, облачный GPU или Google Colab (бесплатно).

## Концепция

```
Ваши варианты:

1. Локальный NVIDIA GPU
   Стоимость: $0 (уже есть)
   Настройка: Установить CUDA + cuDNN
   Лучше всего для: постоянного использования, больших датасетов

2. Google Colab (бесплатный уровень)
   Стоимость: $0
   Настройка: Не нужна
   Лучше всего для: быстрых экспериментов, нет GPU дома

3. Облачный GPU (Lambda, RunPod, Vast.ai)
   Стоимость: $0.20–2.00/час
   Настройка: SSH + установка
   Лучше всего для: серьёзного обучения, больших моделей
```

## Реализация

### Вариант 1: Локальный NVIDIA GPU

Проверьте, есть ли он:

```bash
nvidia-smi
```

Установите PyTorch с CUDA:

```python
import torch

print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"Версия CUDA: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Вариант 2: Google Colab

1. Перейдите на [colab.research.google.com](https://colab.research.google.com)
2. Среда выполнения > Сменить тип среды > T4 GPU
3. Запустите `!nvidia-smi` для проверки

Загружайте ноутбуки из этого курса напрямую в Colab.

### Вариант 3: Облачный GPU

Для Lambda Labs, RunPod или Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Нет GPU? Не беда.

Большинство уроков работают на CPU. Те, которым нужен GPU, будут об этом говорить и включать ссылки на Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется: {device}")
```

## Практика: Бенчмарк GPU vs CPU

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}с")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}с")
    print(f"Ускорение: {cpu_time / gpu_time:.0f}x")
```

## Упражнения

1. Запустите бенчмарк выше и сравните время CPU и GPU
2. Если у вас нет GPU, запустите его в Google Colab и сравните
3. Проверьте, сколько у вас видеопамяти, и оцените максимальный размер модели (правило: 2 байта на параметр для fp16)

## Ключевые термины

| Термин | Что говорят | Что это на самом деле |
|--------|-------------|----------------------|
| CUDA | «Программирование GPU» | Параллельная вычислительная платформа NVIDIA для запуска кода на GPU |
| VRAM | «Память GPU» | Видеопамять на GPU, отдельная от системной RAM. Ограничивает размер модели. |
| fp16 | «Половинная точность» | 16-битное число с плавающей точкой, использует вдвое меньше памяти, чем fp32, с минимальной потерей точности |
| Tensor Core | «Быстрое матричное железо» | Специализированные ядра GPU для матричного умножения, в 4–8 раз быстрее обычных ядер |
