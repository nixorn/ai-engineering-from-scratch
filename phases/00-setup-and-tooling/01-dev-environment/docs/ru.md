# Среда разработки

> Инструменты формируют мышление. Настрой их один раз — и настрой правильно.

**Тип:** Практика
**Языки:** Python, Node.js, Rust
**Необходимые знания:** Нет
**Время:** ~45 минут

## Цели обучения

- Установить с нуля цепочки инструментов Python 3.11+, Node.js 20+ и Rust
- Настроить виртуальные окружения и пакетные менеджеры для воспроизводимых сборок
- Проверить доступ к GPU через CUDA/MPS и выполнить тестовую операцию с тензором
- Понять четырёхуровневый стек: система, пакеты, рантаймы, AI-библиотеки

## Проблема

Тебе предстоит изучать AI-инжиниринг через 200+ уроков с использованием Python, TypeScript, Rust и Julia. Если окружение сломано — каждый урок превращается в борьбу с инструментами вместо обучения.

Большинство людей пропускают настройку окружения. Потом тратят часы на отладку ошибок импорта, конфликтов версий и отсутствующих CUDA-драйверов. Мы сделаем это один раз — и правильно.

## Концепция

Среда AI-инжиниринга состоит из четырёх уровней:

```mermaid
graph TD
    A["4. AI/ML-библиотеки\nPyTorch, JAX, transformers и др."] --> B["3. Рантаймы языков\nPython 3.11+, Node 20+, Rust, Julia"]
    B --> C["2. Пакетные менеджеры\nuv, pnpm, cargo, juliaup"]
    C --> D["1. Системный фундамент\nОС, оболочка, git, редактор, GPU-драйверы"]
```

Установка идёт снизу вверх. Каждый уровень зависит от предыдущего.

## Сборка

### Шаг 1: Системный фундамент

Проверь систему и установи основное.

```bash
# macOS
xcode-select --install
brew install git curl wget

# Ubuntu/Debian
sudo apt update && sudo apt install -y build-essential git curl wget

# Windows (используй WSL2)
wsl --install -d Ubuntu-24.04
```

### Шаг 2: Python с uv

Мы используем `uv` — он в 10–100 раз быстрее pip и автоматически управляет виртуальными окружениями.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv python install 3.12

uv venv
source .venv/bin/activate  # или .venv\Scripts\activate на Windows

uv pip install numpy matplotlib jupyter
```

Проверка:

```python
import sys
print(f"Python {sys.version}")

import numpy as np
print(f"NumPy {np.__version__}")
a = np.array([1, 2, 3])
print(f"Вектор: {a}, скалярное произведение с собой: {np.dot(a, a)}")
```

### Шаг 3: Node.js с pnpm

Для уроков на TypeScript (агенты, MCP-серверы, веб-приложения).

```bash
curl -fsSL https://fnm.vercel.app/install | bash
fnm install 22
fnm use 22

npm install -g pnpm

node -e "console.log('Node', process.version)"
```

### Шаг 4: Rust

Для уроков с высокими требованиями к производительности (инференс, системное программирование).

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rustc --version
cargo --version
```

### Шаг 5: Julia (Необязательно)

Для математически насыщенных уроков, где Julia особенно хороша.

```bash
curl -fsSL https://install.julialang.org | sh

julia -e 'println("Julia ", VERSION)'
```

### Шаг 6: Настройка GPU (если есть)

```bash
# NVIDIA
nvidia-smi

# Установить PyTorch с CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

```python
import torch
print(f"CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Нет GPU? Не страшно. Большинство уроков работают на CPU. Для уроков с тяжёлым обучением используй Google Colab или облачные GPU.

### Шаг 7: Проверка всего

Запусти скрипт верификации:

```bash
python phases/00-setup-and-tooling/01-dev-environment/code/verify.py
```

## Применение

Среда готова для каждого урока курса. Вот где что используется:

| Язык | Используется в | Пакетный менеджер |
|------|---------------|-------------------|
| Python | Фазы 1–12 (ML, DL, NLP, Vision, Audio, LLMs) | uv |
| TypeScript | Фазы 13–17 (Инструменты, Агенты, Рои, Инфра) | pnpm |
| Rust | Фазы 12, 15–17 (Системы с требованиями к производительности) | cargo |
| Julia | Фаза 1 (Математические основы) | Pkg |

## Результат

Этот урок создаёт скрипт верификации, который любой может запустить для проверки своей установки.

См. `outputs/prompt-env-check.md` — промпт, помогающий AI-ассистентам диагностировать проблемы с окружением.

## Упражнения

1. Запусти скрипт верификации и устрани все ошибки
2. Создай виртуальное окружение Python для этого курса и установи PyTorch
3. Напиши «hello world» на всех четырёх языках и запусти каждый
