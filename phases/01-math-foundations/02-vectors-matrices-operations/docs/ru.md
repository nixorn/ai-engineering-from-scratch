# Векторы, матрицы и операции

> Любая нейросеть — это просто умножение матриц с дополнительными шагами.

**Тип:** Практика  
**Языки:** Python, Julia  
**Требования:** Фаза 1, Урок 01 (Интуиция линейной алгебры)  
**Время:** ~60 минут

## Цели обучения

- Построить класс Matrix с поэлементными операциями, умножением матриц, транспонированием, определителем и обратной матрицей
- Различать поэлементное умножение и умножение матриц и объяснять, когда используется каждое
- Реализовать один полносвязный слой нейросети (`relu(W @ x + b)`), используя только самописный класс Matrix
- Объяснить правила broadcasting и то, как в фреймворках нейросетей работает добавление bias

## Проблема

Вы хотите построить нейросеть. Открываете код и видите:

```
output = activation(weights @ input + bias)
```

Этот `@` — умножение матриц. `weights` — матрица. `input` — вектор. Если не понимать, что делают эти операции, строка выглядит магией. Если понимать, то это весь прямой проход слоя в трех операциях.

Каждое изображение, которое обрабатывает модель, — это матрица значений пикселей. Каждый эмбеддинг слова — это вектор. Каждый слой любой нейросети — это матричное преобразование. Нельзя строить AI-системы, не владея матричными операциями так же уверенно, как нельзя писать код без понимания переменных.

Этот урок развивает такое владение с нуля.

## Концепция

### Векторы: упорядоченные списки чисел

Вектор — это список чисел с направлением и длиной. В AI векторы представляют точки данных, признаки или параметры.

```
v = [3, 4]        -- 2D-вектор
w = [1, 0, -2]    -- 3D-вектор
```

2D-вектор `[3, 4]` указывает на координаты (3, 4) на плоскости. Его длина (модуль) равна 5 (треугольник 3-4-5).

### Матрицы: таблицы чисел

Матрица — это двумерная таблица. Строки и столбцы. Матрица размера m x n имеет m строк и n столбцов.

```
A = | 1  2  3 |     -- матрица 2x3 (2 строки, 3 столбца)
    | 4  5  6 |
```

В нейросетях матрицы весов преобразуют входные векторы в выходные. Слой с 784 входами и 128 выходами использует матрицу весов 128x784.

### Почему важны shape

У умножения матриц есть строгое правило: `(m x n) @ (n x p) = (m x p)`. Внутренние размерности должны совпадать.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Внутренние размерности: 784 = 784  -- корректно
```

Если в PyTorch вы получаете ошибку несовпадения shape, причина именно в этом.

### Карта операций

| Операция | Что делает | Использование в нейросетях |
|-----------|------------|-----------------------------|
| Сложение | Поэлементное объединение | Добавление bias к выходу |
| Умножение на скаляр | Масштабирует каждый элемент | Learning rate * gradients |
| Умножение матриц | Преобразует векторы | Прямой проход слоя |
| Транспонирование | Меняет местами строки и столбцы | Обратное распространение |
| Определитель | Сводка матрицы в одно число | Проверка обратимости |
| Обратная матрица | Отменяет преобразование | Решение линейных систем |
| Единичная матрица | Ничего не меняет | Инициализация, residual connections |

### Поэлементное и матричное умножение

Это различие постоянно путает новичков.

Поэлементное: умножаются элементы на одинаковых позициях. Обе матрицы должны иметь одинаковый shape.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Умножение матриц: скалярные произведения строк и столбцов. Внутренние размерности должны совпадать.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Разные операции, разные результаты, разные правила.

### Broadcasting

Когда вы добавляете вектор bias к матрице выходов, shape не совпадают. Broadcasting растягивает меньший массив под больший.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting растягивает вектор по строкам:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Все современные фреймворки делают это автоматически. Понимание этого убирает путаницу, когда shape кажутся неправильными, но код работает.

## Собери это

### Шаг 1: Класс Vector

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Шаг 2: Класс Matrix с основными операциями

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Шаг 3: Смотрим, как это работает

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Шаг 4: Связь с нейросетями

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

Это один полносвязный слой: `output = relu(W @ x + b)`. Каждый dense-слой в любой нейросети делает ровно это.

## Применяй

NumPy делает все вышеперечисленное в меньшем количестве строк и на порядки быстрее.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Оператор `@` в Python вызывает `__matmul__`. NumPy реализует его через оптимизированные BLAS-рутины на C и Fortran. Та же математика, в 100 раз быстрее.

Broadcasting в NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy автоматически растягивает одномерный bias по обеим строкам. Именно так работает добавление bias в любом фреймворке нейросетей.

## Ship It

Этот урок создает промпт для обучения матричным операциям через геометрическую интуицию. См. `outputs/prompt-matrix-operations.md`.

Класс Matrix, который вы построили здесь, — основа мини-фреймворка нейросети, который мы создаем в Фазе 3, Уроке 10.

## Упражнения

1. **Проверьте обратную матрицу.** Перемножьте `A @ A.inverse_2x2()` и убедитесь, что получилась единичная матрица. Попробуйте с тремя разными матрицами 2x2. Что происходит, когда определитель равен нулю?

2. **Реализуйте обратную 3x3.** Расширьте класс Matrix, чтобы вычислять обратные для матриц 3x3 методом присоединенной матрицы. Сверьте результат с `np.linalg.inv` из NumPy.

3. **Соберите двухслойную сеть.** Используя только ваш класс Matrix (без NumPy), создайте двухслойную нейросеть: вход (3) -> скрытый слой (4) -> выход (2). Инициализируйте случайные веса, выполните прямой проход и проверьте, что все shape корректны.

## Ключевые термины

| Термин | Что обычно говорят | Что это реально значит |
|------|---------------------|-------------------------|
| Вектор | "Стрелка" | Упорядоченный список чисел. В AI: точка в пространстве высокой размерности. |
| Матрица | "Таблица чисел" | Линейное преобразование. Оно отображает векторы из одного пространства в другое. |
| Умножение матриц | "Просто перемножить числа" | Скалярные произведения каждой строки первой матрицы и каждого столбца второй. Порядок важен. |
| Транспонирование | "Перевернуть" | Поменять местами строки и столбцы. Превращает матрицу m x n в n x m. Критично для backpropagation. |
| Определитель | "Какое-то число из матрицы" | Показывает, во сколько матрица масштабирует площадь (2D) или объем (3D). Ноль означает, что преобразование схлопывает размерность. |
| Обратная матрица | "Отменяет матрицу" | Матрица, которая обращает преобразование. Существует только когда определитель не равен нулю. |
| Единичная матрица | "Скучная матрица" | Матричный аналог умножения на 1. Используется в residual connections (ResNets). |
| Broadcasting | "Магическое исправление shape" | Растягивание меньшего массива до большего путем повторения по отсутствующим размерностям. |
| Поэлементно | "Обычное умножение" | Умножение элементов на одинаковых позициях. Оба массива должны иметь одинаковый shape (или быть совместимыми для broadcasting). |

## Дополнительное чтение

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - визуальная интуиция для всех операций, рассмотренных здесь
- [Документация NumPy по broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - точные правила, которым следует NumPy
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - краткий справочник по линейной алгебре для ML
