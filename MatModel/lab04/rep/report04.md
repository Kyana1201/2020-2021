---
# Front matter
lang: ru-RU
title: "Отчет лабораторной № 4"
subtitle: "Модель гармонических колебаний"
author: "Хохлачева Яна Дмитриевна"

# Formatting
toc-title: "Содержание"
toc: true # Table of contents
toc_depth: 2
lof: true # List of figures
lot: true # List of tables
fontsize: 12pt
linestretch: 1.5
papersize: a4paper
documentclass: scrreprt
polyglossia-lang: russian
polyglossia-otherlangs: english
mainfont: PT Serif
romanfont: PT Serif
sansfont: PT Sans
monofont: PT Mono
mainfontoptions: Ligatures=TeX
romanfontoptions: Ligatures=TeX
indent: true
pdf-engine: lualatex
header-includes:
  - \linepenalty=10 # the penalty added to the badness of each line within a paragraph (no associated penalty node) Increasing the value makes tex try to have fewer lines in the paragraph.
  - \interlinepenalty=0 # value of the penalty (node) added after each line of a paragraph.
  - \hyphenpenalty=50 # the penalty for line breaking at an automatically inserted hyphen
  - \exhyphenpenalty=50 # the penalty for line breaking at an explicit hyphen
  - \binoppenalty=700 # the penalty for breaking a line at a binary operator
  - \relpenalty=500 # the penalty for breaking a line at a relation
  - \clubpenalty=150 # extra penalty for breaking after first line of a paragraph
  - \widowpenalty=150 # extra penalty for breaking before last line of a paragraph
  - \displaywidowpenalty=50 # extra penalty for breaking before last line before a display math
  - \brokenpenalty=100 # extra penalty for page breaking after a hyphenated line
  - \predisplaypenalty=10000 # penalty for breaking before a display
  - \postdisplaypenalty=0 # penalty for breaking after a display
  - \floatingpenalty = 20000 # penalty for splitting an insertion (can only be split footnote in standard LaTeX)
  - \raggedbottom # or \flushbottom
  - \usepackage{float} # keep figures where there are in the text
  - \floatplacement{figure}{H} # keep figures where there are in the text
---

# Цель работы
* Решить уравнения гармонического осциллятора.
* Построить фазовый портрет гармонического осцилятора.

# Задание

Постройте фазовый портрет гармонического осциллятора и решение уравнения
гармонического осциллятора для следующих случаев

1. Колебания гармонического осциллятора без затуханий и без действий внешней силы $\ddot {x} + 4.3x = 0$

2. Колебания гармонического осциллятора c затуханием и без действий внешней силы $\ddot {x} + 6 \dot {x} + 5x = 0$

3. Колебания гармонического осциллятора c затуханием и под действием внешней силы $\ddot {x} + 10 \dot {x} + 9x = 8sin(7t)$

На интервале $t \in [0; 80]$(шаг 0.05) с начальными условиями $x_0 = 0.8, y_0 = -1.2$



# Выполнение лабораторной работы

$t$ — время

$w$ — частота

$\gamma$ — затухание

   Обозначения:

$$ \ddot{x} = \frac{\partial^2 x}{\partial t^2}, \dot{x} = \frac{\partial x}{\partial t}$$


   При отсутствии потерь в системе получаем уравнение консервативного осциллятора, энергия колебания которого сохраняется во времени:

$$ \ddot {x} + w_0^2x = 0 $$


   Для однозначной разрешимости уравнения второго порядка необходимо задать два начальных условия вида:

$$ \begin{cases} x(t_0) = x_0 \\ \dot{x}(t_0) = y_0 \end{cases} $$

   Уравнение второго порядка можно представить в виде системы двух уравнений первого порядка:

$$ \begin{cases} \dot{x} = y \\ \dot{y} = -w_0^2x \end{cases} $$

   Начальные условия для системы примут вид:

$$ \begin{cases} x(t_0) = x_0 \\ y(t_0) = y_0 \end{cases} $$

   Независимые переменные x, y определяют пространство, в котором «движется» решение. Это фазовое пространство системы, поскольку оно двумерно будем называть его фазовой плоскостью.
   
   Значение фазовых координат x, y в любой момент времени полностью определяет состояние системы. Решению уравнения движения как функции времени отвечает гладкая кривая в фазовой плоскости. Она называется фазовой траекторией. Если множество различных решений (соответствующих различным начальным условиям) изобразить на одной фазовой плоскости, возникает общая картина поведения системы. Такую картину, образованную набором фазовых траекторий, называют фазовым портретом.

# Код на Python: 
```
import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Колебания гармонического осциллятора без затуханий и без действий внешней силы
#Параметры осциллятора

w = math.sqrt(4.3)
g = 0.00

#Правая часть уравнения f(t)

def f(t):
  f = 0
  return f

#Вектор-функция f(t, x) для решения системы дифференциальных уравнений x' = y(t, x), где x - искомый вектор

def y(x,t):
  dx1 = x[1]
  dx2 = -w*w*x[0] - g*x[1] - f(t)
  return dx1,dx2

# Вектор начальных условий x(t0) = x0
x0 = np.array([0.8, -1.2])

#Интервал на котором будет решаться задача
t = np.arange(0, 80, 0.05)

#Решаем дифференциальные уравнения с начальным условием x(t0) = x0 на интервале t с правой частью, заданной y и записываем решение в матрицу x
x = odeint(y, x0,t)

# Перепишим отдельно
y1 = x[:, 0]
y2 = x[:, 1]

plt.plot(y1, y2)
plt.grid(axis='both')
plt.show()

#Колебания гармонического осциллятора c затуханием и без действий внешней силы

w2 = math.sqrt(6)
g2 = 5

#Правая часть уравнения f(t)

def f2(t_2):
  f2 = 0
  return f2

#Вектор-функция f(t, x) для решения системы дифференциальных уравнений x' = y(t, x), где x - искомый вектор

def y22(x_2,t_2):
  dx_21 = x_2[1]
  dx_22 = -w2 * w2 * x_2[0] - 2*g2 * x_2[1] - f2(t_2)
  return dx_21, dx_22

# Вектор начальных условий x(t0) = x0
x_20 = np.array([0.8, -1.2])

#Интервал на котором будет решаться задача
t_2 = np.arange(0, 80, 0.05)

#Решаем дифференциальные уравнения с начальным условием x(t0) = x0 на интервале t с правой частью, заданной y и записываем решение в матрицу x
x_2 = odeint(y22, x_20,t_2)

# Перепишим отдельно
y_21 = x_2[:, 0]
y_22 = x_2[:, 1]

plt.plot(y_21, y_22)
plt.grid(axis='both')
plt.show()

#Колебания гармонического осциллятора c затуханием и под действием внешней силы

w3 = math.sqrt(10)
g3 = 9

#Правая часть уравнения f(t)

def f3(t_3):
  f3 = 8*np.sin(7*t_3)
  return f3

#Вектор-функция f(t, x) для решения системы дифференциальных уравнений x' = y(t, x), где x - искомый вектор

def y33(x_3,t_3):
  dx_31 = x_3[1]
  dx_32 = -w3 * w3 * x_3[0] - 2*g3 * x_3[1] - f3(t_3)
  return dx_31, dx_32

# Вектор начальных условий x(t0) = x0
x_30 = np.array([0.8, -1.2])

#Интервал на котором будет решаться задача
t_3 = np.arange(0, 80, 0.05)

#Решаем дифференциальные уравнения с начальным условием x(t0) = x0 на интервале t с правой частью, заданной y и записываем решение в матрицу x
x_3 = odeint(y33, x_30,t_3)

# Перепишим отдельно
y_31 = x_3[:, 0]
y_32 = x_3[:, 1]

plt.plot(y_31, y_32)
plt.grid(axis='both')
plt.show()
```




(рис. fig:001)

![Колебания гармонического осциллятора без затуханий и без действий внешней
силы](image/Fig1.jpg){ #fig:001 width=70% }


(рис. fig:002)

![Колебания гармонического осциллятора c затуханием и без действий внешней
силы](image/Fig2.jpg){ #fig:002 width=70% }


(рис. -@fig:003)

![Колебания гармонического осциллятора c затуханием и под действием внешней
силы](image/Fig3.jpg){ #fig:003 width=70% }


# Ответы на вопросы

1. Запишите простейшую модель гармонических колебаний

Простейшая модель гармонических колебаний имеет следующий вид: $$ x = x_m cos(\omega t + \phi_0) $$

2. Дайте определение осциллятора

Осциллятор - система, совершающая колебания, показатели которой периодически повторяются во времени.

3. Запишите модель математического маятника

$$\frac{\partial^2 \alpha}{\partial t^2} + \frac{\gamma}{L} sin{\alpha} = 0$$ 

4. Запишите алгоритм перехода от дифференциального уравнения второго порядка к двум дифференциальным уравнениям первого порядка

Пусть у нас есть дифференциальное уравнение 2-го порядка: $$ \ddot{x} + w_0^2x = f(t) $$ 
Для перехода к системе уравнений первого порядка сделаем замену (это метод Ранге-Кутты):
$$ y = \dot{x} $$ 
Тогда получим систему уравнений: $$ \begin{cases} y = \dot{x} \ \dot{y} = - w_0^2x \end{cases} $$
 
5. Что такое фазовый портрет и фазовая траектория?

Фазовый портрет — это то, как величины, описывающие состояние системы, зависят друг от друга.

Фазовая траектория — кривая в фазовом пространстве, составленная из точек, представляющих состояние динамической системы в последовательные моменты времени в течение всего времени эволюции.



# Выводы

Построила фазовый портрет гармонического осциллятора и решила уравнения гармонического осциллятора:

1. Колебания гармонического осциллятора без затуханий и без действий внешней силы.

2. Колебания гармонического осциллятора c затуханием и без действий внешней силы.

3. Колебания гармонического осциллятора c затуханием и под действием внешней силы.