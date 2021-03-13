---
# Front matter
lang: ru-RU
title: "Отчет лабораторной № 5"
subtitle: "Модель хищник-жертва"
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

1. Построить график зависимости $x$ от $y$ и графики функций $x(t), y(t)$

2. Найти стационарное состояние системы

# Задание

Для модели "хищник-жертвва":

$$
\begin{cases}
    \frac{\partial x}{\partial t} = -0.57x(t)+0.047x(t)y(t)
    \\
    \frac{\partial y}{\partial t} = 0.37y(t)-0.027x(t)y(t)
\end{cases}
$$

Построить график зависимости численности хищников от численности жертв,а также графики изменения численности хищников и численности жертв при 
следующих начальных условиях: $x_0 = 11, y_0 = 36$

Найдите стационарное состояние системы.


# Выполнение лабораторной работы

Простейшая модель взаимодействия двух видов типа «хищник-жертва» — модель Лотки-Вольтерры. Данная двувидовая модель основывается на следующих предположениях:
1. Численность популяции жертв x и хищников y зависят только от времени (модель не учитывает пространственное распределение популяции на занимаемой территории)
2. В отсутствии взаимодействия численность видов изменяется по модели Мальтуса (по экспоненциальному закону), при этом число жертв увеличивается, а число хищников падает
3. Естественная смертность жертвы и естественная рождаемость хищника считаются несущественными
4. Эффект насыщения численности обеих популяций не учитывается
5. Скорость роста численности жертв уменьшается пропорционально численности хищников

$$
\begin{cases}
    \frac{\partial x}{\partial t} = -ax(t)+bx(t)y(t)
    \\
    \frac{\partial y}{\partial t} = cy(t)-dx(t)y(t)
\end{cases}
$$

В этой модели $x$ – число жертв, y - число хищников. Коэффициент $a$ описывает скорость естественного прироста числа жертв в отсутствие хищников, $с$ - естественное вымирание хищников, лишенных пищи в виде жертв. Вероятность взаимодействия жертвы и хищника считается пропорциональной как количеству жертв, так и числу самих хищников $(xy)$. Каждый акт взаимодействия уменьшает популяцию жертв, но способствует увеличению популяции хищников (члены $-bxy$ и $dxy$ в правой части уравнения). 

Математический анализ этой (жесткой) модели показывает, что имеется стационарное состояние (положение равновесия, не зависящее от времени решения). Если начальное состояние будет другим, то это приведет к периодическому колебанию численности как жертв, так и хищников, так что по прошествии некоторого времени система возвращается в начальное состояние.
Стационарное состояние системы будет в точке: $x_0 = \frac{c}{d}, y_0 = \frac{a}{b}$

Если начальные значения задать в стационарном состоянии $x(0)=x_0, y(0)=y_0$, то в любой момент времени численность популяций изменяться не будет. При малом отклонении от положения равновесия численности как хищника, так и жертвы с течением времени не возвращаются к равновесным значениям, а совершают периодические колебания вокруг стационарной точки. Амплитуда колебаний и их период определяется начальными значениями численностей $x(0), y(0)$. Колебания совершаются в
противофазе.как функции времени отвечает гладкая кривая в фазовой плоскости. Она называется фазовой траекторией. Если множество различных решений (соответствующих различным начальным условиям) изобразить на одной фазовой плоскости, возникает общая картина поведения системы. Такую картину, образованную набором фазовых траекторий, называют фазовым портретом.

# Код на Python: 

```


a = 0.57 #коэффициент естественной смертности хищников
b = 0.37 #коэффициент естественного прироста жертв
c = 0.047 #коэффициент увеличения числа хищников
d = 0.027 #коэффициент смертности жертв

def syst(x, t):
    dx0 = -a*x[0] + c*x[0]*x[1]
    dx1 = b*x[1] - d*x[0]*x[1]
    return dx0, dx1

x0 = [11, 36] #Начальные занечения x и y (популяция хищников и популяция жертв)

t = np.arange(0, 400, 0.1)

y = odeint(syst, x0, t)

y1 = y[:, 0]
y2 = y[:, 1]

plt.plot(t, y1, label='хищники') #построение графика колебаний изменения числа популяции хищников
plt.plot(t, y2, label='жертвы') #построение графика колебаний изменения числа популяции жертв
plt.legend()
plt.show()

#построение графика зависимости изменения численности хищников от изменения численности жертв
plt.plot(y1,y2)
plt.plot(11, 36, 'ro', label='Начальное состояние')
plt.plot(b/d, a/c, 'go', label='Стационарное состояние')
plt.legend()
plt.grid(axis='both')
plt.show()




```



(рис. fig:001)

![графики изменения численности популяции хищников и численности популяции жертв с течением времени](image/Fig001.jpg){ #fig:001 width=70% }


(рис. fig:002)

![График зависимости численности хищников от численности жертв и стационарное состояние](image/Fig002.jpg){ #fig:002 width=70% }



Для того, чтобы найти стационарное состояние системы, необходимо приравнять производные каждой из функций x и y к нулю и выразить значения y и x соответственно.
Получим следующие значения: $$ x_0 = \frac{b}{d} = \frac{0.37}{0.027} \approx 13.70 $$ $$ y_0 = \frac{a}{c} = \frac{0.57}{0.047} \approx 12.13 $$ При стационарном состоянии значения числа жертв и хищников не меняется во времени.

# Выводы


Ознакомилась с простейшей моделью взаимодействия двух видов типа «хищник — жертва», построив для нее графики и найдя стационарное состояние системы.